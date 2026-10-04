"""Causal Spectral Policy: coarse-to-fine regression of DCT action coefficients.

Implements paper Eq. (2)-(3)::

    c = F a,  c = [c_low; c_high] = [c_{0:lambda}; c_{lambda:K}]
    c_low_hat  = p_low(o)
    c_high_hat = p_high(o, sg(c_low_hat))
    L = ||c_low_hat - c_low||^2 + ||c_high_hat - c_high||^2

F is the orthonormal DCT-II along time, applied independently per spectral
action dimension of the normalized chunk. Gripper widths stay in the time
domain and are regressed by a separate head on the first-stage feature, so
the second stage is supervised by c_high alone (paper App. 7.4). Gripper
squared error is added with the same per-element weight, so the total equals
the per-element time-domain MSE of the reconstructed chunk (Parseval).
"""

import math
from collections import deque

import torch
import torch.nn.functional as F  # noqa: N812
from lerobot.policies.diffusion.modeling_diffusion import DiffusionRgbEncoder
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_STATE
from torch import Tensor, nn

from paw_baseline.csp.configuration_csp import CSPConfig


def dct_matrix(length: int) -> Tensor:
    """Orthonormal DCT-II basis F (K x K): c = F x, x = F^T c."""
    n = torch.arange(length, dtype=torch.float64)
    k = n[:, None]
    basis = torch.cos(math.pi / length * (n[None, :] + 0.5) * k)
    basis[0] *= math.sqrt(1.0 / length)
    basis[1:] *= math.sqrt(2.0 / length)
    return basis.float()


def _baku_weight_init(module: nn.Module) -> None:
    """BAKU's utils.weight_init for linear layers (orthogonal, zero bias)."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.fill_(0.0)


class _CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float):
        super().__init__()
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.dropout = dropout

    def forward(self, x: Tensor) -> Tensor:
        batch, length, channels = x.shape
        q, k, v = self.c_attn(x).split(channels, dim=2)
        shape = (batch, length, self.n_head, channels // self.n_head)
        q, k, v = (t.view(shape).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(batch, length, channels)
        return self.resid_dropout(self.c_proj(y))


class _Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = _CausalSelfAttention(n_embd, n_head, dropout)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class GPTTrunk(nn.Module):
    """BAKU's nanoGPT trunk: token projection, learned positions, causal blocks."""

    def __init__(self, config: CSPConfig):
        super().__init__()
        self.block_size = config.block_size
        self.wte = nn.Linear(config.repr_dim, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            _Block(config.n_embd, config.n_head, config.dropout)
            for _ in range(config.n_layer)
        )
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """nanoGPT init; CSPModel later applies BAKU's orthogonal Linear init."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def forward(self, tokens: Tensor) -> Tensor:
        length = tokens.shape[1]
        if length > self.block_size:
            raise ValueError(f"sequence of {length} tokens exceeds block size {self.block_size}")
        pos = torch.arange(length, device=tokens.device)
        x = self.drop(self.wte(tokens) + self.wpe(pos)[None])
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.ln_f(x))


def _mlp_head(config: CSPConfig, out_dim: int) -> nn.Sequential:
    """BAKU DeterministicHead mean network without tanh squashing.

    DCT coefficients of [-1, 1]-normalized chunks are not bounded by 1.
    """
    sizes = [config.n_embd] + [config.head_hidden_dim] * config.head_num_layers
    layers: list[nn.Module] = []
    for i in range(config.head_num_layers):
        layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU()]
    layers.append(nn.Linear(sizes[-1], out_dim))
    return nn.Sequential(*layers)


class CSPModel(nn.Module):
    def __init__(self, config: CSPConfig):
        super().__init__()
        self.config = config
        action_dim = config.action_feature.shape[0]
        self.time_dims = list(config.time_domain_action_dims)
        self.spec_dims = [d for d in range(action_dim) if d not in self.time_dims]
        self.horizon = config.horizon
        self.n_low = config.n_low_freq
        self.n_spec = len(self.spec_dims)
        self.n_time = len(self.time_dims)
        self.register_buffer("dct", dct_matrix(config.horizon), persistent=False)

        self.camera_keys = list(config.image_features)
        if config.use_separate_rgb_encoder_per_camera:
            self.rgb_encoder = nn.ModuleList(
                DiffusionRgbEncoder(config) for _ in self.camera_keys
            )
        else:
            self.rgb_encoder = DiffusionRgbEncoder(config)
        feature_dim = config.spatial_softmax_num_keypoints * 2
        state_dim = config.robot_state_feature.shape[0]

        # Heads/projections/trunks: BAKU Actor applies orthogonal init after GPT init.
        actor = nn.ModuleDict()
        actor["camera_proj"] = nn.ModuleList(
            nn.Linear(feature_dim, config.repr_dim) for _ in self.camera_keys
        )
        actor["state_proj"] = nn.Sequential(
            nn.Linear(state_dim, config.repr_dim),
            nn.ReLU(),
            nn.Linear(config.repr_dim, config.repr_dim),
        )
        actor["coef_proj"] = nn.Linear(self.n_spec, config.repr_dim)
        actor["low_trunk"] = GPTTrunk(config)
        actor["high_trunk"] = GPTTrunk(config)
        actor["low_head"] = _mlp_head(config, self.n_low * self.n_spec)
        actor["gripper_head"] = _mlp_head(config, self.horizon * self.n_time)
        actor["high_head"] = _mlp_head(
            config, (self.horizon - self.n_low) * self.n_spec
        )
        actor.apply(_baku_weight_init)
        self.actor = actor
        self.low_query = nn.Parameter(torch.randn(1, 1, config.repr_dim))
        self.high_query = nn.Parameter(torch.randn(1, 1, config.repr_dim))
        n_obs = len(self.camera_keys) + 1
        if n_obs + self.n_low + 1 > config.block_size:
            raise ValueError("CSP high-stage token sequence exceeds block_size")

    def observation_tokens(self, batch: dict[str, Tensor]) -> Tensor:
        tokens = []
        for i, key in enumerate(self.camera_keys):
            image = batch[key]
            if image.ndim == 5:  # (B, 1, C, H, W) from a single-step history
                image = image[:, -1]
            encoder = (
                self.rgb_encoder[i]
                if isinstance(self.rgb_encoder, nn.ModuleList)
                else self.rgb_encoder
            )
            tokens.append(self.actor["camera_proj"][i](encoder(image)))
        state = batch[OBS_STATE]
        if state.ndim == 3:
            state = state[:, -1]
        tokens.append(self.actor["state_proj"](state))
        return torch.stack(tokens, dim=1)

    def predict_coefficients(
        self, batch: dict[str, Tensor]
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return (c_low_hat, c_high_hat, time-domain dims hat) in normalized units."""
        obs = self.observation_tokens(batch)
        batch_size = obs.shape[0]
        low_in = torch.cat([obs, self.low_query.expand(batch_size, -1, -1)], dim=1)
        low_feat = self.actor["low_trunk"](low_in)[:, -1]
        c_low = self.actor["low_head"](low_feat).view(batch_size, self.n_low, self.n_spec)
        time_part = self.actor["gripper_head"](low_feat).view(
            batch_size, self.horizon, self.n_time
        )

        # Stop-gradient: fine supervision must not shape the coarse prediction.
        coef_tokens = self.actor["coef_proj"](c_low.detach())
        high_in = torch.cat(
            [obs, coef_tokens, self.high_query.expand(batch_size, -1, -1)], dim=1
        )
        high_feat = self.actor["high_trunk"](high_in)[:, -1]
        c_high = self.actor["high_head"](high_feat).view(
            batch_size, self.horizon - self.n_low, self.n_spec
        )
        return c_low, c_high, time_part

    def to_coefficients(self, actions: Tensor) -> Tensor:
        """(B, K, D_spec) time domain -> (B, K, D_spec) DCT coefficients."""
        return torch.einsum("kn,bnd->bkd", self.dct, actions)

    def to_actions(self, coefficients: Tensor) -> Tensor:
        return torch.einsum("kn,bkd->bnd", self.dct, coefficients)

    def assemble(self, c_low: Tensor, c_high: Tensor, time_part: Tensor) -> Tensor:
        spec = self.to_actions(torch.cat([c_low, c_high], dim=1))
        actions = spec.new_empty(spec.shape[0], self.horizon, self.n_spec + self.n_time)
        actions[..., self.spec_dims] = spec
        actions[..., self.time_dims] = time_part
        return actions

    def compute_loss(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        actions = batch[ACTION]
        if actions.shape[1] != self.horizon:
            raise ValueError(f"expected {self.horizon} actions, got {actions.shape[1]}")
        target = self.to_coefficients(actions[..., self.spec_dims])
        c_low, c_high, time_part = self.predict_coefficients(batch)
        se_low = (c_low - target[:, : self.n_low]).pow(2).sum(dim=(1, 2))
        se_high = (c_high - target[:, self.n_low :]).pow(2).sum(dim=(1, 2))
        se_time = (time_part - actions[..., self.time_dims]).pow(2).sum(dim=(1, 2))
        elements = self.horizon * (self.n_spec + self.n_time)
        per_sample = (se_low + se_high + se_time) / elements
        high_energy = target[:, self.n_low :].pow(2).sum()
        metrics = {
            "loss_low": (se_low.mean() / elements).item(),
            "loss_high": (se_high.mean() / elements).item(),
            "loss_gripper": (se_time.mean() / elements).item(),
            # < 1 when the fine stage beats predicting zero high coefficients.
            "high_rel_err": (se_high.sum() / high_energy.clamp_min(1e-12)).item(),
        }
        return per_sample, metrics


class CSPPolicy(PreTrainedPolicy):
    config_class = CSPConfig
    name = "csp"

    def __init__(self, config: CSPConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.model = CSPModel(config)
        self.reset()

    def get_optim_params(self):
        return self.parameters()

    def reset(self):
        self._action_queue = deque(maxlen=self.config.n_action_steps)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        c_low, c_high, time_part = self.model.predict_coefficients(batch)
        actions = self.model.assemble(c_low, c_high, time_part)
        return actions[:, : self.config.n_action_steps]

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean"):
        per_sample, metrics = self.model.compute_loss(batch)
        if reduction == "none":
            return per_sample, metrics
        return per_sample.mean(), metrics
