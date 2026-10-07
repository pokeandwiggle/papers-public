"""Patch Policy with a VQ-BeT head (Zhou et al. 2026), as a LeRobot policy.

Adapted from ``models/vq_behavior_transformer/bet.py`` of the reference
(https://github.com/gaoyuezhou/patch_policy, MIT). Each history frame's
cameras are encoded by the frozen DINOv2, flattened into one token block per
frame (camera-major, then patch), processed by the block-causal trunk, and the
last token of every frame predicts a residual-VQ code pair and per-code
offsets for the action chunk that starts at that frame. Images only: the
policy never reads ``observation.state``.
"""

import functools
import logging
from collections import deque

import torch
import torch.nn.functional as F
from lerobot.policies.pretrained import PreTrainedPolicy
from torch import nn

from train.patch_policy.configuration_patch_vqbet import PatchVQBeTConfig
from train.patch_policy.encoder import FrozenDinoV2
from train.patch_policy.gpt import GPT, GPTConfig
from train.patch_policy.vqvae import VqVae, fit_vqvae

ENCODER_DIM = 384


class MLP(nn.Sequential):
    """torchvision-style MLP used by the reference heads (ReLU, no norm, dropout 0)."""

    def __init__(self, in_channels: int, hidden_channels: list[int], dropout: float = 0.0):
        layers = []
        in_dim = in_channels
        for hidden in hidden_channels[:-1]:
            layers += [nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = hidden
        layers += [nn.Linear(in_dim, hidden_channels[-1]), nn.Dropout(dropout)]
        super().__init__(*layers)


def batch_idx(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Index x's dimension len(idx.shape) by idx (reference helper)."""
    assert idx.shape == x.shape[: len(idx.shape)]
    remaining = x.shape[len(idx.shape) + 1 :]
    x = x.flatten(start_dim=idx.dim() + 1)
    indices = []
    for i in range(len(idx.shape)):
        shape = [1] * len(idx.shape)
        shape[i] = idx.shape[i]
        indices.append(torch.arange(idx.shape[i], device=idx.device).reshape(shape))
    indices.append(idx)
    return x[tuple(indices)].reshape(idx.shape + remaining)


def focal_loss(logits: torch.Tensor, target: torch.Tensor, gamma: float) -> torch.Tensor:
    logpt = F.log_softmax(logits, dim=-1).gather(1, target.view(-1, 1)).view(-1)
    return (-((1 - logpt.exp()) ** gamma) * logpt).mean()


@functools.cache
def _cuda_bf16_supported() -> bool:
    """bf16 autocast needs native support; older serving GPUs fall back to float32."""
    supported = torch.cuda.is_bf16_supported(including_emulation=False)
    if not supported:
        logging.getLogger(__name__).warning("CUDA device lacks bf16; running the trunk in float32")
    return supported


def history_indices(length: int, offsets: list[int]) -> list[int]:
    """Indices into a buffer of ``length`` 30 Hz frames, newest last.

    Offsets reaching before the first buffered frame clamp to it, matching
    LeRobot's clamping of delta timestamps at an episode's first frame.
    """
    newest = length - 1
    return [max(newest + offset, 0) for offset in offsets]


class PatchVQBeTPolicy(PreTrainedPolicy):
    config_class = PatchVQBeTConfig
    name = "patch_vqbet"

    def __init__(
        self,
        config: PatchVQBeTConfig,
        dataset_stats=None,
        dataset_meta=None,
        *,
        _from_checkpoint: bool = False,
        **kwargs,
    ):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.camera_keys = list(config.image_features)
        self.action_dim = config.action_feature.shape[0]
        self.encoder = FrozenDinoV2(
            config.visual_feature,
            config.image_size,
            pretrained=(
                (config.encoder_repo, config.encoder_revision)
                if config.encoder_pretrained and not _from_checkpoint
                else None
            ),
        )
        self.gpt = GPT(
            GPTConfig(
                n_frames=config.n_obs_steps,
                tokens_per_frame=config.tokens_per_camera * len(self.camera_keys),
                input_dim=ENCODER_DIM,
                output_dim=config.gpt_output_dim,
                n_layer=config.n_layer,
                n_head=config.n_head,
                n_embd=config.n_embd,
                dropout=config.dropout,
            )
        )
        g, c, h, a = config.vqvae_groups, config.vqvae_n_embed, config.chunk_size, self.action_dim
        self.vqvae = self._build_vqvae()
        self.bin_head = MLP(config.gpt_output_dim, [1024, 1024, g * c])
        self.offset_head = MLP(config.gpt_output_dim, [1024, 1024, g * c * h * a])
        self.register_buffer("vqvae_fitted", torch.tensor(False))
        self.reset()

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path, *, config=None, strict: bool = True, **kwargs):
        """Restore every tensor, encoder included, strictly from the checkpoint."""
        kwargs.pop("_from_checkpoint", None)
        policy = super().from_pretrained(
            pretrained_name_or_path, config=config, strict=strict, _from_checkpoint=True, **kwargs
        )
        if not bool(policy.vqvae_fitted):
            raise RuntimeError("checkpoint has no fitted VQ tokenizer")
        policy._freeze_tokenizer()
        return policy

    # ----- tokenizer -------------------------------------------------------

    def _freeze_tokenizer(self) -> None:
        self.vqvae.requires_grad_(False)
        self.vqvae.eval()

    def _build_vqvae(self) -> VqVae:
        cfg = self.config
        return VqVae(
            cfg.chunk_size,
            self.action_dim,
            latent_dim=cfg.vqvae_latent_dim,
            n_embed=cfg.vqvae_n_embed,
            groups=cfg.vqvae_groups,
            hidden_dim=cfg.vqvae_hidden_dim,
            hidden_layers=cfg.vqvae_hidden_layers,
        )

    def fit_tokenizer(self, chunks: torch.Tensor, *, log=print, seed: int = 0) -> dict:
        """Fit the residual VQ-VAE on normalized (N, chunk, action_dim) chunks, then freeze it.

        The tokenizer is rebuilt and fit under a fixed seed, independent of the
        visual feature, so dense and CLS policies of a task share one tokenizer.
        """
        if bool(self.vqvae_fitted):
            raise RuntimeError("VQ tokenizer is already fitted")
        cfg = self.config
        device = self.vqvae_fitted.device
        with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
            torch.manual_seed(seed)
            self.vqvae = self._build_vqvae().to(device)
            info = fit_vqvae(
                self.vqvae,
                chunks.to(device),
                steps=cfg.vqvae_steps,
                batch_size=cfg.vqvae_batch_size,
                lr=cfg.vqvae_lr,
                weight_decay=cfg.vqvae_weight_decay,
                seed=seed,
                log_every=max(1, cfg.vqvae_steps // 10),
                log=log,
            )
        self.vqvae_fitted.fill_(True)
        self._freeze_tokenizer()
        return info

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.train(False)
        if bool(self.vqvae_fitted):
            self.vqvae.eval()
        return self

    # ----- network ---------------------------------------------------------

    def _autocast(self, device: torch.device):
        enabled = (
            device.type == "cuda"
            and self.config.trunk_dtype == "bfloat16"
            and _cuda_bf16_supported()
        )
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled)

    def encode_frames(self, batch: dict) -> torch.Tensor:
        """(B, T, 3, H, W) per camera -> (B, T, cameras*tokens, 384) frame tokens."""
        images = torch.stack([batch[key] for key in self.camera_keys], dim=2)
        b, t, v = images.shape[:3]
        flat = images.flatten(0, 2)
        with self._autocast(flat.device):
            tokens = self.encoder(flat)
        return tokens.reshape(b, t, v * tokens.shape[1], tokens.shape[2])

    def _heads(self, frame_tokens: torch.Tensor):
        with self._autocast(frame_tokens.device):
            trunk = self.gpt(frame_tokens)
        cfg = self.config
        g, c, h, a = cfg.vqvae_groups, cfg.vqvae_n_embed, cfg.chunk_size, self.action_dim
        with torch.autocast(device_type=trunk.device.type, enabled=False):
            trunk = trunk.float()
            logits = self.bin_head(trunk).unflatten(-1, (g, c))
            offsets = self.offset_head(trunk).unflatten(-1, (g, c, h, a))
        return logits, offsets

    def _sample(self, logits: torch.Tensor, offsets: torch.Tensor):
        """Sample codes from the predicted distribution; decode center + offset."""
        b, t, g, c = logits.shape
        probs = torch.softmax(logits, dim=-1)
        sampled = torch.multinomial(probs.reshape(-1, c), num_samples=1).reshape(b, t, g)
        with torch.autocast(device_type=logits.device.type, enabled=False):
            centers = self.vqvae.draw_code_forward(sampled)
            decoded = self.vqvae.get_action_from_latent(centers).detach()
        sampled_offsets = batch_idx(offsets, sampled).sum(dim=2)
        return decoded + sampled_offsets, decoded, sampled

    def unpack_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """(B, (T-1)*stride + chunk, A) -> (B, T, chunk, A): each frame's own chunk."""
        cfg = self.config
        expected = (cfg.n_obs_steps - 1) * cfg.obs_stride + cfg.chunk_size
        if actions.shape[1] != expected:
            raise ValueError(f"expected {expected} action steps, got {actions.shape[1]}")
        return torch.stack(
            [
                actions[:, i * cfg.obs_stride : i * cfg.obs_stride + cfg.chunk_size]
                for i in range(cfg.n_obs_steps)
            ],
            dim=1,
        )

    # ----- training --------------------------------------------------------

    def forward(self, batch: dict) -> tuple[torch.Tensor, dict]:
        if not bool(self.vqvae_fitted):
            raise RuntimeError("fit the VQ tokenizer before training the policy")
        cfg = self.config
        logits, offsets = self._heads(self.encode_frames(batch))
        predicted, decoded, sampled = self._sample(logits, offsets)
        target = self.unpack_actions(batch["action"].float())
        with torch.autocast(device_type=target.device.type, enabled=False):
            _, bins = self.vqvae.get_code(target)

        offset_loss = F.l1_loss(target, predicted)
        flat_logits = logits.flatten(0, 1)
        flat_bins = bins.flatten(0, 1)
        code_losses = [
            focal_loss(flat_logits[:, i], flat_bins[:, i], cfg.focal_gamma)
            for i in range(cfg.vqvae_groups)
        ]
        code_loss = cfg.primary_code_loss_weight * code_losses[0] + cfg.secondary_code_loss_weight * sum(
            code_losses[1:]
        )
        loss = code_loss + cfg.offset_loss_weight * offset_loss

        with torch.no_grad():
            equal = bins == sampled
            info = {
                "code_loss": code_loss.item(),
                "primary_code_loss": code_losses[0].item(),
                "offset_loss": offset_loss.item(),
                "decoded_l1": F.l1_loss(target, decoded).item(),
                "equal_total_code_rate": (equal.sum(-1) == cfg.vqvae_groups).float().mean().item(),
                "equal_primary_code_rate": equal[..., 0].float().mean().item(),
                "last_frame_first_action_mse": F.mse_loss(target[:, -1, 0], predicted[:, -1, 0]).item(),
            }
        return loss, info

    def get_optim_params(self) -> list[dict]:
        cfg = self.config
        groups = self.gpt.param_groups(cfg.optimizer_weight_decay)
        groups.append({"params": list(self.bin_head.parameters()), "weight_decay": cfg.bin_head_weight_decay})
        groups.append({"params": list(self.offset_head.parameters()), "weight_decay": cfg.optimizer_weight_decay})
        return groups

    # ----- inference -------------------------------------------------------

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict, **kwargs) -> torch.Tensor:
        """Normalized (B, n_action_steps, A) chunk from the newest history frame."""
        logits, offsets = self._heads(self.encode_frames(batch))
        predicted, _, _ = self._sample(logits[:, -1:], offsets[:, -1:])
        return predicted[:, 0, : self.config.n_action_steps]

    def reset(self):
        cfg = self.config
        span = (cfg.n_obs_steps - 1) * cfg.obs_stride + 1
        self._frames = {key: deque(maxlen=span) for key in self.camera_keys}
        self._action_queue = deque()

    @torch.no_grad()
    def select_action(self, batch: dict, **kwargs) -> torch.Tensor:
        """Per-step LeRobot API: buffer frames at 30 Hz, replan when the queue empties."""
        for key in self.camera_keys:
            self._frames[key].append(batch[key])
        if not self._action_queue:
            history = {}
            for key in self.camera_keys:
                frames = list(self._frames[key])
                picks = history_indices(len(frames), self.config.history_offsets)
                history[key] = torch.stack([frames[i] for i in picks], dim=1)
            chunk = self.predict_action_chunk(history)
            self._action_queue.extend(chunk.transpose(0, 1))
        return self._action_queue.popleft()
