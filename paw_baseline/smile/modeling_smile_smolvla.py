"""SMILE-SmolVLA: denoise B-spline coefficients instead of raw action chunks.

Following SMILE (Park et al., 2026, Sec. IV), the SmolVLA VLM, conditioning
prefix, action expert, time MLP and flow-matching objective are unchanged.
Only the action-side interface is replaced: K noisy coefficient tokens enter
through a coefficient input projection, translation/rotation/gripper
coefficient heads replace the raw-action output projection, and a fixed
B-spline basis decodes the denoised coefficients into the F-step chunk of
relative actions that the unchanged SmolVLA queue, postprocessor and serving
path execute.
"""

import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from huggingface_hub import hf_hub_download
from lerobot.configs import PreTrainedConfig
from lerobot.policies.common.flow_matching import euler_integrate
from lerobot.policies.common.vla_utils import (
    create_sinusoidal_pos_embedding,
    make_att_2d_masks,
)
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, VLAFlowMatching
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
)
from lerobot.utils.device_utils import resolve_safetensors_device
from safetensors.torch import load_model
from torch import Tensor, nn

from paw_baseline.smile.bspline import bspline_basis, masked_projectors
from paw_baseline.smile.configuration_smile_smolvla import SmileSmolVLAConfig
from paw_baseline.smile.trajectory import cumulative_trajectory, deltas_from_cumulative

# Raw-action projections of smolvla_base that SMILE replaces.
REPLACED_BASE_KEYS = frozenset(
    f"model.{name}.{param}"
    for name in ("action_in_proj", "action_out_proj")
    for param in ("weight", "bias")
)
SMILE_KEY_PREFIXES = ("model.coef_in_proj.", "model.coef_heads.")
# LeRobot NormalizerProcessorStep MEAN_STD epsilon.
NORMALIZER_EPS = 1e-8


class SmileFlowMatching(VLAFlowMatching):
    """SmolVLA flow matching over K normalized B-spline coefficient tokens."""

    def __init__(self, config: SmileSmolVLAConfig, rtc_processor=None):
        super().__init__(config, rtc_processor=rtc_processor)
        hidden = self.vlm_with_expert.expert_hidden_size
        action_dim = config.action_feature.shape[0]
        self.action_dim = action_dim
        self.num_control_points = config.num_control_points
        # Replace the raw-action input projection and output head (Fig. 1a).
        del self.action_in_proj
        del self.action_out_proj
        self.coef_in_proj = nn.Linear(action_dim, hidden)
        modalities = config.modality_dims()
        self.modality_names = tuple(modalities)
        self.coef_heads = nn.ModuleDict(
            {name: nn.Linear(hidden, len(dims)) for name, dims in modalities.items()}
        )
        order = [d for name in self.modality_names for d in modalities[name]]
        self._buffer("head_inverse_order", np.argsort(order), torch.long)
        for name, dims in modalities.items():
            self._buffer(f"dims_{name}", dims, torch.long)
        basis = bspline_basis(
            config.chunk_size, config.num_control_points, config.spline_degree
        )
        projectors, observability, rank = masked_projectors(
            basis, config.observability_tol
        )
        self._buffer("basis", basis, torch.float64)
        self._buffer("projectors", projectors, torch.float64)
        self._buffer("observability", observability, torch.float64)
        self._buffer("observed_rank", rank, torch.float64)
        shape = (config.num_control_points, action_dim)
        self._buffer("coef_mean", np.reshape(config.coef_mean, shape), torch.float64)
        self._buffer("coef_std", np.reshape(config.coef_std, shape), torch.float64)
        self._buffer("action_mean", config.action_mean, torch.float64)
        self._buffer("action_std", config.action_std, torch.float64)
        if config.compile_model:
            # Compile only the network; trajectory maps run eagerly in float64.
            self.sample_actions = SmileFlowMatching.sample_actions.__get__(self)
            self.sample_coefficients = torch.compile(
                self.sample_coefficients, mode=config.compile_mode
            )

    def _buffer(self, name: str, values, dtype) -> None:
        self.register_buffer(
            name, torch.as_tensor(np.asarray(values), dtype=dtype), persistent=False
        )

    # --- Coefficient targets (Sec. IV-D) and decoding (Eq. 4-6) ---

    def coefficient_targets(self, actions: Tensor, valid_length: Tensor) -> Tensor:
        """Normalized Eq. (10) targets from MEAN_STD-normalized (B, F, D) deltas.

        Only the first valid_length steps of each window are expert data; the
        projector ignores padded steps. Coefficient directions those steps do
        not determine are unsupervised; they take the dataset coefficient mean
        so the noisy input stays in distribution.
        """
        raw = actions.to(torch.float64) * (self.action_std + NORMALIZER_EPS) + self.action_mean
        cumulative = cumulative_trajectory(raw)
        coefs = torch.einsum("bkf,bfd->bkd", self.projectors[valid_length], cumulative)
        observed_mean = torch.einsum(
            "bkj,jd->bkd", self.observability[valid_length], self.coef_mean
        )
        coefs = coefs + (self.coef_mean - observed_mean)
        return ((coefs - self.coef_mean) / self.coef_std).to(torch.float32)

    def decode_actions(self, coefs: Tensor) -> Tensor:
        """Normalized (B, F, D) per-step deltas from normalized coefficients."""
        coefs = coefs.to(torch.float64) * self.coef_std + self.coef_mean
        cumulative = torch.einsum("fk,bkd->bfd", self.basis, coefs)
        grip = self.dims_gripper
        cumulative[..., grip] = cumulative[..., grip].clamp(
            self.config.gripper_min, self.config.gripper_max
        )
        deltas = deltas_from_cumulative(cumulative)
        # Exact inverse of the shared postprocessor's x = n * std + mean.
        scale = torch.where(self.action_std > 0, self.action_std, torch.ones_like(self.action_std))
        return ((deltas - self.action_mean) / scale).to(torch.float32)

    def coefficient_heads(self, suffix_out: Tensor) -> Tensor:
        """Per-modality heads, reassembled into action order (B, K, D)."""
        parts = [self.coef_heads[name](suffix_out) for name in self.modality_names]
        return torch.cat(parts, dim=-1)[..., self.head_inverse_order]

    # --- SmolVLA denoiser over coefficient tokens ---

    def embed_suffix(self, noisy_coefs, timestep):
        """SmolVLA's suffix embedding over K coefficient tokens."""
        action_emb = self.coef_in_proj(noisy_coefs)
        device = action_emb.device
        bsize = action_emb.shape[0]
        dtype = action_emb.dtype
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.vlm_with_expert.expert_hidden_size,
            self.config.min_period,
            self.config.max_period,
            device=device,
        )
        time_emb = time_emb.type(dtype=dtype)
        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)
        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        num_tokens = action_time_emb.shape[1]
        pad_masks = torch.ones(bsize, num_tokens, dtype=torch.bool, device=device)
        # Same causal suffix pattern SmolVLA uses for its action tokens.
        att_masks = torch.ones(bsize, num_tokens, dtype=action_time_emb.dtype, device=device)
        return action_time_emb, pad_masks, att_masks

    def forward(
        self, images, img_masks, lang_tokens, lang_masks, state, target, noise=None, time=None
    ) -> tuple[Tensor, Tensor]:
        """Predicted and target velocities (B, K, D) for normalized coefficients."""
        if noise is None:
            noise = self.sample_noise(target.shape, target.device)
        if time is None:
            time = self.sample_time(target.shape[0], target.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * target
        u_t = noise - target
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, time)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        (_, suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
        )
        suffix_out = suffix_out[:, -self.num_control_points :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.coefficient_heads(suffix_out), u_t

    def sample_coefficients(
        self, images, img_masks, lang_tokens, lang_masks, state, noise=None
    ) -> Tensor:
        """Euler-integrate SmolVLA's flow over K normalized coefficient tokens."""
        bsize = state.shape[0]
        device = state.device
        if noise is None:
            noise = self.sample_noise(
                (bsize, self.num_control_points, self.action_dim), device
            )
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        _, past_key_values = self.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
        )
        return euler_integrate(
            lambda input_x_t, current_timestep: self.denoise_step(
                x_t=input_x_t,
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                timestep=current_timestep,
            ),
            noise,
            self.config.num_steps,
        )

    def sample_actions(
        self, images, img_masks, lang_tokens, lang_masks, state, noise=None, **kwargs
    ) -> Tensor:
        """Denoise K coefficient tokens, then decode the full F-step chunk."""
        coefs = self.sample_coefficients(
            images, img_masks, lang_tokens, lang_masks, state, noise=noise
        )
        return self.decode_actions(coefs)

    def denoise_step(self, prefix_pad_masks, past_key_values, x_t, timestep):
        """One SmolVLA denoising step over coefficient tokens."""
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, timestep)
        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
            batch_size, suffix_len, prefix_len
        )
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
        outputs_embeds, _ = self.vlm_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.config.use_cache,
        )
        if past_key_values is not None:
            past_key_values.crop(prefix_len)
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.num_control_points :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.coefficient_heads(suffix_out)


def resolve_base_weights(repo: str, revision: str) -> Path:
    if re.fullmatch("[0-9a-f]{40}", revision or "") is None:
        raise ValueError("SMILE base weights require an immutable commit SHA")
    return Path(hf_hub_download(repo, "model.safetensors", revision=revision))


def valid_prefix_length(is_pad: Tensor | None, batch: int, length: int, device) -> Tensor:
    """Number of in-episode steps per window; padding only follows the episode end."""
    if is_pad is None:
        return torch.full((batch,), length, dtype=torch.long, device=device)
    if bool((is_pad[:, :-1] & ~is_pad[:, 1:]).any()) or bool(is_pad[:, 0].any()):
        raise ValueError("SMILE expects action padding only after the episode end")
    return length - is_pad.sum(dim=1)


class SmileSmolVLAPolicy(SmolVLAPolicy):
    """SmolVLA policy whose action expert denoises B-spline coefficients.

    A fresh policy loads the pinned smolvla_base weights except the replaced
    raw-action projections; resume and serving load a SMILE checkpoint.
    """

    config_class = SmileSmolVLAConfig
    name = "smile_smolvla"

    def supports_rtc(self) -> bool:
        return False

    def __init__(
        self, config: SmileSmolVLAConfig, *, _weights_path: Path | None = None, **kwargs
    ):
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        stats = (kwargs.get("dataset_stats") or {}).get(ACTION)
        if stats is not None:
            for name in ("mean", "std"):
                saved = np.asarray(getattr(config, f"action_{name}"), dtype=np.float64)
                if not np.allclose(np.asarray(stats[name], dtype=np.float64), saved, rtol=1e-5, atol=1e-9):
                    raise ValueError(f"SMILE action_{name} disagrees with the processor statistics")
        self.config = config
        self.init_rtc_processor()
        self.model = SmileFlowMatching(config, rtc_processor=self.rtc_processor)
        self.reset()
        if _weights_path is None:
            self._load_weights(
                resolve_base_weights(config.base_repo, config.base_revision), base=True
            )

    def _load_weights(self, path: Path, *, base: bool) -> None:
        missing, unexpected = load_model(
            self,
            path,
            strict=False,
            device=resolve_safetensors_device(self.config.device or "cpu"),
        )
        missing, unexpected = set(missing), set(unexpected)
        if base:
            new = {key for key in missing if key.startswith(SMILE_KEY_PREFIXES)}
            if unexpected != REPLACED_BASE_KEYS or missing != new or not new:
                raise RuntimeError(
                    f"smolvla_base load mismatch: missing={sorted(missing - new)} "
                    f"unexpected={sorted(unexpected ^ REPLACED_BASE_KEYS)}"
                )
        elif missing or unexpected:
            raise RuntimeError(
                f"SMILE checkpoint load mismatch: missing={sorted(missing)} "
                f"unexpected={sorted(unexpected)}"
            )

    @classmethod
    def from_pretrained(
        cls, pretrained_name_or_path, *, config=None, revision=None, strict=True, **kwargs
    ):
        source = Path(str(pretrained_name_or_path))
        if not source.is_dir():
            raise ValueError("SMILE checkpoints load from a local directory")
        if config is None:
            config = PreTrainedConfig.from_pretrained(source)
        if not isinstance(config, SmileSmolVLAConfig):
            raise TypeError("SMILE loading requires a SmileSmolVLAConfig")
        path = source / "model.safetensors"
        policy = cls(config, _weights_path=path, **kwargs)
        policy._load_weights(path, base=False)
        policy.to(config.device)
        policy.eval()
        return policy

    def forward(
        self, batch: dict[str, Tensor], noise=None, time=None, reduction: str = "mean"
    ) -> tuple[Tensor, dict]:
        """Eq. (8)-(9), supervising only coefficient directions fixed by expert steps.

        A window crossing the episode end is fitted on its valid steps only;
        its velocity error is projected onto the coefficient subspace those
        steps determine. Full windows (all training starts but the last F - 1
        of an episode) reduce to the paper's plain coefficient-space loss.
        """
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        actions = batch[ACTION]
        if actions.ndim != 3 or actions.shape[1] != self.config.chunk_size:
            raise ValueError(
                f"SMILE expects (B, {self.config.chunk_size}, D) action chunks, got {tuple(actions.shape)}"
            )
        valid = valid_prefix_length(
            batch.get("action_is_pad"), actions.shape[0], actions.shape[1], actions.device
        )
        with torch.no_grad():
            target = self.model.coefficient_targets(actions, valid)
        v_t, u_t = self.model.forward(
            images, img_masks, lang_tokens, lang_masks, state, target, noise, time
        )
        sigma = self.model.coef_std
        error = (v_t - u_t).to(torch.float64) * sigma
        error = torch.einsum("bkj,bjd->bkd", self.model.observability[valid], error) / sigma
        squared = error.square()
        weights = {
            "translation": self.config.loss_weight_translation,
            "rotation": self.config.loss_weight_rotation,
            "gripper": self.config.loss_weight_gripper,
        }
        # ||f^m - y^m||_F^2 summed over modalities, per supervised coefficient.
        rank = self.model.observed_rank[valid]
        per_sample = torch.zeros(squared.shape[0], dtype=torch.float64, device=squared.device)
        loss_dict = {}
        denominator = rank.sum() * squared.shape[-1]
        for name in self.model.modality_names:
            term = squared[..., getattr(self.model, f"dims_{name}")].sum(dim=(1, 2))
            per_sample = per_sample + weights[name] * term
            loss_dict[f"loss_{name}"] = (term.sum() / denominator).item()
        loss_dict["partial_window_fraction"] = (valid < actions.shape[1]).float().mean().item()
        if reduction == "none":
            per_sample = (per_sample / (rank * squared.shape[-1])).to(torch.float32)
            loss_dict["loss"] = per_sample.mean().item()
            return per_sample, loss_dict
        loss = (per_sample.sum() / denominator).to(torch.float32)
        loss_dict["loss"] = loss.item()
        return loss, loss_dict
