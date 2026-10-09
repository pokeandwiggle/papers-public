"""The authors' double-encoder cross-attention Diffusion Policy for LeRobot.

``_encode_obs``, ``_loss`` and ``_conditional_sample`` port
DiffusionUnetTimmAttentionPolicy (diffusion_policy/policy/
diffusion_unet_timm_attention_policy.py, commit 6645bd1) line for line on top
of the vendored TimmObsEncoder and StaticAttentionConditionalUnet1D, so the
same normalized batch and random state give the same loss and samples
(tests/test_attn_parity.py). Around that core this module adds the platform
adaptations: 20D model state split into per-part low-dim tokens, images
resized to the reference 224 px, per-step delta actions, in-policy min-max
normalization and EMA weights used for evaluation and serving.
"""

import copy

import timm
import timm.data
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from einops import reduce
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_STATE
from torch import Tensor, nn

from paw_baseline.attn_diffusion.configuration_attn_diffusion import (
    AttnDiffusionConfig,
)
from paw_baseline.attn_diffusion.deltas import DELTA_DIM, absolute_to_deltas
from paw_baseline.attn_diffusion.reference.conditional_unet_1d_static_attention import (
    StaticAttentionConditionalUnet1D,
)
from paw_baseline.attn_diffusion.reference.ema_model import EMAModel
from paw_baseline.attn_diffusion.reference.timm_obs_encoder import (
    _RANGE_SHORT,
    TimmObsEncoder,
)

# 20D model state split into low-dim tokens, like the reference per-arm
# position, orientation and gripper keys (rot6d replaces rpy).
STATE_PARTS: tuple[tuple[str, slice], ...] = (
    ("state_left_grip", slice(9, 10)),
    ("state_left_pos", slice(0, 3)),
    ("state_left_rot6d", slice(3, 9)),
    ("state_right_grip", slice(19, 20)),
    ("state_right_pos", slice(10, 13)),
    ("state_right_rot6d", slice(13, 19)),
)
RANGE_EPS = 1e-4


def camera_name(key: str) -> str:
    """Dot-free encoder key for a LeRobot camera feature (reference naming style)."""
    return key.removeprefix("observation.images.")


def _minmax(low, high) -> tuple[Tensor, Tensor]:
    """Scale and offset mapping [low, high] to [-1, 1]; flat dims map to 0."""
    low = torch.as_tensor(low, dtype=torch.float32)
    high = torch.as_tensor(high, dtype=torch.float32)
    span = high - low
    flat = span < RANGE_EPS
    scale = torch.where(flat, torch.ones_like(span), 2.0 / span.clamp_min(RANGE_EPS))
    offset = torch.where(flat, -(low + high) / 2.0, -1.0 - scale * low)
    return scale, offset


class _Core(nn.Module):
    """The reference policy's trainable modules, under the reference names."""

    def __init__(self, config: AttnDiffusionConfig, camera_keys: list[str]):
        super().__init__()
        size = config.image_resize
        shape_meta = {
            "obs": {
                **{camera_name(key): {"shape": [3, size, size], "type": "rgb"} for key in camera_keys},
                **{
                    name: {"shape": [sl.stop - sl.start], "type": "low_dim"}
                    for name, sl in STATE_PARTS
                },
            },
            "action": {"shape": [DELTA_DIM]},
        }

        def encoder(n_obs_steps: int) -> TimmObsEncoder:
            module = TimmObsEncoder(
                shape_meta=shape_meta,
                model_name=config.vision_backbone,
                pretrained=config.pretrained_backbone,
                n_obs_steps=n_obs_steps,
                feature_aggregation=config.feature_aggregation,
                crop_ratio=config.crop_ratio,
                imagenet_norm=config.imagenet_norm,
            )
            if config.imagenet_norm:
                # The reference normalizes only when it downloads pretrained
                # weights. Serving restores weights from the checkpoint without
                # downloading, so apply the same registry mean/std regardless.
                data_cfg = timm.data.resolve_model_data_config(module.backbone)
                module.normalize = torchvision.transforms.Normalize(
                    mean=data_cfg.get("mean", (0.485, 0.456, 0.406)),
                    std=data_cfg.get("std", (0.229, 0.224, 0.225)),
                )
            return module

        self.obs_encoder = encoder(config.n_obs_steps)
        self.short_range_encoder = encoder(config.short_range_obs_horizon)
        # ModuleAttrMixin's empty device marker never receives gradients.
        for module in (self.obs_encoder, self.short_range_encoder):
            module._dummy_variable.requires_grad_(False)
        self.model = StaticAttentionConditionalUnet1D(
            input_dim=DELTA_DIM,
            global_cond_dim=self.obs_encoder.feature_dim,
            diffusion_step_embed_dim=config.diffusion_step_embed_dim,
            down_dims=list(config.down_dims),
            kernel_size=config.kernel_size,
            n_groups=config.n_groups,
            num_attention_heads=config.num_attention_heads,
            attention_dropout=config.attention_dropout,
            use_temporal_pos_emb=config.use_temporal_pos_emb,
            use_modality_emb=config.use_modality_emb,
            max_modalities=self.obs_encoder.n_modalities,
            use_range_emb=config.use_range_emb,
            max_ranges=3,  # 0: NULL (timestep), 1: LONG, 2: SHORT
        )


class AttnDiffusionPolicy(PreTrainedPolicy):
    config_class = AttnDiffusionConfig
    name = "attn_diffusion"

    def __init__(self, config: AttnDiffusionConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.camera_keys = list(config.image_features)
        self.core = _Core(config, self.camera_keys)
        self.ema_core = copy.deepcopy(self.core) if config.use_ema else None
        self._ema = None
        if self.ema_core is not None:
            self.ema_core.requires_grad_(False)
            self._ema = EMAModel(
                self.ema_core,
                update_after_step=config.ema_update_after_step,
                inv_gamma=config.ema_inv_gamma,
                power=config.ema_power,
                min_value=config.ema_min_value,
                max_value=config.ema_max_value,
            )
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=config.num_train_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule=config.beta_schedule,
            clip_sample=config.clip_sample,
            prediction_type=config.prediction_type,
            variance_type=config.variance_type,
        )
        # DDIM mirrors the DDPM beta schedule for fast inference (reference).
        self.ddim_noise_scheduler = DDIMScheduler(
            num_train_timesteps=config.num_train_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule=config.beta_schedule,
            clip_sample=config.clip_sample,
            prediction_type=config.prediction_type,
        )
        self.ddim_noise_scheduler.set_timesteps(config.num_inference_steps)
        self._register_stats(config)

    # ------------------------------------------------------------ normalization

    def _register_stats(self, config: AttnDiffusionConfig) -> None:
        def stats(prefix: str, size: int) -> tuple[Tensor, Tensor]:
            low, high = getattr(config, f"{prefix}_min"), getattr(config, f"{prefix}_max")
            if low is None:
                return torch.ones(size), torch.zeros(size)
            return _minmax(low, high)

        for prefix, size in (("state", 20), ("delta_first", DELTA_DIM), ("delta_step", DELTA_DIM)):
            scale, offset = stats(prefix, size)
            self.register_buffer(f"{prefix}_scale", scale, persistent=False)
            self.register_buffer(f"{prefix}_offset", offset, persistent=False)

    def normalize_state(self, state: Tensor) -> Tensor:
        return state * self.state_scale + self.state_offset

    def _delta_affine(self, length: int) -> tuple[Tensor, Tensor]:
        """Per-slot scale/offset: the current-step slot uses the first-delta stats."""
        scale = self.delta_step_scale.expand(length, DELTA_DIM).clone()
        offset = self.delta_step_offset.expand(length, DELTA_DIM).clone()
        scale[self.config.past_steps] = self.delta_first_scale
        offset[self.config.past_steps] = self.delta_first_offset
        return scale, offset

    def normalize_deltas(self, deltas: Tensor) -> Tensor:
        scale, offset = self._delta_affine(deltas.shape[-2])
        return deltas * scale + offset

    def unnormalize_deltas(self, ndeltas: Tensor) -> Tensor:
        scale, offset = self._delta_affine(ndeltas.shape[-2])
        return (ndeltas - offset) / scale

    def obs_dict(self, state: Tensor, images: dict[str, Tensor]) -> dict[str, Tensor]:
        """Reference-style normalized obs: (B, T, 3, 224, 224) images in [0, 1]
        and per-part normalized state tokens."""
        nstate = self.normalize_state(state)
        obs = {name: nstate[..., sl] for name, sl in STATE_PARTS}
        for key in self.camera_keys:
            obs[camera_name(key)] = self.resize_images(images[key])
        return obs

    def resize_images(self, image: Tensor) -> Tensor:
        """(B, T, 3, H, W) in [0, 1] to the reference input size (224 px)."""
        size = self.config.image_resize
        if image.shape[-2:] == (size, size):
            return image
        flat = TF.resize(image.flatten(0, 1), [size, size], antialias=True)
        return flat.unflatten(0, image.shape[:2])

    # ------------------------------------------------------------ reference core

    def _encode_obs(self, core: _Core, nobs: dict[str, Tensor]):
        """Reference DiffusionUnetTimmAttentionPolicy._encode_obs."""
        To = core.obs_encoder.n_obs_steps
        long_nobs = {k: v[:, :To] for k, v in nobs.items()}
        enc = core.obs_encoder(long_nobs, output_format="dit")
        tokens = enc["tokens"]
        positions = enc["positions"]
        modalities = enc["modality"]
        ranges = enc["range"]

        if core.short_range_encoder is not None:
            h = self.config.short_range_obs_horizon
            short_nobs = {k: v[:, To - h : To] for k, v in nobs.items()}
            s_enc = core.short_range_encoder(short_nobs, output_format="dit")
            s_tokens = s_enc["tokens"]
            s_positions = s_enc["positions"]
            s_modalities = s_enc["modality"]
            s_ranges = torch.full(
                s_tokens.shape[:2], _RANGE_SHORT, dtype=torch.long, device=s_tokens.device
            )
            if core.training and self.config.short_range_dropout > 0.0:
                B = s_tokens.shape[0]
                keep = torch.rand(B, device=s_tokens.device) >= self.config.short_range_dropout
                s_tokens = s_tokens * keep.float().view(B, 1, 1)
            tokens = torch.cat([tokens, s_tokens], dim=1)
            positions = torch.cat([positions, s_positions], dim=1)
            modalities = torch.cat([modalities, s_modalities], dim=1)
            ranges = torch.cat([ranges, s_ranges], dim=1)
        return tokens, positions, modalities, ranges

    def _loss(self, core: _Core, nobs: dict[str, Tensor], nactions: Tensor) -> Tensor:
        """Reference compute_loss on normalized observations and actions."""
        tokens, positions, modalities, ranges = self._encode_obs(core, nobs)
        trajectory = nactions
        noise = torch.randn(trajectory.shape, device=trajectory.device, dtype=trajectory.dtype)
        noise_new = noise + self.config.input_pertub * torch.randn_like(noise)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (nactions.shape[0],),
            device=trajectory.device,
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise_new, timesteps).detach()
        pred = core.model(
            sample=noisy_trajectory,
            timestep=timesteps,
            global_cond=tokens,
            temporal_positions=positions,
            modality_indices=modalities,
            range_indices=ranges,
        )
        pred_type = self.noise_scheduler.config.prediction_type
        target = noise if pred_type == "epsilon" else trajectory
        loss = reduce(F.mse_loss(pred, target, reduction="none"), "b ... -> b (...)", "mean")
        return loss.mean()

    def _conditional_sample(self, core: _Core, cond, generator=None) -> Tensor:
        """Reference conditional_sample with DDIM and no inpainting."""
        tokens, positions, modalities, ranges = cond
        scheduler = self.ddim_noise_scheduler
        scheduler.set_timesteps(self.config.num_inference_steps)
        trajectory = torch.randn(
            size=(tokens.shape[0], self.config.horizon, DELTA_DIM),
            dtype=torch.float32,
            device=tokens.device,
            generator=generator,
        )
        for t in scheduler.timesteps.to(trajectory.device).long():
            model_output = core.model(
                sample=trajectory,
                timestep=t,
                global_cond=tokens,
                temporal_positions=positions,
                modality_indices=modalities,
                range_indices=ranges,
            )
            trajectory = scheduler.step(model_output, t, trajectory, generator=generator).prev_sample
        return trajectory

    # ------------------------------------------------------------ LeRobot API

    @property
    def eval_core(self) -> _Core:
        return self.ema_core if self.ema_core is not None else self.core

    def get_optim_params(self) -> list[dict]:
        encoders = list(self.core.obs_encoder.parameters()) + list(
            self.core.short_range_encoder.parameters()
        )
        encoder_ids = {id(p) for p in encoders}
        others = [p for p in self.core.parameters() if id(p) not in encoder_ids]
        return [
            {"params": encoders, "lr": self.config.optimizer_lr * self.config.encoder_lr_scale},
            {"params": others, "lr": self.config.optimizer_lr},
        ]

    @torch.no_grad()
    def update(self) -> None:
        """Called by LeRobot after each optimizer step: advance the EMA weights."""
        if self._ema is not None:
            self._ema.step(self.core)
            self.ema_core.eval()

    def reset(self) -> None:
        pass

    def train(self, mode: bool = True):
        super().train(mode)
        if self.ema_core is not None:
            self.ema_core.eval()  # reference EMAModel keeps the averaged model in eval
        return self

    def deltas_from_batch(self, batch: dict[str, Tensor]) -> Tensor:
        # Targets in fp32 even under bf16 autocast: per-step rotation deltas
        # are small enough that bf16 matmuls distort them.
        state, actions = batch[OBS_STATE][:, -1], batch[ACTION]
        with torch.autocast(device_type=state.device.type, enabled=False):
            return absolute_to_deltas(state.float(), actions.float(), self.config.past_steps)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, None]:
        nobs = self.obs_dict(batch[OBS_STATE], {k: batch[k] for k in self.camera_keys})
        nactions = self.normalize_deltas(self.deltas_from_batch(batch))
        return self._loss(self.core, nobs, nactions), None

    @torch.no_grad()
    def sample_deltas(
        self, state: Tensor, images: dict[str, Tensor], generator=None
    ) -> Tensor:
        """Unnormalized (B, horizon, 14) deltas from the EMA weights."""
        core = self.eval_core
        nobs = self.obs_dict(state, images)
        cond = self._encode_obs(core, nobs)
        return self.unnormalize_deltas(self._conditional_sample(core, cond, generator))

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Current and future deltas (B, prediction_steps, 14) for offline use."""
        deltas = self.sample_deltas(batch[OBS_STATE], {k: batch[k] for k in self.camera_keys})
        start = self.config.past_steps
        return deltas[:, start : start + self.config.prediction_steps]

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        raise NotImplementedError("serve attn_diffusion through inference.serve")
