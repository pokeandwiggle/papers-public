"""Configure the authors' real-robot Diffusion Policy on the 30 Hz platform grid.

Architecture and optimization defaults follow the released real-robot config
(config/mundane/8_obs_pill_DINOv3_attention_double_enc.yaml, commit 6645bd1):
DINOv3 ViT-B/16 encoders with CLS tokens fine-tuned at 0.1x the policy LR, a
cross-attention 1D U-Net (down dims 256/512/1024), a second short-range
encoder over the newest 2 frames with 0.3 per-sample dropout, DDPM training
with input perturbation 0.1, DDIM-10 inference, AdamW 1e-4 with cosine decay,
gradient clipping at 1.0 and EMA (power 0.75, max 0.9999).

Timing on the 30 Hz grid: ``n_obs_steps`` frames ``obs_stride`` steps apart
(100 ms), ``prediction_steps`` current+future actions (1.5 s) plus past-token
prediction of every action in the context window, and ``replan_steps``
actions executed before replanning. Actions are 14D per-step deltas (see
``deltas.py``); the first predicted delta is relative to the measured pose.

Low-dimensional normalization lives in the policy (min-max to [-1, 1], as the
reference LinearNormalizer "limits" mode): state statistics, and separate
statistics for the first (measured-pose-relative) delta slot and for all
other per-step delta slots. The trainer fills them from training episodes.
"""

import math
from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamWConfig, DiffuserSchedulerConfig

from paw_baseline.attn_diffusion.deltas import DELTA_DIM


@PreTrainedConfig.register_subclass("attn_diffusion")
@dataclass
class AttnDiffusionConfig(PreTrainedConfig):
    n_obs_steps: int = 2
    obs_stride: int = 3
    prediction_steps: int = 45
    replan_steps: int = 24
    # Actions emitted per plan; always equals prediction_steps (a saved field
    # because validation reads it from config.json).
    n_action_steps: int = 45
    # Linear interpolation from the previous chunk into each new chunk, up to
    # this time after the new chunk's grounding observation (paper A.3).
    interpolation_ms: float = 180.0

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    # Observation encoders (TimmObsEncoder).
    vision_backbone: str = "vit_base_patch16_dinov3.lvd1689m"
    pretrained_backbone: bool = True
    image_resize: int = 224
    crop_ratio: float = 0.9
    feature_aggregation: str = "cls"
    imagenet_norm: bool = True
    short_range_obs_horizon: int = 2
    short_range_dropout: float = 0.3

    # Cross-attention U-Net (StaticAttentionConditionalUnet1D).
    diffusion_step_embed_dim: int = 256
    down_dims: tuple[int, ...] = (256, 512, 1024)
    kernel_size: int = 5
    n_groups: int = 8
    num_attention_heads: int = 8
    attention_dropout: float = 0.1
    use_temporal_pos_emb: bool = True
    use_modality_emb: bool = True
    use_range_emb: bool = True
    input_pertub: float = 0.1

    # Noise schedule: DDPM training, DDIM inference.
    num_train_timesteps: int = 100
    beta_start: float = 0.0001
    beta_end: float = 0.02
    beta_schedule: str = "squaredcos_cap_v2"
    clip_sample: bool = True
    prediction_type: str = "epsilon"
    variance_type: str = "fixed_small"
    num_inference_steps: int = 10

    # Optimization (AdamW, cosine with warmup, encoder LR scale, clipping).
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    encoder_lr_scale: float = 0.1
    grad_clip_norm: float = 1.0
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500

    # Training/serving autocast: "float32" (reference default) or "bfloat16"
    # (the reference's supported bf16 mode; its U-Net always runs in fp32).
    dtype: str = "float32"

    # EMA (reference EMAModel arguments).
    use_ema: bool = True
    ema_update_after_step: int = 0
    ema_inv_gamma: float = 1.0
    ema_power: float = 0.75
    ema_min_value: float = 0.0
    ema_max_value: float = 0.9999

    # Normalization statistics (min/max), filled by the trainer.
    state_min: list[float] | None = None
    state_max: list[float] | None = None
    delta_first_min: list[float] | None = None
    delta_first_max: list[float] | None = None
    delta_step_min: list[float] | None = None
    delta_step_max: list[float] | None = None

    def __post_init__(self):
        super().__post_init__()
        for name in ("obs_stride", "prediction_steps", "replan_steps", "n_obs_steps"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if self.replan_steps > self.prediction_steps:
            raise ValueError("replan_steps cannot exceed prediction_steps")
        self.n_action_steps = self.prediction_steps
        if not 1 <= self.short_range_obs_horizon <= self.n_obs_steps:
            raise ValueError("short_range_obs_horizon must be in [1, n_obs_steps]")
        if not math.isfinite(self.interpolation_ms) or self.interpolation_ms < 0:
            raise ValueError("interpolation_ms must be a nonnegative number")
        if self.dtype not in ("float32", "bfloat16"):
            raise ValueError("dtype must be float32 or bfloat16")
        for prefix, size in (("state", 20), ("delta_first", DELTA_DIM), ("delta_step", DELTA_DIM)):
            low, high = getattr(self, f"{prefix}_min"), getattr(self, f"{prefix}_max")
            if (low is None) != (high is None):
                raise ValueError(f"{prefix}_min and {prefix}_max must be set together")
            if low is not None and (len(low) != size or len(high) != size):
                raise ValueError(f"{prefix} statistics must have {size} values")

    @property
    def past_steps(self) -> int:
        return (self.n_obs_steps - 1) * self.obs_stride

    @property
    def horizon(self) -> int:
        """Denoised sequence length: past tokens plus current and future actions."""
        return self.past_steps + self.prediction_steps

    @property
    def observation_delta_indices(self) -> list:
        return list(range(-self.past_steps, 1, self.obs_stride))

    @property
    def action_delta_indices(self) -> list:
        # One extra leading action anchors the first past-token delta.
        return list(range(-self.past_steps - 1, self.prediction_steps))

    @property
    def reward_delta_indices(self) -> None:
        return None

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.grad_clip_norm,
        )

    def get_scheduler_preset(self) -> DiffuserSchedulerConfig:
        return DiffuserSchedulerConfig(
            name=self.scheduler_name, num_warmup_steps=self.scheduler_warmup_steps
        )

    def validate_features(self) -> None:
        if not self.image_features:
            raise ValueError("attn_diffusion requires camera inputs")
        if self.robot_state_feature is None or self.robot_state_feature.shape != (20,):
            raise ValueError("attn_diffusion requires the 20D model state")
        if self.action_feature is None or self.action_feature.shape != (20,):
            raise ValueError("attn_diffusion requires 20D absolute actions in the dataset")
