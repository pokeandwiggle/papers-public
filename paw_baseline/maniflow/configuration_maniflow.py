"""Configure 2D ManiFlow with multi-token DiT-X conditioning.

Observation tokens follow the official TransformerObsEncoder path
(geyan21/ManiFlow_Policy@ef2f116, maniflow_image_transformer_policy_robotwin.yaml
with a ResNet backbone and feature_aggregation=null): every spatial cell of each
camera's layer-4 feature map is one token, plus one state token. The vision
fields keep LeRobot DiffusionConfig names and defaults, so the backbone
(ImageNet ResNet-18, one per camera) and crop match the Diffusion Policy arm.
The action head, objective, EMA and optimizer follow maniflow_image_policy.py
and the official training workspace.
"""

from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamWConfig, DiffuserSchedulerConfig

FLOW_T_MODES = ("beta", "uniform", "lognorm")
CONSISTENCY_T_MODES = ("discrete", "uniform")
TARGET_T_MODES = ("relative", "absolute")


@PreTrainedConfig.register_subclass("maniflow")
@dataclass
class ManiFlowConfig(PreTrainedConfig):
    # Inputs / output structure.
    n_obs_steps: int = 1
    horizon: int = 64
    n_action_steps: int = 32

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )
    drop_n_last_frames: int = 0

    # Observation backbone and crop: identical fields and defaults to DiffusionConfig.
    vision_backbone: str = "resnet18"
    resize_shape: tuple[int, int] | None = None
    crop_ratio: float = 1.0
    crop_shape: tuple[int, int] | None = None
    crop_is_random: bool = True
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    use_group_norm: bool = False
    use_separate_rgb_encoder_per_camera: bool = True

    # DiT-X action model.
    n_layer: int = 12
    n_head: int = 8
    n_emb: int = 768
    mlp_ratio: float = 4.0
    p_drop_attn: float = 0.1
    qkv_bias: bool = True
    qk_norm: bool = True
    timestep_embed_dim: int = 128
    target_t_embed_dim: int = 128

    # Consistency flow training (paper Alg. 2).
    flow_batch_ratio: float = 0.75
    consistency_batch_ratio: float = 0.25
    sample_t_mode_flow: str = "beta"
    beta_alpha: float = 1.0
    beta_beta: float = 1.5
    beta_cutoff: float = 0.999
    sample_t_mode_consistency: str = "discrete"
    denoise_timesteps: int = 10
    sample_target_t_mode: str = "relative"

    # EMA of the whole network (reference EMAModel settings). It supplies the
    # consistency targets and, as in the official evaluation, the served weights.
    ema_update_after_step: int = 0
    ema_power: float = 0.75
    ema_inv_gamma: float = 1.0
    ema_min_value: float = 0.0
    ema_max_value: float = 0.9999
    use_ema_for_inference: bool = True

    # Inference: Euler steps along the learned flow (paper Table 4 headline: 1-2).
    num_inference_steps: int = 2

    # Training presets (reference optimizer and schedule).
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-3
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500

    def __post_init__(self):
        super().__post_init__()
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(f"vision_backbone must be a ResNet, got {self.vision_backbone}")
        if self.resize_shape is not None and (
            len(self.resize_shape) != 2 or any(d <= 0 for d in self.resize_shape)
        ):
            raise ValueError(f"resize_shape must be two positive integers, got {self.resize_shape}")
        if not 0 < self.crop_ratio <= 1.0:
            raise ValueError(f"crop_ratio must be in (0, 1], got {self.crop_ratio}")
        if self.resize_shape is not None:
            self.crop_shape = (
                (
                    int(self.resize_shape[0] * self.crop_ratio),
                    int(self.resize_shape[1] * self.crop_ratio),
                )
                if self.crop_ratio < 1.0
                else None
            )
        if self.crop_shape is not None and min(self.crop_shape) <= 0:
            raise ValueError(f"crop_shape must be positive, got {self.crop_shape}")
        if self.n_emb % self.n_head:
            raise ValueError("n_emb must be divisible by n_head")
        if abs(self.flow_batch_ratio + self.consistency_batch_ratio - 1.0) > 1e-6:
            raise ValueError("flow and consistency batch ratios must sum to 1")
        if self.sample_t_mode_flow not in FLOW_T_MODES:
            raise ValueError(f"sample_t_mode_flow must be one of {FLOW_T_MODES}")
        if self.sample_t_mode_consistency not in CONSISTENCY_T_MODES:
            raise ValueError(f"sample_t_mode_consistency must be one of {CONSISTENCY_T_MODES}")
        if self.sample_target_t_mode not in TARGET_T_MODES:
            raise ValueError(f"sample_target_t_mode must be one of {TARGET_T_MODES}")
        if self.denoise_timesteps <= 0 or self.num_inference_steps <= 0:
            raise ValueError("denoise_timesteps and num_inference_steps must be positive")
        if self.n_action_steps > self.horizon - self.n_obs_steps + 1:
            raise ValueError("n_action_steps must fit within the predicted horizon")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> DiffuserSchedulerConfig:
        return DiffuserSchedulerConfig(
            name=self.scheduler_name, num_warmup_steps=self.scheduler_warmup_steps
        )

    def validate_features(self) -> None:
        if not self.image_features:
            raise ValueError("ManiFlow 2D requires at least one image input")
        if self.robot_state_feature is None:
            raise ValueError("ManiFlow requires observation.state")
        first = next(iter(self.image_features.values())).shape
        if any(feature.shape != first for feature in self.image_features.values()):
            raise ValueError("all camera inputs must share one shape")
        if self.resize_shape is None and self.crop_shape is not None:
            if self.crop_shape[0] > first[1] or self.crop_shape[1] > first[2]:
                raise ValueError(f"crop_shape {self.crop_shape} exceeds image shape {first}")

    @property
    def observation_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1 - self.n_obs_steps + self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None
