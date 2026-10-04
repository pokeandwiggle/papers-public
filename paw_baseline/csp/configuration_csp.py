"""Configure the Causal Spectral Policy (CSP).

CSP predicts an action chunk of length ``horizon`` (K) in the DCT-II domain.
A first GPT trunk regresses the lowest ``n_low_freq`` (lambda) coefficients of
the continuous pose dimensions from the observation; a second GPT trunk
regresses the remaining K - lambda coefficients conditioned on the observation
and the stop-gradient low-frequency prediction. Gripper dimensions stay in the
time domain (paper App. 7.4: "the gripper channel is concatenated unchanged")
and use a separate head on the first trunk, so the second trunk is supervised
only by the high-frequency coefficients.

lambda defaults to 8 (25% of K=32): inside the paper's stable 20-40% range and
the project page's K=32 cutoff peak (6-10). The App. 7.3 energy heuristic is
degenerate on normalized absolute poses, whose DC term holds >99% of the chunk
energy.

The perception encoder deliberately reuses LeRobot's Diffusion Policy RGB
encoder settings so that the comparison against the ``diffusion`` preset
isolates the action model. Trunk sizes follow paper Table 5 (BAKU GPT).
"""

from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamWConfig, DiffuserSchedulerConfig


@PreTrainedConfig.register_subclass("csp")
@dataclass
class CSPConfig(PreTrainedConfig):
    n_obs_steps: int = 1
    # K: chunk length transformed by the DCT; also the predicted horizon.
    horizon: int = 32
    n_action_steps: int = 32
    # lambda: coefficients c_{0:lambda} form the coarse (low-frequency) part.
    n_low_freq: int = 8
    # Model-space action dims kept in the time domain (gripper widths).
    time_domain_action_dims: tuple[int, ...] = (9, 19)
    # Filled by LeRobot's make_policy from dataset metadata; used for checks.
    action_feature_names: list[str] | None = None

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )
    drop_n_last_frames: int = 0

    # Perception: identical fields and defaults to LeRobot DiffusionConfig.
    vision_backbone: str = "resnet18"
    resize_shape: tuple[int, int] | None = None
    crop_ratio: float = 1.0
    crop_shape: tuple[int, int] | None = None
    crop_is_random: bool = True
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    use_group_norm: bool = False
    spatial_softmax_num_keypoints: int = 32
    use_separate_rgb_encoder_per_camera: bool = True

    # Trunks (paper Table 5; BAKU GPT): one per stage.
    repr_dim: int = 512
    n_embd: int = 256
    n_layer: int = 8
    n_head: int = 4
    block_size: int = 65
    dropout: float = 0.1
    head_hidden_dim: int = 256
    head_num_layers: int = 2

    # Optimizer follows BAKU (AdamW, lr 1e-4, weight decay 1e-4); the cosine
    # warmup schedule matches the Diffusion Policy preset.
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-4
    optimizer_grad_clip_norm: float = 10.0
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500

    def __post_init__(self):
        super().__post_init__()
        if self.n_obs_steps != 1:
            raise ValueError("CSP uses a single observation (n_obs_steps=1)")
        if not 1 <= self.n_low_freq < self.horizon:
            raise ValueError(
                f"n_low_freq must be in [1, horizon), got {self.n_low_freq} for horizon {self.horizon}"
            )
        if not 1 <= self.n_action_steps <= self.horizon:
            raise ValueError("n_action_steps must be in [1, horizon]")
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        if len(set(self.time_domain_action_dims)) != len(self.time_domain_action_dims):
            raise ValueError("time_domain_action_dims must be unique")
        if not (0 < self.crop_ratio <= 1.0):
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

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> DiffuserSchedulerConfig:
        return DiffuserSchedulerConfig(
            name=self.scheduler_name, num_warmup_steps=self.scheduler_warmup_steps
        )

    def validate_features(self) -> None:
        if not self.image_features:
            raise ValueError("CSP requires at least one camera")
        if self.robot_state_feature is None:
            raise ValueError("CSP requires observation.state")
        if self.action_feature is None:
            raise ValueError("CSP requires an action feature")
        action_dim = self.action_feature.shape[0]
        if any(not 0 <= d < action_dim for d in self.time_domain_action_dims):
            raise ValueError(
                f"time_domain_action_dims {self.time_domain_action_dims} outside action dim {action_dim}"
            )
        if len(self.time_domain_action_dims) >= action_dim:
            raise ValueError("CSP needs at least one spectral action dimension")
        if self.action_feature_names is not None:
            if len(self.action_feature_names) != action_dim:
                raise ValueError("action_feature_names length disagrees with action shape")
            for d in self.time_domain_action_dims:
                if not self.action_feature_names[d].endswith("gripper_width"):
                    raise ValueError(
                        f"time-domain dim {d} is {self.action_feature_names[d]!r}, expected a gripper width"
                    )
        shapes = {tuple(ft.shape) for ft in self.image_features.values()}
        if len(shapes) != 1:
            raise ValueError(f"CSP expects equally shaped cameras, got {shapes}")
        if self.resize_shape is None and self.crop_shape is not None:
            (_, height, width) = next(iter(shapes))
            if self.crop_shape[0] > height or self.crop_shape[1] > width:
                raise ValueError("crop_shape must fit within the image shape")

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None
