"""Configure a Florence-2 discrete-token VLA with BIN or DSD action tokens."""

from dataclasses import dataclass, field

from lerobot.configs import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig

from paw_baseline.florence_vla.actions import BIN_Q01_KEY, REPRESENTATIONS, parse_stats
from paw_baseline.layout import CAMERA_KEYS

VLM_REPO = "florence-community/Florence-2-base"
VLM_REVISION = "00921df66db728a9ceb750f5eca43e5c203a2051"
#: BIN percentile setting -> statistics group.
BIN_PERCENTILES = {"0.1-99.9": "bin", "1-99": BIN_Q01_KEY}


@PreTrainedConfig.register_subclass("florence_vla")
@dataclass
class FlorenceVLAConfig(PreTrainedConfig):
    """Florence-2-Base, parallel decoding of one discrete distribution per token.

    ``action_representation`` selects the raw per-step increment encoding
    (``bin``) or Direction-Scale Decomposition (``dsd``); every other setting is
    shared. ``action_stats`` holds training-data bounds (written by training).
    State/action/image features bypass LeRobot normalization: the policy
    integrates increments from the physical 20D state and owns its own image
    and action normalization.
    """

    n_obs_steps: int = 1
    chunk_size: int = 30
    n_action_steps: int = 30

    action_representation: str = "bin"
    # BIN percentile bounds: "0.1-99.9" (default) or OpenVLA's clipped "1-99".
    bin_percentiles: str = "0.1-99.9"
    num_bins: int = 256
    action_stats: str = ""
    # DSD direction-loss masking (Sec. V-C): fraction of the scale range and floor.
    dsd_tau_frac: float = 0.003
    dsd_min_direction_weight: float = 0.1

    vlm_repo: str = VLM_REPO
    vlm_revision: str = VLM_REVISION
    image_size: int = 224
    # Training-only random shift (pixels at source resolution), as BEAST-F's RandomShiftsAug.
    random_shift_pad: int = 8
    prompt_template: str = (
        "Agent Type: 2-arm robot, Action Space: Delta End-Effector, Task Instruction: {task}"
    )
    max_text_tokens: int = 64

    # Drives LeRobot's Accelerate mixed precision; weights and optimizer stay FP32.
    dtype: str = "bfloat16"

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    optimizer_lr: float = 2e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.0
    optimizer_grad_clip_norm: float = 1.0
    scheduler_warmup_steps: int = 1000
    scheduler_decay_steps: int = 40000
    scheduler_decay_lr: float = 2e-6

    def __post_init__(self):
        super().__post_init__()
        if self.action_representation not in REPRESENTATIONS:
            raise ValueError(f"action_representation must be one of {REPRESENTATIONS}")
        if self.bin_percentiles not in BIN_PERCENTILES:
            raise ValueError(f"bin_percentiles must be one of {tuple(BIN_PERCENTILES)}")
        if self.action_representation != "bin" and self.bin_percentiles != "0.1-99.9":
            raise ValueError("bin_percentiles applies only to the bin representation")
        if not 1 <= self.n_action_steps <= self.chunk_size:
            raise ValueError("n_action_steps must be in [1, chunk_size]")
        if self.n_obs_steps != 1:
            raise ValueError("florence_vla uses one observation")
        if self.num_bins < 2:
            raise ValueError("num_bins must be at least 2")
        if self.dtype not in ("bfloat16", "float32"):
            raise ValueError("dtype must be bfloat16 or float32")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    def validate_features(self) -> None:
        if tuple(sorted(self.image_features)) != tuple(sorted(CAMERA_KEYS)):
            raise ValueError(f"florence_vla requires cameras {CAMERA_KEYS}")
        if self.robot_state_feature is None or self.robot_state_feature.shape != (20,):
            raise ValueError("florence_vla requires the 20D TCP state")
        if self.action_feature is None or self.action_feature.shape != (20,):
            raise ValueError("florence_vla requires 20D TCP actions")
        if self.stats_key not in parse_stats(self.action_stats):
            raise ValueError(f"action_stats lack the {self.stats_key!r} bounds")

    @property
    def stats_key(self) -> str:
        """Statistics group bounding the selected representation's tokens."""
        if self.action_representation == "bin":
            return BIN_PERCENTILES[self.bin_percentiles]
        return self.action_representation

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
