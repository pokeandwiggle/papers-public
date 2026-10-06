"""Configure LeRobot PI05 for absolute TCP state and actions."""

from dataclasses import dataclass, field

from lerobot.configs import PreTrainedConfig
from lerobot.optim.schedulers import ConstantWithWarmupSchedulerConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config

from paw_baseline.layout import CAMERA_KEYS
from paw_baseline.pi05.geometry import Pi05Representation, relative_action_mask

BASE_REPO = "lerobot/pi05_base"
BASE_REVISION = "b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba"

TIME_SAMPLING_BETA = "beta"
TIME_SAMPLING_LOGIT_NORMAL = "logit_normal"
# State-relative actions keep only the gripper closed fractions absolute.
RELATIVE_EXCLUDE_JOINTS = ["closed_fraction"]


@PreTrainedConfig.register_subclass("paw_pi05")
@dataclass
class PawPI05Config(PI05Config):
    """Retain the pretrained 32D layers; dataset features remain physical 20D.

    ``paw_time_sampling`` selects the training flow-matching timestep law
    (tau=1 is noise). ``beta`` keeps the native openpi Beta(1.5, 1) sampler;
    ``logit_normal`` draws ``clip(sigmoid(s * z - m), 0.001, 0.999)`` with
    z ~ N(0, 1), m = ``paw_ln_location`` and s = ``paw_ln_scale`` (FACT).
    Inference integration is unchanged.
    """

    base_repo: str = BASE_REPO
    base_revision: str = BASE_REVISION
    paw_representation: dict = field(
        default_factory=lambda: Pi05Representation().to_dict()
    )
    paw_time_sampling: str = TIME_SAMPLING_BETA
    paw_ln_location: float = 1.5
    paw_ln_scale: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        if self.paw_time_sampling not in (
            TIME_SAMPLING_BETA,
            TIME_SAMPLING_LOGIT_NORMAL,
        ):
            raise ValueError(
                f"unknown PI05 time sampling {self.paw_time_sampling!r}"
            )
        if not self.paw_ln_scale > 0:
            raise ValueError("logit-normal scale must be positive")
        if (
            self.use_relative_actions
            and self.relative_exclude_joints != RELATIVE_EXCLUDE_JOINTS
        ):
            raise ValueError(
                "relative PI05 actions keep exactly the closed fractions absolute"
            )

    def get_scheduler_preset(self) -> ConstantWithWarmupSchedulerConfig:
        """Warm up for the configured steps, then hold the optimizer learning rate."""
        return ConstantWithWarmupSchedulerConfig(
            num_warmup_steps=self.scheduler_warmup_steps
        )

    def validate_features(self):
        super().validate_features()
        Pi05Representation.from_dict(self.paw_representation)
        if self.rtc_config is not None:
            raise ValueError("PI05 requires synchronous inference")
        if (
            self.n_obs_steps != 1
            or self.max_state_dim != 32
            or self.max_action_dim != 32
        ):
            raise ValueError(
                "PI05 requires one observation and the pretrained 32D layers"
            )
        if self.robot_state_feature.shape != (20,) or self.action_feature.shape != (
            20,
        ):
            raise ValueError(
                "PI05 requires 20D TCP state and actions before padding"
            )
        if tuple(self.image_features) != CAMERA_KEYS:
            raise ValueError(
                "PI05 requires cameras in static, left wrist, right wrist order"
            )
        if self.use_relative_actions:
            from lerobot.processor import RelativeActionsProcessorStep

            mask = RelativeActionsProcessorStep(
                enabled=True,
                exclude_joints=self.relative_exclude_joints,
                action_names=self.action_feature_names,
            )._build_mask(20)
            if mask != list(relative_action_mask()):
                raise ValueError(
                    "relative PI05 actions require named closed-fraction dimensions"
                )
