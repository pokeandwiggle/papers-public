"""Configure LeRobot PI05 for absolute TCP state and actions."""

from dataclasses import dataclass, field

from lerobot.configs import PreTrainedConfig
from lerobot.optim.schedulers import ConstantWithWarmupSchedulerConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config

from paw_baseline.layout import CAMERA_KEYS
from paw_baseline.pi05.geometry import Pi05Representation

BASE_REPO = "lerobot/pi05_base"
BASE_REVISION = "b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba"


@PreTrainedConfig.register_subclass("paw_pi05")
@dataclass
class PawPI05Config(PI05Config):
    """Retain the pretrained 32D layers; dataset features remain physical 20D."""

    base_repo: str = BASE_REPO
    base_revision: str = BASE_REVISION
    paw_representation: dict = field(
        default_factory=lambda: Pi05Representation().to_dict()
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
