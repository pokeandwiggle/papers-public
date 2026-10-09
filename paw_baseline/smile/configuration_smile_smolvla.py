"""Configure SMILE-SmolVLA: SmolVLA with a B-spline coefficient action interface."""

from dataclasses import dataclass, field

from lerobot.configs import PreTrainedConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

from paw_baseline.layout import GRIPPER_MAX_WIDTH_M
from paw_baseline.smile.bspline import (
    GRIPPER_DIMS,
    OBSERVABILITY_TOL,
    ROTATION_DIMS,
    TRANSLATION_DIMS,
)

BASE_REPO = "lerobot/smolvla_base"
BASE_REVISION = "c83c3163b8ca9b7e67c509fffd9121e66cb96205"


@PreTrainedConfig.register_subclass("smile_smolvla")
@dataclass
class SmileSmolVLAConfig(SmolVLAConfig):
    """SmolVLA fields default to the pinned smolvla_base config.json.

    The raw-action baseline loads that file through ``--policy.path``; these four
    overrides make a fresh SMILE config identical outside the SMILE fields.
    """

    load_vlm_weights: bool = True
    prefix_length: int = 0
    pad_language_to: str = "max_length"
    num_expert_layers: int = 0

    base_repo: str = BASE_REPO
    base_revision: str = BASE_REVISION

    # Paper defaults (Table IV): K = 8 control points, cubic splines.
    num_control_points: int = 8
    spline_degree: int = 3
    # Dataset-level (K, D) coefficient statistics, flattened row-major; set by train.
    coef_mean: list[float] | None = None
    coef_std: list[float] | None = None
    # The per-frame action MEAN_STD statistics of the shared SmolVLA processors;
    # set by train so targets are built from raw deltas.
    action_mean: list[float] | None = None
    action_std: list[float] | None = None
    # 14D relative action: per arm translation delta, rotation-vector delta, gripper.
    translation_dims: list[int] = field(default_factory=lambda: list(TRANSLATION_DIMS))
    rotation_dims: list[int] = field(default_factory=lambda: list(ROTATION_DIMS))
    gripper_dims: list[int] = field(default_factory=lambda: list(GRIPPER_DIMS))
    # Eq. (4): decoded gripper commands are clipped to the controller's range.
    gripper_min: float = 0.0
    gripper_max: float = GRIPPER_MAX_WIDTH_M
    observability_tol: float = OBSERVABILITY_TOL
    # Eq. (9): lambda_tr = lambda_rot = lambda_grip = 1.
    loss_weight_translation: float = 1.0
    loss_weight_rotation: float = 1.0
    loss_weight_gripper: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        if self.num_control_points <= self.spline_degree or self.spline_degree < 1:
            raise ValueError("SMILE needs degree >= 1 and K > degree")
        if self.chunk_size < self.num_control_points:
            raise ValueError("SMILE chunk_size must be at least num_control_points")

    def modality_dims(self) -> dict[str, list[int]]:
        return {
            "translation": list(self.translation_dims),
            "rotation": list(self.rotation_dims),
            "gripper": list(self.gripper_dims),
        }

    def validate_features(self) -> None:
        super().validate_features()
        if self.rtc_config is not None:
            raise ValueError("SMILE uses synchronous fixed-horizon execution, not RTC")
        if self.adapt_to_pi_aloha:
            raise ValueError("SMILE does not support the Aloha adapter")
        dims = self.action_feature.shape[0]
        groups = self.modality_dims().values()
        if sorted(d for group in groups for d in group) != list(range(dims)):
            raise ValueError("SMILE modality dims must partition the action dims")
        if self.translation_dims != list(TRANSLATION_DIMS) or self.rotation_dims != list(
            ROTATION_DIMS
        ):
            raise ValueError("SMILE trajectories require the 14D relative action layout")
        expected = {
            "coef_mean": self.num_control_points * dims,
            "coef_std": self.num_control_points * dims,
            "action_mean": dims,
            "action_std": dims,
        }
        for name, count in expected.items():
            values = getattr(self, name)
            if values is None or len(values) != count:
                raise ValueError(
                    f"SMILE requires {name} with {count} values from training data"
                )
        if min(self.coef_std) <= 0:
            raise ValueError("SMILE coef_std must be positive")
