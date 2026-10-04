"""Convert absolute TCP state/actions between physical 16D and model 20D vectors.

Each model arm uses xyz, two rotation-matrix rows, and gripper width in metres.
Training and serving share this conversion.
"""

import numpy as np

from paw_baseline.layout import PAW_DIM
from paw_baseline.rotation_6d import quat_pose_to_rot6d_pose, rot6d_pose_to_quat_pose
from paw_baseline.rotations import canonical_flat16

MODEL_DIM = 20
MODEL_ARM_DIMS = {"left": tuple(range(10)), "right": tuple(range(10, 20))}
MODEL_ROTATION_DIMS = (*range(3, 9), *range(13, 19))
ROTATION_CONVENTION = "rotation-matrix-first-two-rows-v1"


def encode_model(physical: np.ndarray) -> np.ndarray:
    """Convert (..., 16) train-order physical state/actions to (..., 20)."""
    physical = canonical_flat16(physical)
    if not np.isfinite(physical).all():
        raise ValueError("non-finite physical state/action")
    return np.concatenate(
        [
            quat_pose_to_rot6d_pose(physical[..., :7]),
            physical[..., 7:8],
            quat_pose_to_rot6d_pose(physical[..., 8:15]),
            physical[..., 15:16],
        ],
        axis=-1,
    ).astype(np.float32)


def decode_model(model: np.ndarray) -> np.ndarray:
    """Decode model predictions, rejecting undefined or ill-conditioned rotations."""
    model = np.asarray(model, dtype=np.float64)
    if model.ndim < 1 or model.shape[-1] != MODEL_DIM or not np.isfinite(model).all():
        raise ValueError(f"expected finite (..., {MODEL_DIM}) model actions")
    for start in (3, 13):
        first, second = model[..., start : start + 3], model[..., start + 3 : start + 6]
        first_norm = np.linalg.norm(first, axis=-1, keepdims=True)
        second_norm = np.linalg.norm(second, axis=-1, keepdims=True)
        if np.any(first_norm < 1e-6) or np.any(second_norm < 1e-6):
            raise ValueError("policy emitted a degenerate 6D rotation row")
        sine = np.linalg.norm(
            np.cross(first / first_norm, second / second_norm), axis=-1
        )
        if np.any(sine < 1e-6):
            raise ValueError("policy emitted parallel 6D rotation rows")
    return np.concatenate(
        [
            rot6d_pose_to_quat_pose(model[..., :9]),
            model[..., 9:10],
            rot6d_pose_to_quat_pose(model[..., 10:19]),
            model[..., 19:20],
        ],
        axis=-1,
    ).astype(np.float32)


def physical_bounds(model_bounds: np.ndarray) -> np.ndarray:
    """Map positional/gripper bounds for validation; rotation bounds are unused."""
    bounds = np.asarray(model_bounds)
    if bounds.shape != (MODEL_DIM,):
        raise ValueError(f"expected model bounds ({MODEL_DIM},), got {bounds.shape}")
    result = np.zeros(PAW_DIM, dtype=np.float64)
    result[[0, 1, 2, 7, 8, 9, 10, 15]] = bounds[[0, 1, 2, 9, 10, 11, 12, 19]]
    return result


def model_names() -> list[str]:
    """Names in load-bearing model arm order, with row-major rotation coordinates."""
    return [
        f"{arm}.{name}"
        for arm in ("left", "right")
        for name in (
            "pos_x",
            "pos_y",
            "pos_z",
            "r00",
            "r01",
            "r02",
            "r10",
            "r11",
            "r12",
            "gripper_width",
        )
    ]
