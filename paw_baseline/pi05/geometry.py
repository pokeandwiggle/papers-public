"""Absolute PI TCP poses: matrix columns and closed gripper fractions."""

import numpy as np
from scipy.spatial.transform import Rotation

from paw_baseline.layout import GRIPPER_MAX_WIDTH_M
from paw_baseline.rotations import canonical_flat16

LAYOUT = "pi05-absolute-tcp20-v1"
ROTATION_CONVENTION = "rotation-matrix-first-two-columns-v1"
CLOSED_FRACTION_DIMS = (9, 19)


def relative_action_mask() -> tuple[bool, ...]:
    """State-relative dims: xyz and rotation columns; closed fractions stay absolute.

    Mirrors FACT's openpi DeltaActions mask, ee_pos(3) + rot6(6) per arm.
    """
    return tuple(dim not in CLOSED_FRACTION_DIMS for dim in range(20))


def matrix_to_columns(matrix: np.ndarray) -> np.ndarray:
    """Concatenate columns 0 and 1, in that order (not interleaved rows)."""
    return np.swapaxes(matrix[..., :, :2], -1, -2).reshape(*matrix.shape[:-2], 6)


def columns_to_matrix(columns: np.ndarray) -> np.ndarray:
    """Project predicted columns to SO(3), refusing degenerate directions."""
    first, second = columns[..., :3], columns[..., 3:]
    norm = np.linalg.norm(first, axis=-1, keepdims=True)
    if np.any(norm < 1e-6):
        raise ValueError("degenerate first rotation column")
    first = first / norm
    second = second - (first * second).sum(axis=-1, keepdims=True) * first
    norm = np.linalg.norm(second, axis=-1, keepdims=True)
    if np.any(norm < 1e-6):
        raise ValueError("degenerate parallel rotation columns")
    second = second / norm
    return np.stack((first, second, np.cross(first, second)), axis=-1)


def _matrix(quaternion: np.ndarray) -> np.ndarray:
    return (
        Rotation.from_quat(quaternion.reshape(-1, 4))
        .as_matrix()
        .reshape(*quaternion.shape[:-1], 3, 3)
    )


def _quaternion(matrix: np.ndarray) -> np.ndarray:
    return (
        Rotation.from_matrix(matrix.reshape(-1, 3, 3))
        .as_quat()
        .reshape(*matrix.shape[:-2], 4)
    )


class Pi05Representation:
    """Absolute TCP representation shared by data preparation and serving."""

    def to_dict(self) -> dict:
        """Describe the physical layout stored with the model and dataset."""
        return {
            "gripper_max_width_m": GRIPPER_MAX_WIDTH_M,
            "layout": LAYOUT,
            "rotation_convention": ROTATION_CONVENTION,
        }

    @classmethod
    def from_dict(cls, value: dict):
        """Require the exact absolute TCP representation contract."""
        representation = cls()
        if value != representation.to_dict():
            raise ValueError("incompatible PI05 representation contract")
        return representation

    def encode(self, physical: np.ndarray) -> np.ndarray:
        """Encode absolute TCP poses as rotation columns and closed fractions."""
        physical = canonical_flat16(np.asarray(physical, dtype=np.float64))
        parts = []
        for offset in (0, 8):
            parts.extend(
                (
                    physical[..., offset : offset + 3],
                    matrix_to_columns(_matrix(physical[..., offset + 3 : offset + 7])),
                    1 - physical[..., offset + 7 : offset + 8] / GRIPPER_MAX_WIDTH_M,
                )
            )
        return np.concatenate(parts, axis=-1).astype(np.float32)

    def decode(self, encoded: np.ndarray) -> np.ndarray:
        """Decode absolute model vectors to physical quaternion poses and widths."""
        encoded = np.asarray(encoded, dtype=np.float64)
        if encoded.shape[-1] != 20 or not np.isfinite(encoded).all():
            raise ValueError("PI05 vector must have 20 finite components")
        parts = []
        for offset in (0, 10):
            parts.extend(
                (
                    encoded[..., offset : offset + 3],
                    _quaternion(
                        columns_to_matrix(encoded[..., offset + 3 : offset + 9])
                    ),
                    (1 - encoded[..., offset + 9 : offset + 10]) * GRIPPER_MAX_WIDTH_M,
                )
            )
        return np.concatenate(parts, axis=-1).astype(np.float32)
