"""Physical quaternion checks before conversion to continuous model rotations."""

import numpy as np

from paw_baseline.layout import PAW_DIM, TRAIN_QUAT_SLICES

UNIT_QUAT_TOL = 1e-3


def canonical_quaternions(values: np.ndarray) -> np.ndarray:
    """Return unit xyzw quaternions with their largest-magnitude component positive.

    Equal magnitudes use the first component in xyzw order. No history or
    normalization is used: q and -q produce the same float32 values. This
    convention has discontinuities where the dominant component changes.
    Invalid rotations are rejected rather than repaired.
    """
    quats = np.asarray(values, dtype=np.float32).copy()
    if quats.ndim < 1 or quats.shape[-1] != 4:
        raise ValueError(f"expected (..., 4) quaternion array, got {quats.shape}")
    if not np.isfinite(quats).all():
        raise RuntimeError("non-finite quaternion values")
    norms = np.linalg.norm(quats.astype(np.float64), axis=-1)
    if np.any(np.abs(norms - 1.0) > UNIT_QUAT_TOL):
        raise RuntimeError("quaternions are not unit")
    index = np.argmax(np.abs(quats), axis=-1, keepdims=True)
    pivot = np.take_along_axis(quats, index, axis=-1)
    quats *= np.where(pivot < 0, -1.0, 1.0).astype(np.float32)
    quats[quats == 0] = 0.0  # Signed zero must not preserve the incoming sign.
    return quats


def canonical_flat16(values: np.ndarray) -> np.ndarray:
    """Canonicalize both arm rotations in train-order state or action vectors."""
    result = np.asarray(values, dtype=np.float32).copy()
    if result.ndim < 1 or result.shape[-1] != PAW_DIM:
        raise ValueError(f"expected (..., {PAW_DIM}) state/action, got {result.shape}")
    for sl in TRAIN_QUAT_SLICES:
        result[..., sl] = canonical_quaternions(result[..., sl])
    return result
