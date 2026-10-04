"""Convert xyzw quaternions to rotation-6D using the first two matrix rows.

Decoding uses row-wise Gram-Schmidt. Training and serving must use the
same row convention; taking columns instead represents a different rotation.
"""

from __future__ import annotations

import numpy as np

#: Width of a pose in each encoding: position plus its rotation.
QUAT_POSE_DIM = 7
ROT6D_POSE_DIM = 9


def quat_xyzw_to_matrix(quat: np.ndarray) -> np.ndarray:
    """Rotation matrices for a ``(..., 4)`` array of xyzw quaternions.

    Uses NumPy without requiring SciPy.
    """
    quat = np.asarray(quat, dtype=np.float64)
    if quat.shape[-1] != 4:
        raise ValueError(f"expected xyzw quaternions of width 4, got {quat.shape[-1]}")
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    if np.any(norm == 0):
        raise ValueError("a zero quaternion has no rotation")
    x, y, z, w = np.moveaxis(quat / norm, -1, 0)
    matrix = np.empty(quat.shape[:-1] + (3, 3), dtype=np.float64)
    matrix[..., 0, 0] = 1 - 2 * (y * y + z * z)
    matrix[..., 0, 1] = 2 * (x * y - z * w)
    matrix[..., 0, 2] = 2 * (x * z + y * w)
    matrix[..., 1, 0] = 2 * (x * y + z * w)
    matrix[..., 1, 1] = 1 - 2 * (x * x + z * z)
    matrix[..., 1, 2] = 2 * (y * z - x * w)
    matrix[..., 2, 0] = 2 * (x * z - y * w)
    matrix[..., 2, 1] = 2 * (y * z + x * w)
    matrix[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return matrix


def matrix_to_quat_xyzw(matrix: np.ndarray) -> np.ndarray:
    """xyzw quaternions for a ``(..., 3, 3)`` array of rotation matrices.

    Shepperd's method: pick the branch whose divisor is largest, so no branch divides
    by something near zero. The sign convention is ``w >= 0``, which makes the round
    trip a function rather than a choice between two equivalent answers.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"expected (..., 3, 3) rotation matrices, got {matrix.shape}")
    m = np.moveaxis(np.moveaxis(matrix, -1, 0), -1, 0)
    trace = m[0][0] + m[1][1] + m[2][2]
    candidates = np.stack(
        [
            1.0 + trace,
            1.0 + m[0][0] - m[1][1] - m[2][2],
            1.0 - m[0][0] + m[1][1] - m[2][2],
            1.0 - m[0][0] - m[1][1] + m[2][2],
        ],
        axis=-1,
    )
    branch = np.argmax(candidates, axis=-1)
    scale = 0.5 / np.sqrt(
        np.maximum(np.take_along_axis(candidates, branch[..., None], -1)[..., 0], 1e-12)
    )

    w = np.where(branch == 0, 0.25 / scale, 0.0)
    x = np.zeros_like(w)
    y = np.zeros_like(w)
    z = np.zeros_like(w)

    trace_branch = branch == 0
    x = np.where(trace_branch, (m[2][1] - m[1][2]) * scale, x)
    y = np.where(trace_branch, (m[0][2] - m[2][0]) * scale, y)
    z = np.where(trace_branch, (m[1][0] - m[0][1]) * scale, z)

    x_branch = branch == 1
    w = np.where(x_branch, (m[2][1] - m[1][2]) * scale, w)
    x = np.where(x_branch, 0.25 / scale, x)
    y = np.where(x_branch, (m[0][1] + m[1][0]) * scale, y)
    z = np.where(x_branch, (m[0][2] + m[2][0]) * scale, z)

    y_branch = branch == 2
    w = np.where(y_branch, (m[0][2] - m[2][0]) * scale, w)
    x = np.where(y_branch, (m[0][1] + m[1][0]) * scale, x)
    y = np.where(y_branch, 0.25 / scale, y)
    z = np.where(y_branch, (m[1][2] + m[2][1]) * scale, z)

    z_branch = branch == 3
    w = np.where(z_branch, (m[1][0] - m[0][1]) * scale, w)
    x = np.where(z_branch, (m[0][2] + m[2][0]) * scale, x)
    y = np.where(z_branch, (m[1][2] + m[2][1]) * scale, y)
    z = np.where(z_branch, 0.25 / scale, z)

    quat = np.stack([x, y, z, w], axis=-1)
    quat /= np.linalg.norm(quat, axis=-1, keepdims=True)
    # q and -q are the same rotation; pinning the sign keeps the round trip exact.
    return np.where(quat[..., 3:4] < 0, -quat, quat)


def matrix_to_rot6d(matrix: np.ndarray) -> np.ndarray:
    """The first two rows of each rotation matrix, flattened."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"expected (..., 3, 3) rotation matrices, got {matrix.shape}")
    return matrix[..., :2, :].reshape(matrix.shape[:-2] + (6,))


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Rotation matrices from 6D rows, using Gram-Schmidt."""
    rot6d = np.asarray(rot6d, dtype=np.float64)
    if rot6d.shape[-1] != 6:
        raise ValueError(f"expected 6D rotations of width 6, got {rot6d.shape[-1]}")
    rows = rot6d.reshape(rot6d.shape[:-1] + (2, 3))
    first = rows[..., 0, :]
    first_norm = np.linalg.norm(first, axis=-1, keepdims=True)
    if np.any(first_norm == 0):
        raise ValueError("a zero 6D row names no rotation")
    first = first / first_norm
    second = rows[..., 1, :]
    second = second - np.sum(first * second, axis=-1, keepdims=True) * first
    # Zero here means the two rows were parallel, so the second carries no direction the
    # first does not already give. Predicted rotations can land there, and dividing
    # would return a silently non-finite matrix rather than refusing.
    second_norm = np.linalg.norm(second, axis=-1, keepdims=True)
    if np.any(second_norm == 0):
        raise ValueError("the 6D rows are parallel and name no rotation")
    second = second / second_norm
    third = np.cross(first, second)
    return np.stack([first, second, third], axis=-2)


def quat_pose_to_rot6d_pose(pose: np.ndarray) -> np.ndarray:
    """``(..., 7)`` xyz+quat_xyzw poses as ``(..., 9)`` xyz+rot6d poses."""
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape[-1] != QUAT_POSE_DIM:
        raise ValueError(
            f"expected {QUAT_POSE_DIM}-dim xyz+quat poses, got {pose.shape[-1]}"
        )
    rot6d = matrix_to_rot6d(quat_xyzw_to_matrix(pose[..., 3:]))
    return np.concatenate([pose[..., :3], rot6d], axis=-1)


def rot6d_pose_to_quat_pose(pose: np.ndarray) -> np.ndarray:
    """``(..., 9)`` xyz+rot6d poses as ``(..., 7)`` xyz+quat_xyzw poses."""
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape[-1] != ROT6D_POSE_DIM:
        raise ValueError(
            f"expected {ROT6D_POSE_DIM}-dim xyz+rot6d poses, got {pose.shape[-1]}"
        )
    quat = matrix_to_quat_xyzw(rot6d_to_matrix(pose[..., 3:]))
    return np.concatenate([pose[..., :3], quat], axis=-1)
