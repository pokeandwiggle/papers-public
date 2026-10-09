"""Relative (delta) TCP actions with rotation vectors, and their integration.

SMILE's benchmarks (Park et al., 2026, Sec. IV-B) use relative end-effector
actions: per-step translation and rotation increments with absolute gripper
commands. Our robot is commanded with absolute TCP targets, so each frame's
action is stored relative to the previously commanded target (the first frame
of an episode relative to the measured pose, as no earlier command exists):

    dp_j = p_j - p_ref,    dr_j = log(R_ref^T R_j),    g_j absolute,

with p_ref, R_ref the previous command. Rotation increments are body-frame
rotation vectors, so composing them, exp(R(t)) = exp(R(t-1)) exp(dr_t), is the
exact inverse of the paper's Eq. (6). Serving integrates a predicted chunk from
the robot's current commanded target back to absolute targets.

Per arm the action is [dp(3), dr(3), gripper(1)]; arms are ordered left, right.
"""

import numpy as np

from paw_baseline.layout import PAW_DIM
from paw_baseline.rotation_6d import matrix_to_quat_xyzw, quat_xyzw_to_matrix

DELTA_DIM = 14
DELTA_ARM_DIMS = {"left": tuple(range(0, 7)), "right": tuple(range(7, 14))}
DELTA_TRANSLATION_DIMS = (0, 1, 2, 7, 8, 9)
DELTA_ROTATION_DIMS = (3, 4, 5, 10, 11, 12)
DELTA_GRIPPER_DIMS = (6, 13)
ACTION_REPRESENTATION = "delta_rotvec_v1"
# Train-order flat-16 pose slices: (position, quaternion, gripper) per arm.
_TRAIN_ARMS = (
    (slice(0, 3), slice(3, 7), 7),
    (slice(8, 11), slice(11, 15), 15),
)
_DELTA_ARMS = (
    (slice(0, 3), slice(3, 6), 6),
    (slice(7, 10), slice(10, 13), 13),
)
_SMALL_ANGLE = 1e-6


def delta_names() -> list[str]:
    return [
        f"{arm}.{name}"
        for arm in ("left", "right")
        for name in ("dpos_x", "dpos_y", "dpos_z", "drot_x", "drot_y", "drot_z", "gripper_width")
    ]


def rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    """Rodrigues' formula for (..., 3) rotation vectors."""
    rotvec = np.asarray(rotvec, dtype=np.float64)
    theta = np.linalg.norm(rotvec, axis=-1, keepdims=True)[..., None]
    small = theta < _SMALL_ANGLE
    safe = np.where(small, 1.0, theta)
    a = np.where(small, 1.0 - theta**2 / 6.0, np.sin(safe) / safe)
    b = np.where(small, 0.5 - theta**2 / 24.0, (1.0 - np.cos(safe)) / safe**2)
    x, y, z = np.moveaxis(rotvec, -1, 0)
    zero = np.zeros_like(x)
    k = np.stack(
        [
            np.stack([zero, -z, y], axis=-1),
            np.stack([z, zero, -x], axis=-1),
            np.stack([-y, x, zero], axis=-1),
        ],
        axis=-2,
    )
    eye = np.broadcast_to(np.eye(3), k.shape)
    return eye + a * k + b * (k @ k)


def quat_to_rotvec(quat: np.ndarray) -> np.ndarray:
    """Principal rotation vector (angle in [0, pi]) of (..., 4) xyzw quaternions."""
    quat = np.asarray(quat, dtype=np.float64)
    quat = quat * np.where(quat[..., 3:4] < 0, -1.0, 1.0)
    vec, w = quat[..., :3], quat[..., 3:4]
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(norm, w)
    small = norm < _SMALL_ANGLE
    scale = np.where(small, 2.0 / np.maximum(w, _SMALL_ANGLE), angle / np.where(small, 1.0, norm))
    return vec * scale


def matrix_to_rotvec(matrix: np.ndarray) -> np.ndarray:
    """Principal rotation vector (angle in [0, pi]) of (..., 3, 3) rotations."""
    return quat_to_rotvec(matrix_to_quat_xyzw(matrix))


def rotvec_to_quat(rotvec: np.ndarray) -> np.ndarray:
    """Unit xyzw quaternions exp(rotvec / 2)."""
    rotvec = np.asarray(rotvec, dtype=np.float64)
    theta = np.linalg.norm(rotvec, axis=-1, keepdims=True)
    small = theta < _SMALL_ANGLE
    safe = np.where(small, 1.0, theta)
    half_sinc = np.where(small, 0.5 - theta**2 / 48.0, np.sin(safe / 2.0) / safe)
    return np.concatenate([rotvec * half_sinc, np.cos(theta / 2.0)], axis=-1)


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of xyzw quaternions: R(a b) = R(a) R(b)."""
    ax, ay, az, aw = np.moveaxis(np.asarray(a, dtype=np.float64), -1, 0)
    bx, by, bz, bw = np.moveaxis(np.asarray(b, dtype=np.float64), -1, 0)
    return np.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        axis=-1,
    )


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return np.concatenate([-q[..., :3], q[..., 3:]], axis=-1)


def cumulative_trajectory(deltas: np.ndarray) -> np.ndarray:
    """Cumulative chunk trajectory (..., F, 14) from per-step deltas (Sec. IV-B).

    Translation: P(t) = sum_{i<=t} dp_i. Rotation: exp(R(t)) = exp(R(t-1)) exp(dr_t)
    with R(0) = 0, as rotation vectors. Gripper commands stay absolute.
    """
    deltas = np.asarray(deltas, dtype=np.float64)
    out = deltas.copy()
    for dpos, drot, _ in _DELTA_ARMS:
        out[..., dpos] = np.cumsum(deltas[..., dpos], axis=-2)
        increments = rotvec_to_quat(deltas[..., drot])
        q = np.zeros(deltas.shape[:-2] + (4,))
        q[..., 3] = 1.0
        for t in range(deltas.shape[-2]):
            q = quat_multiply(q, increments[..., t, :])
            out[..., t, drot] = quat_to_rotvec(q)
    return out


def deltas_from_cumulative(cumulative: np.ndarray) -> np.ndarray:
    """Eq. (5)-(6): per-step deltas from a cumulative trajectory with zero start."""
    cumulative = np.asarray(cumulative, dtype=np.float64)
    out = cumulative.copy()
    for dpos, drot, _ in _DELTA_ARMS:
        path = cumulative[..., dpos]
        out[..., dpos] = np.diff(path, axis=-2, prepend=np.zeros_like(path[..., :1, :]))
        quats = rotvec_to_quat(cumulative[..., drot])
        previous = np.concatenate(
            [np.broadcast_to([0.0, 0.0, 0.0, 1.0], quats[..., :1, :].shape), quats[..., :-1, :]],
            axis=-2,
        )
        out[..., drot] = quat_to_rotvec(quat_multiply(quat_conjugate(previous), quats))
    return out


def encode_deltas(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    """Per-frame deltas (T, 14) for one episode of train-order flat-16 vectors.

    Frame 0 is relative to the measured state; frame j > 0 to command j - 1.
    """
    state = np.asarray(state, dtype=np.float64)
    action = np.asarray(action, dtype=np.float64)
    if state.shape != action.shape or state.ndim != 2 or state.shape[1] != PAW_DIM:
        raise ValueError("encode_deltas expects matching (T, 16) state and action")
    reference = np.concatenate([state[:1], action[:-1]], axis=0)
    deltas = np.zeros((len(action), DELTA_DIM), dtype=np.float64)
    for (pos, quat, grip), (dpos, drot, dgrip) in zip(_TRAIN_ARMS, _DELTA_ARMS, strict=True):
        deltas[:, dpos] = action[:, pos] - reference[:, pos]
        ref_rot = quat_xyzw_to_matrix(reference[:, quat])
        rot = quat_xyzw_to_matrix(action[:, quat])
        deltas[:, drot] = matrix_to_rotvec(np.swapaxes(ref_rot, -1, -2) @ rot)
        deltas[:, dgrip] = action[:, grip]
    return deltas


def integrate_deltas(reference: np.ndarray, deltas: np.ndarray) -> np.ndarray:
    """Absolute train-order targets (..., T, 16) from a (..., 16) reference pose.

    p_t = p_{t-1} + dp_t and R_t = R_{t-1} exp(dr_t), starting at the reference;
    gripper widths are absolute and pass through.
    """
    reference = np.asarray(reference, dtype=np.float64)
    deltas = np.asarray(deltas, dtype=np.float64)
    if reference.shape[-1] != PAW_DIM or deltas.shape[-1] != DELTA_DIM:
        raise ValueError("integrate_deltas expects a (..., 16) reference and (..., T, 14) deltas")
    out = np.zeros(deltas.shape[:-1] + (PAW_DIM,), dtype=np.float64)
    for (pos, quat, grip), (dpos, drot, dgrip) in zip(_TRAIN_ARMS, _DELTA_ARMS, strict=True):
        out[..., pos] = reference[..., None, pos] + np.cumsum(deltas[..., dpos], axis=-2)
        rot = quat_xyzw_to_matrix(reference[..., quat])
        increments = rotvec_to_matrix(deltas[..., drot])
        mats = np.empty(increments.shape, dtype=np.float64)
        for t in range(deltas.shape[-2]):
            rot = rot @ increments[..., t, :, :]
            mats[..., t, :, :] = rot
        out[..., quat] = matrix_to_quat_xyzw(mats)
        out[..., grip] = deltas[..., dgrip]
    return out
