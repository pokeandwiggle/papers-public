"""Per-step delta end-effector actions at the 30 Hz grid.

Absolute model vectors (20D: per arm xyz, two rotation-matrix rows, gripper
width) become 14D delta actions: per arm a base-frame translation change
(3), a base-frame rotation-vector change (3, R_k = Exp(dr_k) R_prev) and the
absolute gripper width (1). Each action is the pose change from the previous
step; for the first predicted (current-step) action the previous pose is the
robot's measured pose at the planning observation, so serving accumulates the
chunk from that pose. Past-token actions chain from the preceding recorded
action, which the dataset window supplies as one extra leading action.

Training uses the torch functions; serving accumulates in float64 numpy.
"""

import numpy as np
import torch

DELTA_DIM = 14
# Per arm slices in the 20D model vector and the 14D delta vector.
MODEL_POS = (slice(0, 3), slice(10, 13))
MODEL_ROT = (slice(3, 9), slice(13, 19))
MODEL_GRIP = (9, 19)
DELTA_POS = (slice(0, 3), slice(7, 10))
DELTA_ROT = (slice(3, 6), slice(10, 13))
DELTA_GRIP = (6, 13)


# ---------------------------------------------------------------- torch (training)


def rot6d_to_matrix_torch(rot6d: torch.Tensor) -> torch.Tensor:
    """Gram-Schmidt rows, matching paw_baseline.rotation_6d.rot6d_to_matrix."""
    rows = rot6d.reshape(*rot6d.shape[:-1], 2, 3)
    first = torch.nn.functional.normalize(rows[..., 0, :], dim=-1)
    second = rows[..., 1, :]
    second = second - (first * second).sum(-1, keepdim=True) * first
    second = torch.nn.functional.normalize(second, dim=-1)
    third = torch.cross(first, second, dim=-1)
    return torch.stack([first, second, third], dim=-2)


def matrix_log_torch(matrix: torch.Tensor) -> torch.Tensor:
    """Rotation vectors of rotation matrices (angles in [0, pi))."""
    trace = matrix[..., 0, 0] + matrix[..., 1, 1] + matrix[..., 2, 2]
    cos = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
    angle = torch.acos(cos)
    skew = torch.stack(
        [
            matrix[..., 2, 1] - matrix[..., 1, 2],
            matrix[..., 0, 2] - matrix[..., 2, 0],
            matrix[..., 1, 0] - matrix[..., 0, 1],
        ],
        dim=-1,
    )
    sin = torch.sin(angle)
    # angle / (2 sin angle) -> 1/2 as angle -> 0 (series for small angles).
    small = angle < 1e-4
    factor = torch.where(
        small,
        0.5 + angle**2 / 12.0,
        angle / (2.0 * torch.where(small, torch.ones_like(sin), sin)),
    )
    return skew * factor.unsqueeze(-1)


def pose_deltas(previous: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    """(..., 20) previous and current absolute vectors -> (..., 14) deltas."""
    out = []
    for arm in range(2):
        pos = current[..., MODEL_POS[arm]] - previous[..., MODEL_POS[arm]]
        rot_now = rot6d_to_matrix_torch(current[..., MODEL_ROT[arm]])
        rot_prev = rot6d_to_matrix_torch(previous[..., MODEL_ROT[arm]])
        rotvec = matrix_log_torch(rot_now @ rot_prev.transpose(-1, -2))
        grip = current[..., MODEL_GRIP[arm] : MODEL_GRIP[arm] + 1]
        out.extend([pos, rotvec, grip])
    return torch.cat(out, dim=-1)


def absolute_to_deltas(state: torch.Tensor, actions: torch.Tensor, past_steps: int) -> torch.Tensor:
    """(B, 20) current state and (B, 1 + H, 20) actions -> (B, H, 14) deltas.

    ``actions[:, 0]`` is the action before the window; ``actions[:, 1 + past_steps]``
    is the current-step action, whose previous pose is ``state``.
    """
    previous = actions[:, :-1].clone()
    previous[:, past_steps] = state
    return pose_deltas(previous, actions[:, 1:])


def delta_statistics(episodes: list[tuple[np.ndarray, np.ndarray]]) -> dict[str, list[float]]:
    """Min/max of first-slot and per-step deltas over (state, action) episodes.

    First-slot deltas pair each frame's action with its measured state; per-step
    deltas pair consecutive actions. Window padding (repeated boundary actions)
    yields zero deltas, which the ranges include.
    """
    first, step = [], []
    for state, action in episodes:
        state_t = torch.from_numpy(np.asarray(state, dtype=np.float32))
        action_t = torch.from_numpy(np.asarray(action, dtype=np.float32))
        first.append(pose_deltas(state_t, action_t))
        if len(action_t) > 1:
            step.append(pose_deltas(action_t[:-1], action_t[1:]))
    first_all = torch.cat(first)
    step_all = torch.cat(step + [torch.zeros(1, DELTA_DIM)])
    return {
        "delta_first_min": first_all.min(0).values.tolist(),
        "delta_first_max": first_all.max(0).values.tolist(),
        "delta_step_min": step_all.min(0).values.tolist(),
        "delta_step_max": step_all.max(0).values.tolist(),
    }


# ---------------------------------------------------------------- numpy (serving)


def _rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    rows = np.asarray(rot6d, dtype=np.float64).reshape(*np.shape(rot6d)[:-1], 2, 3)
    first = rows[..., 0, :] / np.linalg.norm(rows[..., 0, :], axis=-1, keepdims=True)
    second = rows[..., 1, :] - np.sum(first * rows[..., 1, :], axis=-1, keepdims=True) * first
    second = second / np.linalg.norm(second, axis=-1, keepdims=True)
    return np.stack([first, second, np.cross(first, second)], axis=-2)


def rotvec_exp(rotvec: np.ndarray) -> np.ndarray:
    """Rotation matrices from (..., 3) rotation vectors (Rodrigues)."""
    rotvec = np.asarray(rotvec, dtype=np.float64)
    angle = np.linalg.norm(rotvec, axis=-1, keepdims=True)[..., None]
    x, y, z = rotvec[..., 0], rotvec[..., 1], rotvec[..., 2]
    zero = np.zeros_like(x)
    skew = np.stack(
        [
            np.stack([zero, -z, y], -1),
            np.stack([z, zero, -x], -1),
            np.stack([-y, x, zero], -1),
        ],
        -2,
    )
    small = angle < 1e-8
    safe = np.where(small, 1.0, angle)
    a = np.where(small, 1.0 - angle**2 / 6.0, np.sin(safe) / safe)
    b = np.where(small, 0.5 - angle**2 / 24.0, (1.0 - np.cos(safe)) / safe**2)
    eye = np.broadcast_to(np.eye(3), skew.shape)
    return eye + a * skew + b * (skew @ skew)


def accumulate(state: np.ndarray, deltas: np.ndarray) -> np.ndarray:
    """(20,) measured pose at the planning observation and (F, 14) deltas -> (F, 20).

    target_0 = state (+) delta_0; target_k = target_{k-1} (+) delta_k, with
    p <- p + dp and R <- Exp(dr) R in the base frame. Grippers are absolute.
    """
    state = np.asarray(state, dtype=np.float64)
    deltas = np.asarray(deltas, dtype=np.float64)
    out = np.empty((deltas.shape[0], 20), dtype=np.float64)
    for arm in range(2):
        pos = state[MODEL_POS[arm]].copy()
        rot = _rot6d_to_matrix(state[MODEL_ROT[arm]])
        steps = rotvec_exp(deltas[:, DELTA_ROT[arm]])
        for k in range(deltas.shape[0]):
            pos = pos + deltas[k, DELTA_POS[arm]]
            rot = steps[k] @ rot
            out[k, MODEL_POS[arm]] = pos
            out[k, MODEL_ROT[arm]] = rot[:2, :].reshape(6)
            out[k, MODEL_GRIP[arm]] = deltas[k, DELTA_GRIP[arm]]
    return out


def rotation_error_deg(rot6d_a: np.ndarray, rot6d_b: np.ndarray) -> np.ndarray:
    """Geodesic angle between rotations given as 6D rows."""
    relative = _rot6d_to_matrix(rot6d_a) @ np.swapaxes(_rot6d_to_matrix(rot6d_b), -1, -2)
    cos = (np.trace(relative, axis1=-2, axis2=-1) - 1.0) / 2.0
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
