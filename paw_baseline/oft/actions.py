"""Per-step delta actions shared by OFT statistics, training and serving.

The prepared dataset stores absolute model vectors: per arm ``[x, y, z, r00, r01,
r02, r10, r11, r12, gripper_width]`` (first two rotation-matrix rows), left arm in
dims 0-9 and right arm in dims 10-19. OpenVLA-OFT and ActionMap predict per-step
end-effector deltas ("per-step velocity") and a binary gripper command. For each
arm and chunk step ``k``::

    dpos_k = p_k - p_{k-1}
    drot_k = log(R_k R_{k-1}^T)   # base-frame rotation vector
    grip_k = +1 (open) if width_k > GRIPPER_THRESHOLD_M else -1 (close)

with ``(p_{-1}, R_{-1})`` the measured TCP pose of the observation. Serving
integrates the deltas from the live pose back into absolute targets. The 14
model dimensions are ``[left dpos(3), drot(3), grip, right dpos(3), drot(3), grip]``.
"""

import torch

from paw_baseline.layout import GRIPPER_MAX_WIDTH_M

ARMS = ("left", "right")
MODEL_ARM_OFFSET = {"left": 0, "right": 10}
ARM_DELTA_DIM = 7
ACTION_DIM = ARM_DELTA_DIM * len(ARMS)
GRIPPER_DIMS = (6, 13)
GRIPPER_THRESHOLD_M = GRIPPER_MAX_WIDTH_M / 2


def rows_to_matrix(rows: torch.Tensor) -> torch.Tensor:
    """Row-wise Gram-Schmidt of ``(..., 6)`` rotation rows to ``(..., 3, 3)``."""
    first, second = rows[..., :3], rows[..., 3:6]
    first = first / first.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    second = second - (first * second).sum(-1, keepdim=True) * first
    second = second / second.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.stack((first, second, torch.linalg.cross(first, second)), dim=-2)


def matrix_to_rows(matrix: torch.Tensor) -> torch.Tensor:
    """First two rows of ``(..., 3, 3)`` matrices as ``(..., 6)``."""
    return matrix[..., :2, :].reshape(*matrix.shape[:-2], 6)


def _matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """Shepperd conversion to ``(..., 4)`` wxyz quaternions with ``w >= 0``."""
    m = matrix
    trace = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    candidates = torch.stack(
        (
            1.0 + trace,
            1.0 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2],
            1.0 - m[..., 0, 0] + m[..., 1, 1] - m[..., 2, 2],
            1.0 - m[..., 0, 0] - m[..., 1, 1] + m[..., 2, 2],
        ),
        dim=-1,
    )
    branch = candidates.argmax(dim=-1, keepdim=True)
    root = candidates.gather(-1, branch).clamp_min(1e-12).sqrt()  # 2 * |q_branch|
    d21 = m[..., 2, 1] - m[..., 1, 2]
    d02 = m[..., 0, 2] - m[..., 2, 0]
    d10 = m[..., 1, 0] - m[..., 0, 1]
    s01 = m[..., 0, 1] + m[..., 1, 0]
    s02 = m[..., 0, 2] + m[..., 2, 0]
    s12 = m[..., 1, 2] + m[..., 2, 1]
    options = torch.stack(
        (
            torch.stack((candidates[..., 0], d21, d02, d10), -1),
            torch.stack((d21, candidates[..., 1], s01, s02), -1),
            torch.stack((d02, s01, candidates[..., 2], s12), -1),
            torch.stack((d10, s02, s12, candidates[..., 3]), -1),
        ),
        dim=-2,
    )
    index = branch[..., None].expand(*branch.shape[:-1], 1, 4)
    quat = options.gather(-2, index).squeeze(-2) / (2.0 * root)
    return quat * torch.where(quat[..., :1] < 0, -1.0, 1.0)


def so3_log(matrix: torch.Tensor) -> torch.Tensor:
    """Rotation vectors of ``(..., 3, 3)`` rotation matrices."""
    quat = _matrix_to_quaternion(matrix)
    w, vec = quat[..., :1], quat[..., 1:]
    norm = vec.norm(dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(norm, w)
    small = norm < 1e-7
    scale = torch.where(small, 2.0 / w.clamp_min(1e-12), angle / norm.clamp_min(1e-12))
    return vec * scale


def so3_exp(rotvec: torch.Tensor) -> torch.Tensor:
    """Rodrigues map from ``(..., 3)`` rotation vectors to matrices."""
    theta2 = (rotvec * rotvec).sum(-1, keepdim=True)[..., None]
    theta = theta2.sqrt()
    small = theta2 < 1e-12
    a = torch.where(small, 1.0 - theta2 / 6.0, torch.sin(theta) / theta.clamp_min(1e-12))
    b = torch.where(
        small, 0.5 - theta2 / 24.0, (1.0 - torch.cos(theta)) / theta2.clamp_min(1e-24)
    )
    x, y, z = rotvec.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack(
        (zero, -z, y, z, zero, -x, -y, x, zero), dim=-1
    ).reshape(*rotvec.shape[:-1], 3, 3)
    eye = torch.eye(3, dtype=rotvec.dtype, device=rotvec.device)
    return eye + a * skew + b * (skew @ skew)


def encode_deltas(state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """``(B, 20)`` state and ``(B, T, 20)`` absolute actions to ``(B, T, 14)`` deltas."""
    parts = []
    for arm in ARMS:
        o = MODEL_ARM_OFFSET[arm]
        pos = actions[..., o : o + 3]
        rot = rows_to_matrix(actions[..., o + 3 : o + 9])
        prev_pos = torch.cat((state[:, None, o : o + 3], pos[:, :-1]), dim=1)
        prev_rot = torch.cat(
            (rows_to_matrix(state[:, None, o + 3 : o + 9]), rot[:, :-1]), dim=1
        )
        grip = torch.where(actions[..., o + 9] > GRIPPER_THRESHOLD_M, 1.0, -1.0)
        parts.extend(
            (
                pos - prev_pos,
                so3_log(rot @ prev_rot.transpose(-1, -2)),
                grip[..., None].to(actions.dtype),
            )
        )
    return torch.cat(parts, dim=-1)


def decode_deltas(
    state: torch.Tensor,
    deltas: torch.Tensor,
    open_width: torch.Tensor,
    close_width: torch.Tensor,
) -> torch.Tensor:
    """Integrate ``(B, T, 14)`` deltas from the ``(B, 20)`` live state to absolute actions.

    Gripper commands open when the binary command is positive. ``open_width`` and
    ``close_width`` hold one commanded width per arm.
    """
    batch, horizon = deltas.shape[:2]
    out = deltas.new_empty(batch, horizon, 20)
    for index, arm in enumerate(ARMS):
        o, d = MODEL_ARM_OFFSET[arm], ARM_DELTA_DIM * index
        out[..., o : o + 3] = state[:, None, o : o + 3] + deltas[..., d : d + 3].cumsum(1)
        rot = rows_to_matrix(state[:, o + 3 : o + 9])
        steps = so3_exp(deltas[..., d + 3 : d + 6])
        for k in range(horizon):
            rot = steps[:, k] @ rot
            out[:, k, o + 3 : o + 9] = matrix_to_rows(rot)
        out[..., o + 9] = torch.where(
            deltas[..., d + 6] > 0, open_width[index], close_width[index]
        )
    return out


class BoundsNormalizer:
    """OFT ``BOUNDS_Q99`` normalization: ``[q01, q99] -> [-1, 1]`` with clipping.

    Dimensions with ``q01 == q99`` normalize to zero, as in OFT. ``raw`` dims pass
    through unchanged (OFT's unnormalized gripper mask).
    """

    def __init__(self, q01, q99, raw_dims=()):
        self.q01 = torch.as_tensor(q01, dtype=torch.float32)
        self.q99 = torch.as_tensor(q99, dtype=torch.float32)
        self.raw = torch.zeros(self.q01.shape[-1], dtype=torch.bool)
        self.raw[list(raw_dims)] = True

    def _to(self, ref: torch.Tensor):
        return (
            self.q01.to(ref.device, ref.dtype),
            self.q99.to(ref.device, ref.dtype),
            self.raw.to(ref.device),
        )

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        low, high, raw = self._to(values)
        scaled = (2 * (values - low) / (high - low + 1e-8) - 1).clamp(-1, 1)
        scaled = torch.where(high == low, torch.zeros_like(scaled), scaled)
        return torch.where(raw, values, scaled)

    def unnormalize(self, values: torch.Tensor) -> torch.Tensor:
        low, high, raw = self._to(values)
        restored = 0.5 * (values + 1) * (high - low + 1e-8) + low
        return torch.where(raw, values, restored)
