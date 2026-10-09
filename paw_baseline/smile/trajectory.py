"""Torch cumulative-trajectory maps for SMILE (Sec. IV-B, Eq. 4-6).

Mirrors paw_baseline.relative: translation increments sum, rotation-vector
increments compose in the body frame, gripper commands stay absolute. Computed
in float64; targets and decoding need no gradients.
"""

import torch
from torch import Tensor

_SMALL_ANGLE = 1e-6
# Per-arm (translation, rotation) slices of the 14D relative action.
ARM_SLICES = ((slice(0, 3), slice(3, 6)), (slice(7, 10), slice(10, 13)))


def rotvec_to_quat(rotvec: Tensor) -> Tensor:
    theta = torch.linalg.norm(rotvec, dim=-1, keepdim=True)
    small = theta < _SMALL_ANGLE
    safe = torch.where(small, torch.ones_like(theta), theta)
    half_sinc = torch.where(small, 0.5 - theta**2 / 48.0, torch.sin(safe / 2.0) / safe)
    return torch.cat([rotvec * half_sinc, torch.cos(theta / 2.0)], dim=-1)


def quat_to_rotvec(quat: Tensor) -> Tensor:
    quat = quat * torch.where(quat[..., 3:4] < 0, -1.0, 1.0)
    vec, w = quat[..., :3], quat[..., 3:4]
    norm = torch.linalg.norm(vec, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(norm, w)
    small = norm < _SMALL_ANGLE
    scale = torch.where(
        small,
        2.0 / torch.clamp(w, min=_SMALL_ANGLE),
        angle / torch.where(small, torch.ones_like(norm), norm),
    )
    return vec * scale


def quat_multiply(a: Tensor, b: Tensor) -> Tensor:
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dim=-1,
    )


def quat_conjugate(q: Tensor) -> Tensor:
    return torch.cat([-q[..., :3], q[..., 3:]], dim=-1)


def cumulative_trajectory(deltas: Tensor) -> Tensor:
    """(B, F, 14) per-step deltas -> cumulative trajectory with R(0) = P(0) = 0."""
    deltas = deltas.to(torch.float64)
    out = deltas.clone()
    for dpos, drot in ARM_SLICES:
        out[..., dpos] = torch.cumsum(deltas[..., dpos], dim=-2)
        increments = rotvec_to_quat(deltas[..., drot])
        q = torch.zeros_like(increments[..., 0, :])
        q[..., 3] = 1.0
        rotations = []
        for t in range(deltas.shape[-2]):
            q = quat_multiply(q, increments[..., t, :])
            rotations.append(quat_to_rotvec(q))
        out[..., drot] = torch.stack(rotations, dim=-2)
    return out


def deltas_from_cumulative(cumulative: Tensor) -> Tensor:
    """Eq. (5)-(6): per-step deltas from a cumulative trajectory with zero start."""
    cumulative = cumulative.to(torch.float64)
    out = cumulative.clone()
    for dpos, drot in ARM_SLICES:
        path = cumulative[..., dpos]
        out[..., dpos] = torch.diff(path, dim=-2, prepend=torch.zeros_like(path[..., :1, :]))
        quats = rotvec_to_quat(cumulative[..., drot])
        identity = torch.zeros_like(quats[..., :1, :])
        identity[..., 3] = 1.0
        previous = torch.cat([identity, quats[..., :-1, :]], dim=-2)
        out[..., drot] = quat_to_rotvec(quat_multiply(quat_conjugate(previous), quats))
    return out
