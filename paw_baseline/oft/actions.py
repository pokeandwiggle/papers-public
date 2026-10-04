"""Per-step delta actions shared by the OpenVLA-OFT L1 head and the ActionMap head.

Both heads predict the same normalized 14D chunk: per arm, a world-frame
translation step (3), a world-frame rotation step as extrinsic xyz Euler angles
(3), and a binary open-gripper target (1). Slot 0 steps from the measured state
to the first commanded pose; later slots step between consecutive commands, so
decoding integrates the chunk from the observed pose. The platform's commands
lead the measured pose by up to ~20 cm, so slot 0 and the per-step velocities
are normalized with separate quantile bounds (OpenVLA-OFT BOUNDS_Q99 per axis).
"""

import torch

from paw_baseline.layout import GRIPPER_MAX_WIDTH_M

ARMS = ("left", "right")
MODEL_ARM_OFFSET = {"left": 0, "right": 10}
ARM_DELTA_DIM = 7
ACTION_DIM = ARM_DELTA_DIM * len(ARMS)
#: Continuous delta dims (translation + rotation) within the 14D chunk.
CONTINUOUS_DIMS = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12)
GRIPPER_DIMS = (6, 13)
GRIPPER_OPEN_THRESHOLD_M = 0.5 * GRIPPER_MAX_WIDTH_M
NORM_EPS = 1e-8
CONSTANT_RANGE = 1e-6


def rot6d_to_matrix(rot6d: torch.Tensor) -> torch.Tensor:
    """Rows-first 6D rotations (..., 6) to matrices (..., 3, 3) via Gram-Schmidt."""
    first = torch.nn.functional.normalize(rot6d[..., 0:3], dim=-1)
    second = rot6d[..., 3:6]
    second = second - (first * second).sum(-1, keepdim=True) * first
    second = torch.nn.functional.normalize(second, dim=-1)
    third = torch.linalg.cross(first, second, dim=-1)
    return torch.stack([first, second, third], dim=-2)


def matrix_to_rot6d(matrix: torch.Tensor) -> torch.Tensor:
    return matrix[..., :2, :].reshape(*matrix.shape[:-2], 6)


def matrix_to_euler_xyz(matrix: torch.Tensor) -> torch.Tensor:
    """Extrinsic xyz Euler angles, R = Rz(c) Ry(b) Rx(a); valid for |b| < 90 deg."""
    b = torch.asin(torch.clamp(-matrix[..., 2, 0], -1.0, 1.0))
    a = torch.atan2(matrix[..., 2, 1], matrix[..., 2, 2])
    c = torch.atan2(matrix[..., 1, 0], matrix[..., 0, 0])
    return torch.stack([a, b, c], dim=-1)


def euler_xyz_to_matrix(euler: torch.Tensor) -> torch.Tensor:
    a, b, c = euler.unbind(-1)
    ca, sa, cb, sb, cc, sc = a.cos(), a.sin(), b.cos(), b.sin(), c.cos(), c.sin()
    row0 = torch.stack([cc * cb, cc * sb * sa - sc * ca, cc * sb * ca + sc * sa], -1)
    row1 = torch.stack([sc * cb, sc * sb * sa + cc * ca, sc * sb * ca - cc * sa], -1)
    row2 = torch.stack([-sb, cb * sa, cb * ca], -1)
    return torch.stack([row0, row1, row2], dim=-2)


def encode_chunk(state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Absolute 20D state (B, 20) and actions (B, T, 20) to raw deltas (B, T, 14).

    Runs in FP32 even inside training autocast: BF16 rotation products would add
    ~4 mrad noise to per-step rotations that are typically below 1 mrad.
    """
    with torch.autocast(device_type=state.device.type, enabled=False):
        return _encode_chunk(state, actions)


def _encode_chunk(state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    state = state.float()
    actions = actions.float()
    parts = []
    for arm in ARMS:
        o = MODEL_ARM_OFFSET[arm]
        pos = actions[..., o : o + 3]
        prev_pos = torch.cat([state[:, None, o : o + 3], pos[:, :-1]], dim=1)
        rot = rot6d_to_matrix(actions[..., o + 3 : o + 9])
        prev_rot = torch.cat([rot6d_to_matrix(state[:, None, o + 3 : o + 9]), rot[:, :-1]], dim=1)
        step = matrix_to_euler_xyz(rot @ prev_rot.transpose(-1, -2))
        grip = (actions[..., o + 9 : o + 10] > GRIPPER_OPEN_THRESHOLD_M).float()
        parts += [pos - prev_pos, step, grip]
    return torch.cat(parts, dim=-1)


def decode_chunk(state: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    """Integrate raw deltas (B, T, 14) from the measured state (B, 20) to absolute actions."""
    with torch.autocast(device_type=state.device.type, enabled=False):
        return _decode_chunk(state, deltas)


def _decode_chunk(state: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    state = state.float()
    deltas = deltas.float()
    batch, horizon = deltas.shape[:2]
    out = deltas.new_zeros(batch, horizon, 20)
    for i, arm in enumerate(ARMS):
        o = MODEL_ARM_OFFSET[arm]
        d = deltas[..., i * ARM_DELTA_DIM : (i + 1) * ARM_DELTA_DIM]
        out[..., o : o + 3] = state[:, None, o : o + 3] + torch.cumsum(d[..., 0:3], dim=1)
        rot = rot6d_to_matrix(state[:, o + 3 : o + 9])
        steps = euler_xyz_to_matrix(d[..., 3:6])
        for t in range(horizon):
            rot = steps[:, t] @ rot
            out[:, t, o + 3 : o + 9] = matrix_to_rot6d(rot)
        out[..., o + 9] = torch.where(d[..., 6] > 0.5, GRIPPER_MAX_WIDTH_M, 0.0)
    return out


def bounded(values: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    """BOUNDS_Q99 to [-1, 1]; constant dimensions (e.g. a frozen arm) map to 0 as in OpenVLA."""
    normalized = torch.clamp(2 * (values - low) / (high - low + NORM_EPS) - 1, -1.0, 1.0)
    return torch.where(high - low < CONSTANT_RANGE, torch.zeros_like(normalized), normalized)


class DeltaNormalizer(torch.nn.Module):
    """OpenVLA-OFT BOUNDS_Q99 normalization of continuous deltas; grippers pass through."""

    def __init__(self, stats: dict):
        super().__init__()
        first, step = stats["first"], stats["step"]
        for name, group in (("first", first), ("step", step)):
            for key in ("q01", "q99"):
                value = torch.tensor(group[key], dtype=torch.float32)
                if value.shape != (len(CONTINUOUS_DIMS),):
                    raise ValueError(f"{name}.{key} must have {len(CONTINUOUS_DIMS)} values")
                self.register_buffer(f"{name}_{key}", value, persistent=False)
        self.register_buffer(
            "continuous", torch.tensor(CONTINUOUS_DIMS), persistent=False
        )

    def _bounds(self, horizon: int) -> tuple[torch.Tensor, torch.Tensor]:
        low = torch.cat([self.first_q01[None], self.step_q01[None].expand(horizon - 1, -1)])
        high = torch.cat([self.first_q99[None], self.step_q99[None].expand(horizon - 1, -1)])
        return low, high

    def normalize(self, deltas: torch.Tensor) -> torch.Tensor:
        low, high = self._bounds(deltas.shape[1])
        out = deltas.clone()
        cont = deltas[..., self.continuous]
        out[..., self.continuous] = bounded(cont, low, high)
        return out

    def unnormalize(self, normalized: torch.Tensor) -> torch.Tensor:
        low, high = self._bounds(normalized.shape[1])
        out = normalized.clone()
        cont = normalized[..., self.continuous]
        out[..., self.continuous] = 0.5 * (cont + 1) * (high - low + NORM_EPS) + low
        return out


class ProprioNormalizer(torch.nn.Module):
    """OpenVLA-OFT BOUNDS_Q99 proprio normalization of the 20D model state."""

    def __init__(self, stats: dict):
        super().__init__()
        for key in ("q01", "q99"):
            value = torch.tensor(stats[key], dtype=torch.float32)
            if value.shape != (20,):
                raise ValueError(f"proprio {key} must have 20 values")
            self.register_buffer(key, value, persistent=False)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return bounded(state.float(), self.q01, self.q99)
