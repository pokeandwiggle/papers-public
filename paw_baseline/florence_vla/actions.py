"""Per-step pose increments with uniform-bin (BIN) or Direction-Scale (DSD) tokens.

Model-space state and actions are 20D: per arm ``[pos(3), rotation row 0 (3),
rotation row 1 (3), gripper width (1)]`` (``paw_baseline.representation``).

Raw action representation (both methods). For an observation at frame ``t``
with measured state ``s_t`` and commanded absolute actions ``a_t .. a_{t+H-1}``,
each arm's chunk step ``k`` is the pose increment from the previous command:

    reference_0 = s_t,  reference_k = a_{t+k-1}  (k >= 1)
    dp_k = pos(a_{t+k}) - pos(reference_k)                 (base frame)
    dR_k = R(a_{t+k}) @ R(reference_k)^T                   (base frame)

with the absolute commanded gripper width ``g_k``. Serving integrates the
decoded increments from the live measured state, so the station still
receives absolute TCP poses. Step 0 (command relative to the lagging measured
state) has its own normalization statistics; steps >= 1 share one set.

* BIN (OpenVLA-style) features per arm: ``(dx, dy, dz, roll, pitch, yaw, g)``
  with ``dR = Rz(yaw) Ry(pitch) Rx(roll)``; every coordinate is affinely mapped
  from its dataset percentile bounds (``QUANTILE_LOW/HIGH``, or OpenVLA's
  1st/99th percentiles in ``BIN_Q01_KEY``) to [-1, 1], clipped, and binned.
* DSD (Eq. 1 of the paper) features per arm:
  ``(s_pos, s_ori, g, n_pos(3), n_ori(3))`` where ``s_pos = |dp|``,
  ``n_pos = dp / s_pos`` (``e3`` at zero), ``omega = log(dR)``,
  ``s_ori = |omega|`` and ``n_ori = omega / s_ori`` (``e3`` at zero). Scales are
  mapped from ``[0, upper percentile]`` to [-1, 1]; unit-vector components use
  the fixed, dataset-independent interval [-1, 1]; the gripper keeps the BIN
  encoding.

Both use ``num_bins`` equal-width bins over [-1, 1] and decode bin centres,
except that the bin containing a zero motion coordinate decodes to zero.
"""

from __future__ import annotations

import base64
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

ARM_OFFSETS = (0, 10)
MODEL_DIM = 20
REPRESENTATIONS = ("bin", "dsd")
FEATURES_PER_ARM = {"bin": 7, "dsd": 9}
STATS_VERSION = 1
#: Minimum normalization width (metres, radians or metres of gripper width).
MIN_RANGE = 1e-4
#: Percentile bounds for every fitted range (BIN coordinates, DSD scales, grippers).
#: OpenVLA uses 1st/99th; with ~50% idle zero increments integrated over a 30-step
#: chunk that clips the fast, task-relevant motions of BIN far more than DSD's
#: norm-based scales, so both representations use 0.1st/99.9th percentiles.
QUANTILE_LOW, QUANTILE_HIGH = 0.001, 0.999
#: OpenVLA's (and the paper's) BIN bounds: every BIN coordinate, gripper included,
#: is mapped from its 1st/99th percentiles and clipped. Stored as ``BIN_Q01_KEY``.
OPENVLA_QUANTILE_LOW, OPENVLA_QUANTILE_HIGH = 0.01, 0.99
BIN_Q01_KEY = "bin_q01"
#: Statistics groups and the representation whose features each one bounds.
STATS_GROUPS = {"bin": "bin", "dsd": "dsd", BIN_Q01_KEY: "bin"}
#: Feature indices of the scale coordinates and their direction components (per arm, DSD).
DSD_SCALE_DIRECTIONS = ((0, (3, 4, 5)), (1, (6, 7, 8)))
DSD_GRIPPER = 2
BIN_GRIPPER = 6


# --------------------------------------------------------------------------- geometry


def rows_to_matrix(rows6: Tensor) -> Tensor:
    """(..., 6) first two rotation rows -> (..., 3, 3), row-wise Gram-Schmidt."""
    first = rows6[..., 0:3]
    second = rows6[..., 3:6]
    first = first / first.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    second = second - (first * second).sum(-1, keepdim=True) * first
    second = second / second.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    third = torch.linalg.cross(first, second, dim=-1)
    return torch.stack([first, second, third], dim=-2)


def matrix_to_rows(matrix: Tensor) -> Tensor:
    """(..., 3, 3) -> (..., 6) first two rows, matching ``matrix_to_rot6d``."""
    return matrix[..., :2, :].reshape(*matrix.shape[:-2], 6)


def matrix_to_rpy(matrix: Tensor) -> Tensor:
    """Roll, pitch, yaw with ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``."""
    pitch = torch.asin((-matrix[..., 2, 0]).clamp(-1.0, 1.0))
    roll = torch.atan2(matrix[..., 2, 1], matrix[..., 2, 2])
    yaw = torch.atan2(matrix[..., 1, 0], matrix[..., 0, 0])
    return torch.stack([roll, pitch, yaw], dim=-1)


def rpy_to_matrix(rpy: Tensor) -> Tensor:
    """Inverse of :func:`matrix_to_rpy`."""
    roll, pitch, yaw = rpy.unbind(-1)
    cr, sr = torch.cos(roll), torch.sin(roll)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    rows = [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]
    return torch.stack([torch.stack(row, dim=-1) for row in rows], dim=-2)


def matrix_to_quat_wxyz(matrix: Tensor) -> Tensor:
    """Shepperd's method with ``w >= 0``; numerically stable for every rotation."""
    m = matrix
    trace = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    candidates = torch.stack(
        [
            1.0 + trace,
            1.0 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2],
            1.0 - m[..., 0, 0] + m[..., 1, 1] - m[..., 2, 2],
            1.0 - m[..., 0, 0] - m[..., 1, 1] + m[..., 2, 2],
        ],
        dim=-1,
    )
    branch = candidates.argmax(dim=-1)
    best = torch.gather(candidates, -1, branch.unsqueeze(-1)).squeeze(-1)
    scale = 0.5 / torch.sqrt(best.clamp_min(1e-12))
    quarter = 0.25 / scale
    d21, d02, d10 = m[..., 2, 1] - m[..., 1, 2], m[..., 0, 2] - m[..., 2, 0], m[..., 1, 0] - m[..., 0, 1]
    s01, s02, s12 = m[..., 0, 1] + m[..., 1, 0], m[..., 0, 2] + m[..., 2, 0], m[..., 1, 2] + m[..., 2, 1]
    options = torch.stack(
        [
            torch.stack([quarter, d21 * scale, d02 * scale, d10 * scale], -1),
            torch.stack([d21 * scale, quarter, s01 * scale, s02 * scale], -1),
            torch.stack([d02 * scale, s01 * scale, quarter, s12 * scale], -1),
            torch.stack([d10 * scale, s02 * scale, s12 * scale, quarter], -1),
        ],
        dim=-2,
    )
    index = branch[..., None, None].expand(*branch.shape, 1, 4)
    quat = torch.gather(options, -2, index).squeeze(-2)
    quat = quat / quat.norm(dim=-1, keepdim=True)
    return torch.where(quat[..., :1] < 0, -quat, quat)


def matrix_to_rotvec(matrix: Tensor) -> Tensor:
    """Axis-angle vector with angle in [0, pi].

    At angle pi the axes n and -n are equivalent; the fixed convention makes
    the largest-magnitude axis component positive.
    """
    quat = matrix_to_quat_wxyz(matrix)
    w, v = quat[..., 0], quat[..., 1:]
    sin_half = v.norm(dim=-1)
    angle = 2.0 * torch.atan2(sin_half, w)
    # angle / sin(angle/2) -> 2 as the angle -> 0.
    factor = torch.where(sin_half > 1e-8, angle / sin_half.clamp_min(1e-12), 2.0 / w.clamp_min(1e-12))
    rotvec = v * factor.unsqueeze(-1)
    pivot = torch.gather(v, -1, v.abs().argmax(dim=-1, keepdim=True))
    flip = (w < 1e-7) & (pivot.squeeze(-1) < 0)
    return torch.where(flip.unsqueeze(-1), -rotvec, rotvec)


def rotvec_to_matrix(rotvec: Tensor) -> Tensor:
    """Rodrigues' formula with small-angle series."""
    theta = rotvec.norm(dim=-1)
    small = theta < 1e-4
    theta_safe = torch.where(small, torch.ones_like(theta), theta)
    a = torch.where(small, 1.0 - theta**2 / 6.0, torch.sin(theta_safe) / theta_safe)
    b = torch.where(small, 0.5 - theta**2 / 24.0, (1.0 - torch.cos(theta_safe)) / theta_safe**2)
    x, y, z = rotvec.unbind(-1)
    zero = torch.zeros_like(x)
    k = torch.stack(
        [
            torch.stack([zero, -z, y], -1),
            torch.stack([z, zero, -x], -1),
            torch.stack([-y, x, zero], -1),
        ],
        dim=-2,
    )
    eye = torch.eye(3, dtype=rotvec.dtype, device=rotvec.device).expand(k.shape)
    return eye + a[..., None, None] * k + b[..., None, None] * (k @ k)


def unit_or_e3(vector: Tensor, scale: Tensor) -> Tensor:
    """``vector / scale`` where the scale is positive, otherwise ``e3``."""
    e3 = torch.zeros_like(vector)
    e3[..., 2] = 1.0
    positive = (scale > 0).unsqueeze(-1)
    return torch.where(positive, vector / torch.where(positive, scale.unsqueeze(-1), torch.ones_like(vector)), e3)


def _renormalize(direction: Tensor) -> Tensor:
    """Project decoded direction components back onto the unit sphere."""
    norm = direction.norm(dim=-1)
    return unit_or_e3(direction, torch.where(norm > 1e-6, norm, torch.zeros_like(norm)))


# --------------------------------------------------------------------------- raw increments


def split_arm(x20: Tensor, arm: int) -> tuple[Tensor, Tensor, Tensor]:
    """Position (...,3), rotation matrix (...,3,3), gripper (...) of one arm."""
    off = ARM_OFFSETS[arm]
    return x20[..., off : off + 3], rows_to_matrix(x20[..., off + 3 : off + 9]), x20[..., off + 9]


def chunk_increments(state: Tensor, actions: Tensor, arm: int) -> tuple[Tensor, Tensor, Tensor]:
    """Per-step increments of one arm for (B, 20) state and (B, H, 20) actions."""
    s_pos, s_rot, _ = split_arm(state, arm)
    a_pos, a_rot, a_grip = split_arm(actions, arm)
    prev_pos = torch.cat([s_pos.unsqueeze(1), a_pos[:, :-1]], dim=1)
    prev_rot = torch.cat([s_rot.unsqueeze(1), a_rot[:, :-1]], dim=1)
    return a_pos - prev_pos, a_rot @ prev_rot.transpose(-1, -2), a_grip


def increments_to_features(representation: str, dp: Tensor, drot: Tensor, grip: Tensor) -> Tensor:
    """Raw increments -> per-arm features (BIN: 7, DSD: 9)."""
    if representation == "bin":
        return torch.cat([dp, matrix_to_rpy(drot), grip.unsqueeze(-1)], dim=-1)
    if representation == "dsd":
        s_pos = dp.norm(dim=-1)
        omega = matrix_to_rotvec(drot)
        s_ori = omega.norm(dim=-1)
        return torch.cat(
            [
                s_pos.unsqueeze(-1),
                s_ori.unsqueeze(-1),
                grip.unsqueeze(-1),
                unit_or_e3(dp, s_pos),
                unit_or_e3(omega, s_ori),
            ],
            dim=-1,
        )
    raise ValueError(f"unknown action representation {representation!r}")


def features_to_increments(representation: str, features: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Per-arm features -> (dp, dR, gripper); DSD directions are renormalized."""
    if representation == "bin":
        return features[..., 0:3], rpy_to_matrix(features[..., 3:6]), features[..., 6]
    if representation == "dsd":
        s_pos = features[..., 0].clamp_min(0.0)
        s_ori = features[..., 1].clamp_min(0.0)
        n_pos = _renormalize(features[..., 3:6])
        n_ori = _renormalize(features[..., 6:9])
        return s_pos.unsqueeze(-1) * n_pos, rotvec_to_matrix(s_ori.unsqueeze(-1) * n_ori), features[..., 2]
    raise ValueError(f"unknown action representation {representation!r}")


def encode_chunk(representation: str, state: Tensor, actions: Tensor) -> Tensor:
    """(B, 20) state + (B, H, 20) absolute actions -> (B, H, 2 * F) features."""
    parts = [
        increments_to_features(representation, *chunk_increments(state, actions, arm))
        for arm in range(len(ARM_OFFSETS))
    ]
    return torch.cat(parts, dim=-1)


def decode_chunk(representation: str, state: Tensor, features: Tensor) -> Tensor:
    """Integrate (B, H, 2 * F) features from (B, 20) measured state -> (B, H, 20)."""
    per_arm = FEATURES_PER_ARM[representation]
    out = []
    for arm in range(len(ARM_OFFSETS)):
        dp, drot, grip = features_to_increments(
            representation, features[..., arm * per_arm : (arm + 1) * per_arm]
        )
        pos, rot, _ = split_arm(state, arm)
        positions, rotations = [], []
        for k in range(features.shape[1]):
            pos = pos + dp[:, k]
            rot = drot[:, k] @ rot
            positions.append(pos)
            rotations.append(rot)
        positions = torch.stack(positions, dim=1)
        rows = matrix_to_rows(torch.stack(rotations, dim=1))
        out.append(torch.cat([positions, rows, grip.unsqueeze(-1)], dim=-1))
    return torch.cat(out, dim=-1)


# --------------------------------------------------------------------------- normalization


class ChunkCodec:
    """Normalization, tokenization and DSD direction-loss weights for one representation.

    ``lo``/``hi`` have shape (H, D): row 0 uses the step-0 (state-anchored)
    statistics and rows >= 1 the increment statistics. ``stats_key`` selects
    the bounds (default: the representation's own group).
    """

    def __init__(
        self,
        representation: str,
        stats: dict,
        horizon: int,
        num_bins: int,
        tau_frac: float,
        min_weight: float,
        stats_key: str | None = None,
    ):
        if representation not in REPRESENTATIONS:
            raise ValueError(f"unknown action representation {representation!r}")
        if stats.get("version") != STATS_VERSION:
            raise ValueError("action statistics version mismatch")
        stats_key = representation if stats_key is None else stats_key
        if STATS_GROUPS.get(stats_key) != representation:
            raise ValueError(f"statistics {stats_key!r} do not bound {representation!r} features")
        if stats_key not in stats:
            raise ValueError(f"action statistics lack the {stats_key!r} bounds")
        self.stats_key = stats_key
        groups = stats[stats_key]
        anchor_lo, anchor_hi = (torch.tensor(groups["anchor"][k], dtype=torch.float64) for k in ("lo", "hi"))
        step_lo, step_hi = (torch.tensor(groups["step"][k], dtype=torch.float64) for k in ("lo", "hi"))
        self.representation = representation
        self.horizon = horizon
        self.num_bins = num_bins
        self.dim = anchor_lo.numel()
        if self.dim != 2 * FEATURES_PER_ARM[representation]:
            raise ValueError("action statistics do not match the representation")
        self.lo = torch.cat([anchor_lo[None], step_lo[None].expand(horizon - 1, -1)])
        self.hi = torch.cat([anchor_hi[None], step_hi[None].expand(horizon - 1, -1)])
        if not torch.all(self.hi - self.lo >= MIN_RANGE * 0.999):
            raise ValueError("degenerate normalization range")
        self.tau_frac = tau_frac
        self.min_weight = min_weight
        # Motion coordinates whose zero-containing bin decodes to exactly zero:
        # BIN translation/rotation coordinates and DSD scales (not grippers or directions).
        per_arm = FEATURES_PER_ARM[representation]
        motion = [0, 1, 2, 3, 4, 5] if representation == "bin" else [0, 1]
        snap = torch.zeros(self.dim, dtype=torch.bool)
        for arm in range(len(ARM_OFFSETS)):
            snap[[arm * per_arm + index for index in motion]] = True
        self.zero_bins = torch.where(
            snap & (self.lo <= 0) & (self.hi > 0),
            self.tokenize(torch.zeros(1, horizon, self.dim, dtype=torch.float64))[0],
            torch.full((horizon, self.dim), -1, dtype=torch.long),
        )

    def normalize(self, features: Tensor) -> Tensor:
        lo, hi = self.lo.to(features), self.hi.to(features)
        return (2.0 * (features - lo) / (hi - lo) - 1.0).clamp(-1.0, 1.0)

    def denormalize(self, normalized: Tensor) -> Tensor:
        lo, hi = self.lo.to(normalized), self.hi.to(normalized)
        return lo + (normalized + 1.0) * 0.5 * (hi - lo)

    def tokenize(self, features: Tensor) -> Tensor:
        """(B, H, D) features -> (B, H, D) bin indices in [0, num_bins)."""
        normalized = self.normalize(features)
        return torch.floor((normalized + 1.0) * 0.5 * self.num_bins).clamp(0, self.num_bins - 1).long()

    def detokenize(self, tokens: Tensor) -> Tensor:
        """(B, H, D) bin indices -> features at the bin centres.

        The bin containing a zero motion coordinate decodes to exactly zero:
        integrated absolute poses have no controller dead zone, so a bin-centre
        residual would otherwise make idle arms creep (both representations).
        """
        centres = -1.0 + (tokens.to(self.lo.dtype) + 0.5) * (2.0 / self.num_bins)
        values = self.denormalize(centres)
        zero = tokens == self.zero_bins.to(tokens.device)
        return torch.where(zero, torch.zeros_like(values), values)

    def loss_weights(self, features: Tensor) -> Tensor:
        """(B, H, D) token weights; DSD direction tokens are down-weighted at small scale.

        ``w = clip(s / tau, w_min, 1)`` with ``tau = tau_frac * (scale range)``
        for the scale ``s`` that owns each unit vector (Sec. V-C of the paper).
        """
        weights = torch.ones_like(features)
        if self.representation != "dsd":
            return weights
        per_arm = FEATURES_PER_ARM["dsd"]
        lo, hi = self.lo.to(features), self.hi.to(features)
        for arm in range(len(ARM_OFFSETS)):
            for scale_index, direction_indices in DSD_SCALE_DIRECTIONS:
                column = arm * per_arm + scale_index
                tau = self.tau_frac * (hi[:, column] - lo[:, column])
                weight = (features[..., column] / tau).clamp(self.min_weight, 1.0)
                for index in direction_indices:
                    weights[..., arm * per_arm + index] = weight
        return weights


# --------------------------------------------------------------------------- statistics


def _quantiles(values: np.ndarray, low: float, high: float) -> tuple[np.ndarray, np.ndarray]:
    return np.quantile(values, low, axis=0), np.quantile(values, high, axis=0)


def _widen(lo: np.ndarray, hi: np.ndarray) -> tuple[list[float], list[float]]:
    lo, hi = lo.astype(np.float64).copy(), hi.astype(np.float64).copy()
    narrow = hi - lo < MIN_RANGE
    centre = 0.5 * (lo + hi)
    lo[narrow] = centre[narrow] - MIN_RANGE / 2
    hi[narrow] = centre[narrow] + MIN_RANGE / 2
    return lo.tolist(), hi.tolist()


def _group_stats(
    representation: str,
    features: np.ndarray,
    grip_lo: np.ndarray,
    grip_hi: np.ndarray,
    low: float = QUANTILE_LOW,
    high: float = QUANTILE_HIGH,
) -> dict:
    """Bounds for one group of (N, 2F) features."""
    per_arm = FEATURES_PER_ARM[representation]
    q_low, q_high = _quantiles(features, low, high)
    lo, hi = q_low.copy(), q_high.copy()
    for arm in range(len(ARM_OFFSETS)):
        base = arm * per_arm
        if representation == "bin":
            lo[base + BIN_GRIPPER], hi[base + BIN_GRIPPER] = grip_lo[arm], grip_hi[arm]
        else:
            for scale_index, direction_indices in DSD_SCALE_DIRECTIONS:
                lo[base + scale_index] = 0.0
                hi[base + scale_index] = max(q_high[base + scale_index], MIN_RANGE)
                for index in direction_indices:
                    lo[base + index], hi[base + index] = -1.0, 1.0
            lo[base + DSD_GRIPPER], hi[base + DSD_GRIPPER] = grip_lo[arm], grip_hi[arm]
    lo_list, hi_list = _widen(lo, hi)
    return {"lo": lo_list, "hi": hi_list}


def compute_action_stats(state: np.ndarray, action: np.ndarray, episode_index: np.ndarray) -> dict:
    """Fit both representations' bounds on frame-ordered (N, 20) model-space data.

    Step-0 statistics use every frame's command relative to its measured state.
    Increment statistics use consecutive commands within each episode, which
    are exactly the step >= 1 increments of chunks (padded steps are masked).
    ``BIN_Q01_KEY`` holds the BIN bounds at OpenVLA's 1st/99th percentiles.
    """
    state_t = torch.from_numpy(np.asarray(state, dtype=np.float64))
    action_t = torch.from_numpy(np.asarray(action, dtype=np.float64))
    episode_index = np.asarray(episode_index)
    same = torch.from_numpy(episode_index[1:] == episode_index[:-1])
    grip = action_t[:, [ARM_OFFSETS[0] + 9, ARM_OFFSETS[1] + 9]].numpy()
    grip_lo, grip_hi = _quantiles(grip, QUANTILE_LOW, QUANTILE_HIGH)
    grip_lo_q01, grip_hi_q01 = _quantiles(grip, OPENVLA_QUANTILE_LOW, OPENVLA_QUANTILE_HIGH)
    result: dict = {"version": STATS_VERSION, "frames": int(len(episode_index))}
    for representation in REPRESENTATIONS:
        anchor, step = [], []
        for arm in range(len(ARM_OFFSETS)):
            s_pos, s_rot, _ = split_arm(state_t, arm)
            a_pos, a_rot, a_grip = split_arm(action_t, arm)
            anchor.append(
                increments_to_features(
                    representation, a_pos - s_pos, a_rot @ s_rot.transpose(-1, -2), a_grip
                )
            )
            step.append(
                increments_to_features(
                    representation,
                    (a_pos[1:] - a_pos[:-1])[same],
                    (a_rot[1:] @ a_rot[:-1].transpose(-1, -2))[same],
                    a_grip[1:][same],
                )
            )
        anchor_features = torch.cat(anchor, -1).numpy()
        step_features = torch.cat(step, -1).numpy()
        result[representation] = {
            "anchor": _group_stats(representation, anchor_features, grip_lo, grip_hi),
            "step": _group_stats(representation, step_features, grip_lo, grip_hi),
        }
        if representation == "bin":
            bounds = (grip_lo_q01, grip_hi_q01, OPENVLA_QUANTILE_LOW, OPENVLA_QUANTILE_HIGH)
            result[BIN_Q01_KEY] = {
                "anchor": _group_stats("bin", anchor_features, *bounds),
                "step": _group_stats("bin", step_features, *bounds),
            }
    return result


def stats_from_prepared(root: Path, episodes: list[int] | tuple[int, ...]) -> dict:
    """Read a prepared 20D LeRobot view and fit statistics on ``episodes`` only."""
    import pyarrow.parquet as pq

    states, actions, indices, frames = [], [], [], []
    for path in sorted((Path(root) / "data").glob("chunk-*/file-*.parquet")):
        table = pq.read_table(path, columns=["observation.state", "action", "episode_index", "frame_index"])
        states.append(np.asarray(table["observation.state"].combine_chunks().flatten()).reshape(-1, MODEL_DIM))
        actions.append(np.asarray(table["action"].combine_chunks().flatten()).reshape(-1, MODEL_DIM))
        indices.append(table["episode_index"].to_numpy())
        frames.append(table["frame_index"].to_numpy())
    state, action = np.concatenate(states), np.concatenate(actions)
    episode_index, frame_index = np.concatenate(indices), np.concatenate(frames)
    keep = np.isin(episode_index, np.asarray(list(episodes)))
    if not keep.any():
        raise ValueError("no frames for the selected statistics episodes")
    order = np.lexsort((frame_index[keep], episode_index[keep]))
    state, action, episode_index = state[keep][order], action[keep][order], episode_index[keep][order]
    stats = compute_action_stats(state, action, episode_index)
    stats["episodes"] = len(set(episode_index.tolist()))
    return stats


def encode_stats(stats: dict) -> str:
    """Base64 JSON: draccus would YAML-parse (and mangle) a raw JSON CLI value."""
    return base64.urlsafe_b64encode(json.dumps(stats, separators=(",", ":")).encode()).decode()


def parse_stats(text: str) -> dict:
    if not text:
        raise ValueError("florence_vla requires action_stats fitted on the training data")
    stats = json.loads(base64.urlsafe_b64decode(text.encode()).decode())
    if not isinstance(stats, dict) or stats.get("version") != STATS_VERSION:
        raise ValueError("invalid florence_vla action_stats")
    # Checkpoints fitted before BIN_Q01_KEY existed lack it; it is optional unless used.
    for stats_key, representation in STATS_GROUPS.items():
        if stats_key not in stats and stats_key not in REPRESENTATIONS:
            continue
        for group in ("anchor", "step"):
            for key in ("lo", "hi"):
                values = stats[stats_key][group][key]
                if len(values) != 2 * FEATURES_PER_ARM[representation] or not all(
                    isinstance(v, (int, float)) and math.isfinite(v) for v in values
                ):
                    raise ValueError("invalid florence_vla action_stats bounds")
    return stats
