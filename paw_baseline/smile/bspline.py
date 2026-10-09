"""Fixed B-spline basis, least-squares coefficient targets and their statistics.

SMILE (Park et al., 2026, Sec. IV-B/D) represents an F-step action chunk by K
control points of a degree-P B-spline evaluated at F uniformly spaced action
times. Targets are least-squares projections (Eq. 10) of the chunk's cumulative
relative translation and rotation trajectories and its absolute gripper
trajectory; dataset-level coefficient statistics normalize every modality.

Near an episode end only the steps inside the episode are expert data. A
window with v < F valid steps is fitted on those steps alone, and only the
coefficient directions they determine are supervised (validity masks).
"""

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from paw_baseline.relative import (
    DELTA_DIM,
    DELTA_GRIPPER_DIMS,
    DELTA_ROTATION_DIMS,
    DELTA_TRANSLATION_DIMS,
    cumulative_trajectory,
)

TRANSLATION_DIMS = DELTA_TRANSLATION_DIMS
ROTATION_DIMS = DELTA_ROTATION_DIMS
GRIPPER_DIMS = DELTA_GRIPPER_DIMS
assert sorted(TRANSLATION_DIMS + ROTATION_DIMS + GRIPPER_DIMS) == list(range(DELTA_DIM))

# Coefficient spreads below this are constant (e.g. a frozen arm) and keep scale 1.
MIN_COEF_STD = 1e-8
# Coefficient directions a truncated window determines with singular value below
# this are unobserved (the full F=50, K=8 basis has smallest singular value 0.59).
OBSERVABILITY_TOL = 0.1
# Cumulative rotation vectors are continuous only below pi.
MAX_CUMULATIVE_ANGLE = np.pi - 0.1


def clamped_uniform_knots(num_control_points: int, degree: int) -> np.ndarray:
    """Open-uniform knot vector on [0, 1] with K + P + 1 knots."""
    if degree < 1 or num_control_points <= degree:
        raise ValueError("B-spline needs degree >= 1 and more control points than degree")
    interior = np.linspace(0.0, 1.0, num_control_points - degree + 1)[1:-1]
    return np.concatenate(
        [np.zeros(degree + 1), interior, np.ones(degree + 1)]
    ).astype(np.float64)


def bspline_basis(length: int, num_control_points: int, degree: int) -> np.ndarray:
    """Cox-de Boor basis Phi (F, K) at F uniformly spaced times in [0, 1]."""
    if length < num_control_points:
        raise ValueError("chunk length must be at least the number of control points")
    knots = clamped_uniform_knots(num_control_points, degree)
    times = np.linspace(0.0, 1.0, length)
    intervals = len(knots) - 1
    basis = np.zeros((length, intervals))
    for i in range(intervals):
        basis[:, i] = (knots[i] <= times) & (times < knots[i + 1])
    # The right end belongs to the last non-degenerate interval (index K - 1).
    basis[times >= 1.0, :] = 0.0
    basis[times >= 1.0, num_control_points - 1] = 1.0
    for p in range(1, degree + 1):
        nxt = np.zeros((length, intervals - p))
        for i in range(intervals - p):
            left = knots[i + p] - knots[i]
            right = knots[i + p + 1] - knots[i + 1]
            if left > 0:
                nxt[:, i] += (times - knots[i]) / left * basis[:, i]
            if right > 0:
                nxt[:, i] += (knots[i + p + 1] - times) / right * basis[:, i + 1]
        basis = nxt
    if basis.shape != (length, num_control_points):
        raise AssertionError(basis.shape)
    return basis


def projection_matrix(basis: np.ndarray) -> np.ndarray:
    """Least-squares projector Phi^+ (K, F): argmin ||Phi Theta - Y||_F = Phi^+ Y."""
    return np.linalg.pinv(basis)


def masked_projectors(
    basis: np.ndarray, tol: float = OBSERVABILITY_TOL
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per valid-prefix length v = 0..F: projector, observability, rank.

    projectors[v] (K, F) is the minimum-norm least-squares fit to the first v
    steps (zero columns for padded steps); observability[v] (K, K) projects onto
    the coefficient directions those steps determine; for v = F both reduce to
    the ordinary fit and the identity.
    """
    length, k = basis.shape
    projectors = np.zeros((length + 1, k, length))
    observability = np.zeros((length + 1, k, k))
    rank = np.zeros(length + 1, dtype=np.int64)
    for v in range(1, length + 1):
        u, s, vt = np.linalg.svd(basis[:v], full_matrices=False)
        keep = s >= tol
        projectors[v, :, :v] = (vt[keep].T / s[keep]) @ u[:, keep].T
        observability[v] = vt[keep].T @ vt[keep]
        rank[v] = int(keep.sum())
    if rank[length] != k:
        raise ValueError("the full basis must determine every coefficient")
    return projectors, observability, rank


def _episode_actions(prepared_root: Path, episodes) -> dict[int, np.ndarray]:
    wanted = set(int(e) for e in episodes)
    parts: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    for path in sorted((Path(prepared_root) / "data").glob("chunk-*/file-*.parquet")):
        table = pq.read_table(path, columns=["episode_index", "frame_index", "action"])
        index = np.asarray(table["episode_index"])
        frames = np.asarray(table["frame_index"])
        actions = np.asarray(table["action"].combine_chunks().flatten()).reshape(
            len(table), -1
        )
        if actions.shape[1] != DELTA_DIM:
            raise ValueError("SMILE expects 14D relative actions")
        for episode in np.unique(index):
            if int(episode) in wanted:
                mask = index == episode
                parts.setdefault(int(episode), []).append((frames[mask], actions[mask]))
    if set(parts) != wanted:
        raise ValueError("training episodes are missing from the prepared data")
    result = {}
    for episode, chunks in parts.items():
        frames = np.concatenate([f for f, _ in chunks])
        actions = np.concatenate([a for _, a in chunks]).astype(np.float64)
        result[episode] = actions[np.argsort(frames)]
    return result


def coefficient_stats(
    prepared_root: Path,
    episodes: tuple[int, ...] | list[int],
    *,
    chunk_size: int,
    num_control_points: int,
    degree: int,
) -> dict:
    """Per-coefficient (K, D) mean/std over every eligible training chunk.

    A chunk start is eligible when all F actions lie inside the episode. Its
    cumulative trajectory is built from the raw per-step deltas (translation
    sums, composed rotations, absolute gripper) and projected onto the basis.
    """
    projector = projection_matrix(
        bspline_basis(chunk_size, num_control_points, degree)
    )
    total = 0
    first = np.zeros((num_control_points, DELTA_DIM))
    second = np.zeros((num_control_points, DELTA_DIM))
    max_angle = 0.0
    for actions in _episode_actions(prepared_root, episodes).values():
        # Windows crossing the episode end use only their valid steps.
        for start in range(max(0, len(actions) - chunk_size + 1), len(actions)):
            tail = cumulative_trajectory(actions[start:])
            for drot in (slice(3, 6), slice(10, 13)):
                max_angle = max(
                    max_angle, float(np.linalg.norm(tail[..., drot], axis=-1).max())
                )
        if len(actions) < chunk_size:
            continue
        windows = np.lib.stride_tricks.sliding_window_view(
            actions, chunk_size, axis=0
        ).swapaxes(-1, -2)  # (S, F, D)
        cumulative = cumulative_trajectory(windows)
        for drot in (slice(3, 6), slice(10, 13)):
            max_angle = max(
                max_angle, float(np.linalg.norm(cumulative[..., drot], axis=-1).max())
            )
        coefs = np.einsum("kf,sfd->skd", projector, cumulative)
        total += len(coefs)
        first += coefs.sum(axis=0)
        second += np.square(coefs).sum(axis=0)
    if total == 0:
        raise ValueError("no training episode is as long as the SMILE chunk")
    if max_angle > MAX_CUMULATIVE_ANGLE:
        raise ValueError(
            f"cumulative chunk rotation reaches {np.degrees(max_angle):.1f} deg; "
            "rotation vectors would wrap"
        )
    coef_mean = first / total
    coef_std = np.sqrt(np.maximum(second / total - np.square(coef_mean), 0.0))
    coef_std = np.where(coef_std < MIN_COEF_STD, 1.0, coef_std)
    return {
        "coef_mean": coef_mean.reshape(-1).tolist(),
        "coef_std": coef_std.reshape(-1).tolist(),
        "windows": int(total),
        "max_cumulative_angle_deg": float(np.degrees(max_angle)),
    }


def action_stats(prepared_root: Path) -> dict:
    """The prepared per-frame action normalization the policy processors use."""
    stats = json.loads((Path(prepared_root) / "meta/stats.json").read_text())["action"]
    if len(stats["mean"]) != DELTA_DIM:
        raise ValueError("SMILE expects 14D relative action statistics")
    return {"action_mean": stats["mean"], "action_std": stats["std"]}
