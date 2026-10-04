"""Fit OFT normalization statistics on the selected training episodes.

Action statistics are q01/q99 of the per-step delta targets at each chunk position.
The first position is relative to the measured pose and therefore includes the
platform's tracking lead (commands lead the measurement by up to ~18 cm), whereas
later positions are pure per-step velocities; per-position bounds keep the first
step from being clipped to velocity range. Chunks follow LeRobot's episode-end
padding (the last frame repeats), matching the training targets.
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from paw_baseline.layout import GRIPPER_MAX_WIDTH_M
from paw_baseline.oft.actions import (
    ARMS,
    GRIPPER_DIMS,
    GRIPPER_THRESHOLD_M,
    MODEL_ARM_OFFSET,
    encode_deltas,
)
from paw_baseline.representation import MODEL_ROTATION_DIMS

STATS_VERSION = 1


def _read_prepared(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    states, actions, episodes = [], [], []
    for path in sorted((root / "data").glob("chunk-*/file-*.parquet")):
        table = pq.read_table(path, columns=["observation.state", "action", "episode_index"])
        states.append(np.asarray(table["observation.state"].combine_chunks().flatten()).reshape(-1, 20))
        actions.append(np.asarray(table["action"].combine_chunks().flatten()).reshape(-1, 20))
        episodes.append(table["episode_index"].to_numpy())
    return np.concatenate(states), np.concatenate(actions), np.concatenate(episodes)


def chunk_deltas(
    state: np.ndarray, action: np.ndarray, episodes: np.ndarray, selected, horizon: int
) -> np.ndarray:
    """Delta targets ``(anchors, horizon, 14)`` for every anchor of the selected episodes."""
    parts = []
    for episode in sorted(selected):
        rows = np.flatnonzero(episodes == episode)
        if len(rows) == 0:
            raise ValueError(f"episode {episode} has no frames")
        anchors = np.arange(len(rows))
        targets = np.minimum(anchors[:, None] + np.arange(horizon), len(rows) - 1)
        s = torch.from_numpy(state[rows].astype(np.float64))
        a = torch.from_numpy(action[rows][targets].astype(np.float64))
        parts.append(encode_deltas(s, a).numpy())
    return np.concatenate(parts)


def fit_stats(
    state: np.ndarray, action: np.ndarray, episodes: np.ndarray, selected, horizon: int
) -> dict:
    fit = np.isin(episodes, list(selected))
    deltas = chunk_deltas(state, action, episodes, selected, horizon)
    q01 = np.quantile(deltas, 0.01, axis=0)
    q99 = np.quantile(deltas, 0.99, axis=0)
    # Binary grippers are not normalized (OFT mask); record the fixed +-1 range.
    for dim in GRIPPER_DIMS:
        q01[:, dim], q99[:, dim] = -1.0, 1.0
    state_q01 = np.quantile(state[fit], 0.01, axis=0)
    state_q99 = np.quantile(state[fit], 0.99, axis=0)
    # Rotation rows have the known domain [-1, 1], as in the baseline statistics.
    state_q01[list(MODEL_ROTATION_DIMS)] = -1.0
    state_q99[list(MODEL_ROTATION_DIMS)] = 1.0
    open_width, close_width = [], []
    for arm in ARMS:
        widths = action[fit, MODEL_ARM_OFFSET[arm] + 9]
        opened = widths[widths > GRIPPER_THRESHOLD_M]
        closed = widths[widths <= GRIPPER_THRESHOLD_M]
        open_width.append(float(np.median(opened)) if len(opened) else GRIPPER_MAX_WIDTH_M)
        close_width.append(float(np.median(closed)) if len(closed) else 0.0)
    return {
        "version": STATS_VERSION,
        "chunk_size": horizon,
        "num_anchors": int(len(deltas)),
        "action_q01": q01.tolist(),
        "action_q99": q99.tolist(),
        "state_q01": state_q01.tolist(),
        "state_q99": state_q99.tolist(),
        "gripper_open_width": open_width,
        "gripper_close_width": close_width,
    }


def write_oft_stats(root: Path, train_episodes, horizon: int) -> Path:
    """Write (once) and return statistics for this prepared view, selection and horizon."""
    selected = sorted(int(e) for e in train_episodes)
    key = json.dumps({"version": STATS_VERSION, "episodes": selected, "horizon": horizon})
    digest = hashlib.sha256(key.encode()).hexdigest()[:16]
    path = root / ".paw" / f"oft_stats-{digest}.json"
    if path.is_file():
        return path
    state, action, episodes = _read_prepared(root)
    stats = fit_stats(state, action, episodes, selected, horizon)
    stats["train_episode_indices"] = selected
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(stats) + "\n")
    temporary.replace(path)
    return path
