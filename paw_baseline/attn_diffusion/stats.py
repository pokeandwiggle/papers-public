"""In-policy normalization statistics for attn_diffusion from training episodes."""

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from paw_baseline.attn_diffusion.deltas import delta_statistics


def training_episodes(root: Path, episodes: tuple[int, ...]) -> list[tuple[np.ndarray, np.ndarray]]:
    """(state, action) 20D model arrays per selected episode, in frame order."""
    wanted = set(episodes)
    collected: dict[int, list] = {}
    for path in sorted((root / "data").glob("chunk-*/file-*.parquet")):
        table = pq.read_table(
            path, columns=["episode_index", "frame_index", "observation.state", "action"]
        )
        index = table["episode_index"].to_numpy()
        frame = table["frame_index"].to_numpy()
        state = np.asarray(table["observation.state"].combine_chunks().flatten()).reshape(len(table), -1)
        action = np.asarray(table["action"].combine_chunks().flatten()).reshape(len(table), -1)
        for episode in np.unique(index):
            if int(episode) not in wanted:
                continue
            mask = index == episode
            order = np.argsort(frame[mask])
            collected[int(episode)] = (state[mask][order], action[mask][order])
    missing = wanted - set(collected)
    if missing:
        raise ValueError(f"training episodes missing from data parquets: {sorted(missing)[:5]}")
    return [collected[e] for e in sorted(collected)]


def training_statistics(root: Path, episodes: tuple[int, ...]) -> dict[str, list[float]]:
    """State min/max from the prepared statistics (fit on training episodes,
    rotation rows fixed to [-1, 1], parked arms squashed) and delta min/max."""
    stats = json.loads((root / "meta/stats.json").read_text())["observation.state"]
    return {
        "state_min": [float(v) for v in stats["min"]],
        "state_max": [float(v) for v in stats["max"]],
        **delta_statistics(training_episodes(root, episodes)),
    }


def override_args(statistics: dict[str, list[float]]) -> list[str]:
    return [
        f"--policy.{key}=" + json.dumps(values, separators=(",", ":"))
        for key, values in statistics.items()
    ]
