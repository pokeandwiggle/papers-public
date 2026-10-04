"""Fit OpenVLA-OFT delta-action and proprio quantile bounds on training episodes.

Slot 0 (measured state to first command) and later slots (command to command)
get separate BOUNDS_Q99 bounds; see ``paw_baseline.oft.actions``.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from paw_baseline.oft.actions import CONTINUOUS_DIMS, encode_chunk

STATS_VERSION = 1


def _load_frames(root: Path, episodes: set[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tables = [
        pq.read_table(path, columns=["episode_index", "frame_index", "observation.state", "action"])
        for path in sorted((root / "data").glob("chunk-*/file-*.parquet"))
    ]
    episode = np.concatenate([t["episode_index"].to_numpy() for t in tables])
    frame = np.concatenate([t["frame_index"].to_numpy() for t in tables])
    state = np.concatenate(
        [np.stack(t["observation.state"].to_numpy(zero_copy_only=False)) for t in tables]
    )
    action = np.concatenate(
        [np.stack(t["action"].to_numpy(zero_copy_only=False)) for t in tables]
    )
    keep = np.isin(episode, sorted(episodes))
    order = np.lexsort((frame[keep], episode[keep]))
    return episode[keep][order], state[keep][order].astype(np.float32), action[keep][order].astype(np.float32)


def _bounds(values: np.ndarray) -> dict:
    return {
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def oft_statistics(root: Path, episodes) -> dict:
    episodes = set(int(e) for e in episodes)
    episode, state, action = _load_frames(root, episodes)
    if not len(episode):
        raise ValueError("no training frames for OFT statistics")
    state_t = torch.from_numpy(state)
    action_t = torch.from_numpy(action)
    # Slot 0: every frame's command relative to its own measured state.
    first = encode_chunk(state_t, action_t[:, None])[:, 0].numpy()
    # Later slots: consecutive commands within an episode.
    same = episode[1:] == episode[:-1]
    prev = torch.from_numpy(action[:-1][same])
    nxt = torch.from_numpy(action[1:][same])
    # encode_chunk(prev_as_state, [prev, next]) yields next - prev in slot 1.
    step = encode_chunk(prev, torch.stack([prev, nxt], dim=1))[:, 1].numpy()
    dims = list(CONTINUOUS_DIMS)
    return {
        "action": {"first": _bounds(first[:, dims]), "step": _bounds(step[:, dims])},
        "proprio": _bounds(state),
    }


def write_oft_statistics(root: Path, episodes) -> Path:
    """Write statistics for this prepared view and episode selection; return the path."""
    episodes = sorted(int(e) for e in episodes)
    digest = hashlib.sha256(json.dumps(episodes).encode()).hexdigest()[:16]
    path = root / ".paw" / f"oft_stats_v{STATS_VERSION}_{digest}.json"
    if not path.exists():
        payload = {
            "version": STATS_VERSION,
            "episodes_sha256_16": digest,
            "stats": oft_statistics(root, episodes),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        temporary.replace(path)
    return path
