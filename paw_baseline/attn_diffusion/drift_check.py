"""Check that accumulated delta actions reproduce recorded held-out trajectories.

For each held-out episode (the trainer's default 5 spread episodes) and each
anchor frame t with a full chunk, compute the 45 delta actions exactly as
training does (torch float32, first delta relative to the measured pose at t),
keep them in float32 as the model emits them, accumulate them in float64 from
the measured pose at t as serving does, and compare with the recorded
absolute actions a_t .. a_{t+44}.

Usage: python -m paw_baseline.attn_diffusion.drift_check <hf_repo_id> [...]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from paw_baseline.attn_diffusion.deltas import (
    DELTA_POS,
    DELTA_ROT,
    MODEL_POS,
    MODEL_ROT,
    absolute_to_deltas,
    accumulate,
    rotation_error_deg,
)
from paw_baseline.layout import ACTION_KEY_DIMS, STATE_KEY_DIMS
from paw_baseline.representation import encode_model
from train.__main__ import holdout_episodes

CHUNK = 45
ANCHOR_STRIDE = 5


def _fold(table, key_dims) -> np.ndarray:
    parts = []
    for key, dim in key_dims:
        column = table[key].combine_chunks()
        values = (
            column.to_numpy(zero_copy_only=False).reshape(-1, 1)
            if dim == 1
            else column.flatten().to_numpy(zero_copy_only=False).reshape(-1, dim)
        )
        parts.append(values)
    return encode_model(np.concatenate(parts, axis=1).astype(np.float32))


def load_episodes(root: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    episodes: dict[int, list] = {}
    for path in sorted((root / "data").glob("chunk-*/file-*.parquet")):
        table = pq.read_table(path)
        state, action = _fold(table, STATE_KEY_DIMS), _fold(table, ACTION_KEY_DIMS)
        index = table["episode_index"].to_numpy()
        frame = table["frame_index"].to_numpy()
        for episode in np.unique(index):
            mask = index == episode
            order = np.argsort(frame[mask])
            episodes[int(episode)] = (state[mask][order], action[mask][order])
    return episodes


def check(root: Path) -> dict:
    episodes = load_episodes(root)
    holdout = holdout_episodes(len(episodes))
    pos_err, rot_err, anchor_pos, step_pos, step_rot, lag_pos = [], [], [], [], [], []
    for episode in holdout:
        state, action = episodes[episode]
        for t in range(1, len(action) - CHUNK, ANCHOR_STRIDE):
            window = torch.from_numpy(action[t - 1 : t + CHUNK][None])
            deltas = absolute_to_deltas(torch.from_numpy(state[t][None]), window, 0)[0]
            deltas = deltas.numpy().astype(np.float32)
            truth = action[t : t + CHUNK].astype(np.float64)
            rebuilt = accumulate(state[t], deltas)
            for arm in range(2):
                pos_err.append(
                    np.linalg.norm(rebuilt[:, MODEL_POS[arm]] - truth[:, MODEL_POS[arm]], axis=-1).max()
                )
                rot_err.append(
                    rotation_error_deg(rebuilt[:, MODEL_ROT[arm]], truth[:, MODEL_ROT[arm]]).max()
                )
                anchor_pos.append(np.linalg.norm(deltas[0, DELTA_POS[arm]]))
                step_pos.extend(np.linalg.norm(deltas[1:, DELTA_POS[arm]], axis=-1))
                step_rot.extend(np.degrees(np.linalg.norm(deltas[1:, DELTA_ROT[arm]], axis=-1)))
                # Offset if consecutive deltas were chained from the measured pose.
                lag_pos.append(
                    np.linalg.norm(action[t - 1, MODEL_POS[arm]] - state[t, MODEL_POS[arm]])
                )

    def pct(values, scale=1.0):
        values = np.asarray(values, dtype=np.float64) * scale
        return {q: float(np.percentile(values, q)) for q in (50, 99, 100)}

    return {
        "holdout_episodes": holdout,
        "anchors": len(pos_err) // 2,
        "chunk_max_position_error_mm": pct(pos_err, 1000),
        "chunk_max_rotation_error_deg": pct(rot_err),
        "first_delta_position_mm": pct(anchor_pos, 1000),
        "step_delta_position_mm": pct(step_pos, 1000),
        "step_delta_rotation_deg": pct(step_rot),
        "previous_action_to_measured_pose_mm": pct(lag_pos, 1000),
    }


def main() -> None:
    from huggingface_hub import snapshot_download

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repos", nargs="+")
    parser.add_argument("--cache", type=Path, default=Path("../tmp/drift_data"))
    args = parser.parse_args()
    report = {}
    for repo in args.repos:
        root = Path(
            snapshot_download(
                repo_id=repo,
                repo_type="dataset",
                revision="v3.0",
                allow_patterns=["data/**", "meta/info.json"],
                local_dir=args.cache / repo.split("/")[-1],
            )
        )
        report[repo] = check(root)
        print(json.dumps({repo: report[repo]}), flush=True)


if __name__ == "__main__":
    main()
