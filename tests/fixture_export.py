"""Synthetic structured_v1 dataset for CPU tests.

Mirrors the real export layout — per-side sub-feature columns (plus one
wrench column prepare must drop), v3.0 meta, per-camera mp4s holding the
episodes back to back — at 64×64 px and a few dozen frames so the whole
train → serve → validate path runs in minutes without a GPU or HF access.
"""

import json
from pathlib import Path

import av
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

CAMERAS = (
    "observation.images.rgb_static",
    "observation.images.rgb_left",
    "observation.images.rgb_right",
)
FPS = 30
IMAGE_SIZE = 64
TASK = "Poke the wiggle target."

_POSE_NAMES = ["pos_x", "pos_y", "pos_z", "quat_x", "quat_y", "quat_z", "quat_w"]


def _episode_states(frames: int, episode: int, *, frozen_right_arm: bool) -> np.ndarray:
    """(frames, 16) smooth flat-16 trajectory in training order."""
    t = np.arange(frames, dtype=np.float64)
    phase = 0.1 * episode
    state = np.zeros((frames, 16))
    state[:, 0] = 0.30 + 0.002 * t + phase  # left moves dominantly in +x
    state[:, 1] = 0.10 + 0.001 * t
    state[:, 2] = 0.40 - 0.001 * t
    state[:, 6] = 1.0  # identity quaternions, w last
    state[:, 7] = 0.02 + 0.04 * t / max(frames - 1, 1)
    state[:, 8] = 0.30 if frozen_right_arm else 0.30 + 0.002 * t + phase
    state[:, 9] = -0.10 if frozen_right_arm else -0.10 - 0.001 * t
    state[:, 10] = 0.40
    state[:, 14] = 1.0
    state[:, 15] = 0.05
    return state


def _fixed_list(values: np.ndarray) -> pa.FixedSizeListArray:
    flat = pa.array(values.ravel(), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, values.shape[1])


def _vector_stats(values: np.ndarray) -> dict:
    v = values.astype(np.float64)
    if v.ndim == 1:
        v = v.reshape(-1, 1)
    return {
        "min": v.min(axis=0).tolist(),
        "max": v.max(axis=0).tolist(),
        "mean": v.mean(axis=0).tolist(),
        "std": v.std(axis=0).tolist(),
        "count": [int(v.shape[0])],
    }


def _write_video(path: Path, total_frames: int, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 200, size=(IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=FPS)
        stream.width = IMAGE_SIZE
        stream.height = IMAGE_SIZE
        stream.pix_fmt = "yuv420p"
        for index in range(total_frames):
            image = np.clip(base.astype(np.int16) + index, 0, 255).astype(np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def build_export(
    root: Path, *, episodes: int = 2, frames: int = 48, frozen_right_arm: bool = False
) -> Path:
    """Write a complete structured_v1 export under ``root`` and return it."""
    per_episode = [
        _episode_states(frames, e, frozen_right_arm=frozen_right_arm)
        for e in range(episodes)
    ]
    state = np.concatenate(per_episode)
    # The demonstrator's action leads the state by one frame, held on the last.
    action = np.concatenate([np.concatenate([s[1:], s[-1:]]) for s in per_episode])
    total = episodes * frames

    columns: dict = {
        "observation.state.tcp_pose_left": _fixed_list(state[:, 0:7]),
        "observation.state.gripper_width_left": pa.array(
            state[:, 7], type=pa.float32()
        ),
        "observation.state.tcp_pose_right": _fixed_list(state[:, 8:15]),
        "observation.state.gripper_width_right": pa.array(
            state[:, 15], type=pa.float32()
        ),
        # Present in real exports, dropped by prepare.
        "observation.state.tcp_wrench_left": _fixed_list(np.zeros((total, 6))),
        "action.tcp_pose_left": _fixed_list(action[:, 0:7]),
        "action.gripper_width_left": pa.array(action[:, 7], type=pa.float32()),
        "action.tcp_pose_right": _fixed_list(action[:, 8:15]),
        "action.gripper_width_right": pa.array(action[:, 15], type=pa.float32()),
        "timestamp": pa.array(
            np.tile(np.arange(frames) / FPS, episodes), type=pa.float32()
        ),
        "frame_index": pa.array(np.tile(np.arange(frames), episodes), type=pa.int64()),
        "episode_index": pa.array(
            np.repeat(np.arange(episodes), frames), type=pa.int64()
        ),
        "index": pa.array(np.arange(total), type=pa.int64()),
        "task_index": pa.array(np.zeros(total, dtype=np.int64)),
    }
    data_dir = root / "data/chunk-000"
    data_dir.mkdir(parents=True)
    pq.write_table(pa.table(columns), data_dir / "file-000.parquet")

    features: dict = {}
    for side in ("left", "right"):
        features[f"observation.state.tcp_pose_{side}"] = {
            "dtype": "float32",
            "shape": [7],
            "names": _POSE_NAMES,
        }
        features[f"observation.state.gripper_width_{side}"] = {
            "dtype": "float32",
            "shape": [1],
            "names": ["width"],
        }
        features[f"action.tcp_pose_{side}"] = {
            "dtype": "float32",
            "shape": [7],
            "names": _POSE_NAMES,
        }
        features[f"action.gripper_width_{side}"] = {
            "dtype": "float32",
            "shape": [1],
            "names": ["width"],
        }
    features["observation.state.tcp_wrench_left"] = {
        "dtype": "float32",
        "shape": [6],
        "names": None,
    }
    for key, dtype in (
        ("timestamp", "float32"),
        ("frame_index", "int64"),
        ("episode_index", "int64"),
        ("index", "int64"),
        ("task_index", "int64"),
    ):
        features[key] = {"dtype": dtype, "shape": [1], "names": None}
    for camera in CAMERAS:
        features[camera] = {
            "dtype": "video",
            "shape": [IMAGE_SIZE, IMAGE_SIZE, 3],
            "names": ["height", "width", "channel"],
            "info": {
                "video.height": IMAGE_SIZE,
                "video.width": IMAGE_SIZE,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": FPS,
                "video.channels": 3,
                "has_audio": False,
            },
        }
    info = {
        "codebase_version": "v3.0",
        "robot_type": "dual_fr3_pedestal",
        "total_episodes": episodes,
        "total_frames": total,
        "total_tasks": 1,
        "chunks_size": 1000,
        "data_files_size_in_mb": 100.0,
        "video_files_size_in_mb": 200.0,
        "fps": FPS,
        "splits": {"train": f"0:{episodes}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": features,
    }
    meta_dir = root / "meta"
    meta_dir.mkdir(parents=True)
    (meta_dir / "info.json").write_text(json.dumps(info, indent=2) + "\n")

    stats: dict = {
        "observation.state.tcp_pose_left": _vector_stats(state[:, 0:7]),
        "observation.state.gripper_width_left": _vector_stats(state[:, 7]),
        "observation.state.tcp_pose_right": _vector_stats(state[:, 8:15]),
        "observation.state.gripper_width_right": _vector_stats(state[:, 15]),
        "observation.state.tcp_wrench_left": _vector_stats(np.zeros((total, 6))),
        "action.tcp_pose_left": _vector_stats(action[:, 0:7]),
        "action.gripper_width_left": _vector_stats(action[:, 7]),
        "action.tcp_pose_right": _vector_stats(action[:, 8:15]),
        "action.gripper_width_right": _vector_stats(action[:, 15]),
        "timestamp": _vector_stats(np.tile(np.arange(frames) / FPS, episodes)),
        "frame_index": _vector_stats(np.tile(np.arange(frames), episodes)),
        "episode_index": _vector_stats(np.repeat(np.arange(episodes), frames)),
        "index": _vector_stats(np.arange(total)),
        "task_index": _vector_stats(np.zeros(total)),
    }
    for camera in CAMERAS:
        shaped = lambda value: (np.full((3, 1, 1), value)).tolist()  # noqa: E731
        stats[camera] = {
            "min": shaped(0.0),
            "max": shaped(1.0),
            "mean": shaped(0.5),
            "std": shaped(0.29),
            "count": [total],
        }
    (meta_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")

    pq.write_table(
        pa.Table.from_pandas(
            pd.DataFrame({"task_index": [0]}, index=pd.Index([TASK], name="task"))
        ),
        meta_dir / "tasks.parquet",
    )

    episode_rows: dict = {
        "episode_index": pa.array(np.arange(episodes), type=pa.int64()),
        "tasks": pa.array([[TASK]] * episodes),
        "length": pa.array(np.full(episodes, frames), type=pa.int64()),
        "data/chunk_index": pa.array(np.zeros(episodes), type=pa.int64()),
        "data/file_index": pa.array(np.zeros(episodes), type=pa.int64()),
        "dataset_from_index": pa.array(np.arange(episodes) * frames, type=pa.int64()),
        "dataset_to_index": pa.array(
            (np.arange(episodes) + 1) * frames, type=pa.int64()
        ),
        "meta/episodes/chunk_index": pa.array(np.zeros(episodes), type=pa.int64()),
        "meta/episodes/file_index": pa.array(np.zeros(episodes), type=pa.int64()),
    }
    for camera in CAMERAS:
        episode_rows[f"videos/{camera}/chunk_index"] = pa.array(
            np.zeros(episodes), type=pa.int64()
        )
        episode_rows[f"videos/{camera}/file_index"] = pa.array(
            np.zeros(episodes), type=pa.int64()
        )
        episode_rows[f"videos/{camera}/from_timestamp"] = pa.array(
            np.arange(episodes) * frames / FPS
        )
        episode_rows[f"videos/{camera}/to_timestamp"] = pa.array(
            (np.arange(episodes) + 1) * frames / FPS
        )
    episodes_dir = meta_dir / "episodes/chunk-000"
    episodes_dir.mkdir(parents=True)
    pq.write_table(pa.table(episode_rows), episodes_dir / "file-000.parquet")

    for index, camera in enumerate(CAMERAS):
        _write_video(
            root / f"videos/{camera}/chunk-000/file-000.mp4", total, seed=index
        )
    return root
