"""Fit RGB normalization on selected export episodes without holdout leakage."""

from collections import defaultdict
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq

from paw_baseline.layout import CAMERA_KEYS, FPS


def selected_visual_stats(source: Path, episodes: tuple[int, ...]) -> dict:
    """Decode selected intervals from v3 videos and return per-channel moments.

    The exporter supplies a shared video file plus per-episode timestamps.
    Frame indices avoid float comparisons at adjacent episode boundaries.
    """
    selected = set(episodes)
    rows = [
        row
        for path in sorted((source / "meta/episodes").glob("chunk-*/file-*.parquet"))
        for row in pq.read_table(path).to_pylist()
        if row["episode_index"] in selected
    ]
    if {row["episode_index"] for row in rows} != selected:
        raise ValueError("episode metadata does not cover the normalization selection")
    result = {}
    expected_frames = sum(row["length"] for row in rows)
    for camera in CAMERA_KEYS:
        intervals = defaultdict(list)
        for row in rows:
            prefix = f"videos/{camera}"
            path = (
                source
                / prefix
                / f"chunk-{row[prefix + '/chunk_index']:03d}"
                / f"file-{row[prefix + '/file_index']:03d}.mp4"
            )
            start = round(row[prefix + "/from_timestamp"] * FPS)
            intervals[path].append((start, start + row["length"]))
        count = 0
        pixels = 0
        total = np.zeros(3, dtype=np.float64)
        squares = np.zeros(3, dtype=np.float64)
        minimum = np.full(3, np.inf)
        maximum = np.full(3, -np.inf)
        for path, spans in sorted(intervals.items()):
            with av.open(str(path)) as container:
                for index, frame in enumerate(container.decode(video=0)):
                    if not any(start <= index < end for start, end in spans):
                        continue
                    values = (
                        frame.to_ndarray(format="rgb24")
                        .reshape(-1, 3)
                        .astype(np.float64)
                        / 255.0
                    )
                    count += 1
                    pixels += len(values)
                    total += values.sum(axis=0)
                    squares += np.square(values).sum(axis=0)
                    minimum = np.minimum(minimum, values.min(axis=0))
                    maximum = np.maximum(maximum, values.max(axis=0))
        if count != expected_frames:
            raise ValueError(
                f"{camera}: decoded {count} training frames, expected {expected_frames}"
            )
        mean = total / pixels
        result[camera] = {
            "count": [count],
            "max": maximum.reshape(3, 1, 1).tolist(),
            "mean": mean.reshape(3, 1, 1).tolist(),
            "min": minimum.reshape(3, 1, 1).tolist(),
            "std": np.sqrt(np.maximum(squares / pixels - np.square(mean), 0))
            .reshape(3, 1, 1)
            .tolist(),
        }
    return result
