"""Packed RGB video timestamps need a tolerance representable in float32."""

from functools import wraps
from pathlib import Path

import numpy as np
import torch
from lerobot.datasets import dataset_reader, video_utils

_original_decode = video_utils.decode_video_frames


def _representable_tolerance(tolerance_s: float, timestamps: list[float]) -> float:
    # Above 1024 seconds one float32 ULP exceeds 1e-4.
    grid = float(np.spacing(np.float32(max(timestamps))))
    return tolerance_s + grid


def _decode_video_frames(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    backend: str | None = None,
    return_uint8: bool = False,
    is_depth: bool = False,
) -> torch.Tensor:
    return _original_decode(
        video_path,
        timestamps,
        tolerance_s if is_depth else _representable_tolerance(tolerance_s, timestamps),
        backend,
        return_uint8=return_uint8,
        is_depth=is_depth,
    )


def _install_decoder_patch(worker_id: int | None = None) -> None:
    video_utils.decode_video_frames = _decode_video_frames
    dataset_reader.decode_video_frames = _decode_video_frames


def install_video_timestamp_patch(trainer) -> None:
    """Apply the RGB tolerance correction in training and its DataLoader workers.

    Depth retains its original tolerance. Spawn/forkserver workers install the
    same module-level decoder as the parent before loading any samples.
    """
    _install_decoder_patch()
    original = trainer._dataloader_worker_kwargs
    if getattr(original, "_paw_video_timestamps", False):
        return

    @wraps(original)
    def worker_kwargs(cfg):
        return {**original(cfg), "worker_init_fn": _install_decoder_patch}

    worker_kwargs._paw_video_timestamps = True
    trainer._dataloader_worker_kwargs = worker_kwargs
