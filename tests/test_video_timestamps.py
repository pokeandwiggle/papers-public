"""Long packed videos must not lose matching frames to float32 rounding."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from lerobot.datasets import dataset_reader, video_utils
from lerobot.scripts import lerobot_train

from train import video_timestamps


class _DecoderCache:
    def __init__(self, loaded):
        self.loaded = loaded

    def get_decoder(self, path):
        return SimpleNamespace(
            metadata=SimpleNamespace(average_fps=30),
            get_frames_at=lambda indices: SimpleNamespace(
                data=torch.full((1, 3, 2, 2), 127, dtype=torch.uint8),
                pts_seconds=torch.tensor([self.loaded], dtype=torch.float64),
            ),
        )


def _timestamps(offset):
    below = np.float32(offset)
    above = np.nextafter(below, np.float32(float("inf")))
    midpoint = (float(below) + float(above)) / 2
    return midpoint + 1e-9, midpoint - 1e-9


class _PackedVideo(torch.utils.data.Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, index):
        query, loaded = _timestamps(1941.2333333333333)
        video_utils._default_decoder_cache = _DecoderCache(loaded)
        return dataset_reader.decode_video_frames(
            "packed.mp4", [query], 1e-4, "torchcodec", return_uint8=True
        )


@pytest.fixture
def installed_patch(monkeypatch):
    # Restore the real entry points after each test, including the trainer hook.
    monkeypatch.setattr(
        video_utils, "decode_video_frames", video_utils.decode_video_frames
    )
    monkeypatch.setattr(
        dataset_reader, "decode_video_frames", dataset_reader.decode_video_frames
    )
    monkeypatch.setattr(
        lerobot_train,
        "_dataloader_worker_kwargs",
        lerobot_train._dataloader_worker_kwargs,
    )
    video_timestamps.install_video_timestamp_patch(lerobot_train)


@pytest.mark.parametrize(
    "offset", [1370.2333333333333, 1941.2333333333333, 2900.2333333333333]
)
def test_matching_frame_survives_float32_rounding(offset, installed_patch, monkeypatch):
    query, loaded = _timestamps(offset)
    assert abs(query - loaded) < 1e-4
    monkeypatch.setattr(video_utils, "_default_decoder_cache", _DecoderCache(loaded))
    for return_uint8 in (False, True):
        frames = dataset_reader.decode_video_frames(
            "packed.mp4", [query], 1e-4, "torchcodec", return_uint8=return_uint8
        )
        expected = torch.full((1, 3, 2, 2), 127, dtype=torch.uint8)
        torch.testing.assert_close(
            frames, expected if return_uint8 else expected / 255.0
        )


def test_wrong_frame_is_still_rejected(installed_patch, monkeypatch):
    query, loaded = _timestamps(1941.2333333333333)
    monkeypatch.setattr(
        video_utils, "_default_decoder_cache", _DecoderCache(loaded + 1 / 30)
    )
    with pytest.raises(video_utils.FrameTimestampError):
        dataset_reader.decode_video_frames("packed.mp4", [query], 1e-4, "torchcodec")


def test_depth_keeps_original_tolerance(installed_patch, monkeypatch):
    seen = {}

    def decode(path, timestamps, tolerance, backend, **kwargs):
        seen.update(tolerance=tolerance, **kwargs)
        return "depth"

    monkeypatch.setattr(video_timestamps, "_original_decode", decode)
    assert (
        dataset_reader.decode_video_frames(
            "depth.mp4", [1941.23], 1e-4, "pyav", is_depth=True
        )
        == "depth"
    )
    assert seen == {"tolerance": 1e-4, "return_uint8": False, "is_depth": True}


@pytest.mark.parametrize("context", ["fork", "forkserver", "spawn"])
def test_dataloader_workers_apply_patch(context, installed_patch):
    cfg = SimpleNamespace(
        num_workers=1,
        prefetch_factor=2,
        persistent_workers=False,
        dataloader_multiprocessing_context=context,
    )
    loader = torch.utils.data.DataLoader(
        _PackedVideo(), num_workers=1, **lerobot_train._dataloader_worker_kwargs(cfg)
    )
    batches = list(loader)
    assert len(batches) == 1
    torch.testing.assert_close(
        batches[0], torch.full((1, 1, 3, 2, 2), 127, dtype=torch.uint8)
    )


def test_install_is_idempotent(installed_patch):
    original_hook = lerobot_train._dataloader_worker_kwargs
    video_timestamps.install_video_timestamp_patch(lerobot_train)
    assert lerobot_train._dataloader_worker_kwargs is original_hook
