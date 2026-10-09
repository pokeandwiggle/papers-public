"""Strided-history Diffusion: timing contract, U-Net lengths, serving parity."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from lerobot.configs import FeatureType, PolicyFeature

import paw_baseline.hist_diffusion  # noqa: F401
from paw_baseline.hist_diffusion.configuration_hist_diffusion import (
    HistDiffusionConfig,
)
from paw_baseline.hist_diffusion.modeling_hist_diffusion import HistDiffusionPolicy
from paw_baseline.hist_diffusion.serving import (
    ReplanSchedule,
    StridedHistory,
    blend_seam,
)
from train.__main__ import load_preset

CAMERA = "observation.images.rgb_static"


def _preset_config(name: str) -> HistDiffusionConfig:
    args = dict(arg[2:].split("=", 1) for arg in load_preset(name)["args"])
    fields = {
        key.removeprefix("policy."): value
        for key, value in args.items()
        if key.startswith("policy.") and key != "policy.type"
    }
    kwargs = {
        key: (json.loads(value) if value not in ("default", "DDIM") else value)
        for key, value in fields.items()
    }
    return HistDiffusionConfig(**kwargs)


def _small_config(**kwargs) -> HistDiffusionConfig:
    config = HistDiffusionConfig(
        down_dims=(32, 64, 128),
        diffusion_step_embed_dim=16,
        pretrained_backbone_weights=None,
        spatial_softmax_num_keypoints=4,
        num_inference_steps=2,
        noise_scheduler_type="DDIM",
        **kwargs,
    )
    config.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(20,)),
        CAMERA: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
    }
    config.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(20,))
    }
    return config


@pytest.mark.parametrize(
    ("preset", "n_obs", "replan", "horizon"),
    [("hist_diffusion_to2_exec24", 2, 24, 48), ("hist_diffusion_to8_exec6", 8, 6, 66)],
)
def test_presets_hold_prediction_fixed(preset, n_obs, replan, horizon):
    config = _preset_config(preset)
    assert (config.n_obs_steps, config.replan_steps, config.horizon) == (
        n_obs,
        replan,
        horizon,
    )
    assert config.prediction_steps == config.n_action_steps == 45
    assert config.observation_delta_indices == list(range(-3 * (n_obs - 1), 1, 3))
    assert config.action_delta_indices == list(range(-3 * (n_obs - 1), 45))
    assert config.num_inference_steps == 16 and config.compile_model


def test_presets_differ_only_in_context_and_execution():
    a, b = (
        set(load_preset(name)["args"])
        for name in ("hist_diffusion_to2_exec24", "hist_diffusion_to8_exec6")
    )
    assert a - b == {"--policy.n_obs_steps=2", "--policy.replan_steps=24"}
    assert b - a == {"--policy.n_obs_steps=8", "--policy.replan_steps=6"}


@pytest.mark.parametrize("n_obs", [1, 2, 3, 8])
def test_unet_trains_and_samples_any_length(n_obs):
    torch.manual_seed(0)
    config = _small_config(n_obs_steps=n_obs, prediction_steps=7, replan_steps=2)
    policy = HistDiffusionPolicy(config)
    batch = {
        "observation.state": torch.randn(2, n_obs, 20),
        CAMERA: torch.rand(2, n_obs, 3, 64, 64),
        "action": torch.randn(2, config.horizon, 20),
        "action_is_pad": torch.zeros(2, config.horizon, dtype=torch.bool),
    }
    loss, _ = policy.forward(batch)
    loss.backward()
    assert torch.isfinite(loss)
    policy.eval()
    chunk = policy.predict_history_chunk(
        batch["observation.state"], {CAMERA: batch[CAMERA]}
    )
    assert chunk.shape == (2, 7, 20)


def test_emitted_actions_start_at_current_observation():
    config = _small_config(n_obs_steps=3, prediction_steps=5, replan_steps=2)
    policy = HistDiffusionPolicy(config).eval()
    sequence = torch.arange(config.horizon, dtype=torch.float32)
    policy.diffusion.conditional_sample = lambda batch_size, **_: sequence.view(
        1, -1, 1
    ).expand(batch_size, -1, 20)
    chunk = policy.predict_history_chunk(
        torch.zeros(1, 3, 20), {CAMERA: torch.zeros(1, 3, 3, 64, 64)}
    )
    # Past tokens cover (3 - 1) * 3 = 6 grid steps before the current one.
    assert chunk[0, :, 0].tolist() == [6.0, 7.0, 8.0, 9.0, 10.0]


def test_history_strides_and_repeats_first_frame():
    history = StridedHistory(n_obs_steps=3, obs_stride=3)
    for index in range(10):
        history.append({"x": np.array([index])})
        context = history.context()["x"][:, 0].tolist()
        expected = [max(index + offset, 0) for offset in (-6, -3, 0)]
        assert context == expected
    history.reset()
    history.append({"x": np.array([42])})
    assert history.context()["x"][:, 0].tolist() == [42, 42, 42]


def test_schedule_replans_on_grid_and_after_late_plans():
    schedule = ReplanSchedule(6)
    planned = []
    for index in range(20):
        # Inference occupies observations 6-8, so the plan due at 6 starts at 9.
        if schedule.due(index) and index not in (6, 7, 8):
            schedule.planned(index)
            planned.append(index)
    assert planned == [0, 9, 15]


def test_blend_seam_crossfades_only_replaceable_slots():
    chunk = np.zeros((1, 10, 16), dtype=np.float32)
    chunk[0, :, 6] = chunk[0, :, 13] = 1.0
    chunk[0, :, 0] = 1.0
    old = np.zeros((1, 5, 16), dtype=np.float32)
    old[0, :, 3] = old[0, :, 10] = -1.0  # same rotation, opposite sign
    blended = blend_seam(chunk, committed_count=2, replaceable=old, steps=4)
    np.testing.assert_allclose(blended[0, :2], chunk[0, :2])
    np.testing.assert_allclose(blended[0, 2:6, 0], [0.2, 0.4, 0.6, 0.8], rtol=1e-6)
    np.testing.assert_allclose(blended[0, 6:], chunk[0, 6:])
    norms = np.linalg.norm(blended[0, :, 3:7], axis=1)
    np.testing.assert_allclose(norms, 1.0, rtol=1e-6)
    assert blend_seam(chunk, 2, np.zeros((1, 0, 16)), 4) is not chunk
    np.testing.assert_array_equal(blend_seam(chunk, 2, np.zeros((1, 0, 16)), 4), chunk)


def test_serving_context_matches_training_sample(export_dir, cache_root, monkeypatch):
    """Frames assembled from the 30 Hz stream equal LeRobot's strided sample."""
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    from paw_baseline.prepare import VIDEO_BACKEND, prepare_dataset

    monkeypatch.setenv("PAW_BASELINE_CACHE", str(cache_root))
    root: Path = prepare_dataset(export_dir).root
    config = _small_config(n_obs_steps=4, prediction_steps=6, replan_steps=2)
    meta = LeRobotDatasetMetadata(root.name, root=root)
    deltas = resolve_delta_timestamps(config, meta)
    strided = LeRobotDataset(
        root.name, root=root, delta_timestamps=deltas, video_backend=VIDEO_BACKEND
    )
    single = LeRobotDataset(root.name, root=root, video_backend=VIDEO_BACKEND)
    keys = ["observation.state", *meta.camera_keys]
    episode_start = int(meta.episodes[1]["dataset_from_index"])
    for frame in (0, 4, 20):  # padded start, partly padded, fully inside
        history = StridedHistory(config.n_obs_steps, config.obs_stride)
        for index in range(frame + 1):
            item = single[episode_start + index]
            history.append({key: item[key].numpy() for key in keys})
        context = history.context()
        sample = strided[episode_start + frame]
        for key in keys:
            np.testing.assert_array_equal(context[key], sample[key].numpy())
        assert sample["action"].shape[0] == config.horizon


def test_validation_scores_are_logged_and_final_checkpoint_served(
    export_dir, cache_root, monkeypatch, tmp_path
):
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    from paw_baseline.hist_diffusion.checkpoint_selection import ValidationSelector
    from paw_baseline.hist_diffusion.processor_hist_diffusion import (
        make_hist_diffusion_pre_post_processors,
    )
    from paw_baseline.prepare import prepare_dataset

    monkeypatch.setenv("PAW_BASELINE_CACHE", str(cache_root))
    root = prepare_dataset(export_dir).root
    meta = LeRobotDatasetMetadata(root.name, root=root)
    config = _small_config(n_obs_steps=3, prediction_steps=6, replan_steps=2)
    config.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(20,)),
        **{
            key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64))
            for key in meta.camera_keys
        },
    }
    config.device = "cpu"
    torch.manual_seed(0)
    policy = HistDiffusionPolicy(config)
    preprocessor, _ = make_hist_diffusion_pre_post_processors(config, meta.stats)
    selector = ValidationSelector(root, [1])
    first = selector.score(tmp_path / "checkpoints/000002", 2, policy, preprocessor)
    again = selector.score(tmp_path / "checkpoints/000002", 2, policy, preprocessor)
    assert first == again and np.isfinite(first)  # fixed anchors and noise
    assert policy.training  # training mode restored
    assert selector.scores[0]["anchors"] == 32
    assert (
        tmp_path / "checkpoints/000002/pretrained_model/.paw/validation_score.json"
    ).is_file()
    with torch.no_grad():
        for parameter in policy.diffusion.unet.parameters():
            parameter.mul_(1.5)
    selector.score(tmp_path / "checkpoints/000004", 4, policy, preprocessor)
    selection = selector.write(tmp_path, tmp_path / "checkpoints/000004")
    assert selection["served"]["step"] == 4
    assert [s["step"] for s in selection["scores"]] == [2, 2, 4]
    assert selection["lowest"]["val_ddim_mse"] == min(
        s["val_ddim_mse"] for s in selection["scores"]
    )
    assert json.loads((tmp_path / ".paw/checkpoint_selection.json").read_text())
