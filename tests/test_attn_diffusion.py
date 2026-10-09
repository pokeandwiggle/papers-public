"""attn_diffusion: timing contract, deltas, cached serving, seam interpolation, top-5."""

import json

import numpy as np
import pytest
import torch
from lerobot.configs import FeatureType, PolicyFeature

import paw_baseline.attn_diffusion  # noqa: F401
from paw_baseline.attn_diffusion.configuration_attn_diffusion import AttnDiffusionConfig
from paw_baseline.attn_diffusion.deltas import (
    absolute_to_deltas,
    accumulate,
    delta_statistics,
    rotation_error_deg,
)
from paw_baseline.attn_diffusion.modeling_attn_diffusion import AttnDiffusionPolicy
from paw_baseline.attn_diffusion.serving import AttnPlanner, interpolate_into_chunk, mix_wire
from paw_baseline.representation import encode_model
from train.__main__ import load_preset

CAMERAS = ("observation.images.rgb_left", "observation.images.rgb_right", "observation.images.rgb_static")


def _preset_config(name: str) -> AttnDiffusionConfig:
    args = dict(arg[2:].split("=", 1) for arg in load_preset(name)["args"])
    kwargs = {}
    for key, value in args.items():
        if key.startswith("policy.") and key != "policy.type":
            try:
                kwargs[key.removeprefix("policy.")] = json.loads(value)
            except json.JSONDecodeError:
                kwargs[key.removeprefix("policy.")] = value
    return AttnDiffusionConfig(**kwargs)


@pytest.mark.parametrize(
    ("preset", "n_obs", "replan", "horizon"),
    [("attn_dp_to2_exec24", 2, 24, 48), ("attn_dp_to8_exec6", 8, 6, 66)],
)
def test_presets(preset, n_obs, replan, horizon):
    config = _preset_config(preset)
    assert (config.n_obs_steps, config.replan_steps, config.horizon) == (n_obs, replan, horizon)
    assert config.prediction_steps == config.n_action_steps == 45
    assert config.observation_delta_indices == list(range(-3 * (n_obs - 1), 1, 3))
    assert config.action_delta_indices == list(range(-3 * (n_obs - 1) - 1, 45))
    assert config.vision_backbone == "vit_base_patch16_dinov3.lvd1689m"
    assert config.short_range_obs_horizon == 2 and config.short_range_dropout == 0.3
    assert config.encoder_lr_scale == 0.1 and config.use_ema and config.ema_power == 0.75
    assert config.interpolation_ms == 180


@pytest.mark.parametrize("preset", ["attn_dp_to2_exec24", "attn_dp_to8_exec6"])
def test_saved_config_carries_validation_fields(tmp_path, preset):
    # inference.validate reads n_action_steps straight from config.json.
    _preset_config(preset)._save_pretrained(tmp_path)
    saved = json.loads((tmp_path / "config.json").read_text())
    assert saved["n_action_steps"] == saved["prediction_steps"] == 45
    loaded = AttnDiffusionConfig.from_pretrained(tmp_path)
    assert loaded.n_action_steps == 45 and loaded.replan_steps == _preset_config(preset).replan_steps


def test_presets_differ_only_in_context_and_execution():
    a, b = (set(load_preset(n)["args"]) for n in ("attn_dp_to2_exec24", "attn_dp_to8_exec6"))
    assert a - b == {"--policy.n_obs_steps=2", "--policy.replan_steps=24"}
    assert b - a == {"--policy.n_obs_steps=8", "--policy.replan_steps=6"}
    assert "--steps=200000" in a


def _random_poses(rng, n):
    physical = np.zeros((n, 16), dtype=np.float32)
    physical[:, [0, 1, 2, 8, 9, 10]] = rng.normal(0, 0.2, (n, 6))
    quats = rng.normal(size=(n, 2, 4))
    quats /= np.linalg.norm(quats, axis=-1, keepdims=True)
    physical[:, 3:7], physical[:, 11:15] = quats[:, 0], quats[:, 1]
    physical[:, [7, 15]] = rng.uniform(0, 0.08, (n, 2))
    return encode_model(physical)


def test_deltas_accumulate_back_to_actions():
    rng = np.random.default_rng(0)
    state = _random_poses(rng, 1)[0]
    # Smooth trajectory: small consecutive changes from a random start.
    start = _random_poses(rng, 1)
    actions = [start[0]]
    for _ in range(50):
        step = actions[-1].copy()
        step[[0, 1, 2, 10, 11, 12]] += rng.normal(0, 0.002, 6)
        actions.append(step)
    actions = np.stack(actions).astype(np.float32)
    window = torch.from_numpy(actions[None])  # index 0 = action before the window
    past = 3
    deltas = absolute_to_deltas(torch.from_numpy(state[None]), window, past)[0].numpy()
    assert deltas.shape == (50, 14)
    rebuilt = accumulate(state, deltas[past:])
    truth = actions[1 + past :]
    np.testing.assert_allclose(rebuilt[:, [0, 1, 2, 10, 11, 12]], truth[:, [0, 1, 2, 10, 11, 12]], atol=1e-5)
    assert rotation_error_deg(rebuilt[:, 3:9], truth[:, 3:9]).max() < 1e-3
    np.testing.assert_allclose(rebuilt[:, [9, 19]], truth[:, [9, 19]], atol=1e-7)
    stats = delta_statistics([(np.repeat(state[None], 51, 0), actions)])
    assert all(len(v) == 14 for v in stats.values())


def _tiny_policy(n_obs_steps=3, replan=2, bf16=False):
    rng = np.random.default_rng(1)
    config = AttnDiffusionConfig(
        n_obs_steps=n_obs_steps,
        replan_steps=replan,
        vision_backbone="vit_tiny_patch16_224",
        pretrained_backbone=False,
        diffusion_step_embed_dim=32,
        down_dims=(32, 64, 128),
        num_attention_heads=4,
        num_inference_steps=3,
        state_min=[-1.0] * 20,
        state_max=[1.0] * 20,
        delta_first_min=list(rng.uniform(-0.2, -0.1, 14)),
        delta_first_max=list(rng.uniform(0.1, 0.2, 14)),
        delta_step_min=[-0.02] * 14,
        delta_step_max=[0.02] * 14,
        dtype="bfloat16" if bf16 else "float32",
    )
    config.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(20,)),
        **{k: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)) for k in CAMERAS},
    }
    config.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(20,))}
    return AttnDiffusionPolicy(config).eval()


def test_delta_targets_ignore_bf16_autocast():
    policy = _tiny_policy()
    rng = np.random.default_rng(3)
    batch = {
        "observation.state": torch.from_numpy(_random_poses(rng, 2 * policy.config.n_obs_steps)).view(2, -1, 20),
        "action": torch.from_numpy(_random_poses(rng, 2 * (policy.config.horizon + 1))).view(2, -1, 20),
    }
    expected = policy.deltas_from_batch(batch)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = policy.deltas_from_batch(batch)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected)


def test_normalization_round_trip_uses_first_slot_stats():
    policy = _tiny_policy()
    deltas = torch.randn(2, policy.config.horizon, 14) * 0.01
    torch.testing.assert_close(policy.unnormalize_deltas(policy.normalize_deltas(deltas)), deltas)
    scale, _ = policy._delta_affine(policy.config.horizon)
    torch.testing.assert_close(scale[policy.config.past_steps], policy.delta_first_scale)
    torch.testing.assert_close(scale[0], policy.delta_step_scale)


def test_cached_planner_matches_policy_sampling():
    policy = _tiny_policy()
    planner = AttnPlanner(policy)
    rng = np.random.default_rng(2)
    states = _random_poses(rng, 12)
    frames = [
        {"observation.state": states[i], **{k: rng.random((3, 64, 64), dtype=np.float32) for k in CAMERAS}}
        for i in range(12)
    ]
    for newest in range(12):
        planner.append(frames[newest])
        if newest not in (0, 4, 6, 11):
            continue
        snapshot = planner.snapshot(newest)
        indices = planner.context_indices(newest)
        state = torch.from_numpy(np.stack([frames[i]["observation.state"] for i in indices]))[None]
        images = {k: torch.from_numpy(np.stack([frames[i][k] for i in indices]))[None] for k in CAMERAS}
        expected = policy.sample_deltas(state, images, torch.Generator().manual_seed(newest))
        actual = planner.sample(snapshot, torch.Generator().manual_seed(newest))
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    # Strided context: each plan encodes only frames it has not seen before.
    assert planner.encoded < 4 * (policy.config.n_obs_steps + policy.config.short_range_obs_horizon)
    absolute = planner.plan(planner.snapshot(11))
    assert absolute.shape == (45, 20)


def _wire(x):
    row = np.zeros(16, dtype=np.float32)
    row[[6, 13]] = 1.0
    row[0] = x
    return row


def test_desired_anchor_continues_from_the_current_command():
    from paw_baseline.attn_diffusion.serving import DESIRED_STATE, anchor_chunk

    rng = np.random.default_rng(5)
    measured, desired = _random_poses(rng, 2)
    future = rng.normal(0, 0.01, (45, 14))
    future[:, [6, 13]] = rng.uniform(0, 0.08, (45, 2))  # absolute gripper widths
    frame = {"observation.state": measured, DESIRED_STATE: desired}
    chunk, used = anchor_chunk(frame, future, "desired")
    assert used == "desired"
    np.testing.assert_allclose(chunk[0, [0, 1, 2, 10, 11, 12]], desired[[0, 1, 2, 10, 11, 12]], atol=1e-6)
    assert rotation_error_deg(chunk[0, 3:9], desired[3:9]) < 1e-4
    np.testing.assert_allclose(chunk[:, [9, 19]], future[:, [6, 13]], atol=1e-7)  # predicted grippers
    # Later slots chain only the per-step deltas from the desired pose.
    steps = future.copy()
    steps[0, [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]] = 0.0
    np.testing.assert_allclose(chunk, accumulate(desired, steps), atol=1e-6)
    # The measured anchor (training semantics) and the fallback without a desired pose.
    np.testing.assert_allclose(anchor_chunk(frame, future, "measured")[0], accumulate(measured, future))
    fallback, used = anchor_chunk({"observation.state": measured}, future, "desired")
    assert used == "measured"
    np.testing.assert_allclose(fallback, accumulate(measured, future))


def test_serve_folds_the_desired_pose_into_attn_frames():
    from inference.serve import BaselinePolicyServer, _fold_state
    from paw_baseline.attn_diffusion.serving import DESIRED_STATE
    from paw_baseline.layout import STATE_KEY_DIMS

    rng = np.random.default_rng(6)

    def pose():
        quat = rng.normal(size=4)
        return np.concatenate([rng.normal(0, 0.3, 3), quat / np.linalg.norm(quat)])[None].astype(np.float32)

    observation = {
        key: pose() if dim == 7 else rng.uniform(0, 0.08, (1, 1)).astype(np.float32)
        for key, dim in STATE_KEY_DIMS
    }
    for side in ("left", "right"):
        observation[f"observation.state.desired_tcp_pose_{side}"] = pose()
    frames = []
    server = object.__new__(BaselinePolicyServer)
    server._attn = type("Planner", (), {"append": lambda self, frame: frames.append(frame) or 1})()
    server._history = None
    server._cameras = []
    server._pending = None
    server._schedule = type("Schedule", (), {"due": lambda self, index: False})()
    server._on_history_observation(observation)
    swapped = {
        **observation,
        **{f"observation.state.tcp_pose_{s}": observation[f"observation.state.desired_tcp_pose_{s}"] for s in ("left", "right")},
    }
    np.testing.assert_array_equal(frames[0]["observation.state"], _fold_state(observation)[0])
    np.testing.assert_array_equal(frames[0][DESIRED_STATE], _fold_state(swapped)[0])
    # No desired pose from the station: no anchor entry (the planner falls back).
    for side in ("left", "right"):
        del observation[f"observation.state.desired_tcp_pose_{side}"]
    server._on_history_observation(observation)
    assert DESIRED_STATE not in frames[1]


def test_interpolation_enters_chunk_until_180ms():
    chunk = np.stack([_wire(float(i)) for i in range(45)])[None]
    previous = _wire(-1.0)
    out = interpolate_into_chunk(chunk, seam=2, previous=previous, interpolation_ms=180.0)
    target = 5.4  # 180 ms at 30 Hz, linearly between slots 5 and 6
    expected = [-1.0 + (s - 1) / (target - 1) * (target + 1) for s in range(2, 6)]
    np.testing.assert_allclose(out[0, 2:6, 0], expected, atol=1e-5)
    np.testing.assert_allclose(out[0, :2, 0], chunk[0, :2, 0])
    np.testing.assert_allclose(out[0, 6:, 0], chunk[0, 6:, 0])
    # A seam past 180 ms or no previous action leaves the chunk unchanged.
    np.testing.assert_array_equal(interpolate_into_chunk(chunk, 6, previous, 180.0), chunk)
    np.testing.assert_array_equal(interpolate_into_chunk(chunk, 2, None, 180.0), chunk)
    mixed = mix_wire(previous, chunk[0, 3], 0.5)
    assert abs(np.linalg.norm(mixed[3:7]) - 1.0) < 1e-6


@pytest.mark.parametrize(
    ("final_score", "served", "top"),
    [(0.95, 20, [20, 10, 30, 15, 25]), (0.1, 35, [35, 20, 10, 30, 15])],
)
def test_top_k_keeps_five_and_serves_best(tmp_path, monkeypatch, final_score, served, top):
    from safetensors.torch import load_file, save_file

    from paw_baseline.attn_diffusion import selection

    keeper = selection.TopKCheckpoints(tmp_path, [0])
    scores = {5: 0.9, 10: 0.5, 15: 0.7, 20: 0.4, 25: 0.8, 30: 0.6, 35: final_score}
    monkeypatch.setattr(
        keeper,
        "validation_mse",
        lambda policy, pre: {"val_ddim_mse": scores[keeper._step], "val_ddim_delta_mse": 1.0},
    )
    keeper._batch = {"action": torch.zeros(3)}
    for step in sorted(scores):
        pretrained = tmp_path / f"checkpoints/{step:06d}/pretrained_model"
        pretrained.mkdir(parents=True)
        save_file(
            {"core.w": torch.full((2,), float(step)), "ema_core.w": torch.full((2,), float(step))},
            str(pretrained / "model.safetensors"),
        )
        keeper._step = step
        keeper.score(pretrained.parent, step, None, None, final=step == 35)
    final = tmp_path / "checkpoints/000035/pretrained_model"
    tensors = load_file(str(final / "model.safetensors"))
    assert tensors["ema_core.w"].tolist() == [float(served)] * 2  # best (lowest MSE) EMA weights
    assert tensors["core.w"].tolist() == [35.0, 35.0]
    record = json.loads((final / ".paw/checkpoint_selection.json").read_text())
    assert record["served_step"] == served
    assert [r["step"] for r in record["top_k"]] == top
    assert not keeper.scratch.exists()
