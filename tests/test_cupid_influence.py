import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from paw_baseline.cupid import influence
from paw_baseline.cupid.rollouts import (
    _resize_jpeg,
    cartesian_action_to_train,
    cover_crop_box,
    match_requests,
    observation_state_to_train,
)
from paw_baseline.cupid.score import episode_source_ids


def test_sign_sketch_is_seeded_linear_and_unbiased():
    p, d = 5003, 64
    x = torch.randn(p, dtype=torch.float64)
    y = x + 0.5 * torch.randn(p, dtype=torch.float64)
    first = influence.SignSketch(p, d, seed=3, device="cpu")
    again = influence.SignSketch(p, d, seed=3, device="cpu")
    assert torch.equal(first(x), again(x))
    assert torch.allclose(first(2 * x + y), 2 * first(x) + first(y))
    estimates = [
        float(influence.SignSketch(p, d, seed=s, device="cpu")(x)
              @ influence.SignSketch(p, d, seed=s, device="cpu")(y))
        for s in range(200)
    ]
    exact = float(x @ y)
    assert abs(np.mean(estimates) - exact) < 0.05 * abs(exact)


def test_performance_influence_matches_pairwise_sum():
    rng = np.random.default_rng(0)
    d = 6
    train = rng.normal(size=(40, d))
    demo_of = rng.integers(0, 4, size=40)
    tests = rng.normal(size=(15, d))
    rollout_of = rng.integers(0, 3, size=15)
    success = np.array([True, False, True])
    h_inv = influence.gauss_newton_inverse(influence.gram(train))
    assert np.allclose(h_inv, np.linalg.inv(train.T @ train))
    demo_sums = np.stack([train[demo_of == k].sum(0) for k in range(4)])
    rollout_sums = np.stack([tests[rollout_of == r].sum(0) for r in range(3)])
    scores, traj = influence.performance_influence(rollout_sums, success, demo_sums, h_inv)
    brute = np.zeros(4)
    for i, r in enumerate(rollout_of):
        sign = 1.0 if success[r] else -1.0
        for j, k in enumerate(demo_of):
            brute[k] += sign * tests[i] @ h_inv @ train[j]
    assert np.allclose(scores, brute)
    assert traj.shape == (3, 4)


def _pairwise_net(train, demo_of, tests, rollout_of, success, n_demos):
    h_inv = np.linalg.inv(train.T @ train)
    brute = np.zeros(n_demos)
    for i, r in enumerate(rollout_of):
        sign = 1.0 if success[r] else -1.0
        for j, k in enumerate(demo_of):
            brute[k] += sign * tests[i] @ h_inv @ train[j]
    return brute


def test_scores_from_saved_gram_use_inverse_once(tmp_path):
    """The score stage consumes an accumulated X^T X; it must invert it exactly once."""
    from paw_baseline.cupid import score

    rng = np.random.default_rng(1)
    d, n_demos, n_rollouts = 5, 4, 3
    train = rng.normal(size=(60, d)) * rng.uniform(0.2, 5.0, size=d)
    demo_of = np.repeat(np.arange(n_demos), 15)
    tests = rng.normal(size=(12, d))
    rollout_of = rng.integers(0, n_rollouts, size=12)
    success = np.array([True, False, False])
    expected = _pairwise_net(train, demo_of, tests, rollout_of, success, n_demos)
    xtx = np.zeros((d, d))
    for row in train:  # accumulate like run_train does
        xtx += np.outer(row, row)
    saved = {
        "xtx": xtx,
        "sums": np.stack([train[demo_of == k].sum(0) for k in range(n_demos)]),
        "source_ids": np.array([11, 12, 13, 14]),
    }
    rollout_sums = np.stack([tests[rollout_of == r].sum(0) for r in range(n_rollouts)])
    scores, _ = influence.cupid_scores(saved["xtx"], saved["sums"], rollout_sums, success)
    assert np.allclose(scores, expected)
    squared = influence.performance_influence(
        rollout_sums, success, saved["sums"], np.linalg.inv(xtx @ xtx)
    )[0]
    assert not np.allclose(squared, expected)
    score.write_scores(
        tmp_path,
        saved,
        {"num_pairs": 60, "proj_dim": d},
        rollout_sums=rollout_sums,
        success=success,
        episodes=np.arange(n_rollouts),
        rollout_stats={"eval_dir": "x"},
        filter_ratio=0.5,
        ridge=0.0,
    )
    written = json.loads((tmp_path / score.SCORE_FILE).read_text())
    assert np.allclose([written["scores"][str(i)] for i in (11, 12, 13, 14)], expected)
    order = np.argsort(expected)
    assert written["train_episode_ids"] == sorted(int(11 + k) for k in order[2:])
    assert written["scoring_version"] == 2


def test_gauss_newton_inverse_rejects_feature_matrices():
    with pytest.raises(ValueError):
        influence.gauss_newton_inverse(np.ones((10, 3)))


def test_freeze_action_dims_zeroes_only_the_parked_arm():
    loaded = SimpleNamespace(frozen_arm="left")
    action = torch.ones(2, 3, 20)
    frozen = influence.freeze_action_dims(loaded, action)
    assert torch.all(frozen[..., :10] == 0) and torch.all(frozen[..., 10:] == 1)
    assert torch.all(action == 1)
    assert influence.freeze_action_dims(SimpleNamespace(frozen_arm=None), action) is action


def _stamp(ns):
    return SimpleNamespace(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)


def _pose(x):
    return SimpleNamespace(
        position=SimpleNamespace(x=x, y=x + 1, z=x + 2),
        orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
    )


def test_match_requests_joins_on_capture_stamp_and_skips_unmatched():
    obs = [
        SimpleNamespace(capture_stamp=_stamp(t), chunk_requested=req)
        for t, req in ((100, True), (200, False), (300, True), (400, True))
    ]
    chunks = [
        SimpleNamespace(chunk_target_stamp=_stamp(t), name=t) for t in (300, 999, 100)
    ]
    pairs = match_requests(obs, chunks)
    assert [(o.capture_stamp.nanosec, c.name) for o, c in pairs] == [(100, 100), (300, 300)]


def test_wire_messages_map_to_train_order():
    action = SimpleNamespace(
        left_arm_pose=_pose(1.0),
        right_arm_pose=_pose(5.0),
        left_gripper_width=0.01,
        right_gripper_width=0.02,
    )
    vec = cartesian_action_to_train(action)
    assert vec.tolist() == [1, 2, 3, 0, 0, 0, 1, 0.01, 5, 6, 7, 0, 0, 0, 1, 0.02]
    obs = SimpleNamespace(
        tcp_left=[_pose(1.0)],
        tcp_right=[_pose(5.0)],
        gripper_left_width=0.01,
        gripper_right_width=0.02,
    )
    assert observation_state_to_train(obs).tolist() == vec.tolist()


@pytest.mark.parametrize(
    ("native", "fx", "cx", "expected_fx", "expected_cx", "box"),
    [
        # Static camera: native 1280x720 -> export calibration of the 256x256 view.
        ((720, 1280), 708.8812384594567, 639.5518975572207, 251.98512773363498, 128.01844795979332, (256, 455, 0, 99)),
        # Wrist camera: native 640x480.
        ((480, 640), 389.88812255859375, 313.9997253417969, 207.73726530075075, 125.06938490867614, (256, 341, 0, 42)),
    ],
)
def test_cover_crop_matches_station_delivery(native, fx, cx, expected_fx, expected_cx, box):
    scaled_h, scaled_w, top, left = cover_crop_box(*native, 256, 256)
    assert (scaled_h, scaled_w, top, left) == box
    sx = scaled_w / native[1]
    assert fx * sx == pytest.approx(expected_fx, rel=1e-9)
    assert (cx + 0.5) * sx - 0.5 - left == pytest.approx(expected_cx, rel=1e-9)
    image = np.zeros((*native, 3), dtype=np.uint8)
    image[:, native[1] // 2 :] = 255
    out = _resize_jpeg(image, 256, 256, jpeg_quality=None)
    assert out.shape == (256, 256, 3)


def test_episode_source_ids_require_matching_rating_fingerprints(tmp_path):
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta/info.json").write_text(json.dumps({"total_episodes": 2}))
    episodes = {
        "0": {"subtasks": [{"start": 1, "rating": "ok"}, {"start": 0, "rating": "excellent"}],
              "high_levels": [{"rating": "ok"}]},
        "1": {"subtasks": [{"start": 0, "rating": "excellent"}], "high_levels": [{"rating": "excellent"}]},
    }
    (tmp_path / "meta/pokeandwiggle_annotations.json").write_text(
        json.dumps({"version": 1, "episodes": episodes})
    )
    members = [
        {"id": 20, "rating": "excellent", "subtask_ratings": ["excellent"]},
        {"id": 10, "rating": "ok", "subtask_ratings": ["excellent", "ok"]},
    ]
    assert episode_source_ids(tmp_path, members) == [10, 20]
    members[0]["rating"] = "ok"
    with pytest.raises(ValueError):
        episode_source_ids(tmp_path, members)
