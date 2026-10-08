"""BIN/DSD increment representations: geometry, round trips, tokens and statistics."""

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from paw_baseline.florence_vla.actions import (
    ChunkCodec,
    compute_action_stats,
    decode_chunk,
    encode_chunk,
    encode_stats,
    increments_to_features,
    matrix_to_rotvec,
    matrix_to_rpy,
    parse_stats,
    rotvec_to_matrix,
    rpy_to_matrix,
)
from paw_baseline.representation import encode_model


def _random_trajectory(rng, episodes=4, length=80):
    """Smooth physical 16D trajectories -> (state, action, episode) in 20D model space."""
    rows_s, rows_a, eps = [], [], []
    for ep in range(episodes):
        for arm_rows in (rows_s,):
            pass
        pos = rng.normal(size=(2, 3)) * 0.1 + np.array([0.4, 0.0, 0.3])
        rot = Rotation.random(2, random_state=rng.integers(1 << 30))
        grip = rng.uniform(0, 0.085, size=2)
        for t in range(length):
            step = rng.normal(size=(2, 3)) * 0.004 * (t % 7 != 0)
            pos = pos + step
            rot = Rotation.from_rotvec(rng.normal(size=(2, 3)) * 0.01) * rot
            grip = np.clip(grip + rng.normal(size=2) * 0.002, 0, 0.085)
            lag_pos = pos - step * 3
            action = []
            state = []
            for arm in range(2):
                q = rot[arm].as_quat()
                action += [*pos[arm], *q, grip[arm]]
                state += [*lag_pos[arm], *(Rotation.from_rotvec(rng.normal(size=3) * 0.005) * rot[arm]).as_quat(), grip[arm]]
            rows_a.append(action)
            rows_s.append(state)
            eps.append(ep)
    return (
        encode_model(np.asarray(rows_s, dtype=np.float32)),
        encode_model(np.asarray(rows_a, dtype=np.float32)),
        np.asarray(eps),
    )


def test_rpy_and_rotvec_round_trip():
    rng = np.random.default_rng(0)
    rotations = Rotation.from_rotvec(rng.normal(size=(64, 3)) * 0.3)
    matrices = torch.from_numpy(rotations.as_matrix())
    rpy = matrix_to_rpy(matrices)
    # R = Rz(yaw) Ry(pitch) Rx(roll) is SciPy's extrinsic "xyz" convention.
    np.testing.assert_allclose(rpy.numpy(), rotations.as_euler("xyz"), atol=1e-9)
    np.testing.assert_allclose(rpy_to_matrix(rpy).numpy(), matrices.numpy(), atol=1e-9)
    rotvec = matrix_to_rotvec(matrices)
    np.testing.assert_allclose(rotvec.numpy(), rotations.as_rotvec(), atol=1e-9)
    np.testing.assert_allclose(rotvec_to_matrix(rotvec).numpy(), matrices.numpy(), atol=1e-9)
    tiny = torch.from_numpy(Rotation.from_rotvec([[1e-9, -2e-9, 3e-9]]).as_matrix())
    np.testing.assert_allclose(matrix_to_rotvec(tiny).numpy(), [[1e-9, -2e-9, 3e-9]], atol=1e-15)
    half_turn = torch.from_numpy(Rotation.from_rotvec([[0.0, -np.pi, 0.0]]).as_matrix())
    np.testing.assert_allclose(matrix_to_rotvec(half_turn).numpy(), [[0.0, np.pi, 0.0]], atol=1e-6)


@pytest.mark.parametrize("representation", ["bin", "dsd"])
def test_features_round_trip_exactly(representation):
    rng = np.random.default_rng(1)
    state, action, _ = _random_trajectory(rng, episodes=1, length=40)
    state_t = torch.from_numpy(state[:1]).double()
    chunk = torch.from_numpy(action[None, :30]).double()
    features = encode_chunk(representation, state_t, chunk)
    assert features.shape == (1, 30, 14 if representation == "bin" else 18)
    decoded = decode_chunk(representation, state_t, features)
    np.testing.assert_allclose(decoded.numpy(), chunk.numpy(), atol=2e-6)


def test_dsd_features_match_definition():
    dp = torch.tensor([[0.003, -0.004, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float64)
    omega = torch.tensor([[0.0, 0.0, 0.02], [0.0, 0.0, 0.0]], dtype=torch.float64)
    features = increments_to_features("dsd", dp, rotvec_to_matrix(omega), torch.tensor([0.05, 0.01]))
    np.testing.assert_allclose(features[0].numpy(), [0.005, 0.02, 0.05, 0.6, -0.8, 0.0, 0.0, 0.0, 1.0], atol=1e-12)
    # Zero scales use the fixed default direction e3.
    np.testing.assert_allclose(features[1].numpy(), [0, 0, 0.01, 0, 0, 1, 0, 0, 1], atol=1e-12)
    # Scaling the motion changes only the scale coordinates (speed invariance).
    faster = increments_to_features("dsd", 2 * dp, rotvec_to_matrix(2 * omega), torch.tensor([0.05, 0.01]))
    np.testing.assert_allclose(faster[0, 3:].numpy(), features[0, 3:].numpy(), atol=1e-12)


@pytest.mark.parametrize("representation", ["bin", "dsd"])
def test_tokenized_round_trip_error_is_bounded(representation):
    rng = np.random.default_rng(2)
    state, action, episodes = _random_trajectory(rng)
    stats = parse_stats(encode_stats(compute_action_stats(state, action, episodes)))
    codec = ChunkCodec(representation, stats, 30, 256, 0.003, 0.1)
    state_t = torch.from_numpy(state[5:6]).double()
    chunk = torch.from_numpy(action[None, 5:35]).double()
    features = encode_chunk(representation, state_t, chunk)
    tokens = codec.tokenize(features)
    assert tokens.min() >= 0 and tokens.max() <= 255
    decoded = decode_chunk(representation, state_t, codec.detokenize(tokens))
    position_error = (decoded[..., [0, 1, 2, 10, 11, 12]] - chunk[..., [0, 1, 2, 10, 11, 12]]).abs().max()
    assert position_error < 0.01  # accumulated quantization over 30 steps (clipping at q01/q99)
    grip_error = (decoded[..., [9, 19]] - chunk[..., [9, 19]]).abs().max()
    assert grip_error < 0.001


@pytest.mark.parametrize("representation", ["bin", "dsd"])
def test_idle_chunk_decodes_without_drift(representation):
    rng = np.random.default_rng(5)
    state, action, episodes = _random_trajectory(rng)
    stats = compute_action_stats(state, action, episodes)
    codec = ChunkCodec(representation, stats, 30, 256, 0.003, 0.1)
    state_t = torch.from_numpy(state[7:8]).double()
    # Command equals measured state and stays still: every increment is zero.
    idle = state_t[:, None].expand(1, 30, 20).clone()
    tokens = codec.tokenize(encode_chunk(representation, state_t, idle))
    decoded = decode_chunk(representation, state_t, codec.detokenize(tokens))
    motion = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    np.testing.assert_allclose(decoded[..., motion].numpy(), idle[..., motion].numpy(), atol=1e-6)
    # Values in other bins still decode to the bin centre.
    centres = codec.detokenize(torch.full((1, 30, codec.dim), 200))
    assert torch.all(centres[..., 0] != 0)


def test_codec_statistics_groups_and_bounds():
    rng = np.random.default_rng(3)
    state, action, episodes = _random_trajectory(rng)
    stats = compute_action_stats(state, action, episodes)
    for representation, width in (("bin", 14), ("dsd", 18)):
        for group in ("anchor", "step"):
            assert len(stats[representation][group]["lo"]) == width
    dsd = stats["dsd"]["step"]
    for base in (0, 9):
        assert dsd["lo"][base] == 0.0 and dsd["lo"][base + 1] == 0.0
        assert dsd["lo"][base + 3 : base + 9] == [-1.0] * 6
        assert dsd["hi"][base + 3 : base + 9] == [1.0] * 6
    # Gripper bounds are shared by BIN and DSD (same encoding).
    assert stats["bin"]["step"]["lo"][6] == stats["dsd"]["step"]["lo"][2]
    assert stats["bin"]["anchor"]["hi"][13] == stats["dsd"]["anchor"]["hi"][11]
    codec = ChunkCodec("bin", stats, 30, 256, 0.003, 0.1)
    assert torch.equal(codec.lo[1], codec.lo[29]) and not torch.equal(codec.lo[0], codec.lo[1])


def test_openvla_bin_bounds_are_narrower_and_selectable():
    rng = np.random.default_rng(6)
    state, action, episodes = _random_trajectory(rng)
    stats = parse_stats(encode_stats(compute_action_stats(state, action, episodes)))
    wide, q01 = stats["bin"], stats["bin_q01"]
    features = encode_chunk("bin", torch.from_numpy(state[:1]).double(), torch.from_numpy(action[None, :30]).double())
    for group in ("anchor", "step"):
        motion = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
        assert all(np.array(q01[group]["hi"])[motion] < np.array(wide[group]["hi"])[motion])
        assert all(np.array(q01[group]["lo"])[motion] > np.array(wide[group]["lo"])[motion])
    codec = ChunkCodec("bin", stats, 30, 256, 0.003, 0.1, stats_key="bin_q01")
    assert codec.lo[1, 0] == q01["step"]["lo"][0] and codec.hi[0, 0] == q01["anchor"]["hi"][0]
    assert codec.tokenize(features).shape == features.shape
    with pytest.raises(ValueError):
        ChunkCodec("dsd", stats, 30, 256, 0.003, 0.1, stats_key="bin_q01")
    # Statistics fitted before the q01 group existed still parse for BIN/DSD.
    legacy = {k: v for k, v in stats.items() if k != "bin_q01"}
    assert "bin_q01" not in parse_stats(encode_stats(legacy))
    ChunkCodec("bin", legacy, 30, 256, 0.003, 0.1)
    with pytest.raises(ValueError):
        ChunkCodec("bin", legacy, 30, 256, 0.003, 0.1, stats_key="bin_q01")


def test_config_selects_bin_statistics():
    from paw_baseline.florence_vla.configuration_florence_vla import FlorenceVLAConfig

    assert FlorenceVLAConfig(action_representation="bin").stats_key == "bin"
    assert FlorenceVLAConfig(action_representation="bin", bin_percentiles="1-99").stats_key == "bin_q01"
    assert FlorenceVLAConfig(action_representation="dsd").stats_key == "dsd"
    with pytest.raises(ValueError):
        FlorenceVLAConfig(action_representation="dsd", bin_percentiles="1-99")
    with pytest.raises(ValueError):
        FlorenceVLAConfig(action_representation="bin", bin_percentiles="5-95")


def test_dsd_direction_weights():
    rng = np.random.default_rng(4)
    state, action, episodes = _random_trajectory(rng)
    stats = compute_action_stats(state, action, episodes)
    codec = ChunkCodec("dsd", stats, 2, 256, 0.003, 0.1)
    features = torch.zeros(1, 2, 18, dtype=torch.float64)
    tau_pos = 0.003 * stats["dsd"]["step"]["hi"][0]
    features[0, 1, 0] = tau_pos / 2  # left translation scale at half the threshold
    features[0, 1, 1] = 1.0  # left rotation scale far above threshold
    weights = codec.loss_weights(features)
    np.testing.assert_allclose(weights[0, 1, 3:6].numpy(), 0.5)
    np.testing.assert_allclose(weights[0, 1, 6:9].numpy(), 1.0)
    np.testing.assert_allclose(weights[0, 1, 12:15].numpy(), 0.1)  # right arm at zero scale
    np.testing.assert_allclose(weights[0, :, [0, 1, 2, 9, 10, 11]].numpy(), 1.0)
    assert torch.all(ChunkCodec("bin", stats, 2, 256, 0.003, 0.1).loss_weights(torch.zeros(1, 2, 14)) == 1)
