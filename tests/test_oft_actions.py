import numpy as np
import torch
from scipy.spatial.transform import Rotation

from paw_baseline.oft.actions import (
    GRIPPER_DIMS,
    BoundsNormalizer,
    decode_deltas,
    encode_deltas,
    so3_exp,
    so3_log,
)
from paw_baseline.oft.prepare import chunk_deltas, fit_stats


def _random_model_poses(rng, shape):
    """Random absolute 20D model vectors (xyz, rotation rows, gripper) per arm."""
    out = np.zeros((*shape, 20))
    for offset in (0, 10):
        out[..., offset : offset + 3] = rng.normal(0.4, 0.2, (*shape, 3))
        matrix = Rotation.random(int(np.prod(shape)), random_state=rng).as_matrix()
        out[..., offset + 3 : offset + 9] = matrix[:, :2, :].reshape(*shape, 6)
        out[..., offset + 9] = rng.choice([0.0, 0.085], size=shape)
    return out


def test_so3_maps_match_scipy():
    rng = np.random.default_rng(0)
    rotvec = rng.normal(0, 1, (64, 3))
    rotvec = np.concatenate([rotvec, rng.normal(0, 1e-9, (4, 3))])
    expected = Rotation.from_rotvec(rotvec).as_matrix()
    matrix = so3_exp(torch.from_numpy(rotvec))
    np.testing.assert_allclose(matrix.numpy(), expected, atol=1e-10)
    np.testing.assert_allclose(so3_log(matrix).numpy(), rotvec, atol=1e-8)


def test_deltas_round_trip_and_use_live_state_as_first_anchor():
    rng = np.random.default_rng(1)
    state = torch.from_numpy(_random_model_poses(rng, (5,)))
    actions = torch.from_numpy(_random_model_poses(rng, (5, 7)))
    deltas = encode_deltas(state, actions)
    assert deltas.shape == (5, 7, 14)
    np.testing.assert_allclose(
        deltas[:, 0, :3].numpy(), (actions[:, 0, :3] - state[:, :3]).numpy(), atol=1e-12
    )
    first = Rotation.from_matrix(np.stack([
        actions[:, 0, 3:6].numpy(), actions[:, 0, 6:9].numpy(),
        np.cross(actions[:, 0, 3:6].numpy(), actions[:, 0, 6:9].numpy())], axis=1))
    live = Rotation.from_matrix(np.stack([
        state[:, 3:6].numpy(), state[:, 6:9].numpy(),
        np.cross(state[:, 3:6].numpy(), state[:, 6:9].numpy())], axis=1))
    np.testing.assert_allclose(deltas[:, 0, 3:6].numpy(), (first * live.inv()).as_rotvec(), atol=1e-8)
    widths = torch.tensor([0.085, 0.085]), torch.tensor([0.0, 0.0])
    restored = decode_deltas(state, deltas, *widths)
    np.testing.assert_allclose(restored.numpy(), actions.numpy(), atol=1e-9)
    assert set(deltas[..., list(GRIPPER_DIMS)].unique().tolist()) <= {-1.0, 1.0}


def test_bounds_normalizer_clips_masks_constant_and_raw_dims():
    norm = BoundsNormalizer([[-1.0, 0.0, 2.0]], [[1.0, 0.0, 4.0]], raw_dims=[2])
    values = torch.tensor([[[0.5, 3.0, 7.0], [5.0, 0.0, -1.0]]])
    out = norm.normalize(values)
    torch.testing.assert_close(out, torch.tensor([[[0.5, 0.0, 7.0], [1.0, 0.0, -1.0]]]))
    back = norm.unnormalize(torch.tensor([[[0.5, 0.0, 1.0]]]))
    torch.testing.assert_close(back, torch.tensor([[[0.5, 0.0, 1.0]]]), atol=1e-6, rtol=0)


def test_stats_pad_episode_ends_with_last_frame():
    rng = np.random.default_rng(2)
    state = _random_model_poses(rng, (6,))
    action = _random_model_poses(rng, (6,))
    episodes = np.array([0, 0, 0, 1, 1, 1])
    deltas = chunk_deltas(state, action, episodes, [0, 1], horizon=4)
    assert deltas.shape == (6, 4, 14)
    # The last anchor of each episode repeats its final action: zero motion after step 0.
    np.testing.assert_allclose(deltas[2, 1:, [0, 1, 2, 3, 4, 5]], 0.0, atol=1e-12)
    stats = fit_stats(state, action, episodes, [0], horizon=4)
    assert np.asarray(stats["action_q01"]).shape == (4, 14)
    assert stats["gripper_open_width"][0] in (0.085, float(np.median(action[:3, 9][action[:3, 9] > 0.04])))
