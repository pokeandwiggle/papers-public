"""CPU checks for the OFT delta representation and the ActionMap head."""

import math

import numpy as np
import pytest
import torch

from paw_baseline.oft.actions import (
    DeltaNormalizer,
    decode_chunk,
    encode_chunk,
    euler_xyz_to_matrix,
    matrix_to_euler_xyz,
)
from paw_baseline.oft.heads import HeatmapActionHead, L1RegressionActionHead, make_grid
from paw_baseline.representation import encode_model


def _random_quat(rng, n):
    q = rng.normal(size=(n, 4))
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def _trajectory(rng, batch, horizon):
    """Random smooth absolute 16D trajectories, encoded to the 20D model layout."""
    from scipy.spatial.transform import Rotation as R

    states, actions = [], []
    for _ in range(batch):
        frames = []
        base = [R.from_quat(_random_quat(rng, 1)[0]) for _ in range(2)]
        pos = rng.uniform(-0.5, 0.5, size=(2, 3))
        for t in range(horizon + 1):
            row = []
            for arm in range(2):
                rot = R.from_rotvec(rng.normal(scale=0.02, size=3) * t) * base[arm]
                row += list(pos[arm] + 0.01 * t * rng.normal(size=3)) + list(rot.as_quat())
                row += [0.085 if rng.random() > 0.5 else 0.0]
            frames.append(row)
        frames = np.asarray(frames, dtype=np.float64)
        states.append(encode_model(frames[0]))
        actions.append(encode_model(frames[1:]))
    return torch.tensor(np.stack(states)), torch.tensor(np.stack(actions))


def test_euler_round_trip():
    euler = torch.tensor([[0.3, -0.7, 2.5], [-1.0, 0.2, -3.0]])
    assert torch.allclose(matrix_to_euler_xyz(euler_xyz_to_matrix(euler)), euler, atol=1e-5)


def test_euler_matches_scipy_extrinsic_xyz():
    from scipy.spatial.transform import Rotation as R

    euler = np.array([0.3, -0.4, 1.2])
    expected = R.from_euler("xyz", euler).as_matrix()
    actual = euler_xyz_to_matrix(torch.tensor(euler)).numpy()
    assert np.allclose(actual, expected, atol=1e-6)


def test_delta_round_trip_reconstructs_absolute_chunk():
    rng = np.random.default_rng(0)
    state, actions = _trajectory(rng, batch=3, horizon=32)
    deltas = encode_chunk(state, actions)
    assert deltas.shape == (3, 32, 14)
    decoded = decode_chunk(state, deltas)
    for o in (0, 10):
        assert torch.allclose(decoded[..., o : o + 9], actions[..., o : o + 9].float(), atol=1e-4)
        grip = actions[..., o + 9].float()
        assert torch.equal(decoded[..., o + 9], torch.where(grip > 0.0425, 0.085, 0.0))


def test_normalizer_round_trip_and_slot_zero_bounds():
    stats = {
        "first": {"q01": [-0.2] * 12, "q99": [0.2] * 12},
        "step": {"q01": [-0.01] * 12, "q99": [0.01] * 12},
    }
    norm = DeltaNormalizer(stats)
    deltas = torch.zeros(2, 4, 14)
    deltas[:, 0, 0] = 0.1
    deltas[:, 1, 0] = 0.005
    deltas[:, :, 6] = 1.0
    normalized = norm.normalize(deltas)
    assert torch.allclose(normalized[:, 0, 0], torch.tensor(0.5))
    assert torch.allclose(normalized[:, 1, 0], torch.tensor(0.5))
    assert torch.equal(normalized[..., 6], deltas[..., 6])
    assert torch.allclose(norm.unnormalize(normalized), deltas, atol=1e-6)


def _reference_head_class():
    import importlib.util
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[2]
        / "tmp/references/ActionMap/actionmap_oft/actionmap/heatmap_action_head.py"
    )
    if not path.exists():
        pytest.skip("ActionMap reference checkout is absent")
    spec = importlib.util.spec_from_file_location("actionmap_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.HeatmapActionHead


@pytest.mark.parametrize("decode_mode,top_k", [("topk", 10), ("soft_argmax", 0)])
def test_heatmap_head_matches_reference_single_arm(decode_mode, top_k):
    Reference = _reference_head_class()
    torch.manual_seed(0)
    dim, chunk, grid, rot_grid = 8, 3, (6, 5, 4), (4, 4, 3)
    reference = Reference(
        input_dim=dim, hidden_dim=16, action_dim=7, num_actions_chunk=chunk,
        trans_grid=grid, rot_grid=rot_grid, trans_sigma=0.1, rot_sigma=0.1,
        decode_mode=decode_mode, decode_top_k=top_k, grid_pad=0.2,
    )
    ours = HeatmapActionHead(
        dim, 16, 7, chunk, trans_grid=grid, rot_grid=rot_grid, trans_sigma=0.1,
        rot_sigma=0.1, grid_pad=0.2, decode_mode=decode_mode, decode_top_k=top_k,
    )
    ours.trunk.load_state_dict(reference.trunk.state_dict())
    ours.trans_heads[0].load_state_dict(reference.trans_head.state_dict())
    ours.rot_heads[0].load_state_dict(reference.rot_head.state_dict())
    ours.grip_heads[0].load_state_dict(reference.grip_head.state_dict())
    # The reference pads only translation; compare on its rotation grid convention.
    ours.rot_centers = make_grid(rot_grid, 0.0)
    hidden = torch.randn(2, chunk * 7, dim)
    gt = torch.rand(2, chunk, 7) * 2 - 1
    # Soft-label targets and cross-entropy on the (padded) translation grid.
    trans_logits = reference.trans_head(reference._forward_trunk(hidden))
    ref_trans = Reference._gaussian_target(gt[..., :3], reference.trans_centers, 0.1)
    ref_trans_ce = -(ref_trans * torch.log_softmax(trans_logits, -1)).sum(-1).mean()
    our_trans = HeatmapActionHead.gaussian_log_target(gt[..., :3], grid, 0.2, 0.1)
    our_trans_ce = HeatmapActionHead.soft_label_ce(trans_logits, our_trans)
    assert torch.allclose(our_trans.exp(), ref_trans, atol=1e-6)
    assert torch.allclose(our_trans_ce, ref_trans_ce, atol=1e-5)
    # Rotation targets on the reference's unpadded grid.
    ref_rot = Reference._gaussian_target(gt[..., 3:6], reference.rot_centers, 0.1)
    our_rot = HeatmapActionHead.gaussian_log_target(gt[..., 3:6], rot_grid, 0.0, 0.1)
    assert torch.allclose(our_rot.exp(), ref_rot, atol=1e-6)
    with torch.no_grad():
        ref = reference.predict_action(hidden)
        mine = ours.predict_action(hidden)
    assert torch.allclose(mine[..., :6], ref[..., :6], atol=1e-5)
    assert torch.equal(mine[..., 6], (ref[..., 6] > 0).float())


def test_two_arm_heads_shapes_and_loss():
    torch.manual_seed(0)
    hidden = torch.randn(2, 4 * 14, 8)
    gt = torch.rand(2, 4, 14) * 2 - 1
    gt[..., [6, 13]] = (gt[..., [6, 13]] > 0).float()
    l1 = L1RegressionActionHead(8, 16, 14, 4)
    pred, loss = l1.predict_action_with_loss(hidden, gt)
    assert pred.shape == (2, 4, 14) and loss.ndim == 0
    heat = HeatmapActionHead(8, 16, 14, 4, trans_grid=(6, 6, 6), rot_grid=(4, 4, 4))
    pred, loss = heat.predict_action_with_loss(hidden, gt)
    assert pred.shape == (2, 4, 14) and math.isfinite(loss.item())
    loss.backward()
    assert heat.trans_heads[1][2].weight.grad is not None
    out = heat.predict_action(hidden)
    assert out.shape == (2, 4, 14)
    assert set(out[..., [6, 13]].unique().tolist()) <= {0.0, 1.0}
    assert out[..., :6].abs().max() <= 1.2 + 1e-6


def test_gaussian_target_peaks_at_truth():
    grid = (11, 11, 11)
    truth = torch.tensor([[[0.4, -0.6, 1.0]]])
    target = HeatmapActionHead.gaussian_log_target(truth, grid, 0.2, 0.1).exp()
    centers = make_grid(grid, 0.2)
    peak = centers[target[0, 0].argmax()]
    assert torch.allclose(target.sum(-1), torch.ones(1, 1))
    assert torch.linalg.norm(peak - truth[0, 0]) < 0.25
    expected = target[0, 0] @ centers
    assert torch.linalg.norm(expected - truth[0, 0]) < 0.05


def test_encode_ignores_bf16_autocast():
    rng = np.random.default_rng(1)
    state, actions = _trajectory(rng, batch=2, horizon=8)
    exact = encode_chunk(state, actions)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        under_autocast = encode_chunk(state, actions)
    assert under_autocast.dtype == torch.float32
    assert torch.allclose(exact, under_autocast, atol=1e-6)


def test_constant_dimensions_normalize_to_zero():
    from paw_baseline.oft.actions import ProprioNormalizer

    q01 = [0.0] * 20
    q99 = [1.0] * 10 + [0.3] * 10  # right arm frozen at 0.3
    q01 = q01[:10] + [0.3] * 10
    norm = ProprioNormalizer({"q01": q01, "q99": q99})
    state = torch.full((1, 20), 0.3)
    state[0, 10:] += 1e-4  # live measurement noise on the frozen arm
    out = norm(state)
    assert torch.equal(out[0, 10:], torch.zeros(10))
    stats = {
        "first": {"q01": [0.0] * 12, "q99": [0.0] * 6 + [0.1] * 6},
        "step": {"q01": [0.0] * 12, "q99": [0.0] * 6 + [0.1] * 6},
    }
    delta_norm = DeltaNormalizer(stats)
    deltas = torch.zeros(1, 3, 14)
    deltas[..., :6] = 1e-5
    normalized = delta_norm.normalize(deltas)
    assert torch.equal(normalized[..., :6], torch.zeros(1, 3, 6))
    assert torch.allclose(delta_norm.unnormalize(normalized)[..., :6], torch.zeros(1, 3, 6), atol=1e-7)
