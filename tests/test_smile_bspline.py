"""SMILE B-spline basis, masked least-squares targets and coefficient statistics."""

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from paw_baseline.relative import (
    DELTA_DIM,
    cumulative_trajectory,
    deltas_from_cumulative,
)
from paw_baseline.smile import trajectory
from paw_baseline.smile.bspline import (
    GRIPPER_DIMS,
    ROTATION_DIMS,
    TRANSLATION_DIMS,
    bspline_basis,
    clamped_uniform_knots,
    coefficient_stats,
    masked_projectors,
    projection_matrix,
)


def test_modalities_follow_relative_action_layout():
    assert TRANSLATION_DIMS == (0, 1, 2, 7, 8, 9)
    assert ROTATION_DIMS == (3, 4, 5, 10, 11, 12)
    assert GRIPPER_DIMS == (6, 13)


@pytest.mark.parametrize("length,k,p", [(50, 8, 3), (32, 8, 3), (50, 6, 3), (50, 8, 5)])
def test_basis_matches_scipy_and_partitions_unity(length, k, p):
    from scipy.interpolate import BSpline

    basis = bspline_basis(length, k, p)
    knots = clamped_uniform_knots(k, p)
    times = np.linspace(0.0, 1.0, length)
    reference = np.stack(
        [BSpline(knots, np.eye(k)[i], p, extrapolate=False)(times) for i in range(k)],
        axis=1,
    )
    # SciPy's half-open convention leaves the right endpoint undefined.
    reference[-1] = np.eye(k)[-1]
    np.testing.assert_allclose(basis, np.nan_to_num(reference), atol=1e-12)
    np.testing.assert_allclose(basis.sum(axis=1), 1.0, atol=1e-12)
    assert np.all(basis >= -1e-12)
    np.testing.assert_allclose(basis[0], np.eye(k)[0])
    np.testing.assert_allclose(basis[-1], np.eye(k)[-1])


def test_projection_recovers_spline_and_suppresses_jitter():
    rng = np.random.default_rng(0)
    basis = bspline_basis(50, 8, 3)
    projector = projection_matrix(basis)
    coefs = rng.normal(size=(8, DELTA_DIM))
    np.testing.assert_allclose(projector @ (basis @ coefs), coefs, atol=1e-9)
    smoother = basis @ projector
    t = np.linspace(0, 1, 50)
    smooth = np.sin(np.pi * t)
    jitter = np.where(np.arange(50) % 2 == 0, 1.0, -1.0) * 0.01
    np.testing.assert_allclose(smoother @ smooth, smooth, atol=2e-3)
    rms = np.sqrt(np.mean(np.square(smoother @ jitter)))
    assert rms < 0.15 * np.sqrt(np.mean(np.square(jitter)))


def test_masked_projectors_use_only_valid_steps():
    rng = np.random.default_rng(1)
    basis = bspline_basis(50, 8, 3)
    projectors, observability, rank = masked_projectors(basis)
    np.testing.assert_allclose(projectors[50], projection_matrix(basis), atol=1e-10)
    np.testing.assert_allclose(observability[50], np.eye(8), atol=1e-10)
    assert rank[50] == 8 and rank[1] == 1
    assert np.all(np.diff(rank[1:]) >= 0)
    for v in (1, 10, 23, 37, 44, 49):
        # Padded steps never influence the target.
        assert np.all(projectors[v][:, v:] == 0)
        g = observability[v]
        np.testing.assert_allclose(g @ g, g, atol=1e-10)
        np.testing.assert_allclose(g, g.T, atol=1e-10)
        assert round(np.trace(g)) == rank[v]
        # A spline observed on v steps is recovered on its observed directions,
        # and its fit reproduces those steps whenever they determine it.
        coefs = rng.normal(size=(8, 3))
        y = basis @ coefs
        y[v:] = 1e6  # padding garbage
        fitted = projectors[v] @ y
        np.testing.assert_allclose(fitted, g @ coefs, atol=1e-6)
        np.testing.assert_allclose(basis[:v] @ fitted, basis[:v] @ g @ coefs, atol=1e-6)


def test_torch_trajectories_match_numpy():
    rng = np.random.default_rng(2)
    deltas = rng.normal(scale=0.01, size=(4, 50, DELTA_DIM))
    deltas[..., [3, 4, 5, 10, 11, 12]] = rng.normal(scale=0.05, size=(4, 50, 6))
    cumulative = cumulative_trajectory(deltas)
    torch_cumulative = trajectory.cumulative_trajectory(torch.as_tensor(deltas)).numpy()
    np.testing.assert_allclose(torch_cumulative, cumulative, atol=1e-12)
    torch_deltas = trajectory.deltas_from_cumulative(torch.as_tensor(cumulative)).numpy()
    np.testing.assert_allclose(torch_deltas, deltas_from_cumulative(cumulative), atol=1e-12)
    np.testing.assert_allclose(torch_deltas, deltas, atol=1e-9)


def _write_prepared(root, episodes):
    (root / "data/chunk-000").mkdir(parents=True)
    (root / "meta").mkdir()
    rows, idx, frames = [], [], []
    for episode, actions in episodes.items():
        rows.append(actions)
        idx += [episode] * len(actions)
        frames += list(range(len(actions)))
    actions = np.concatenate(rows).astype(np.float32)
    flat = pa.array(actions.ravel(), type=pa.float32())
    pq.write_table(
        pa.table(
            {
                "action": pa.FixedSizeListArray.from_arrays(flat, DELTA_DIM),
                "episode_index": pa.array(idx, type=pa.int64()),
                "frame_index": pa.array(frames, type=pa.int64()),
            }
        ),
        root / "data/chunk-000/file-000.parquet",
    )
    stats = {"mean": actions.mean(axis=0).tolist(), "std": actions.std(axis=0).tolist()}
    (root / "meta/stats.json").write_text(json.dumps({"action": stats}))


def test_coefficient_stats_use_eligible_cumulative_windows_for_all_modalities(tmp_path):
    rng = np.random.default_rng(3)
    episodes = {
        0: rng.normal(scale=0.01, size=(60, DELTA_DIM)),
        1: rng.normal(scale=0.01, size=(55, DELTA_DIM)),
        2: rng.normal(scale=0.01, size=(70, DELTA_DIM)),
        3: rng.normal(scale=0.01, size=(30, DELTA_DIM)),  # shorter than F
    }
    for actions in episodes.values():
        actions[:, [6, 13]] = rng.choice([0.0, 0.085], size=(len(actions), 2))
    episodes[0][:, 13] = 0.04  # a constant gripper keeps a unit scale
    _write_prepared(tmp_path, episodes)
    stats = coefficient_stats(
        tmp_path, [0, 1, 3], chunk_size=50, num_control_points=8, degree=3
    )
    # Only starts whose full F-step window lies inside the episode are eligible.
    assert stats["windows"] == (60 - 49) + (55 - 49)
    projector = projection_matrix(bspline_basis(50, 8, 3))
    coefs = []
    for episode in (0, 1):
        actions = episodes[episode].astype(np.float32).astype(np.float64)
        for start in range(len(actions) - 49):
            coefs.append(projector @ cumulative_trajectory(actions[start : start + 50]))
    coefs = np.stack(coefs)
    got_mean = np.asarray(stats["coef_mean"]).reshape(8, DELTA_DIM)
    got_std = np.asarray(stats["coef_std"]).reshape(8, DELTA_DIM)
    np.testing.assert_allclose(got_mean, coefs.mean(axis=0), atol=1e-9)
    expected_std = coefs.std(axis=0)
    varying = expected_std > 1e-8
    np.testing.assert_allclose(got_std[varying], expected_std[varying], rtol=1e-6)
    assert np.all(got_std[~varying] == 1.0)
    # Rotation coefficients are normalized by their own dataset statistics.
    rot = list(ROTATION_DIMS)
    assert not np.allclose(got_std[:, rot], 1.0)
    assert not np.allclose(got_mean[:, rot], 0.0)
