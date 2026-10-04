"""Physical rotations remain continuous at the observed SmolVLA sign boundary."""

import numpy as np
import pytest

from paw_baseline.representation import decode_model, encode_model


def _physical(quats):
    state = np.zeros((*quats.shape[:-1], 16), dtype=np.float32)
    state[..., 3:7] = quats
    state[..., 11:15] = quats
    state[..., 7] = 0.02
    state[..., 15] = 0.08
    return state


def test_observed_boundary_is_continuous_and_sign_invariant():
    q = np.array([0.7014793754, -0.6943339109, -0.1272907257, 0.09810288996])
    mid = (abs(q[0]) + abs(q[1])) / 2
    quats = np.stack([q, q])
    for row, delta in zip(quats, (-1e-4, 1e-4), strict=True):
        row[0], row[1] = mid + delta, -(mid - delta)
    quats /= np.linalg.norm(quats, axis=-1, keepdims=True)
    encoded = encode_model(_physical(quats))
    assert encoded.shape == (2, 20)
    assert np.linalg.norm(encoded[0] - encoded[1]) < 0.002
    np.testing.assert_array_equal(encoded, encode_model(_physical(-quats)))


def test_random_round_trip_preserves_rotation_position_and_gripper():
    rng = np.random.default_rng(2627)
    quats = rng.normal(size=(2, 50, 4))
    quats /= np.linalg.norm(quats, axis=-1, keepdims=True)
    physical = _physical(quats)
    physical[..., :3] = rng.normal(size=(2, 50, 3))
    physical[..., 8:11] = rng.normal(size=(2, 50, 3))
    actual = decode_model(encode_model(physical))
    np.testing.assert_allclose(
        actual[..., [0, 1, 2, 7, 8, 9, 10, 15]],
        physical[..., [0, 1, 2, 7, 8, 9, 10, 15]],
    )
    for sl in (slice(3, 7), slice(11, 15)):
        np.testing.assert_allclose(
            abs(np.sum(actual[..., sl] * physical[..., sl], axis=-1)), 1, atol=2e-7
        )


def test_rotation_6d_uses_matrix_rows():
    # +90 degrees about z: rows (0, -1, 0), (1, 0, 0), not columns.
    q = np.array([0, 0, 2**-0.5, 2**-0.5])
    np.testing.assert_allclose(
        encode_model(_physical(q))[3:9], [0, -1, 0, 1, 0, 0], atol=1e-7
    )


@pytest.mark.parametrize(
    "rows", [[0] * 6, [1, 0, 0, 2, 0, 0], [float("nan"), 0, 0, 0, 1, 0]]
)
def test_degenerate_prediction_fails_without_fallback(rows):
    value = encode_model(_physical(np.array([0, 0, 0, 1.0])))
    value[3:9] = rows
    with pytest.raises(ValueError):
        decode_model(value)
