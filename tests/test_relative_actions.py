"""Relative TCP actions: SO(3) maps, per-frame deltas, integration, Eq. (5)-(6)."""

import numpy as np
import pytest
from paw_baseline.relative import (
    DELTA_DIM,
    cumulative_trajectory,
    deltas_from_cumulative,
    encode_deltas,
    integrate_deltas,
    matrix_to_rotvec,
    quat_multiply,
    quat_to_rotvec,
    rotvec_to_matrix,
    rotvec_to_quat,
)
from paw_baseline.rotation_6d import quat_xyzw_to_matrix
from scipy.spatial.transform import Rotation


def _random_rotvecs(rng, n, max_angle):
    axis = rng.normal(size=(n, 3))
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    return axis * rng.uniform(0, max_angle, size=(n, 1))


def test_so3_maps_match_scipy_including_small_and_large_angles():
    rng = np.random.default_rng(0)
    rotvecs = np.concatenate(
        [
            _random_rotvecs(rng, 200, np.pi - 1e-3),
            _random_rotvecs(rng, 50, 1e-7),
            np.zeros((1, 3)),
        ]
    )
    np.testing.assert_allclose(
        rotvec_to_matrix(rotvecs), Rotation.from_rotvec(rotvecs).as_matrix(), atol=1e-12
    )
    np.testing.assert_allclose(
        matrix_to_rotvec(rotvec_to_matrix(rotvecs)), rotvecs, atol=1e-9
    )
    quats = rotvec_to_quat(rotvecs)
    np.testing.assert_allclose(
        np.abs(np.sum(quats * Rotation.from_rotvec(rotvecs).as_quat(), axis=1)), 1.0, atol=1e-12
    )
    np.testing.assert_allclose(quat_to_rotvec(quats), rotvecs, atol=1e-9)
    np.testing.assert_allclose(quat_to_rotvec(-quats), rotvecs, atol=1e-9)
    a, b = Rotation.random(20, random_state=1), Rotation.random(20, random_state=2)
    np.testing.assert_allclose(
        quat_xyzw_to_matrix(quat_multiply(a.as_quat(), b.as_quat())),
        (a * b).as_matrix(),
        atol=1e-12,
    )


def _random_episode(rng, length):
    """Smooth absolute command and lagging measured trajectories (train order)."""
    action = np.zeros((length, 16))
    state = np.zeros((length, 16))
    for pos, quat, grip in ((slice(0, 3), slice(3, 7), 7), (slice(8, 11), slice(11, 15), 15)):
        path = np.cumsum(rng.normal(scale=0.004, size=(length, 3)), axis=0)
        rot = Rotation.from_rotvec(np.cumsum(rng.normal(scale=0.03, size=(length, 3)), axis=0))
        quats = rot.as_quat()
        quats *= np.where(rng.random((length, 1)) < 0.5, -1.0, 1.0)  # sign noise
        action[:, pos], action[:, quat] = path, quats
        action[:, grip] = np.where(np.arange(length) < length // 2, 0.085, 0.0)
        lag = Rotation.from_rotvec(rng.normal(scale=0.05, size=3))
        state[:, pos] = path - 0.05
        state[:, quat] = (rot * lag).as_quat()
        state[:, grip] = action[:, grip]
    return state, action


def test_deltas_reference_previous_command_and_integrate_exactly():
    rng = np.random.default_rng(3)
    state, action = _random_episode(rng, 120)
    deltas = encode_deltas(state, action)
    assert deltas.shape == (120, DELTA_DIM)
    # Frame 0 is relative to the measured pose, later frames to the last command.
    np.testing.assert_allclose(deltas[0, 0:3], action[0, 0:3] - state[0, 0:3], atol=1e-12)
    np.testing.assert_allclose(deltas[5, 0:3], action[5, 0:3] - action[4, 0:3], atol=1e-12)
    expected = (
        Rotation.from_quat(action[4, 3:7]).inv() * Rotation.from_quat(action[5, 3:7])
    ).as_rotvec()
    np.testing.assert_allclose(deltas[5, 3:6], expected, atol=1e-10)
    np.testing.assert_allclose(deltas[:, 6], action[:, 7])
    # Integrating from the episode's first measured pose recovers every command.
    absolute = integrate_deltas(state[0], deltas)
    np.testing.assert_allclose(absolute[:, [0, 1, 2, 8, 9, 10]], action[:, [0, 1, 2, 8, 9, 10]], atol=1e-9)
    for quat in (slice(3, 7), slice(11, 15)):
        dots = np.abs(np.sum(absolute[:, quat] * action[:, quat], axis=1))
        np.testing.assert_allclose(dots, 1.0, atol=1e-9)
    # A chunk anchored mid-episode integrates from the previous command.
    chunk = integrate_deltas(action[39], deltas[40:90])
    np.testing.assert_allclose(chunk[:, 8:11], action[40:90, 8:11], atol=1e-9)


@pytest.mark.parametrize("max_angle", [0.02, 0.3])
def test_cumulative_trajectory_inverts_through_eq6(max_angle):
    rng = np.random.default_rng(4)
    deltas = rng.normal(scale=0.01, size=(7, 50, DELTA_DIM))
    for drot in (slice(3, 6), slice(10, 13)):
        deltas[..., drot] = _random_rotvecs(rng, 7 * 50, max_angle).reshape(7, 50, 3)
    cumulative = cumulative_trajectory(deltas)
    np.testing.assert_allclose(cumulative[..., 0:3], np.cumsum(deltas[..., 0:3], axis=1), atol=1e-12)
    np.testing.assert_allclose(cumulative[..., 6], deltas[..., 6])
    # exp(R(t)) is the composed rotation of the first t increments.
    composed = Rotation.identity()
    for t in range(50):
        composed = composed * Rotation.from_rotvec(deltas[0, t, 3:6])
        np.testing.assert_allclose(
            Rotation.from_rotvec(cumulative[0, t, 3:6]).as_matrix(), composed.as_matrix(), atol=1e-9
        )
    np.testing.assert_allclose(deltas_from_cumulative(cumulative), deltas, atol=1e-9)
    # Cumulative rotation equals the command relative to the chunk reference.
    reference = np.zeros(16)
    reference[6] = reference[14] = 1.0
    absolute = integrate_deltas(reference, deltas[0])
    np.testing.assert_allclose(
        Rotation.from_quat(absolute[:, 3:7]).as_rotvec(), cumulative[0, :, 3:6], atol=1e-9
    )
