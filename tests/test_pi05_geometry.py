"""Physical examples of the absolute PI TCP contract, independent of weights."""

import numpy as np
import pytest
from paw_baseline.pi05.geometry import Pi05Representation
from scipy.spatial.transform import Rotation


def physical(rotations=(0, 0)):
    value = np.zeros(16)
    for offset, angle in zip((0, 8), rotations, strict=True):
        value[offset : offset + 3] = [0.3, -0.2, 0.4]
        value[offset + 3 : offset + 7] = Rotation.from_euler("z", angle).as_quat()
        value[offset + 7] = 0.085
    return value


def test_absolute_pose_roundtrip():
    rep = Pi05Representation()
    target = np.stack([physical((0.8, -0.6)), physical((-0.5, 0.7))])
    target[:, [0, 9]] += [[0.02, -0.04], [0.08, 0.03]]
    target[:, [7, 15]] = [[0.04, 0.01], [0.085, 0]]
    np.testing.assert_allclose(rep.decode(rep.encode(target)), target, atol=1e-6)


def test_columns_and_closed_fraction():
    rep = Pi05Representation()
    state = physical((np.pi / 2, 0))
    state[15] = 0
    encoded = rep.encode(state)
    np.testing.assert_allclose(encoded[3:9], [0, 1, 0, -1, 0, 0], atol=1e-6)
    np.testing.assert_allclose(encoded[[9, 19]], [0, 1])


def test_invalid_rotations_fail():
    with pytest.raises(ValueError, match="degenerate"):
        Pi05Representation().decode(np.zeros(20))


def test_representation_contract():
    rep = Pi05Representation()
    assert Pi05Representation.from_dict(rep.to_dict()).to_dict() == rep.to_dict()
    with pytest.raises(ValueError, match="incompatible"):
        Pi05Representation.from_dict({**rep.to_dict(), "layout": "unknown"})


def test_server_emits_absolute_tcp_targets():
    from inference.serve import BaselinePolicyServer
    from paw_baseline.layout import STATE_KEY_DIMS, WIRE_FROM_TRAIN

    rep = Pi05Representation()
    state = physical((0.3, -0.2))
    target = physical((0.4, -0.3))
    target[0] += 0.05
    server = object.__new__(BaselinePolicyServer)
    server._representation, server._frozen_arm, server._cameras = rep, None, ()
    server._task_instruction = "test"
    observation = {
        "actions_discarded": False,
        "chunk_requested": True,
        "capture_time_ns": 123,
    }
    offset = 0
    for key, size in STATE_KEY_DIMS:
        observation[key] = state[None, offset : offset + size].copy()
        offset += size
    server._predict = lambda batch: rep.encode(target)[None, None]
    emitted = []
    server.emit_chunk = lambda value, timestamp, *, discard_actions: emitted.append(
        (value["action"], timestamp)
    )
    server.on_observation(observation)
    np.testing.assert_allclose(
        emitted[0][0][0, 0], target[list(WIRE_FROM_TRAIN)], atol=1e-6
    )
    assert emitted[0][1] == 123
