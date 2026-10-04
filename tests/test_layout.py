import numpy as np

from paw_baseline.layout import (
    ACTION_KEY_DIMS,
    PAW_DIM,
    STATE_KEY_DIMS,
    TRAIN_FROM_WIRE,
    WIRE_FROM_TRAIN,
    flat_names,
    pin_frozen_arm,
)


def test_flat_names_cover_all_dims():
    names = flat_names(STATE_KEY_DIMS)
    assert len(names) == PAW_DIM
    assert names[0] == "tcp_pose_left.pos_x"
    assert names[7] == "gripper_width_left.width"
    assert names[14] == "tcp_pose_right.quat_w"
    assert flat_names(ACTION_KEY_DIMS) == names


def test_wire_permutations_are_inverse():
    train = np.arange(PAW_DIM)
    wire = train[list(WIRE_FROM_TRAIN)]
    assert wire[list(TRAIN_FROM_WIRE)].tolist() == train.tolist()


def test_wire_order_puts_grippers_last():
    train = np.arange(PAW_DIM, dtype=float)
    wire = train[list(WIRE_FROM_TRAIN)]
    assert wire[0:7].tolist() == train[0:7].tolist()  # left pose unchanged
    assert wire[7:14].tolist() == train[8:15].tolist()  # right pose shifts forward
    assert wire[14] == train[7]  # left gripper
    assert wire[15] == train[15]  # right gripper


def _observation():
    return {
        "observation.state.tcp_pose_left": np.array([[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]]),
        "observation.state.gripper_width_left": np.array([[0.02]]),
        "observation.state.tcp_pose_right": np.array([[0.4, 0.5, 0.6, 0.0, 0.0, 0.0, 1.0]]),
        "observation.state.gripper_width_right": np.array([[0.07]]),
    }


def test_pin_frozen_left_arm_in_wire_order_without_touching_right():
    wire = np.arange(2 * PAW_DIM, dtype=np.float32).reshape(1, 2, PAW_DIM)
    before = wire.copy()
    pinned = pin_frozen_arm(wire, _observation(), "left")
    expected_pose = np.repeat(_observation()["observation.state.tcp_pose_left"], 2, axis=0)
    np.testing.assert_allclose(pinned[0, :, 0:7], expected_pose)
    np.testing.assert_allclose(pinned[0, :, 14], 0.02)
    np.testing.assert_array_equal(pinned[..., 7:14], before[..., 7:14])
    np.testing.assert_array_equal(pinned[..., 15], before[..., 15])
    np.testing.assert_array_equal(wire, before)


def test_pin_frozen_right_arm_in_wire_order_without_touching_left():
    wire = np.arange(2 * PAW_DIM, dtype=np.float32).reshape(1, 2, PAW_DIM)
    before = wire.copy()
    pinned = pin_frozen_arm(wire, _observation(), "right")
    expected_pose = np.repeat(_observation()["observation.state.tcp_pose_right"], 2, axis=0)
    np.testing.assert_allclose(pinned[0, :, 7:14], expected_pose)
    np.testing.assert_allclose(pinned[0, :, 15], 0.07)
    np.testing.assert_array_equal(pinned[..., 0:7], before[..., 0:7])
    np.testing.assert_array_equal(pinned[..., 14], before[..., 14])


def test_pin_frozen_arm_rejects_unsafe_live_state():
    import pytest

    observation = _observation()
    observation["observation.state.gripper_width_left"] = np.array([[0.2]])
    with pytest.raises(ValueError, match="physical range"):
        pin_frozen_arm(np.zeros((1, 2, PAW_DIM)), observation, "left")
