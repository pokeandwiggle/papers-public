"""Physical state/action layout.

The flat 16-dim *training* order mirrors the platform's ``policy_layout``
convention: ``[tcp_pose_left(7), gripper_width_left(1), tcp_pose_right(7),
gripper_width_right(1)]`` with poses as ``pos_xyz + quat_xyzw`` (w last) and
gripper widths in metres.

The policy-toolkit *wire* order for the same 16 quantities puts both grippers
last: ``[tcp_pose_left(7), tcp_pose_right(7), gripper_width_left(1),
gripper_width_right(1)]`` (see ``DefaultActionTranslator.from_dict`` and the
flat ``observation.state`` built by ``DefaultObservationTranslator``). Serve
permutes model output train→wire before emitting; validate permutes received
chunks wire→train before comparing against dataset actions.
"""

# Sub-feature keys in canonical concatenation order; order is load-bearing.
STATE_KEY_DIMS: tuple[tuple[str, int], ...] = (
    ("observation.state.tcp_pose_left", 7),
    ("observation.state.gripper_width_left", 1),
    ("observation.state.tcp_pose_right", 7),
    ("observation.state.gripper_width_right", 1),
)
ACTION_KEY_DIMS: tuple[tuple[str, int], ...] = tuple(
    (key.replace("observation.state.", "action."), dim) for key, dim in STATE_KEY_DIMS
)

PAW_DIM = 16
POSE_NAMES = ("pos_x", "pos_y", "pos_z", "quat_x", "quat_y", "quat_z", "quat_w")

CAMERA_KEYS: tuple[str, ...] = (
    "observation.images.rgb_static",
    "observation.images.rgb_left",
    "observation.images.rgb_right",
)

GRIPPER_MAX_WIDTH_M = 0.085
FPS = 30

# Component locations within a flat-16 train-order vector.
TRAIN_GRIP_DIMS: tuple[int, ...] = (7, 15)
TRAIN_QUAT_SLICES: tuple[slice, ...] = (slice(3, 7), slice(11, 15))

# Index permutations between the two orders: wire[i] = train[WIRE_FROM_TRAIN[i]]
# and train[i] = wire[TRAIN_FROM_WIRE[i]].
WIRE_FROM_TRAIN: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 7, 15)
TRAIN_FROM_WIRE: tuple[int, ...] = tuple(WIRE_FROM_TRAIN.index(i) for i in range(PAW_DIM))

# Component locations within a toolkit wire-order vector. Unlike training
# order, both poses come first and both grippers come last.
WIRE_ARM_DIMS: dict[str, tuple[int, ...]] = {
    "left": (*range(0, 7), 14),
    "right": (*range(7, 14), 15),
}


def pin_frozen_arm(chunk, observation: dict, arm: str):
    """Pin one wire-order arm, including its gripper, to live observed state.

    The wire contract always carries both arms. A frozen side is therefore a
    hold-position command, not an omitted command. The copy also prevents this
    serving safety operation from mutating a caller-owned model output.
    """
    import numpy as np

    if arm not in WIRE_ARM_DIMS:
        raise ValueError(f"frozen_arm must be 'left' or 'right', got {arm!r}")
    arm_dims = WIRE_ARM_DIMS[arm]
    pinned = np.asarray(chunk, dtype=np.float32).copy()
    if pinned.ndim != 3 or pinned.shape[0] != 1 or pinned.shape[-1] != PAW_DIM:
        raise ValueError(f"wire chunk must have shape (1, T, {PAW_DIM}), got {pinned.shape}")

    pose_key = f"observation.state.tcp_pose_{arm}"
    grip_key = f"observation.state.gripper_width_{arm}"
    pose = np.asarray(observation[pose_key], dtype=np.float32).reshape(-1)
    grip = np.asarray(observation[grip_key], dtype=np.float32).reshape(-1)
    if pose.shape != (7,) or grip.shape != (1,):
        raise ValueError(
            f"live {arm} state must be pose (7,) and gripper (1,), got {pose.shape} / {grip.shape}"
        )
    if not np.isfinite(pose).all() or not np.isfinite(grip).all():
        raise ValueError(f"live {arm} state contains non-finite values")
    quat_norm = float(np.linalg.norm(pose[3:7]))
    if abs(quat_norm - 1.0) > 0.01:
        raise ValueError(f"live {arm} pose has non-unit quaternion (norm {quat_norm:.3f})")
    if not 0.0 <= float(grip[0]) <= GRIPPER_MAX_WIDTH_M:
        raise ValueError(f"live {arm} gripper width {float(grip[0]):.6f} is outside physical range")

    pinned[..., list(arm_dims[:7])] = pose
    pinned[..., arm_dims[7]] = grip[0]
    return pinned


def flat_names(key_dims: tuple[tuple[str, int], ...]) -> list[str]:
    """Per-dimension names for a folded vector, e.g. ``tcp_pose_left.pos_x``."""
    names: list[str] = []
    for key, dim in key_dims:
        short = key.rsplit(".", 1)[-1] if key.startswith("action.") else key.split(".", 2)[-1]
        subnames = POSE_NAMES if dim == 7 else ("width",)
        names.extend(f"{short}.{sub}" for sub in subnames)
    return names
