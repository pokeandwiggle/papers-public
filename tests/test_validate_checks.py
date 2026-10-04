import numpy as np
import pytest
from paw_policy_toolkit.action import DefaultActionTranslator
from paw_policy_toolkit.policy_service import ActionChunk

from inference.validate import (
    MAX_FIRST_ACTION_CMD_ERR_M,
    MAX_STEP_POS_M,
    aggregate_plausibility,
    check_plausibility,
    check_protocol,
    check_safety,
    chunk_to_train_rows,
)
from paw_baseline.layout import WIRE_FROM_TRAIN


@pytest.mark.parametrize("enabled", [False, True])
def test_simulation_waiver_help_requires_feature_flag(monkeypatch, capsys, enabled):
    from inference import validate

    monkeypatch.setenv("PAW_ENABLE_SIMULATION", "1" if enabled else "0")
    monkeypatch.setattr("sys.argv", ["validate", "--help"])
    with pytest.raises(SystemExit) as result:
        validate.main()
    assert result.value.code == 0
    assert (
        "--allow-simulation-gripper-reversals" in capsys.readouterr().out
    ) is enabled


def test_simulation_waiver_rejected_before_checkpoint_io_when_disabled(monkeypatch):
    from inference import validate

    monkeypatch.delenv("PAW_ENABLE_SIMULATION", raising=False)
    monkeypatch.setattr(
        "sys.argv",
        [
            "validate",
            "--server",
            "localhost:1",
            "--checkpoint",
            "/missing",
            "--allow-simulation-gripper-reversals",
        ],
    )
    with pytest.raises(SystemExit) as result:
        validate.main()
    assert result.value.code == 2


def _train_rows(horizon: int = 5, step: float = 0.001) -> np.ndarray:
    """A compliant chunk: tiny steps from a fixed pose, unit quats, sane grippers."""
    rows = np.zeros((horizon, 16))
    t = np.arange(horizon).reshape(-1, 1)
    rows[:, 0:3] = np.array([0.3, 0.1, 0.4]) + step * t
    rows[:, 6] = 1.0
    rows[:, 7] = 0.04
    rows[:, 8:11] = np.array([0.3, -0.1, 0.4]) + step * t
    rows[:, 14] = 1.0
    rows[:, 15] = 0.04
    return rows


def _to_chunk(train: np.ndarray) -> ActionChunk:
    """Encode train-order rows the way serve does: permute to wire, then wrap."""
    wire = train[:, list(WIRE_FROM_TRAIN)][np.newaxis]
    return DefaultActionTranslator().from_dict({"action": wire})


def test_chunk_round_trip_through_wire_layout():
    train = _train_rows()
    decoded = chunk_to_train_rows(_to_chunk(train))
    np.testing.assert_allclose(decoded, train, atol=1e-6)


def test_empty_chunk_rejected():
    with pytest.raises(ValueError, match="zero actions"):
        chunk_to_train_rows(ActionChunk())


def test_protocol_checks():
    train = _train_rows()
    chunk = _to_chunk(train)
    chunk.target_time_ns = 42
    rows = chunk_to_train_rows(chunk)
    assert check_protocol(chunk, rows, horizon=5, capture_time_ns=42)["pass"]
    assert not check_protocol(chunk, rows, horizon=6, capture_time_ns=42)[
        "horizon_matches"
    ]
    assert not check_protocol(chunk, rows, horizon=5, capture_time_ns=41)[
        "target_time_echoed"
    ]
    bad = train.copy()
    bad[:, 3:7] *= 2.0  # non-unit quaternions
    assert not check_protocol(chunk, bad, horizon=5, capture_time_ns=42)[
        "unit_quaternions"
    ]


def test_protocol_requires_frozen_arm_to_hold_live_observation():
    train = _train_rows()
    observed = train[0].copy()
    train[:, 0:8] = observed[0:8]
    chunk = _to_chunk(train)
    rows = chunk_to_train_rows(chunk)
    assert check_protocol(
        chunk,
        rows,
        horizon=5,
        capture_time_ns=0,
        frozen_arm="left",
        observed_state=observed,
    )["frozen_arm_holds_observation"]
    rows[2, 0] += 0.01
    assert not check_protocol(
        chunk,
        rows,
        horizon=5,
        capture_time_ns=0,
        frozen_arm="left",
        observed_state=observed,
    )["pass"]


def test_safety_passes_compliant_chunk():
    train = _train_rows()
    bbox = (train.min(axis=0), train.max(axis=0))
    assert check_safety(train, first_command=train[0], action_bbox=bbox)["pass"]


def test_safety_catches_each_violation():
    train = _train_rows()
    bbox = (train.min(axis=0), train.max(axis=0))
    command = train[0]

    teleport = train.copy()
    teleport[2, 0] += 2 * MAX_STEP_POS_M
    result = check_safety(teleport, first_command=command, action_bbox=bbox)
    assert not result["pass"] and result["max_pos_step_m"] > MAX_STEP_POS_M

    drifted = command.copy()
    drifted[0] -= (
        2 * MAX_FIRST_ACTION_CMD_ERR_M
    )  # chunk starts far from the demo's command
    result = check_safety(train, first_command=drifted, action_bbox=bbox)
    assert (
        not result["pass"]
        and result["first_action_cmd_err_m"] > MAX_FIRST_ACTION_CMD_ERR_M
    )

    spin = train.copy()
    spin[3:, 3] = np.sin(np.radians(45))  # ~90 deg flip mid-chunk
    spin[3:, 6] = np.cos(np.radians(45))
    assert (
        check_safety(spin, first_command=command, action_bbox=bbox)["max_rot_step_deg"]
        > 15
    )

    escape = train.copy()
    escape[:, 1] += 1.0  # outside bbox + margin
    assert not check_safety(escape, first_command=command, action_bbox=bbox)[
        "within_bbox"
    ]

    crush = train.copy()
    crush[:, 7] = 0.2  # beyond any physical gripper
    assert not check_safety(crush, first_command=command, action_bbox=bbox)[
        "grippers_in_range"
    ]


def test_safety_gripper_tolerance_accepts_float32_full_open():
    train = _train_rows()
    # float32(0.085) sits a hair above 0.085 in float64, like wire values do.
    train[:, [7, 15]] = np.float64(np.float32(0.085))
    bbox = (train.min(axis=0), train.max(axis=0))
    assert check_safety(train, first_command=train[0], action_bbox=bbox)[
        "grippers_in_range"
    ]


def test_simulation_waiver_only_allows_gripper_reversals():
    train = _train_rows()
    train[:, 7] = [0.01, 0.07, 0.01, 0.07, 0.01]
    bbox = (train.min(axis=0), train.max(axis=0))
    assert not check_safety(train, train[0], bbox)["pass"]
    result = check_safety(train, train[0], bbox, allow_gripper_reversals=True)
    assert result["pass"] and result["gripper_reversals_waived"]
    assert result["gripper_reversals_over_20mm"] == 3
    train[2, 0] += 2 * MAX_STEP_POS_M
    assert not check_safety(train, train[0], bbox, allow_gripper_reversals=True)["pass"]


def test_safety_ignores_motion_checks_on_pinned_arm():
    train = _train_rows()
    bbox = (train.min(axis=0), train.max(axis=0))
    train[:, 0] += np.arange(len(train))  # absurd left motion would normally fail
    assert check_safety(
        train, first_command=train[0], action_bbox=bbox, frozen_arm="left"
    )["pass"]


def test_plausibility_measures():
    truth = _train_rows(step=0.002)
    matching = check_plausibility(truth.copy(), truth)
    assert matching["magnitude_ok"] and matching["dominant_sign_ok"]

    backwards = truth.copy()
    backwards[:, [0, 8]] = truth[::-1, [0, 8]]  # dominant axis runs the other way
    assert check_plausibility(backwards, truth)["dominant_sign_ok"] is False

    static_truth = _train_rows(step=0.0)
    undecidable = check_plausibility(truth, static_truth)
    assert (
        undecidable["magnitude_ok"] is None and undecidable["dominant_sign_ok"] is None
    )


def test_plausibility_only_scores_driven_arm():
    truth = _train_rows(step=0.002)
    prediction = truth.copy()
    prediction[:, 0] = prediction[::-1, 0]  # frozen left arm is nonsensical
    result = check_plausibility(prediction, truth, frozen_arm="left")
    assert result["magnitude_ok"] and result["dominant_sign_ok"]


def test_aggregate_plausibility_majority():
    def measure(magnitude, sign):
        return {"magnitude_ok": magnitude, "dominant_sign_ok": sign}

    passing = aggregate_plausibility(
        [measure(True, True), measure(True, False), measure(True, True)]
    )
    assert passing["pass"]
    failing = aggregate_plausibility(
        [measure(True, False), measure(True, False), measure(True, True)]
    )
    assert not failing["pass"]
    all_undecidable = aggregate_plausibility([measure(None, None)])
    assert all_undecidable["pass"]  # nothing decidable is not a failure


def test_safety_rejects_gripper_flicker_but_allows_sustained_grasp():
    train = _train_rows(horizon=7)
    bbox = (train.min(axis=0), train.max(axis=0))
    train[:, 7] = [0.085, 0, 0.085, 0, 0.085, 0, 0.085]
    result = check_safety(train, train[0], bbox)
    assert result["gripper_reversals_over_20mm"] == 5
    assert not result["pass"]
    train[:, 7] = [0.085, 0.085, 0, 0, 0, 0.085, 0.085]
    assert check_safety(train, train[0], bbox)["pass"]
