"""Training/serving quaternion representation must not depend on stream history."""

from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pytest
from inference.serve import BaselinePolicyServer
from inference.validate import _build_observation
from paw_baseline.layout import STATE_KEY_DIMS, TRAIN_QUAT_SLICES
from paw_baseline.prepare import _fold
from paw_baseline.representation import encode_model
from paw_policy_toolkit.observation import DefaultObservationTranslator
from paw_policy_toolkit.policy_service import ObservationRequirements


def _states():
    angles = np.deg2rad(np.linspace(120, 240, 17))
    states = np.zeros((len(angles), 16), dtype=np.float32)
    for sl in TRAIN_QUAT_SLICES:
        states[:, sl.start] = np.sin(angles / 2)
        states[:, sl.stop - 1] = np.cos(angles / 2)
    states[:, 7] = states[:, 15] = 0.04
    return states


def _training_state(states):
    columns = {}
    offset = 0
    for key, width in STATE_KEY_DIMS:
        values = states[:, offset : offset + width]
        columns[key] = (
            pa.array(values[:, 0])
            if width == 1
            else pa.FixedSizeListArray.from_arrays(pa.array(values.ravel()), width)
        )
        offset += width
    return _fold(pa.table(columns), STATE_KEY_DIMS)


@pytest.mark.parametrize("reset_each_frame", [False, True])
@pytest.mark.parametrize(
    "flip_left,flip_right", [(False, False), (True, False), (False, True), (True, True)]
)
def test_training_matches_transport_after_start_and_reset(
    reset_each_frame, flip_left, flip_right
):
    states = _states()
    expected = encode_model(_training_state(states))
    captured = []
    server = object.__new__(BaselinePolicyServer)
    server._representation = None
    server._configured = True
    server._cameras = ()
    server._frozen_arm = None
    server._task_instruction = "test"
    server._policy = SimpleNamespace(reset=lambda: None)
    server.emit_chunk = lambda *args, discard_actions: None

    def predict(batch):
        captured.append(batch["observation.state"].copy())
        return batch["observation.state"][:, None, :].copy()

    server._predict = predict
    sent = states.copy()
    for flip, sl in zip((flip_left, flip_right), TRAIN_QUAT_SLICES, strict=True):
        if flip:
            sent[:, sl] *= -1
    # Each suffix models starting the real translator in the middle of a trajectory.
    for start in range(len(states)):
        translator = DefaultObservationTranslator()
        translator.reset(ObservationRequirements())
        server.activate()
        for index in range(start, len(states)):
            if reset_each_frame:
                translator.reset(ObservationRequirements())
                server.activate()
            observation = _build_observation(sent[index], {}, index)
            server.on_observation(translator.to_dict(observation))
            np.testing.assert_array_equal(captured[-1][0], expected[index])


def test_training_ignores_independent_quaternion_signs():
    states = _states()
    flipped = states.copy()
    flipped[::2, 3:7] *= -1
    flipped[1::2, 11:15] *= -1
    np.testing.assert_array_equal(_training_state(states), _training_state(flipped))


def test_training_uses_largest_component_with_fixed_tie_break():
    states = np.zeros((4, 16), dtype=np.float32)
    quats = np.array(
        [[-1, 0, 0, 0], [0, -1, 0, 0], [-0.5, 0.5, -0.5, 0.5], [0, 0, 0, -1]],
        dtype=np.float32,
    )
    for sl in TRAIN_QUAT_SLICES:
        states[:, sl] = quats
    prepared = _training_state(states)
    for sl in TRAIN_QUAT_SLICES:
        np.testing.assert_array_equal(prepared[:, sl], -quats)
        # q q^T preserves the represented physical rotation when the sign changes.
        np.testing.assert_array_equal(
            prepared[:, sl, None] * prepared[:, None, sl],
            quats[:, :, None] * quats[:, None, :],
        )


def test_random_rotations_are_sign_invariant_and_idempotent():
    from paw_baseline.rotations import canonical_quaternions

    rng = np.random.default_rng(20260906)
    values = rng.normal(size=(4096, 4))
    values = (values / np.linalg.norm(values, axis=-1, keepdims=True)).astype(
        np.float32
    )
    before = values.copy()
    result = canonical_quaternions(values)
    assert result.tobytes() == canonical_quaternions(-values).tobytes()
    assert result.tobytes() == canonical_quaternions(result).tobytes()
    np.testing.assert_array_equal(values, before)
    np.testing.assert_allclose(
        result[:, :, None] * result[:, None, :],
        values[:, :, None] * values[:, None, :],
        atol=0,
    )


@pytest.mark.parametrize("quat", [[0, 0, 0, 0], [0, 0, 0, 2], [float("nan"), 0, 0, 1]])
def test_invalid_rotations_are_rejected(quat):
    from paw_baseline.rotations import canonical_quaternions

    with pytest.raises(RuntimeError):
        canonical_quaternions(np.asarray(quat))
