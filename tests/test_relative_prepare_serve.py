"""Relative-action datasets, serving metadata, serve integration and validation."""

import json

import numpy as np
import pyarrow.parquet as pq
import pytest
from paw_baseline.layout import STATE_KEY_DIMS, WIRE_FROM_TRAIN
from paw_baseline.prepare import (
    DELTA_LAYOUT_NAME,
    PROVENANCE_PATH,
    prepare_dataset,
)
from paw_baseline.relative import (
    ACTION_REPRESENTATION,
    DELTA_DIM,
    delta_names,
    encode_deltas,
    integrate_deltas,
)
from paw_baseline.representation import decode_model
from paw_baseline.serving_metadata import metadata_from_prepared

from inference import serve as serve_module
from inference.validate import _build_observation, _episode_actions, _load_episode_meta
from tests.fixture_export import build_export


def _source_rows(export_dir, prefix):
    table = pq.read_table(export_dir / "data/chunk-000/file-000.parquet")
    parts = []
    for key, dim in STATE_KEY_DIMS:
        column = table[key.replace("observation.state.", prefix)].combine_chunks()
        parts.append(np.asarray(column.flatten() if dim > 1 else column).reshape(-1, dim))
    return np.concatenate(parts, axis=1).astype(np.float64), np.asarray(table["episode_index"])


@pytest.fixture(scope="module")
def relative(export_dir, cache_root):
    return prepare_dataset(
        export_dir, cache_root=cache_root, action_representation=ACTION_REPRESENTATION
    )


def test_relative_view_stores_per_frame_deltas(export_dir, relative):
    assert relative.root.name.endswith(DELTA_LAYOUT_NAME)
    info = json.loads((relative.root / "meta/info.json").read_text())
    assert info["features"]["action"]["shape"] == [DELTA_DIM]
    assert info["features"]["action"]["names"] == delta_names()
    assert info["features"]["observation.state"]["shape"] == [20]
    stamp = json.loads((relative.root / PROVENANCE_PATH).read_text())
    assert stamp["action_representation"] == ACTION_REPRESENTATION
    assert len(stamp["physical_action_bounds"]["min"]) == 16

    view = pq.read_table(relative.root / "data/chunk-000/file-000.parquet")
    deltas = np.asarray(view["action"].combine_chunks().flatten()).reshape(-1, DELTA_DIM)
    state, episodes = _source_rows(export_dir, "observation.state.")
    action, _ = _source_rows(export_dir, "action.")
    for episode in np.unique(episodes):
        rows = episodes == episode
        expected = encode_deltas(state[rows], action[rows])
        np.testing.assert_allclose(deltas[rows], expected, atol=1e-6)
    stats = json.loads((relative.root / "meta/stats.json").read_text())["action"]
    np.testing.assert_allclose(stats["mean"], deltas.astype(np.float64).mean(axis=0), atol=1e-6)


def test_validation_truth_recovers_demonstrated_commands(export_dir, relative):
    action, episodes = _source_rows(export_dir, "action.")
    for episode in _load_episode_meta(relative.root):
        absolute = _episode_actions(relative.root, episode)
        expected = action[episodes == episode["episode_index"]]
        np.testing.assert_allclose(absolute[:, [0, 1, 2, 7, 8, 9, 10, 15]], expected[:, [0, 1, 2, 7, 8, 9, 10, 15]], atol=1e-5)
        for quat in (slice(3, 7), slice(11, 15)):
            dots = np.abs(np.sum(absolute[:, quat] * expected[:, quat], axis=1))
            np.testing.assert_allclose(dots, 1.0, atol=1e-5)


def test_relative_serving_metadata(relative):
    metadata = metadata_from_prepared(relative.root, model="smile_smolvla")
    assert metadata["action_representation"] == ACTION_REPRESENTATION
    assert metadata["action_dim"] == DELTA_DIM
    assert metadata["layout"] == DELTA_LAYOUT_NAME


def test_frozen_arm_relative_deltas_are_zero(tmp_path):
    export = build_export(tmp_path / "frozen_export", frozen_right_arm=True)
    prepared = prepare_dataset(
        export, cache_root=tmp_path / "cache", action_representation=ACTION_REPRESENTATION
    )
    stamp = json.loads((prepared.root / PROVENANCE_PATH).read_text())
    assert stamp["frozen_arm"] == "right"
    view = pq.read_table(prepared.root / "data/chunk-000/file-000.parquet")
    deltas = np.asarray(view["action"].combine_chunks().flatten()).reshape(-1, DELTA_DIM)
    assert np.all(deltas[:, 7:13] == 0)
    stats = json.loads((prepared.root / "meta/stats.json").read_text())["action"]
    np.testing.assert_allclose(stats["std"][7:14], 1.0)


def _observation(state16, committed=None, replaceable=None, discarded=False):
    obs = {"actions_discarded": discarded}
    offset = 0
    for key, dim in STATE_KEY_DIMS:
        obs[key] = state16[offset : offset + dim].reshape(1, dim).astype(np.float32)
        offset += dim
    obs["committed_actions"] = (
        np.zeros((1, 0, 16)) if committed is None else committed[None, :, WIRE_FROM_TRAIN]
    )
    obs["replaceable_actions"] = (
        np.zeros((1, 0, 16)) if replaceable is None else replaceable[None, :, WIRE_FROM_TRAIN]
    )
    return obs


def _pose(rng):
    from scipy.spatial.transform import Rotation

    pose = np.zeros(16)
    pose[[0, 1, 2, 8, 9, 10]] = rng.normal(size=6)
    pose[3:7] = Rotation.random(random_state=1).as_quat()
    pose[11:15] = Rotation.random(random_state=2).as_quat()
    pose[[7, 15]] = 0.04
    return pose


def test_command_reference_prefers_robot_reported_hold():
    rng = np.random.default_rng(0)
    measured, held, other = _pose(rng), _pose(rng), _pose(rng)
    reference, source = serve_module.command_reference(_observation(measured))
    assert source == "measured"
    np.testing.assert_allclose(reference, measured, atol=1e-6)
    reference, source = serve_module.command_reference(
        _observation(measured, replaceable=np.stack([held, other]))
    )
    assert source == "replaceable"
    np.testing.assert_allclose(reference, held)
    reference, source = serve_module.command_reference(
        _observation(measured, committed=np.stack([other, held]), replaceable=other[None])
    )
    assert source == "committed"
    np.testing.assert_allclose(reference, held)
    reference, source = serve_module.command_reference(
        _observation(measured, replaceable=held[None], discarded=True)
    )
    assert source == "measured"
    observation = _observation(measured)
    observation["observation.state.desired_tcp_pose_left"] = held[0:7].reshape(1, 7)
    observation["observation.state.desired_tcp_pose_right"] = held[8:15].reshape(1, 7)
    reference, source = serve_module.command_reference(observation)
    assert source == "desired"
    np.testing.assert_allclose(reference[[*range(0, 7), *range(8, 15)]], held[[*range(0, 7), *range(8, 15)]])
    np.testing.assert_allclose(reference[[7, 15]], measured[[7, 15]], atol=1e-6)


def test_validation_observation_reports_previous_command_as_hold():
    from paw_policy_toolkit.observation import DefaultObservationTranslator
    from paw_policy_toolkit.policy_service import ObservationRequirements

    rng = np.random.default_rng(1)
    state, previous = _pose(rng), _pose(rng)
    observation = _build_observation(state, {}, 123, previous)
    translator = DefaultObservationTranslator()
    translator.reset(ObservationRequirements())
    as_dict = translator.to_dict(observation)
    reference, source = serve_module.command_reference(as_dict)
    assert source == "replaceable"
    np.testing.assert_allclose(reference[[0, 1, 2, 7, 8, 9, 10, 15]], previous[[0, 1, 2, 7, 8, 9, 10, 15]], atol=1e-6)
    for quat in (slice(3, 7), slice(11, 15)):
        assert abs(abs(np.dot(reference[quat], previous[quat])) - 1) < 1e-6


class _RecordingServer(serve_module.BaselinePolicyServer):
    def __init__(self, prediction):
        self._prediction = prediction
        self._delta = True
        self._reference_source = None
        self._frozen_arm = None
        self._representation = None
        self._cameras = ()
        self._task_instruction = "task"
        self.emitted = []

    def _predict(self, batch):
        return self._prediction

    def emit_chunk(self, chunk, target_time_ns, *, discard_actions):
        self.emitted.append((chunk["action"], target_time_ns, discard_actions))


def test_server_integrates_deltas_from_reference_and_echoes_capture_time():
    rng = np.random.default_rng(2)
    measured = _pose(rng)
    deltas = np.zeros((1, 15, DELTA_DIM), dtype=np.float32)
    deltas[0, :, 0] = 0.002  # 2 mm per step along x, left arm
    deltas[0, :, 12] = 0.01  # right arm yaw increments
    deltas[0, :, [6, 13]] = 0.03
    server = _RecordingServer(deltas)
    observation = _observation(measured)
    observation.update({"chunk_requested": True, "capture_time_ns": 99})
    server.on_observation(observation)
    wire, target, discard = server.emitted[-1]
    # The robot matches replies to requests by the echoed capture time.
    assert target == 99 and discard is False and wire.shape == (1, 15, 16)
    train = wire[0][:, [WIRE_FROM_TRAIN.index(i) for i in range(16)]]
    expected = integrate_deltas(measured, deltas[0].astype(np.float64))
    np.testing.assert_allclose(train[:, 0], measured[0] + 0.002 * np.arange(1, 16), atol=1e-6)
    np.testing.assert_allclose(train[:, [7, 15]], 0.03, atol=1e-7)
    for quat in (slice(3, 7), slice(11, 15)):
        dots = np.abs(np.sum(train[:, quat] * expected[:, quat], axis=1))
        np.testing.assert_allclose(dots, 1.0, atol=1e-6)
    # In synchronous execution the robot reports the held last command; the
    # next chunk continues from it rather than from the lagging measured pose.
    held = train[-1].astype(np.float64)
    observation = _observation(measured, replaceable=held[None])
    observation.update({"chunk_requested": True, "capture_time_ns": 130})
    server.on_observation(observation)
    second = server.emitted[-1][0][0][:, [WIRE_FROM_TRAIN.index(i) for i in range(16)]]
    np.testing.assert_allclose(second[0, 0], held[0] + 0.002, atol=1e-6)
    assert server._reference_source == "replaceable"
    # Without robot reports (e.g. after a previewed chunk) it never trusts its own output.
    observation = _observation(measured)
    observation.update({"chunk_requested": True, "capture_time_ns": 160})
    server.on_observation(observation)
    third = server.emitted[-1][0][0][:, [WIRE_FROM_TRAIN.index(i) for i in range(16)]]
    np.testing.assert_allclose(third[0, 0], measured[0] + 0.002, atol=1e-6)
    # decode_model is untouched for absolute checkpoints.
    assert decode_model is serve_module.decode_model
