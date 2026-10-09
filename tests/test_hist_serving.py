"""Server pacing for hist_diffusion: strided context, grid cadence, seams."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from inference.serve import BaselinePolicyServer
from paw_baseline.hist_diffusion.serving import ReplanSchedule, StridedHistory
from paw_baseline.representation import encode_model


def _physical(x: float) -> np.ndarray:
    """Train-order flat-16 pose with left x position ``x`` and identity rotations."""
    row = np.zeros(16, dtype=np.float32)
    row[[6, 14]] = 1.0
    row[[7, 15]] = 0.04
    row[0] = x
    return row


def _observation(index: int, *, committed: int = 0, replaceable: int = 0) -> dict:
    left = np.array([[float(index), 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    right = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    wire_row = np.zeros(16, dtype=np.float32)
    wire_row[[6, 13]] = 1.0
    wire_row[0] = -1.0
    return {
        "actions_discarded": False,
        "chunk_requested": index == 0,
        "capture_time_ns": 1000 + index,
        "observation.state.tcp_pose_left": left,
        "observation.state.gripper_width_left": np.array([[0.04]], dtype=np.float32),
        "observation.state.tcp_pose_right": right,
        "observation.state.gripper_width_right": np.array([[0.04]], dtype=np.float32),
        "committed_actions": np.zeros((1, committed, 16), dtype=np.float32),
        "replaceable_actions": np.tile(wire_row, (1, replaceable, 1)),
    }


class _Emitter:
    def __init__(self):
        self.chunks = []

    def emit_chunk(self, chunk, target_time_ns, *, discard_actions):
        self.chunks.append((chunk["action"], target_time_ns, discard_actions))


def _server(n_obs_steps=3, obs_stride=3, replan_steps=6, seam_steps=0):
    server = object.__new__(BaselinePolicyServer)
    server._representation = None
    server._frozen_arm = None
    server._cameras = ()
    server._history = StridedHistory(n_obs_steps, obs_stride)
    server._schedule = ReplanSchedule(replan_steps)
    server._seam_steps = seam_steps
    server._plan_stats = []
    server._inference_executor = ThreadPoolExecutor(max_workers=1)
    server._emitter = _Emitter()
    server.contexts = []

    def predict(context):
        server.contexts.append(context["observation.state"].copy())
        # Echo the newest observed left x as a constant 5-step plan.
        newest = float(context["observation.state"][-1, 0])
        return encode_model(np.stack([_physical(newest)] * 5)[None])

    server._predict_history = predict
    return server


def test_plans_on_grid_with_strided_context_and_grounding():
    server = _server()
    for index in range(14):
        server.on_observation(_observation(index))
        server._wait_for_plan(raise_errors=True)
    chunks = server._emitter.chunks
    assert [target for _, target, _ in chunks] == [1000, 1006, 1012]
    # Context x positions: strided by 3, repeating the first frame at the start.
    assert [c[:, 0].tolist() for c in server.contexts] == [
        [0.0, 0.0, 0.0],
        [0.0, 3.0, 6.0],
        [6.0, 9.0, 12.0],
    ]
    np.testing.assert_allclose(chunks[1][0][0, :, 0], 6.0)
    server._inference_executor.shutdown()


def test_seam_blends_from_previous_emitted_chunk():
    server = _server(replan_steps=2, seam_steps=2)
    # No earlier chunk: the first plan is emitted unchanged.
    server.on_observation(_observation(0, committed=1, replaceable=3))
    np.testing.assert_allclose(server._emitter.chunks[0][0][0, :, 0], 0.0)
    server.on_observation(_observation(1, committed=1, replaceable=3))
    # Robot executes chunk 0 (x=0); the new plan (x=2) lands after 1 slot.
    server.on_observation(_observation(2, committed=1, replaceable=3))
    server._wait_for_plan(raise_errors=True)
    wire = server._emitter.chunks[1][0]
    np.testing.assert_allclose(wire[0, :, 0], [2.0, 2 / 3, 4 / 3, 2.0, 2.0], atol=1e-6)
    # Without a robot plan in flight (e.g. discarded), there is no seam to blend.
    server.on_observation(_observation(3))
    server.on_observation(_observation(4))
    server._wait_for_plan(raise_errors=True)
    np.testing.assert_allclose(server._emitter.chunks[2][0][0, :, 0], 4.0)
    server._inference_executor.shutdown()


def test_first_plan_is_emitted_before_intake_returns():
    server = _server()
    server.on_observation(_observation(0))
    assert server._pending is None and len(server._emitter.chunks) == 1
    server._inference_executor.shutdown()
