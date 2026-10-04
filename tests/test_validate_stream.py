"""Serving protocol and validation stream lifecycle."""

import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from types import SimpleNamespace

import grpc
import numpy as np
import pytest
from inference import validate
from inference.serve import BaselinePolicyServer
from paw_baseline.representation import encode_model
from paw_policy_toolkit import ObservationRequirements, PolicyServer
from paw_policy_toolkit.policy_service import (
    ActionChunk,
    ClientMessage,
    ConfigureRequest,
    Observation,
    PolicyServiceStub,
    ServerMessage,
    add_PolicyServiceServicer_to_server,
)
from paw_policy_toolkit.server import _PolicyServiceServicer


@pytest.mark.parametrize("horizon", [4, 16, 32])
def test_baseline_preserves_preview_execution_flags_on_wire(
    monkeypatch, tmp_path, horizon
):
    state = np.zeros(16, dtype=np.float32)
    state[[6, 14]] = 1.0
    state[[7, 15]] = [0.02, 0.06]
    prediction = encode_model(np.tile(state, (1, horizon, 1)))

    def load(policy):
        policy._cameras = ()
        policy._frozen_arm = None
        policy._policy = SimpleNamespace(reset=lambda: None)
        policy._representation = None

    monkeypatch.setattr(BaselinePolicyServer, "_load", load)
    monkeypatch.setattr(
        BaselinePolicyServer, "_predict", lambda self, batch: prediction
    )
    policy = BaselinePolicyServer(tmp_path, "cpu", "model")
    # Preview replies must arrive without becoming executable after a resume.
    discard_flags = [True, False, True]
    observations = []
    for capture_time, discard in enumerate(discard_flags, start=100):
        observation = validate._build_observation(state, {}, capture_time)
        observation.actions_discarded = discard
        observations.append(ClientMessage(observation=observation))

    with ThreadPoolExecutor(max_workers=2) as workers:
        server = grpc.server(workers)
        add_PolicyServiceServicer_to_server(_PolicyServiceServicer(policy), server)
        port = server.add_insecure_port("127.0.0.1:0")
        server.start()
        try:
            with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
                stub = PolicyServiceStub(channel)
                stub.Configure(
                    ConfigureRequest(model_id="model", task_instruction="pick"),
                    timeout=5,
                )
                chunks = [
                    message.action_chunk
                    for message in stub.Run(iter(observations), timeout=5)
                    if message.WhichOneof("payload") == "action_chunk"
                ]
            assert [chunk.discard_actions for chunk in chunks] == discard_flags
            assert [chunk.target_time_ns for chunk in chunks] == [100, 101, 102]
            assert [len(chunk.actions) for chunk in chunks] == [horizon] * 3
            for chunk in chunks:
                for action in chunk.actions:
                    assert [
                        arm.target_pose.orientation.w for arm in action.arms
                    ] == [1, 1]
                    assert [
                        gripper.target_width for gripper in action.grippers
                    ] == pytest.approx([0.02, 0.06])
        finally:
            server.stop(grace=0).wait()
            policy.close()


def test_request_chunk_drains_stream_and_keeps_first_chunk_latency(monkeypatch):
    clock = [10.0]
    drained = []
    monkeypatch.setattr(validate.time, "monotonic", lambda: clock[0])

    def run(requests, *, timeout):
        assert len(list(requests)) == 1
        assert timeout == 5.0
        clock[0] = 10.125
        yield ServerMessage(action_chunk=ActionChunk(target_time_ns=1))
        clock[0] = 12.0
        yield ServerMessage(action_chunk=ActionChunk(target_time_ns=2))
        drained.append(True)

    chunk, latency = validate._request_chunk(
        SimpleNamespace(Run=run), Observation(), timeout=5.0
    )

    assert drained == [True]
    assert chunk.target_time_ns == 1
    assert latency == 0.125


def test_request_chunk_does_not_hide_error_after_first_chunk():
    def run(requests, *, timeout):
        yield ServerMessage(action_chunk=ActionChunk())
        raise grpc.RpcError("stream teardown failed")

    with pytest.raises(grpc.RpcError, match="stream teardown failed"):
        validate._request_chunk(SimpleNamespace(Run=run), Observation(), timeout=5)


class _TeardownPolicy(PolicyServer):
    def __init__(self):
        super().__init__()
        self.teardown_started = threading.Event()
        self.release_teardown = threading.Event()
        self.deactivations = 0

    def configure(self, request):
        return ObservationRequirements()

    def activate(self):
        pass

    def on_observation(self, observation):
        actions = np.zeros((1, 1, 16), dtype=np.float32)
        actions[0, 0, [6, 13]] = 1.0
        self.emit_chunk(
            {"action": actions},
            target_time_ns=observation["capture_time_ns"],
            discard_actions=observation["actions_discarded"],
        )

    def deactivate(self):
        self.teardown_started.set()
        assert self.release_teardown.wait(timeout=5)
        self.deactivations += 1


def test_request_chunk_waits_for_sdk_teardown_before_next_probe():
    policy = _TeardownPolicy()
    with ThreadPoolExecutor(max_workers=4) as workers:
        server = grpc.server(workers)
        add_PolicyServiceServicer_to_server(_PolicyServiceServicer(policy), server)
        port = server.add_insecure_port("127.0.0.1:0")
        server.start()
        try:
            with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
                stub = PolicyServiceStub(channel)
                stub.Configure(ConfigureRequest(model_id="validation-probe"))
                state = np.zeros(16)
                state[[6, 14]] = 1.0
                observation = validate._build_observation(state, {}, 42)
                first = workers.submit(validate._request_chunk, stub, observation, 5)
                try:
                    assert policy.teardown_started.wait(timeout=5)
                    with pytest.raises(TimeoutError):
                        first.result(timeout=0.2)
                finally:
                    policy.release_teardown.set()
                chunk, _ = first.result(timeout=5)
                assert chunk.target_time_ns == 42
                assert policy.deactivations == 1

                next_chunk, _ = validate._request_chunk(stub, observation, timeout=5)
                assert next_chunk.target_time_ns == 42
                assert policy.deactivations == 2
        finally:
            policy.release_teardown.set()
            server.stop(grace=0).wait()
