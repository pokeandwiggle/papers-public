from threading import Event

import numpy as np
import pytest
import torch
from inference.serve import BaselinePolicyServer, _emit_hygiene
from paw_baseline.layout import GRIPPER_MAX_WIDTH_M, TRAIN_GRIP_DIMS, TRAIN_QUAT_SLICES
from paw_policy_toolkit.policy_service import ConfigureRequest


def _chunk(horizon: int = 4) -> np.ndarray:
    chunk = np.zeros((1, horizon, 16), dtype=np.float32)
    for sl in TRAIN_QUAT_SLICES:
        chunk[..., sl.stop - 1] = 1.0  # identity quaternions, w last
    chunk[..., list(TRAIN_GRIP_DIMS)] = 0.04
    return chunk


def test_clean_chunk_passes_through():
    chunk = _chunk()
    np.testing.assert_array_equal(_emit_hygiene(chunk), chunk)


def test_quats_renormalized_direction_preserved():
    chunk = _chunk()
    for sl in TRAIN_QUAT_SLICES:
        chunk[..., sl] *= 1.1  # model drifted off the unit sphere
    clean = _emit_hygiene(chunk)
    for sl in TRAIN_QUAT_SLICES:
        np.testing.assert_allclose(
            np.linalg.norm(clean[..., sl], axis=-1), 1.0, atol=1e-6
        )
        assert (clean[..., sl.stop - 1] > 0).all()  # w component kept its sign


@pytest.mark.parametrize("norm", [0.1, 0.2, 0.408, 0.5])
@pytest.mark.parametrize("arm", [0, 1])
def test_small_quaternions_normalized_without_changing_rotation(norm, arm):
    chunk = _chunk()
    sl = TRAIN_QUAT_SLICES[arm]
    direction = np.array([0.5, -0.5, 0.5, -0.5], dtype=np.float32)
    chunk[0, 1, sl] = direction * norm
    before = chunk.copy()

    clean = _emit_hygiene(chunk)

    np.testing.assert_allclose(clean[0, 1, sl], direction, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(clean[0, 1, sl]), 1.0, atol=1e-6)
    np.testing.assert_array_equal(chunk, before)


@pytest.mark.parametrize(
    "norm", [0.0, 1e-8, 0.099, np.nextafter(np.float32(0.1), np.float32(0))]
)
@pytest.mark.parametrize("arm", [0, 1])
def test_quaternion_below_minimum_norm_rejected(norm, arm):
    chunk = _chunk()
    chunk[0, 1, TRAIN_QUAT_SLICES[arm]] *= norm
    with pytest.raises(ValueError, match="degenerate"):
        _emit_hygiene(chunk)


def test_grippers_clamped_to_physical_range():
    chunk = _chunk()
    chunk[0, 0, TRAIN_GRIP_DIMS[0]] = -0.005
    chunk[0, 1, TRAIN_GRIP_DIMS[1]] = 0.2
    clean = _emit_hygiene(chunk)
    assert clean[0, 0, TRAIN_GRIP_DIMS[0]] == 0.0
    assert clean[0, 1, TRAIN_GRIP_DIMS[1]] == np.float32(GRIPPER_MAX_WIDTH_M)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_non_finite_rejected(value):
    chunk = _chunk()
    chunk[0, 2, 0] = value
    with pytest.raises(ValueError, match="non-finite"):
        _emit_hygiene(chunk)


def test_degenerate_quaternion_rejected():
    chunk = _chunk()
    chunk[0, 1, TRAIN_QUAT_SLICES[0]] = 0.0  # nothing to renormalize toward
    with pytest.raises(ValueError, match="degenerate"):
        _emit_hygiene(chunk)


def test_input_chunk_not_mutated():
    chunk = _chunk()
    chunk[..., list(TRAIN_GRIP_DIMS)] = 0.2
    before = chunk.copy()
    _emit_hygiene(chunk)
    np.testing.assert_array_equal(chunk, before)


def test_configure_rejects_a_different_model_id():
    server = object.__new__(BaselinePolicyServer)
    server._representation = None
    server._model_id = "123"
    server._loaded = True
    server.stop_event = Event()
    server._cameras = ()
    with pytest.raises(ValueError, match="pinned to model_id"):
        server.configure(ConfigureRequest(model_id="456", task_instruction="pick"))

    server.configure(ConfigureRequest(model_id="123", task_instruction="pick"))
    assert server._task_instruction == "pick"


def test_predict_uses_queue_populating_public_action_api():
    class IdentityProcessor:
        def __call__(self, value):
            return value

    class FakePolicy:
        def __init__(self):
            self.calls = 0

        def select_action(self, batch):
            self.calls += 1
            return torch.full((1, 16), float(self.calls))

        def predict_action_chunk(self, batch):
            raise AssertionError("direct chunk API bypasses policy queues")

    server = object.__new__(BaselinePolicyServer)
    server._representation = None
    server._policy = FakePolicy()
    server._preprocess = IdentityProcessor()
    server._postprocess = IdentityProcessor()
    server._n_action_steps = 3
    server._inference_executor = None

    chunk = server._predict({"observation.state": np.zeros((1, 16), dtype=np.float32)})

    assert chunk.shape == (1, 3, 16)
    np.testing.assert_array_equal(chunk[0, :, 0], [1.0, 2.0, 3.0])
    assert server._policy.calls == 3


@pytest.mark.parametrize("frozen_arm", ["left", "right"])
def test_server_pins_full_frozen_side_to_live_observation(frozen_arm):
    server = object.__new__(BaselinePolicyServer)
    server._representation = None
    server._frozen_arm = frozen_arm
    server._cameras = ()
    server._task_instruction = "test"
    predicted = _chunk(horizon=3)
    predicted[..., 0:3] = [0.8, 0.8, 0.8]
    predicted[..., 8:11] = [-0.8, -0.8, -0.8]
    from paw_baseline.representation import encode_model

    server._predict = lambda batch: encode_model(_emit_hygiene(predicted))
    emitted = {}
    server.emit_chunk = lambda chunk, target, *, discard_actions: emitted.update(
        chunk=chunk["action"], target=target
    )

    observation = {
        "actions_discarded": False,
        "chunk_requested": True,
        "capture_time_ns": 123,
        "observation.state.tcp_pose_left": np.array(
            [[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32
        ),
        "observation.state.gripper_width_left": np.array([[0.02]], dtype=np.float32),
        "observation.state.tcp_pose_right": np.array(
            [[0.4, 0.5, 0.6, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32
        ),
        "observation.state.gripper_width_right": np.array([[0.07]], dtype=np.float32),
    }
    server.on_observation(observation)

    wire = emitted["chunk"]
    assert emitted["target"] == 123
    if frozen_arm == "left":
        expected_pose = np.repeat(
            observation["observation.state.tcp_pose_left"], 3, axis=0
        )
        np.testing.assert_allclose(wire[0, :, 0:7], expected_pose)
        np.testing.assert_allclose(wire[0, :, 14], 0.02)
        np.testing.assert_allclose(wire[0, :, 7:10], np.full((3, 3), -0.8))
    else:
        expected_pose = np.repeat(
            observation["observation.state.tcp_pose_right"], 3, axis=0
        )
        np.testing.assert_allclose(wire[0, :, 7:14], expected_pose)
        np.testing.assert_allclose(wire[0, :, 15], 0.07)
        np.testing.assert_allclose(wire[0, :, 0:3], np.full((3, 3), 0.8))


def test_server_preserves_predicted_gripper_pulses():
    from paw_baseline.layout import STATE_KEY_DIMS
    from paw_baseline.representation import encode_model

    predicted = _emit_hygiene(_chunk(horizon=3))
    predicted[0, :, 7] = [0.084, 0.0209, 0.0767]
    predicted[0, :, 15] = [0.01, 0.08, 0.02]
    server = object.__new__(BaselinePolicyServer)
    server._representation = None
    server._cameras = ()
    server._frozen_arm = None
    server._task_instruction = "test"
    server._predict = lambda batch: encode_model(predicted)
    emitted = {}
    server.emit_chunk = lambda chunk, target, *, discard_actions: emitted.update(
        chunk=chunk["action"]
    )
    observation = {
        "actions_discarded": False,
        "chunk_requested": True,
        "capture_time_ns": 123,
    }
    offset = 0
    for key, width in STATE_KEY_DIMS:
        observation[key] = predicted[:, 0, offset : offset + width]
        offset += width

    server.on_observation(observation)

    np.testing.assert_array_equal(emitted["chunk"][0, :, 14], predicted[0, :, 7])
    np.testing.assert_array_equal(emitted["chunk"][0, :, 15], predicted[0, :, 15])


def test_compiled_predictions_keep_one_thread_across_request_streams():
    import threading
    from concurrent.futures import ThreadPoolExecutor

    threads = []

    class FakePolicy:
        def select_action(self, batch):
            threads.append(threading.get_ident())
            if batch.get("fail"):
                raise ValueError("inference failed")
            return torch.zeros((1, 20))

    server = object.__new__(BaselinePolicyServer)
    server._policy = FakePolicy()
    server._preprocess = lambda value: value
    server._postprocess = lambda value: value
    server._n_action_steps = 2
    with ThreadPoolExecutor(max_workers=1) as owner:
        server._inference_executor = owner
        warmup = server._predict({})
        for _ in range(2):
            with ThreadPoolExecutor(max_workers=1) as rpc:
                result = rpc.submit(server._predict, {}).result()
                np.testing.assert_array_equal(result, warmup)
        with pytest.raises(ValueError, match="inference failed"):
            server._predict({"fail": True})

    assert len(set(threads)) == 1
    assert threads[0] != threading.get_ident()


@pytest.mark.parametrize(
    "policy_type,compiled,override,precision,expected_compiled,expected",
    [
        ("act", None, None, None, False, "high"),
        ("diffusion", False, None, None, True, "highest"),
        ("diffusion", True, None, None, True, "highest"),
        ("diffusion", True, False, None, False, "high"),
        ("multi_task_dit", None, None, None, False, "high"),
        ("paw_pi05", False, None, None, True, "highest"),
        ("paw_pi05", True, None, None, True, "highest"),
        ("paw_pi05", False, True, None, True, "highest"),
        ("paw_pi05", False, False, None, False, "high"),
        ("paw_pi05", True, False, None, False, "high"),
        ("paw_pi05", False, None, "high", True, "high"),
        ("paw_pi05", False, None, "highest", True, "highest"),
        ("pi05", False, None, None, True, "highest"),
        ("pi05", True, None, None, True, "highest"),
        ("smolvla", False, None, None, True, "highest"),
        ("smolvla", True, None, None, True, "highest"),
        ("smolvla", False, False, None, False, "high"),
        ("smolvla", True, False, None, False, "high"),
        ("smolvla", False, None, "high", True, "high"),
    ],
)
@pytest.mark.parametrize("compile_mode", [None, "default"])
def test_serving_compile_defaults_and_precision(
    monkeypatch,
    tmp_path,
    policy_type,
    compiled,
    override,
    precision,
    expected_compiled,
    expected,
    compile_mode,
):
    import json
    from types import SimpleNamespace

    import inference.serve as serving
    import lerobot.policies

    config = {
        "type": policy_type,
        "compile_model": compiled,
        "input_features": {
            "observation.state": {"type": "STATE", "shape": [20]},
            "observation.images.rgb_static": {"type": "VISUAL", "shape": [3, 4, 4]},
        },
        "output_features": {"action": {"shape": [20]}},
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    native = SimpleNamespace(
        compile_model=compiled,
        compile_mode="default",
        n_action_steps=32,
        n_obs_steps=1,
    )
    if compiled is None:
        del native.compile_model
    monkeypatch.setattr(
        serving, "load_serving_metadata", lambda _: {"frozen_arm": None}
    )
    monkeypatch.setattr(serving, "load_policy_config", lambda _: native)

    class FakePolicy:
        config = native
        diffusion = SimpleNamespace(unet=SimpleNamespace(_orig_mod=object()))

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            expected_load_compiled = (
                compiled if policy_type == "diffusion" else expected_compiled
            )
            assert (
                getattr(kwargs["config"], "compile_model", False)
                is expected_load_compiled
            )
            if policy_type == "diffusion":
                assert kwargs["strict"] is True
            if expected_compiled:
                expected_mode = compile_mode or (
                    "max-autotune" if policy_type in {"paw_pi05", "pi05"} else "default"
                )
                assert kwargs["config"].compile_mode == expected_mode
            # LeRobot overrides the process setting while constructing PI05.
            torch.set_float32_matmul_precision("high")
            return cls()

        def to(self, device):
            pass

        def eval(self):
            pass

    monkeypatch.setattr(lerobot.policies, "get_policy_class", lambda _: FakePolicy)
    monkeypatch.setattr(
        torch, "compile", lambda model, **kwargs: SimpleNamespace(_orig_mod=model)
    )
    monkeypatch.setattr(
        lerobot.policies, "make_pre_post_processors", lambda *a, **kw: (None, None)
    )
    observed = []
    monkeypatch.setattr(
        serving.BaselinePolicyServer,
        "_warmup",
        lambda self: observed.append(
            self._inference_executor.submit(torch.get_float32_matmul_precision).result()
            if self._inference_executor is not None
            else torch.get_float32_matmul_precision()
        ),
    )
    previous = torch.get_float32_matmul_precision()
    try:
        server = serving.BaselinePolicyServer(
            tmp_path,
            "cpu",
            compile_model=override,
            compile_mode=compile_mode,
            matmul_precision=precision,
        )
        server.do_configure(ConfigureRequest(task_instruction="pick"))
        server.close()
        assert (server._inference_executor is not None) is expected_compiled
        if server._inference_executor is not None:
            with pytest.raises(RuntimeError, match="shutdown"):
                server._inference_executor.submit(lambda: None)
        assert observed == [expected]
    finally:
        torch.set_float32_matmul_precision(previous)
