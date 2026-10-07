"""CPU end-to-end for Patch Policy: train both presets, serve with history, validate.

A tiny random-encoder configuration keeps this offline; the assertion is that
tokenizer fitting, checkpointing, strict restore, history serving and the
validation protocol all work, not that the policy is good.
"""

import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import grpc
import numpy as np
import pytest

from tests.test_e2e_cpu import _run, env  # noqa: F401 — shared CPU-only env fixture

pytestmark = pytest.mark.e2e

BASELINE = Path(__file__).resolve().parents[1]
HORIZON = 5
OVERRIDES = {
    "steps": 2,
    "batch_size": 2,
    "num_workers": 0,
    "log_freq": 1,
    "save_freq": 2,
    "policy.device": "cpu",
    "policy.encoder_pretrained": False,
    "policy.image_size": 28,
    "policy.chunk_size": HORIZON,
    "policy.n_action_steps": HORIZON,
    "policy.n_layer": 1,
    "policy.n_head": 2,
    "policy.n_embd": 32,
    "policy.vqvae_steps": 3,
    "policy.vqvae_batch_size": 16,
    "policy.trunk_dtype": "float32",
}


@pytest.fixture(scope="module", params=["patch_vqbet_dense", "patch_vqbet_cls"])
def trained(export_dir, env, tmp_path_factory, request):  # noqa: F811
    output_dir = tmp_path_factory.mktemp("patch-train-out")
    params = {"preset": request.param, "overrides": OVERRIDES}
    result = _run(
        ["train", f"--dataset={export_dir}", f"--output={output_dir}", "--params=" + json.dumps(params)],
        {**env, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        timeout=900,
    )
    assert result.returncode == 0, f"train failed:\n{result.stdout[-4000:]}\n{result.stderr[-6000:]}"
    return output_dir, request.param


def test_two_process_ddp_training_shares_rank_zero_tokenizer(export_dir, env, tmp_path):  # noqa: F811
    output_dir = tmp_path / "ddp-out"
    output_dir.mkdir()
    params = {"preset": "patch_vqbet_dense", "overrides": OVERRIDES}
    result = _run(
        [
            "torch.distributed.run",
            "--standalone",
            "--local-addr=127.0.0.1",
            "--nproc-per-node=2",
            "--module",
            "train",
            f"--dataset={export_dir}",
            f"--output={output_dir}",
            "--params=" + json.dumps(params),
        ],
        {**env, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        timeout=900,
    )
    assert result.returncode == 0, f"ddp train failed:\n{result.stdout[-4000:]}\n{result.stderr[-6000:]}"
    stamp = json.loads((output_dir / ".paw/baseline_train.json").read_text())
    assert stamp["world_size"] == 2
    from safetensors import safe_open

    with safe_open(Path(stamp["checkpoint_dir"]) / "model.safetensors", framework="pt") as weights:
        assert bool(weights.get_tensor("vqvae_fitted"))
        assert all(not key.startswith("module.") for key in weights.keys())


def _serve(checkpoint, env):  # noqa: F811
    with socket.socket() as probe:
        probe.bind(("", 0))
        port = probe.getsockname()[1]
    server = subprocess.Popen(
        [sys.executable, "-m", "inference.serve", f"--checkpoint={checkpoint}", f"--port={port}", "--device=cpu"],
        cwd=BASELINE,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        if server.poll() is not None:
            pytest.fail(f"server died during startup:\n{server.stdout.read()[-3000:]}")
        try:
            socket.create_connection(("localhost", port), timeout=1).close()
            return server, port
        except OSError:
            time.sleep(1)
    server.kill()
    pytest.fail("server never bound its port")


def test_checkpoint_carries_fitted_tokenizer_and_config(trained):
    output_dir, preset = trained
    stamp = json.loads((output_dir / ".paw/baseline_train.json").read_text())
    checkpoint = Path(stamp["checkpoint_dir"])
    config = json.loads((checkpoint / "config.json").read_text())
    assert config["type"] == "patch_vqbet"
    assert config["visual_feature"] == ("patch" if preset.endswith("dense") else "cls")
    assert config["n_obs_steps"] == 2 and config["obs_stride"] == 3
    from safetensors import safe_open

    with safe_open(checkpoint / "model.safetensors", framework="pt") as weights:
        assert bool(weights.get_tensor("vqvae_fitted"))
        assert any(key.startswith("encoder.model.") for key in weights.keys())


def test_serve_validate_and_stream_history(trained, env, tmp_path):  # noqa: F811
    output_dir, _ = trained
    checkpoint = json.loads((output_dir / ".paw/baseline_train.json").read_text())["checkpoint_dir"]
    server, port = _serve(checkpoint, env)
    try:
        report_path = tmp_path / "report.json"
        result = _run(
            [
                "inference.validate",
                f"--server=localhost:{port}",
                f"--checkpoint={checkpoint}",
                "--episodes=2",
                f"--output={report_path}",
            ],
            env,
            timeout=600,
        )
        assert result.returncode in (0, 1), f"validate crashed:\n{result.stdout}\n{result.stderr}"
        report = json.loads(report_path.read_text())
        assert all(probe["protocol"]["pass"] for probe in report["probes"]), report["probes"]

        # A 7-observation stream must yield chunks at observations 0, 3 and 6.
        from inference.validate import _build_observation
        from paw_policy_toolkit.policy_service import ClientMessage, ConfigureRequest, PolicyServiceStub

        channel = grpc.insecure_channel(f"localhost:{port}")
        stub = PolicyServiceStub(channel)
        stub.Configure(ConfigureRequest(model_id=checkpoint, task_instruction="task"), timeout=300)
        state = np.zeros(16)
        state[6] = state[14] = 1.0
        images = {
            f"observation.images.{slot}": np.full((3, 64, 64), 0.5, dtype=np.float32)
            for slot in ("rgb_static", "rgb_left", "rgb_right")
        }
        observations = [_build_observation(state, images, 1_000_000 + i * 33_333_333) for i in range(7)]
        for observation in observations:
            observation.chunk_requested = False
        chunks = list(stub.Run(iter([ClientMessage(observation=o) for o in observations]), timeout=60))
        channel.close()
        targets = [message.action_chunk.target_time_ns for message in chunks]
        assert targets == [observations[i].capture_time_ns for i in (0, 3, 6)]
        assert all(len(message.action_chunk.actions) == HORIZON for message in chunks)
    finally:
        server.terminate()
        server.wait(timeout=30)
