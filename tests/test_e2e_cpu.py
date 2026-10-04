"""CPU end-to-end: train on the synthetic export, serve it, validate over gRPC.

Exercises the exact commands the harness runs (``python -m train`` /
``inference.serve`` / ``inference.validate``) as subprocesses. The 2-step
checkpoint is garbage by design — the assertion is that the machinery works
and reports honestly, not that the policy is good.
"""

import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e

BASELINE = Path(__file__).resolve().parents[1]
HORIZON = 5
TRAIN_PARAMS = {
    "preset": "act",
    "overrides": {
        "steps": 2,
        "batch_size": 2,
        "policy.device": "cpu",
        "policy.chunk_size": HORIZON,
        "policy.n_action_steps": HORIZON,
        "policy.dim_model": 64,
        "policy.n_encoder_layers": 1,
        "policy.n_decoder_layers": 1,
        "policy.dim_feedforward": 128,
        "policy.use_amp": False,
        "num_workers": 0,
        "log_freq": 1,
        "save_freq": 2,
    },
}


def _run(args: list[str], env: dict, timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", *args],
        cwd=BASELINE,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


@pytest.fixture(scope="module")
def env(cache_root):
    # No WANDB_* vars: the entrypoint's own defaults must keep wandb offline,
    # or the run would demand an API key on any machine without a login.
    # These tests must remain CPU-only even on a workstation with a GPU.
    return {
        **os.environ,
        "ACCELERATE_TORCH_DEVICE": "cpu",
        "ACCELERATE_USE_CPU": "true",
        "CUDA_VISIBLE_DEVICES": "",
        "PAW_BASELINE_CACHE": str(cache_root),
    }


@pytest.fixture(scope="module", params=[1, 2])
def trained(export_dir, env, tmp_path_factory, request):
    output_dir = tmp_path_factory.mktemp("train-out")
    world_size = request.param
    modules = (
        ["train"]
        if world_size == 1
        else [
            "torch.distributed.run",
            "--standalone",
            "--local-addr=127.0.0.1",
            "--nproc-per-node=2",
            "--module",
            "train",
        ]
    )
    result = _run(
        [
            *modules,
            f"--dataset={export_dir}",
            f"--output={output_dir}",
            "--params=" + json.dumps(TRAIN_PARAMS),
        ],
        {**env, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        timeout=900,
    )
    assert result.returncode == 0, (
        f"train failed:\n{result.stdout}\n{result.stderr}"
    )
    return output_dir


def test_train_stamp_and_checkpoint(trained):
    stamp = json.loads((trained / ".paw/baseline_train.json").read_text())
    checkpoint = Path(stamp["checkpoint_dir"])
    assert stamp["model"] == "act"
    from safetensors import safe_open

    with safe_open(checkpoint / "model.safetensors", framework="pt") as weights:
        keys = weights.keys()
        assert all(not key.startswith("module.") for key in keys)
    assert stamp["jpeg_quality"] == 85
    assert json.loads((checkpoint / ".paw/image_transforms.json").read_text()) == {
        "jpeg_quality": 85
    }
    assert stamp["holdout_episodes"] == []  # 2 episodes are too few to spare any
    for artifact in ("config.json", "model.safetensors", "train_config.json"):
        assert (checkpoint / artifact).exists(), artifact


@pytest.mark.parametrize("action_steps", [None, 2, 1])
def test_serve_and_validate(trained, env, tmp_path, action_steps):
    checkpoint = json.loads((trained / ".paw/baseline_train.json").read_text())[
        "checkpoint_dir"
    ]
    source = Path(checkpoint)
    source_config = (source / "config.json").read_bytes()
    source_probes = (source / ".paw/validation_probes.npz").read_bytes()
    expected_horizon = HORIZON if action_steps is None else action_steps
    if action_steps is not None:
        checkpoint = str(tmp_path / "variant")
        result = _run(
            [
                "paw_baseline.variant",
                f"--checkpoint={source}",
                f"--output={checkpoint}",
                "--config="
                + json.dumps({"schema_version": 1, "n_action_steps": action_steps}),
            ],
            env,
            timeout=120,
        )
        assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
        assert (Path(checkpoint) / "config.json").read_bytes() == source_config
        assert (
            Path(checkpoint) / ".paw/validation_probes.npz"
        ).read_bytes() == source_probes
    with socket.socket() as probe:
        probe.bind(("", 0))
        port = probe.getsockname()[1]

    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "inference.serve",
            f"--checkpoint={checkpoint}",
            f"--port={port}",
            "--device=cpu",
        ],
        cwd=BASELINE,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            if server.poll() is not None:
                pytest.fail(
                    f"server died during startup:\n{server.stdout.read()[-3000:]}"
                )
            try:
                socket.create_connection(("localhost", port), timeout=1).close()
                break
            except OSError:
                time.sleep(1)
        else:
            pytest.fail("server never bound its port")

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
        # Exit 1 (hard checks failing) is expected for a 2-step model; anything
        # else means the machinery itself broke.
        assert result.returncode in (0, 1), (
            f"validate crashed:\n{result.stdout}\n{result.stderr}"
        )

        assert report_path.is_file(), (
            f"validate produced no report:\n{result.stdout}\n{result.stderr}"
        )
        report = json.loads(report_path.read_text())
        assert report["horizon"] == expected_horizon
        assert report["native_action_steps"] == HORIZON
        assert report["probe_pool"] == "all"  # no holdout on the tiny fixture
        assert len(report["probes"]) == 6  # 2 episodes x 3 anchors
        for probe_result in report["probes"]:
            assert probe_result["protocol"]["actions"] == expected_horizon
            assert probe_result["protocol"]["horizon_matches"]
            assert probe_result["protocol"]["target_time_echoed"]
            assert probe_result["latency_s"] > 0
        assert report["aggregate"]["hard_pass"] == (result.returncode == 0)
        assert (source / "config.json").read_bytes() == source_config
        assert (source / ".paw/validation_probes.npz").read_bytes() == source_probes
    finally:
        server.terminate()
        server.wait(timeout=30)


def test_resume_at_final_step_publishes_without_last_pointer(
    trained, export_dir, env, tmp_path
):
    output = tmp_path / "restored"
    shutil.copytree(trained, output, symlinks=True)
    stamp = json.loads((trained / ".paw/baseline_train.json").read_text())
    relative = Path(stamp["checkpoint_dir"]).relative_to(trained)
    checkpoint = output / relative
    weights = checkpoint / "model.safetensors"
    before = hashlib.sha256(weights.read_bytes()).hexdigest()
    (output / "checkpoints/last").unlink()
    (output / "training_result.json").unlink()
    result = _run(
        [
            "train",
            f"--dataset={export_dir}",
            f"--output={output}",
            f"--resume-from={checkpoint.parent}",
            "--params=" + json.dumps(TRAIN_PARAMS),
        ],
        env,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    assert hashlib.sha256(weights.read_bytes()).hexdigest() == before
    assert (checkpoint.parent / "COMPLETE").is_file()
    assert (
        json.loads((output / "training_result.json").read_text())["checkpoint_path"]
        == relative.as_posix()
    )
