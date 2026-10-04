"""CPU end-to-end for the CSP preset: train, serve over gRPC, validate."""

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.test_e2e_cpu import BASELINE, _run

pytestmark = pytest.mark.e2e

HORIZON = 6
TRAIN_PARAMS = {
    "preset": "csp",
    "overrides": {
        "steps": 2,
        "batch_size": 2,
        "policy.device": "cpu",
        "policy.horizon": HORIZON,
        "policy.n_action_steps": HORIZON,
        "policy.n_low_freq": 2,
        "policy.n_layer": 1,
        "policy.pretrained_backbone_weights": None,
        "num_workers": 0,
        "log_freq": 1,
        "save_freq": 2,
    },
}


@pytest.fixture(scope="module")
def env(cache_root):
    return {
        **os.environ,
        "ACCELERATE_TORCH_DEVICE": "cpu",
        "ACCELERATE_USE_CPU": "true",
        "CUDA_VISIBLE_DEVICES": "",
        "PAW_BASELINE_CACHE": str(cache_root),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }


def test_csp_train_serve_validate(export_dir, env, tmp_path):
    output = tmp_path / "train-out"
    result = _run(
        [
            "train",
            f"--dataset={export_dir}",
            f"--output={output}",
            "--params=" + json.dumps(TRAIN_PARAMS),
        ],
        env,
        timeout=900,
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    stamp = json.loads((output / ".paw/baseline_train.json").read_text())
    checkpoint = Path(stamp["checkpoint_dir"])
    config = json.loads((checkpoint / "config.json").read_text())
    assert config["type"] == "csp"
    assert config["n_low_freq"] == 2
    assert config["action_feature_names"][9].endswith("gripper_width")

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
                pytest.fail(f"server died:\n{server.stdout.read()[-3000:]}")
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
        assert result.returncode in (0, 1), result.stdout + result.stderr
        report = json.loads(report_path.read_text())
        assert report["horizon"] == HORIZON
        for probe_result in report["probes"]:
            assert probe_result["protocol"]["actions"] == HORIZON
            assert probe_result["protocol"]["horizon_matches"]
            assert probe_result["protocol"]["target_time_echoed"]
    finally:
        server.terminate()
        server.wait(timeout=30)
