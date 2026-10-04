"""Portable validation retains real observation and ground-truth arrays."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from inference import validate


def test_portable_probes_preserve_arrays_and_holdout(tmp_path, monkeypatch):
    sample = {
        "probe": validate.Probe(7, 0.5, 12, 112, "pick up"),
        "state": np.arange(16, dtype=np.float64),
        "truth": np.arange(32, dtype=np.float64).reshape(2, 16),
        "images": {"observation.images.top": np.ones((3, 4, 4), dtype=np.float32)},
    }
    bbox = (np.zeros(16), np.ones(16))
    monkeypatch.setattr(validate, "dataset_probes", lambda *args: ([sample], bbox))
    config = {
        "input_features": {"observation.images.top": {"type": "VISUAL"}},
        "n_action_steps": 2,
    }
    path = tmp_path / "validation_probes.npz"
    path.write_bytes(validate.make_validation_bundle(tmp_path, config, {7}))
    samples, loaded_bbox, pool = validate.load_validation_bundle(
        path, ["observation.images.top"], 2, 3
    )
    assert pool == "holdout"
    assert samples[0]["probe"] == sample["probe"]
    for key in ("state", "truth"):
        np.testing.assert_array_equal(samples[0][key], sample[key])
    np.testing.assert_array_equal(
        samples[0]["images"]["observation.images.top"],
        sample["images"]["observation.images.top"],
    )
    np.testing.assert_array_equal(loaded_bbox, bbox)
    with pytest.raises(ValueError, match="match"):
        validate.load_validation_bundle(path, ["observation.images.top"], 3, 3)


@pytest.mark.parametrize("portable", [False, True])
@pytest.mark.parametrize("action_steps", [1, 2])
@pytest.mark.parametrize(
    "timeout_args,configure_timeout,run_timeout",
    [
        ([], 900, 5),
        (["--configure-timeout=60"], 60, 5),
        (["--timeout=2"], 900, 2),
    ],
)
def test_shortened_variant_validates_native_probes(
    tmp_path,
    monkeypatch,
    portable,
    action_steps,
    timeout_args,
    configure_timeout,
    run_timeout,
):
    from paw_policy_toolkit.action import DefaultActionTranslator

    from paw_baseline.layout import WIRE_FROM_TRAIN

    config = {
        "type": "act",
        "device": "cpu",
        "chunk_size": 5,
        "n_action_steps": 5,
        "input_features": {
            "observation.images.rgb_static": {"type": "VISUAL", "shape": [3, 4, 4]}
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / ".paw").mkdir()
    (tmp_path / ".paw/inference.json").write_text(
        json.dumps({"schema_version": 1, "n_action_steps": action_steps})
    )
    truth = np.zeros((5, 16), dtype=np.float64)
    truth[:, [6, 14]] = 1
    truth[:, [7, 15]] = 0.04
    # The native tail reverses direction; only the returned prefix is relevant.
    truth[:, 0] = [0.3, 0.31, 0.29, 0.28, 0.27]
    sample = {
        "probe": validate.Probe(7, 0.5, 12, 112, "pick up"),
        "state": truth[0].copy(),
        "truth": truth,
        "images": {
            "observation.images.rgb_static": np.ones((3, 4, 4), dtype=np.float32)
        },
    }
    bbox = truth.min(axis=0), truth.max(axis=0)

    def dataset_probes(root, cameras, horizon, count, holdout):
        assert horizon == 5
        return [sample], bbox

    monkeypatch.setattr(validate, "dataset_probes", dataset_probes)
    bundle = tmp_path / ".paw/validation_probes.npz"
    original_bundle = validate.make_validation_bundle(tmp_path, config, set())
    bundle.write_bytes(original_bundle)
    monkeypatch.setattr(validate, "_resolve_dataset_root", lambda *args: tmp_path)
    monkeypatch.setattr(
        validate, "load_serving_metadata", lambda checkpoint: {"frozen_arm": None}
    )

    def configure(request, *, timeout, metadata):
        assert timeout == configure_timeout
        assert metadata == (("x-paw-model-id", request.model_id),)

    monkeypatch.setattr(
        validate,
        "PolicyServiceStub",
        lambda channel: SimpleNamespace(Configure=configure),
    )

    def request_chunk(stub, observation, timeout):
        assert timeout == run_timeout
        wire = truth[:action_steps, list(WIRE_FROM_TRAIN)][None]
        chunk = DefaultActionTranslator().from_dict({"action": wire})
        chunk.target_time_ns = observation.capture_time_ns
        return chunk, 0.001

    monkeypatch.setattr(validate, "_request_chunk", request_chunk)
    output = tmp_path / "report.json"
    argv = [
        "validate",
        "--server=localhost:1",
        f"--checkpoint={tmp_path}",
        f"--output={output}",
        *timeout_args,
    ]
    if not portable:
        argv.append("--dataset=fixture")
    monkeypatch.setattr(validate.sys, "argv", argv)
    with pytest.raises(SystemExit) as exited:
        validate.main()
    assert exited.value.code == 0
    report = json.loads(output.read_text())
    assert report["horizon"] == action_steps
    assert report["native_action_steps"] == 5
    assert report["replan_budget_s"] == action_steps / validate.FPS / 2
    assert report["probes"][0]["protocol"]["horizon_matches"]
    assert report["probes"][0]["plausibility"]["truth_path_m"] == pytest.approx(
        0.01 if action_steps == 2 else 0
    )
    if action_steps == 1:
        assert report["aggregate"]["plausibility"]["magnitude_ok"]["decidable"] == 0
    assert bundle.read_bytes() == original_bundle
    assert sample["truth"].shape == (5, 16)
