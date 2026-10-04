import json

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from paw_baseline.prepare import LAYOUT_NAME
from paw_baseline.representation import MODEL_ARM_DIMS, ROTATION_CONVENTION
from paw_baseline.serving_metadata import (
    SERVING_METADATA_PATH,
    SERVING_SCHEMA_VERSION,
    load_serving_metadata,
    write_serving_metadata,
)


def _checkpoint(tmp_path, frozen_arm):
    checkpoint = tmp_path / "pretrained_model"
    checkpoint.mkdir()
    metadata = {
        "schema_version": SERVING_SCHEMA_VERSION,
        "layout": LAYOUT_NAME,
        "rotation_convention": ROTATION_CONVENTION,
        "action_dim": 20,
        "frozen_arm": frozen_arm,
        "model": "act",
        "dataset_repo_id": "test/export",
        "dataset_revision": "abc",
    }
    write_serving_metadata(checkpoint, metadata)
    state_file = "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
    (checkpoint / "policy_postprocessor.json").write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "registry_name": "unnormalizer_processor",
                        "state_file": state_file,
                    }
                ]
            }
        )
    )
    mean = np.linspace(-0.2, 0.2, 20, dtype=np.float32)
    arrays = {
        "action.mean": mean.copy(),
        "action.min": mean - 0.2,
        "action.max": mean + 0.2,
        "action.std": np.full(20, 0.2, dtype=np.float32),
        "action.q01": mean - 0.1,
        "action.q99": mean + 0.1,
    }
    if frozen_arm is not None:
        dims = list(MODEL_ARM_DIMS[frozen_arm])
        arrays["action.min"][dims] = mean[dims] - 0.5
        arrays["action.max"][dims] = mean[dims] + 0.5
        arrays["action.std"][dims] = 1.0
        arrays["action.q01"][dims] = mean[dims] - 0.5
        arrays["action.q99"][dims] = mean[dims] + 0.5
    save_file(
        {key: torch.from_numpy(value) for key, value in arrays.items()},
        checkpoint / state_file,
    )
    return checkpoint


@pytest.mark.parametrize("frozen_arm", [None, "left", "right"])
def test_load_serving_metadata_cross_checks_normalizer(tmp_path, frozen_arm):
    checkpoint = _checkpoint(tmp_path, frozen_arm)
    assert load_serving_metadata(checkpoint)["frozen_arm"] == frozen_arm


def test_load_serving_metadata_fails_closed_when_missing(tmp_path):
    checkpoint = tmp_path / "pretrained_model"
    checkpoint.mkdir()
    with pytest.raises(RuntimeError, match="refusing to serve"):
        load_serving_metadata(checkpoint)


def test_load_serving_metadata_rejects_stats_disagreement(tmp_path):
    checkpoint = _checkpoint(tmp_path, "left")
    metadata_path = checkpoint / SERVING_METADATA_PATH
    metadata = json.loads(metadata_path.read_text())
    metadata["frozen_arm"] = "right"
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(RuntimeError, match="disagree"):
        load_serving_metadata(checkpoint)


@pytest.mark.parametrize(
    "convention",
    [None, "largest-component-positive-xyzw-v1", "episode-continuous", "unknown"],
)
def test_rejects_incompatible_quaternion_contract(tmp_path, convention):
    checkpoint = _checkpoint(tmp_path, None)
    path = checkpoint / SERVING_METADATA_PATH
    metadata = json.loads(path.read_text())
    if convention is None:
        metadata.pop("rotation_convention")
    else:
        metadata["rotation_convention"] = convention
    path.write_text(json.dumps(metadata))
    with pytest.raises(RuntimeError, match="incompatible rotation convention"):
        load_serving_metadata(checkpoint)


def test_resume_rejects_old_contract_before_preparing_data(tmp_path, monkeypatch):
    import sys

    from train import __main__ as train

    checkpoint = _checkpoint(tmp_path, None)
    path = checkpoint / SERVING_METADATA_PATH
    metadata = json.loads(path.read_text())
    metadata.pop("rotation_convention")
    path.write_text(json.dumps(metadata))
    (tmp_path / "COMPLETE").touch()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--dataset",
            str(tmp_path),
            "--output",
            str(tmp_path / "out"),
            "--params",
            '{"preset":"act"}',
            "--resume-from",
            str(tmp_path),
        ],
    )
    monkeypatch.setattr(
        train,
        "prepare_dataset",
        lambda *a, **kw: pytest.fail("prepared data before checking resume contract"),
    )
    with pytest.raises(RuntimeError, match="incompatible rotation convention"):
        train.main()
