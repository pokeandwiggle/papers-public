"""Checkpoint metadata required for baseline serving."""

import json
from pathlib import Path

import numpy as np
from safetensors.torch import load_file

from paw_baseline.prepare import DELTA_LAYOUT_NAME, LAYOUT_NAME, PROVENANCE_PATH
from paw_baseline.relative import (
    ACTION_REPRESENTATION as DELTA_REPRESENTATION,
)
from paw_baseline.relative import DELTA_ARM_DIMS, DELTA_DIM
from paw_baseline.representation import MODEL_ARM_DIMS, MODEL_DIM, ROTATION_CONVENTION

SERVING_METADATA_PATH = Path(".paw/serving.json")
SERVING_SCHEMA_VERSION = 3
# Relative-action checkpoints integrate deltas from the commanded pose and
# ground chunks to start on arrival (synchronous fixed-horizon execution).
DELTA_SERVING_SCHEMA_VERSION = 5


def is_delta_metadata(metadata: dict) -> bool:
    return metadata.get("action_representation") == DELTA_REPRESENTATION


def metadata_from_prepared(prepared_root: Path, *, model: str) -> dict:
    provenance_path = prepared_root / PROVENANCE_PATH
    if not provenance_path.exists():
        raise RuntimeError(f"prepared dataset has no provenance: {provenance_path}")
    provenance = json.loads(provenance_path.read_text())
    if "pi05" in provenance:
        from paw_baseline.pi05.geometry import (
            LAYOUT,
            Pi05Representation,
        )
        from paw_baseline.pi05.geometry import (
            ROTATION_CONVENTION as PI_ROTATION,
        )

        representation = Pi05Representation.from_dict(
            provenance["pi05"]["representation"]
        )
        return {
            "schema_version": 4,
            "layout": LAYOUT,
            "rotation_convention": PI_ROTATION,
            "action_dim": MODEL_DIM,
            "frozen_arm": provenance["frozen_arm"],
            "model": model,
            "dataset_repo_id": provenance["source_repo_id"],
            "dataset_revision": provenance["source_revision"],
            "pi05": {**provenance["pi05"], "representation": representation.to_dict()},
        }
    _require_rotation_convention(provenance)
    frozen_arm = provenance.get("frozen_arm")
    if provenance.get("action_representation") == DELTA_REPRESENTATION:
        if provenance.get("layout") != DELTA_LAYOUT_NAME:
            raise RuntimeError("relative-action dataset has an unexpected layout")
        if frozen_arm not in (None, "left", "right"):
            raise RuntimeError(f"prepared dataset has invalid frozen_arm {frozen_arm!r}")
        return {
            "schema_version": DELTA_SERVING_SCHEMA_VERSION,
            "layout": DELTA_LAYOUT_NAME,
            "action_representation": DELTA_REPRESENTATION,
            "rotation_convention": ROTATION_CONVENTION,
            "action_dim": DELTA_DIM,
            "frozen_arm": frozen_arm,
            "model": model,
            "dataset_repo_id": provenance["source_repo_id"],
            "dataset_revision": provenance["source_revision"],
        }
    if provenance.get("layout") != LAYOUT_NAME:
        raise RuntimeError(
            f"prepared dataset layout is {provenance.get('layout')!r}, expected {LAYOUT_NAME!r}"
        )
    if frozen_arm not in (None, "left", "right"):
        raise RuntimeError(f"prepared dataset has invalid frozen_arm {frozen_arm!r}")
    return {
        "schema_version": SERVING_SCHEMA_VERSION,
        "layout": LAYOUT_NAME,
        "rotation_convention": ROTATION_CONVENTION,
        "action_dim": MODEL_DIM,
        "frozen_arm": frozen_arm,
        "model": model,
        "dataset_repo_id": provenance["source_repo_id"],
        "dataset_revision": provenance["source_revision"],
    }


def write_serving_metadata(pretrained_dir: Path, metadata: dict) -> Path:
    path = pretrained_dir / SERVING_METADATA_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2) + "\n")
    return path


def _normalizer_state_file(checkpoint: Path) -> Path:
    config_path = checkpoint / "policy_postprocessor.json"
    if not config_path.exists():
        raise RuntimeError(
            f"checkpoint lacks saved postprocessor config: {config_path}"
        )
    config = json.loads(config_path.read_text())
    matches = [
        step.get("state_file")
        for step in config.get("steps", [])
        if step.get("registry_name") == "unnormalizer_processor"
    ]
    if len(matches) != 1 or not matches[0]:
        raise RuntimeError(
            "checkpoint must contain exactly one saved unnormalizer state"
        )
    return checkpoint / matches[0]


def _squashed_arms(
    checkpoint: Path, action_dim: int = MODEL_DIM, arm_dims=MODEL_ARM_DIMS
) -> set[str]:
    """Recognize prepare's deliberate mean +/- 0.5 frozen-arm fingerprint."""
    state_path = _normalizer_state_file(checkpoint)
    if not state_path.exists():
        raise RuntimeError(f"checkpoint lacks saved unnormalizer state: {state_path}")
    tensors = load_file(state_path)
    required = (
        "action.mean",
        "action.min",
        "action.max",
        "action.std",
        "action.q01",
        "action.q99",
    )
    missing = [key for key in required if key not in tensors]
    if missing:
        raise RuntimeError(f"saved unnormalizer lacks action statistics: {missing}")
    values = {key: tensors[key].detach().cpu().numpy() for key in required}
    if any(value.shape != (action_dim,) for value in values.values()):
        shapes = {key: value.shape for key, value in values.items()}
        raise RuntimeError(f"saved action statistics have unexpected shapes: {shapes}")

    result = set()
    for arm, dims_tuple in arm_dims.items():
        dims = list(dims_tuple)
        mean = values["action.mean"][dims]
        fingerprint = (
            np.allclose(values["action.min"][dims], mean - 0.5, atol=1e-6)
            and np.allclose(values["action.max"][dims], mean + 0.5, atol=1e-6)
            and np.allclose(values["action.q01"][dims], mean - 0.5, atol=1e-6)
            and np.allclose(values["action.q99"][dims], mean + 0.5, atol=1e-6)
            and np.allclose(values["action.std"][dims], 1.0, atol=1e-6)
        )
        if fingerprint:
            result.add(arm)
    return result


def _require_rotation_convention(metadata: dict) -> None:
    if metadata.get("rotation_convention") != ROTATION_CONVENTION:
        raise RuntimeError(
            "incompatible rotation convention; retrain with the current baseline "
            "or use the checkpoint's original runtime; do not relabel old checkpoints"
        )


def load_serving_metadata(checkpoint: Path) -> dict:
    """Require the checkpoint's input representation and frozen-arm statistics."""
    path = checkpoint / SERVING_METADATA_PATH
    if not path.exists():
        raise RuntimeError(
            f"checkpoint lacks {SERVING_METADATA_PATH}; refusing to serve without frozen-arm metadata"
        )
    metadata = json.loads(path.read_text())
    if "pi05" in metadata:
        from paw_baseline.pi05.geometry import (
            LAYOUT,
            Pi05Representation,
        )
        from paw_baseline.pi05.geometry import (
            ROTATION_CONVENTION as PI_ROTATION,
        )

        Pi05Representation.from_dict(metadata["pi05"]["representation"])
        if (
            metadata.get("schema_version") != 4
            or metadata.get("layout") != LAYOUT
            or metadata.get("rotation_convention") != PI_ROTATION
            or metadata.get("action_dim") != MODEL_DIM
        ):
            raise RuntimeError("incompatible PI05 serving metadata")
        native = json.loads((checkpoint / "config.json").read_text())
        if (
            native.get("type") != "paw_pi05"
            or native.get("paw_representation") != metadata["pi05"]["representation"]
            or native.get("chunk_size") != metadata["pi05"]["chunk_size"]
        ):
            raise RuntimeError("PI05 model configuration and serving metadata disagree")
        expected = set() if metadata["frozen_arm"] is None else {metadata["frozen_arm"]}
        if _squashed_arms(checkpoint) != expected:
            raise RuntimeError("PI05 frozen-arm metadata/statistics disagree")
        return metadata
    _require_rotation_convention(metadata)
    delta = is_delta_metadata(metadata)
    schema, layout, action_dim = (
        (DELTA_SERVING_SCHEMA_VERSION, DELTA_LAYOUT_NAME, DELTA_DIM)
        if delta
        else (SERVING_SCHEMA_VERSION, LAYOUT_NAME, MODEL_DIM)
    )
    if metadata.get("schema_version") != schema:
        raise RuntimeError(
            f"unsupported serving metadata schema: {metadata.get('schema_version')!r}"
        )
    if metadata.get("layout") != layout or metadata.get("action_dim") != action_dim:
        raise RuntimeError(
            f"serving metadata layout mismatch: {metadata.get('layout')!r} / {metadata.get('action_dim')!r}"
        )
    frozen_arm = metadata.get("frozen_arm")
    if frozen_arm not in (None, "left", "right"):
        raise RuntimeError(f"invalid checkpoint frozen_arm {frozen_arm!r}")
    expected = set() if frozen_arm is None else {frozen_arm}
    derived = _squashed_arms(
        checkpoint, action_dim, DELTA_ARM_DIMS if delta else MODEL_ARM_DIMS
    )
    if derived != expected:
        raise RuntimeError(
            f"frozen-arm metadata/statistics disagree: metadata={frozen_arm!r}, stats={sorted(derived)}"
        )
    return metadata
