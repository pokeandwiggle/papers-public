r"""Prepare an inference-only checkpoint package, then validate/register normally.

Examples:
    python -m paw_baseline.variant --checkpoint CHECKPOINT --output VARIANT \
        --config '{"schema_version": 1, "n_action_steps": 8}'
    python -m paw_baseline.variant --checkpoint CHECKPOINT --output VARIANT \
        --config '{"schema_version": 1, "num_integration_steps": 5}'

Config is a JSON object with schema_version=1 and at least one positive integer:
  n_action_steps: actions emitted per plan, supported by ACT, Diffusion and DiT.
    Range: 1 through the original config.json n_action_steps. The trained
    prediction horizon stays fixed; the station may skip or replace actions,
    so emitted length is not a guarantee of physically executed steps.
  num_integration_steps: sampler steps for flow-matching DiT.
  num_inference_steps: sampler steps for Diffusion or diffusion-objective DiT;
    cannot exceed config.json num_train_timesteps (default: 100).
Use at most one sampler field, optionally together with n_action_steps.

Config replaces the complete inference sidecar. Omitted fields use config.json
values, even when the source is already a variant; source overrides are not merged.
The native config, weights, auxiliary files and portable probes are preserved.
The output must be a new directory outside the source. Source validation and
registration receipts are removed from the copy: validate and register it anew.
This command checks settings and package structure; it does not run validation.
It prints the new package path on success.
"""

import argparse
import dataclasses
import json
import shutil
from pathlib import Path

INFERENCE_PATH = Path(".paw/inference.json")
SUPPORTED_POLICIES = ("diffusion", "multi_task_dit")
_LOCAL_IDENTITY_PATHS = (
    Path(".paw/model_registration.json"),
    Path(".paw/validation_receipt.json"),
)


@dataclasses.dataclass(frozen=True)
class InferenceConfig:
    """Native inference overrides; omitted fields keep checkpoint defaults."""

    schema_version: int
    n_action_steps: int | None = None
    num_inference_steps: int | None = None
    num_integration_steps: int | None = None

    def settings(self) -> dict:
        """Return exactly the configured native fields, excluding absent ones."""
        return {
            key: value
            for key, value in dataclasses.asdict(self).items()
            if value is not None
        }


def parse_inference_config(value: object) -> InferenceConfig:
    """Parse the complete v1 sidecar without silently accepting unused settings."""
    fields = {field.name for field in dataclasses.fields(InferenceConfig)}
    if (
        not isinstance(value, dict)
        or "schema_version" not in value
        or not set(value) <= fields
        or len(value) < 2
        or {"num_inference_steps", "num_integration_steps"} <= set(value)
    ):
        raise ValueError(
            "inference config requires schema_version and n_action_steps and/or one native sampler step count"
        )
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("inference config schema_version must be 1")
    for field, steps in value.items():
        if field != "schema_version" and (type(steps) is not int or steps <= 0):
            raise ValueError(f"{field} must be a positive integer")
    return InferenceConfig(**value)


def _require_supported_policy(native: dict, settings: InferenceConfig) -> None:
    policy_type = native["type"]
    if settings.n_action_steps is not None:
        if policy_type not in ("act", *SUPPORTED_POLICIES):
            raise ValueError(f"n_action_steps is not supported for {policy_type!r}")
        native_steps = native.get("n_action_steps")
        if type(native_steps) is not int or native_steps <= 0:
            raise ValueError("checkpoint n_action_steps must be a positive integer")
        if settings.n_action_steps > native_steps:
            raise ValueError("n_action_steps cannot exceed checkpoint n_action_steps")
    if settings.num_inference_steps is None and settings.num_integration_steps is None:
        return
    if policy_type not in SUPPORTED_POLICIES:
        raise ValueError(
            f"sampler steps support only {SUPPORTED_POLICIES}, got {policy_type!r}"
        )
    flow_matching = (
        policy_type == "multi_task_dit"
        and native.get("objective", "diffusion") == "flow_matching"
    )
    field = "num_integration_steps" if flow_matching else "num_inference_steps"
    if getattr(settings, field) is None:
        raise ValueError(
            f"{policy_type} objective {native.get('objective', 'diffusion')!r} requires {field}"
        )
    if not flow_matching and settings.num_inference_steps > native.get(
        "num_train_timesteps", 100
    ):
        raise ValueError("num_inference_steps cannot exceed num_train_timesteps")


def load_policy_config(checkpoint: Path):
    """Load native config and apply supported settings before policy construction."""
    import lerobot.policies  # noqa: F401 — registers native configuration types.
    from lerobot.configs.policies import PreTrainedConfig

    policy_type = json.loads((checkpoint / "config.json").read_text())["type"]
    if policy_type == "paw_pi05":
        from paw_baseline.pi05 import (  # noqa: F401
            configuration_paw_pi05,
            processor_paw_pi05,
        )
    if policy_type == "artok":
        from paw_baseline.artok import (  # noqa: F401
            configuration_artok,
            processor_artok,
        )

    config = PreTrainedConfig.from_pretrained(checkpoint)
    path = checkpoint / INFERENCE_PATH
    if path.exists():
        settings = parse_inference_config(json.loads(path.read_text()))
        _require_supported_policy(
            json.loads((checkpoint / "config.json").read_text()), settings
        )
        for field, value in settings.settings().items():
            if field != "schema_version":
                setattr(config, field, value)
    return config


def build_variant(checkpoint: Path, output: Path, config: object) -> Path:
    """Copy a package with fresh local identity and explicit inference settings.

    Auxiliary files and portable validation probes travel with the weights.
    Existing output paths are refused; source packages are never modified.
    """
    settings = parse_inference_config(config)
    checkpoint = checkpoint.resolve()
    output = output.resolve()
    if output == checkpoint or checkpoint in output.parents:
        raise ValueError("variant output must be outside the source checkpoint")
    native = json.loads((checkpoint / "config.json").read_text())
    _require_supported_policy(native, settings)
    if not (checkpoint / "model.safetensors").is_file():
        raise ValueError("source checkpoint has no model.safetensors weights")
    if output.exists():
        raise FileExistsError(output)
    shutil.copytree(checkpoint, output)
    for relative in _LOCAL_IDENTITY_PATHS:
        (output / relative).unlink(missing_ok=True)
    path = output / INFERENCE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings.settings(), indent=2) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="source saved policy directory"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new variant directory outside the source",
    )
    parser.add_argument(
        "--config",
        type=json.loads,
        required=True,
        help="complete inference overrides JSON (fields and examples above)",
    )
    args = parser.parse_args()
    print(build_variant(args.checkpoint, args.output, args.config))


if __name__ == "__main__":
    main()
