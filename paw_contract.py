"""Protected, stdlib-only command and checkpoint contract for external trainers."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

CONTRACT_VERSION = 1
TRAIN_ENTRYPOINT = ("python", "-m", "train")
COMPLETE_MARKER = "COMPLETE"
TRAINING_RESULT = "training_result.json"


@dataclass(frozen=True)
class TrainArgs:
    """Inputs supplied explicitly by the local launcher or cloud adapter."""

    dataset: Path
    output: Path
    resume_from: Path | None
    params: dict


def _params(value: str) -> dict:
    try:
        result = json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not isinstance(result, dict):
        raise argparse.ArgumentTypeError("params must be a JSON object")
    return result


def parse_args(argv: list[str]) -> TrainArgs:
    """Parse the training invocation; paths and params have no environment fallback."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--params", type=_params, required=True)
    parser.add_argument("--resume-from", type=Path)
    return TrainArgs(**vars(parser.parse_args(argv)))


def mark_checkpoint_complete(checkpoint_dir: Path) -> None:
    """Commit a local checkpoint after all trainer and serving files are closed."""
    if not checkpoint_dir.is_dir():
        raise ValueError(f"checkpoint directory does not exist: {checkpoint_dir}")
    (checkpoint_dir / COMPLETE_MARKER).touch()


def read_training_result(output: Path) -> dict:
    """Read a result naming a complete checkpoint within the run's output tree."""
    result = json.loads((output / TRAINING_RESULT).read_text())
    if not isinstance(result, dict) or set(result) != {"checkpoint_path", "config"}:
        raise ValueError("training result must contain checkpoint_path and config")
    _check_result(output, **result)
    return result


def _check_result(output: Path, *, checkpoint_path: str, config: dict) -> None:
    if not isinstance(checkpoint_path, str) or not isinstance(config, dict):
        raise ValueError("checkpoint_path must be a string and config a JSON object")
    path = PurePosixPath(checkpoint_path)
    if (
        path.as_posix() != checkpoint_path
        or "\\" in checkpoint_path
        or "\0" in checkpoint_path
    ):
        raise ValueError("checkpoint_path must be a canonical relative POSIX path")
    if path.is_absolute() or ".." in path.parts or len(path.parts) < 2:
        raise ValueError("checkpoint_path must stay inside checkpoints/<step>")
    if (
        path.parts[0] != "checkpoints"
        or not path.parts[1].isascii()
        or not path.parts[1].isdigit()
    ):
        raise ValueError("checkpoint_path must start with checkpoints/<numeric step>")
    checkpoint = output.joinpath(*path.parts)
    component = output
    for part in path.parts:
        component = component / part
        if component.is_symlink():
            raise ValueError("checkpoint_path must not contain a symlink")
    if (
        not checkpoint.resolve().is_relative_to(output.resolve())
        or not checkpoint.is_dir()
    ):
        raise ValueError(
            "checkpoint_path must name an existing directory inside output"
        )
    if not (output / path.parts[0] / path.parts[1] / COMPLETE_MARKER).is_file():
        raise ValueError("checkpoint has no COMPLETE marker")


def write_training_result(output: Path, *, checkpoint_path: str, config: dict) -> None:
    """Atomically commit the final checkpoint selection after training succeeds."""
    _check_result(output, checkpoint_path=checkpoint_path, config=config)
    temporary = output / f".{TRAINING_RESULT}.tmp"
    temporary.write_text(
        json.dumps(
            {"checkpoint_path": checkpoint_path, "config": config}, sort_keys=True
        )
        + "\n"
    )
    temporary.replace(output / TRAINING_RESULT)
