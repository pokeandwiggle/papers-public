"""Resolve nominal epoch budgets against the selected training frames."""

import math
from pathlib import Path

import pyarrow.parquet as pq


def epoch_step_args(
    preset: dict,
    effective_args: list[str],
    dataset: Path,
    episodes: tuple[int, ...],
    *,
    resume: bool = False,
    world_size: int = 1,
) -> list[str]:
    """Derive steps for epoch presets; explicit steps and saved resume targets win.

    Presets declare epochs and an optional nonnegative integer min_steps floor.
    Resolve max(min_steps, ceil(epochs * selected_frames / global_batch)) once,
    then pass ordinary --steps to LeRobot without changing its training loop.
    Epochs count anchor frames, not the action slots predicted for each anchor.
    The result is a total optimizer-step target, not additional resume steps.
    """
    args = dict(arg[2:].split("=", 1) for arg in effective_args)
    if "epochs" not in preset or "steps" in args or resume:
        return []
    epochs = preset["epochs"]
    min_steps = preset.get("min_steps", 0)
    if type(min_steps) is not int or min_steps < 0:
        raise ValueError("min_steps must be a nonnegative integer")
    global_batch = int(args["batch_size"]) * world_size
    if not math.isfinite(epochs) or epochs <= 0 or global_batch <= 0:
        raise ValueError("epochs and global batch must be positive")
    lengths = {
        row["episode_index"]: row["length"]
        for path in sorted((dataset / "meta/episodes").glob("chunk-*/file-*.parquet"))
        for row in pq.read_table(path, columns=["episode_index", "length"]).to_pylist()
    }
    if not episodes or any(lengths.get(index, 0) <= 0 for index in episodes):
        raise ValueError("selected training episode lengths must exist and be positive")
    frames = sum(lengths[index] for index in episodes)
    steps = max(min_steps, math.ceil(epochs * frames / global_batch))
    return [f"--steps={steps}"]
