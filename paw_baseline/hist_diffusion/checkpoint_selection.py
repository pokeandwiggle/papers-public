"""Score every checkpoint by held-out DDIM action MSE.

The paper (App. B.2, A.3, I) selects checkpoints by validation DDIM MSE. Every
saved checkpoint is scored on fixed anchors from held-out episodes:
DDIM-sampled action sequences (inference step count, fixed noise) against
demonstrated actions, in normalized action space, over the full denoised
sequence as in the reference implementation. The cloud runner retains only
the final checkpoint, so every arm serves its final step; the scores are
recorded so the training trajectory can be reported.
"""

import json
import logging
from pathlib import Path

import numpy as np
import torch
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

SELECTION_PATH = Path(".paw/checkpoint_selection.json")
SCORE_PATH = Path(".paw/validation_score.json")
ANCHORS_PER_EPISODE = 32
BATCH_SIZE = 32
SEED = 0

logger = logging.getLogger(__name__)


class ValidationSelector:
    """Score checkpoints on held-out episodes and remember the best one."""

    def __init__(self, prepared_root: Path, holdout: list[int]):
        if not holdout:
            raise ValueError("checkpoint selection requires held-out episodes")
        self.root = Path(prepared_root)
        self.holdout = sorted(holdout)
        self.scores: list[dict] = []
        self._batch: dict[str, torch.Tensor] | None = None

    def _load_anchors(self, config) -> dict[str, torch.Tensor]:
        from lerobot.datasets.factory import resolve_delta_timestamps
        from lerobot.datasets.lerobot_dataset import (
            LeRobotDataset,
            LeRobotDatasetMetadata,
        )

        from paw_baseline.prepare import VIDEO_BACKEND

        meta = LeRobotDatasetMetadata(self.root.name, root=self.root)
        dataset = LeRobotDataset(
            self.root.name,
            root=self.root,
            delta_timestamps=resolve_delta_timestamps(config, meta),
            video_backend=VIDEO_BACKEND,
        )
        keys = [OBS_STATE, *config.image_features, ACTION]
        items = []
        for episode in self.holdout:
            row = meta.episodes[episode]
            start, length = int(row["dataset_from_index"]), int(row["length"])
            frames = np.linspace(0, length - 1, min(ANCHORS_PER_EPISODE, length))
            for frame in np.unique(frames.round().astype(int)):
                item = dataset[start + int(frame)]
                items.append({key: item[key] for key in keys})
        return {key: torch.stack([item[key] for item in items]) for key in keys}

    @torch.no_grad()
    def score(self, checkpoint_dir: Path, step: int, policy, preprocessor) -> float:
        """Score one checkpoint, record it beside its weights, and return the MSE."""
        config = policy.config
        if self._batch is None:
            self._batch = self._load_anchors(config)
        model = policy.diffusion
        unet = getattr(model.unet, "_orig_mod", model.unet)  # eager, no recompiles
        scheduler = model.noise_scheduler
        device = next(policy.parameters()).device
        generator = torch.Generator(device=device).manual_seed(SEED)
        was_training = policy.training
        policy.eval()
        errors = []
        try:
            total = len(self._batch[ACTION])
            for begin in range(0, total, BATCH_SIZE):
                batch = {
                    key: value[begin : begin + BATCH_SIZE]
                    for key, value in self._batch.items()
                }
                for key in config.image_features:
                    if batch[key].dtype == torch.uint8:
                        batch[key] = batch[key].to(torch.float32) / 255.0
                batch = preprocessor(batch)
                batch[OBS_IMAGES] = torch.stack(
                    [batch[key] for key in config.image_features], dim=-4
                )
                global_cond = model._prepare_global_conditioning(batch)
                target = batch[ACTION]
                sample = torch.randn(
                    target.shape, generator=generator, device=device, dtype=target.dtype
                )
                scheduler.set_timesteps(model.num_inference_steps)
                for t in scheduler.timesteps:
                    timestep = torch.full(
                        sample.shape[:1], t, dtype=torch.long, device=device
                    )
                    output = unet(sample, timestep, global_cond=global_cond)
                    sample = scheduler.step(
                        output, t, sample, generator=generator
                    ).prev_sample
                errors.append(((sample - target) ** 2).mean(dim=(1, 2)).cpu())
        finally:
            policy.train(was_training)
        mse = float(torch.cat(errors).mean())
        record = {
            "step": step,
            "checkpoint_dir": str(checkpoint_dir),
            "val_ddim_mse": mse,
            "anchors": int(len(self._batch[ACTION])),
            "holdout_episodes": self.holdout,
        }
        self.scores.append(record)
        path = Path(checkpoint_dir) / "pretrained_model" / SCORE_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2) + "\n")
        logger.info("checkpoint step %d val_ddim_mse=%.6f", step, mse)
        print(f"checkpoint step {step} val_ddim_mse={mse:.6f}", flush=True)
        return mse

    def write(self, output_dir: Path, served: Path) -> dict:
        """Record this process's scores and the served (final) checkpoint."""
        scores = sorted(self.scores, key=lambda record: record["step"])
        served_scores = [s for s in scores if Path(s["checkpoint_dir"]) == served]
        best = min(scores, key=lambda record: record["val_ddim_mse"]) if scores else None
        selection = {
            "rule": "final checkpoint served; held-out DDIM MSE logged",
            "served": served_scores[-1] if served_scores else None,
            "lowest": best,
            "scores": scores,
        }
        path = Path(output_dir) / SELECTION_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(selection, indent=2) + "\n")
        if selection["served"] is not None and best is not None:
            print(
                f"served final checkpoint step {selection['served']['step']} "
                f"(val_ddim_mse={selection['served']['val_ddim_mse']:.6f}; "
                f"lowest {best['val_ddim_mse']:.6f} at step {best['step']})",
                flush=True,
            )
        return selection
