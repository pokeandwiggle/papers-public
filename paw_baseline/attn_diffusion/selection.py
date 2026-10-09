"""Keep the 5 checkpoints with the lowest validation DDIM MSE; serve the best.

The authors keep the 5 checkpoints with the lowest validation DDIM MSE (the
MSE between DDIM-sampled and demonstrated actions on held-out episodes) and
screen them on hardware. Robot screening is not available here, so the best
of the 5 is served. Every saved checkpoint is scored with the EMA weights on
fixed anchors from the held-out episodes, with fixed sampling noise.

The ranking MSE is computed on what the robot executes: the sampled current
and future deltas accumulated from the measured pose into absolute 20D
targets, against the recorded absolute actions (the reference's metric for
a policy predicting absolute actions). The MSE of the raw per-step deltas
over the full sequence is logged alongside; it is dominated by the gripper
and first-slot terms and barely sees accumulated drift.

The cloud runner registers only the final checkpoint and prunes earlier ones,
so the EMA weights of the current top 5 are kept in a scratch directory and,
when the final checkpoint is saved, the best weights replace its EMA weights
before it is marked complete. The selection is recorded in the package.
"""

import json
import logging
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
from lerobot.utils.constants import ACTION, OBS_STATE
from safetensors.torch import load_file, save_file

from paw_baseline.attn_diffusion.deltas import accumulate

KEEP = 5
ANCHORS_PER_EPISODE = 32
BATCH_SIZE = 8
SEED = 0
EMA_PREFIX = "ema_core."
SCORE_PATH = Path(".paw/validation_score.json")
SELECTION_PATH = Path(".paw/checkpoint_selection.json")

logger = logging.getLogger(__name__)


class TopKCheckpoints:
    def __init__(self, prepared_root: Path, holdout: list[int], keep: int = KEEP):
        if not holdout:
            raise ValueError("checkpoint selection requires held-out episodes")
        self.root = Path(prepared_root)
        self.holdout = sorted(holdout)
        self.keep = keep
        self.scores: list[dict] = []
        self.kept: dict[int, Path] = {}
        self.scratch = Path(tempfile.mkdtemp(prefix="paw_topk_"))
        self._batch = None

    def _anchors(self, config) -> dict:
        from lerobot.datasets.factory import resolve_delta_timestamps
        from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

        from paw_baseline.prepare import VIDEO_BACKEND

        meta = LeRobotDatasetMetadata(self.root.name, root=self.root)
        dataset = LeRobotDataset(
            self.root.name,
            root=self.root,
            delta_timestamps=resolve_delta_timestamps(config, meta),
            video_backend=VIDEO_BACKEND,
        )
        keys = [OBS_STATE, ACTION, *config.image_features]
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
    def validation_mse(self, policy, preprocessor) -> dict[str, float]:
        config = policy.config
        if self._batch is None:
            self._batch = self._anchors(config)
        device = next(policy.parameters()).device
        generator = torch.Generator(device=device).manual_seed(SEED)
        trajectory_errors, delta_errors = [], []
        total = len(self._batch[ACTION])
        bf16 = config.dtype == "bfloat16" and device.type == "cuda"
        future = slice(config.past_steps, config.past_steps + config.prediction_steps)
        for begin in range(0, total, BATCH_SIZE):
            batch = {k: v[begin : begin + BATCH_SIZE] for k, v in self._batch.items()}
            for key in config.image_features:
                if batch[key].dtype == torch.uint8:
                    batch[key] = batch[key].to(torch.float32) / 255.0
            batch = preprocessor(batch)
            target = policy.deltas_from_batch(batch)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
                sample = policy.sample_deltas(
                    batch[OBS_STATE], {k: batch[k] for k in config.image_features}, generator
                )
            sample = sample.float()
            delta_errors.append(((sample - target) ** 2).mean(dim=(1, 2)).cpu())
            # Action window index 0 is one step before the oldest past token.
            states = batch[OBS_STATE][:, -1].double().cpu().numpy()
            recorded = batch[ACTION][:, 1:][:, future].double().cpu().numpy()
            deltas = sample[:, future].double().cpu().numpy()
            for state, delta, truth in zip(states, deltas, recorded, strict=True):
                trajectory_errors.append(float(((accumulate(state, delta) - truth) ** 2).mean()))
        return {
            "val_ddim_mse": float(np.mean(trajectory_errors)),
            "val_ddim_delta_mse": float(torch.cat(delta_errors).mean()),
        }

    def score(self, checkpoint_dir: Path, step: int, policy, preprocessor, final: bool) -> None:
        """Score one saved checkpoint, update the top-k, and finalize on the last save."""
        metrics = self.validation_mse(policy, preprocessor)
        mse = metrics["val_ddim_mse"]
        record = {"step": step, **metrics, "anchors": int(len(self._batch[ACTION]))}
        self.scores.append(record)
        pretrained = Path(checkpoint_dir) / "pretrained_model"
        (pretrained / SCORE_PATH).parent.mkdir(parents=True, exist_ok=True)
        (pretrained / SCORE_PATH).write_text(json.dumps(record, indent=2) + "\n")
        print(
            f"checkpoint step {step} val_ddim_mse={mse:.8f} "
            f"(per-step deltas {metrics['val_ddim_delta_mse']:.8f})",
            flush=True,
        )
        self._update_top_k(record, pretrained / "model.safetensors")
        if final:
            self._finalize(pretrained)

    def top_k(self) -> list[dict]:
        ranked = sorted(self.scores, key=lambda r: (r["val_ddim_mse"], -r["step"]))
        return [r for r in ranked if r["step"] in self.kept][: self.keep]

    def _update_top_k(self, record: dict, weights: Path) -> None:
        ranked = sorted(self.scores, key=lambda r: (r["val_ddim_mse"], -r["step"]))
        best_steps = {r["step"] for r in ranked[: self.keep]}
        if record["step"] in best_steps:
            tensors = {k: v for k, v in load_file(str(weights)).items() if k.startswith(EMA_PREFIX)}
            path = self.scratch / f"{record['step']}.safetensors"
            save_file(tensors, str(path), metadata={"format": "pt"})
            self.kept[record["step"]] = path
        for step in list(self.kept):
            if step not in best_steps:
                self.kept.pop(step).unlink(missing_ok=True)

    def _finalize(self, pretrained: Path) -> None:
        top = self.top_k()
        if not top:
            raise ValueError("no checkpoint was kept")
        best = top[0]
        final_step = self.scores[-1]["step"]
        # Always rewrite (identical tensors when the final step is best) so
        # every run exercises the same finalization path.
        weights = pretrained / "model.safetensors"
        tensors = load_file(str(weights))
        replacement = load_file(str(self.kept[best["step"]]))
        if set(replacement) != {k for k in tensors if k.startswith(EMA_PREFIX)}:
            raise ValueError("kept EMA weights do not match the checkpoint layout")
        tensors.update(replacement)
        save_file(tensors, str(weights), metadata={"format": "pt"})
        selection = {
            "rule": (
                "lowest validation DDIM MSE of the accumulated absolute targets among "
                "the 5 kept checkpoints (EMA weights)"
            ),
            "served_step": best["step"],
            "served_val_ddim_mse": best["val_ddim_mse"],
            "final_step": final_step,
            "top_k": top,
            "scores": self.scores,
            "holdout_episodes": self.holdout,
        }
        (pretrained / SELECTION_PATH).write_text(json.dumps(selection, indent=2) + "\n")
        print(
            f"serving EMA weights of step {best['step']} (val_ddim_mse={best['val_ddim_mse']:.8f}) "
            f"in final checkpoint {final_step}",
            flush=True,
        )
        shutil.rmtree(self.scratch, ignore_errors=True)
