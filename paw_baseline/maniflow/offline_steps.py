"""Compare ManiFlow chunks sampled with different Euler step counts against held-out ground truth.

Usage (from baseline/):
  uv run python -m paw_baseline.maniflow.offline_steps \
      --checkpoint <pretrained_model dir> --holdout <json with holdout_lerobot_indices> \
      --steps 1 2 4 10 --seeds 3 --anchors 10 --output <report.json>

Pre-specified criterion: 2 steps is "clearly worse" than 10 steps if the paired
relative increase in executed-chunk (first n_action_steps) normalized-action MSE
exceeds 10% and its 95% CI excludes zero. Every sampler uses identical initial
noise per (anchor, seed), the served EMA weights, and eval-mode preprocessing.
Holds the global GPU flock (/data/locks/gpu.lock) like paw-ais workers.
"""

import argparse
import contextlib
import fcntl
import json
import os
from pathlib import Path

import numpy as np
import torch

from paw_baseline.prepare import VIDEO_BACKEND
from paw_baseline.representation import decode_model
from paw_baseline.variant import load_policy_config

GPU_LOCK = Path(os.environ.get("PAW_GPU_LOCK", "/data/locks/gpu.lock"))


@contextlib.contextmanager
def gpu_lock():
    fd = os.open(GPU_LOCK, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        print(f"waiting for GPU lock {GPU_LOCK}", flush=True)
        fcntl.flock(fd, fcntl.LOCK_EX)
        print("GPU lock acquired", flush=True)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def rotation_angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Geodesic angle between quaternions (..., 4)."""
    dot = np.abs(np.sum(a * b, axis=-1)).clip(0.0, 1.0)
    return np.degrees(2 * np.arccos(dot))


def physical_errors(pred_model: np.ndarray, true_model: np.ndarray, mask: np.ndarray) -> dict:
    """Errors on decoded 16-D TCP actions: position (mm), rotation (deg), gripper (mm)."""
    pred = decode_model(pred_model)
    true = decode_model(true_model)
    out = {}
    for arm, (pos, quat, grip) in {"left": (slice(0, 3), slice(3, 7), 7), "right": (slice(8, 11), slice(11, 15), 15)}.items():
        pos_err = 1000 * np.linalg.norm(pred[..., pos] - true[..., pos], axis=-1)
        rot_err = rotation_angle_deg(pred[..., quat], true[..., quat])
        grip_err = 1000 * np.abs(pred[..., grip] - true[..., grip])
        out[f"{arm}_pos_mm"] = float(pos_err[mask].mean())
        out[f"{arm}_rot_deg"] = float(rot_err[mask].mean())
        out[f"{arm}_grip_mm"] = float(grip_err[mask].mean())
    return out


def jerk(pred_model: np.ndarray) -> float:
    """Mean |second difference| of positions (mm/step^2) along each chunk."""
    pos = pred_model[..., [0, 1, 2, 10, 11, 12]] * 1000
    return float(np.abs(np.diff(pos, n=2, axis=1)).mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 2, 4, 10])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--anchors", type=int, default=10)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies import make_pre_post_processors

    from paw_baseline.maniflow.modeling_maniflow import ManiFlowPolicy
    from train.video_timestamps import _install_decoder_patch

    _install_decoder_patch()
    checkpoint = args.checkpoint.resolve()
    config = load_policy_config(checkpoint)
    train_config = json.loads((checkpoint / "train_config.json").read_text())
    root = Path(train_config["dataset"]["root"])
    holdout = json.loads(args.holdout.read_text())["holdout_lerobot_indices"]
    trained_on = set(train_config["dataset"]["episodes"])
    assert not trained_on & set(holdout), "held-out episodes were used for training"

    fps = 30
    horizon, executed = config.horizon, config.n_action_steps
    delta = {"action": [i / fps for i in range(horizon)]}
    dataset = LeRobotDataset(repo_id=root.name, root=root, video_backend=VIDEO_BACKEND, delta_timestamps=delta)
    episodes = dataset.meta.episodes
    anchors, anchor_episode = [], []
    for index in holdout:
        start, end = int(episodes["dataset_from_index"][index]), int(episodes["dataset_to_index"][index])
        length = end - start
        for frac in np.linspace(0.05, 0.95, args.anchors):
            anchors.append(start + int(round(frac * (length - 1))))
            anchor_episode.append(index)

    results = {n: {"mse_exec": [], "mse_full": [], "phys": [], "jerk": [], "episode": []} for n in args.steps}
    with gpu_lock():
        device = "cuda"
        config.device = device
        policy = ManiFlowPolicy.from_pretrained(checkpoint, config=config).to(device).eval()
        model = policy.inference_model()
        assert model is policy.ema_model
        preprocess, postprocess = make_pre_post_processors(
            config,
            pretrained_path=checkpoint,
            preprocessor_overrides={"device_processor": {"device": device}},
            postprocessor_overrides={"device_processor": {"device": device}},
        )
        for begin in range(0, len(anchors), args.batch):
            items = [dataset[i] for i in anchors[begin : begin + args.batch]]
            raw = {
                key: torch.stack([item[key] for item in items])
                for key in ["observation.state", "action", "action_is_pad", *config.image_features]
            }
            for key in config.image_features:
                if raw[key].dtype == torch.uint8:
                    raw[key] = raw[key].float() / 255
            raw["task"] = [item["task"] for item in items]
            with torch.inference_mode():
                batch = preprocess(raw)
                obs = {
                    "observation.state": batch["observation.state"].reshape(len(items), 1, -1),
                    "observation.images": torch.stack(
                        [batch[k].reshape(len(items), *batch[k].shape[-3:]) for k in config.image_features], dim=1
                    ).unsqueeze(1),
                }
                target = batch["action"]  # normalized (B, horizon, A)
                mask = ~raw["action_is_pad"].to(device)
                batch_episodes = anchor_episode[begin : begin + len(items)]
                vis_cond = model._vis_cond(obs)
                for seed in range(args.seeds):
                    generator = torch.Generator(device=device).manual_seed(1000 * seed + begin)
                    noise = torch.randn(target.shape, generator=generator, device=device)
                    for n in args.steps:
                        x = noise.clone()
                        dt = 1.0 / n
                        for i in range(n):
                            t = torch.full((len(items),), i / n, device=device)
                            x = x + model.dit(x, t, torch.full_like(t, dt), vis_cond) * dt
                        served = model.generate_actions(obs, noise=noise.clone(), num_inference_steps=n)
                        torch.testing.assert_close(served, x[:, :executed])
                        err = ((x - target) ** 2).mean(-1)  # (B, horizon)
                        m = mask.float()
                        results[n]["mse_exec"].extend(
                            ((err[:, :executed] * m[:, :executed]).sum(1) / m[:, :executed].sum(1)).tolist()
                        )
                        results[n]["mse_full"].extend(((err * m).sum(1) / m.sum(1)).tolist())
                        results[n]["episode"].extend(batch_episodes)
                        pred_phys = torch.stack([postprocess(x[:, j]) for j in range(executed)], 1).cpu().numpy()
                        true_phys = torch.stack([postprocess(target[:, j]) for j in range(executed)], 1).cpu().numpy()
                        results[n]["phys"].append(physical_errors(pred_phys, true_phys, mask[:, :executed].cpu().numpy()))
                        results[n]["jerk"].append(jerk(pred_phys))
                        results[n].setdefault("jerk_truth", []).append(jerk(true_phys))

    summary = {"anchors": len(anchors), "held_out_episodes": len(holdout), "seeds": args.seeds, "steps": {}}
    for n in args.steps:
        mse = np.array(results[n]["mse_exec"])
        phys = {k: float(np.mean([p[k] for p in results[n]["phys"]])) for k in results[n]["phys"][0]}
        summary["steps"][n] = {
            "mse_exec_mean": float(mse.mean()),
            "mse_exec_sem": float(mse.std(ddof=1) / np.sqrt(len(mse))),
            "mse_full_mean": float(np.mean(results[n]["mse_full"])),
            "jerk_mm": float(np.mean(results[n]["jerk"])),
            "jerk_ground_truth_mm": float(np.mean(results[n]["jerk_truth"])),
            **phys,
        }
    if 2 in results and 10 in results:
        # Paired by (anchor, seed, noise); CI from a bootstrap over held-out episodes.
        a, b = np.array(results[2]["mse_exec"]), np.array(results[10]["mse_exec"])
        episode = np.array(results[2]["episode"])
        assert episode.tolist() == results[10]["episode"]
        groups = sorted(set(episode.tolist()))
        diff_by_ep = np.array([(a[episode == g] - b[episode == g]).mean() for g in groups])
        b_by_ep = np.array([b[episode == g].mean() for g in groups])
        rng = np.random.default_rng(0)
        boot = []
        for _ in range(10000):
            pick = rng.integers(0, len(groups), len(groups))
            boot.append(diff_by_ep[pick].mean() / b_by_ep[pick].mean())
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rel = diff_by_ep.mean() / b_by_ep.mean()
        summary["two_vs_ten"] = {
            "relative_mse_increase": float(rel),
            "ci95": [float(lo), float(hi)],
            "episodes_where_2_steps_better": int((diff_by_ep < 0).sum()),
            "episodes": len(groups),
            "clearly_worse": bool(rel > 0.10 and lo > 0),
        }
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
