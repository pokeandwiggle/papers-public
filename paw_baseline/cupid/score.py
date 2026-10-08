"""Score demonstrations with CUPID and write the filter-k selection.

Two resumable stages share one output directory:

``train``  projected l_square gradients of strided state-action pairs of every
           demonstration in the base policy's export, summed per demonstration,
           with the Gauss-Newton matrix X^T X of those pairs.
``score``  projected gradients of every chunk request of a base-policy station
           eval, summed per rollout, then the performance influence of each
           demonstration and the filter-k selection (top 1 - filter_ratio).

Example, for a base policy trained on every demonstration of dataset:1435:

    uv run --directory baseline python -m paw_baseline.cupid.score train \\
        --checkpoint CKPT --export EXPORT --members experiments/cupid/members_1435.json \\
        --cache-root CACHE --output OUT --stride 10
    uv run --directory baseline python -m paw_baseline.cupid.score score \\
        --checkpoint CKPT --eval-dir EVAL_DOWNLOAD --expected-model model:ID \\
        --output OUT --filter-ratio 0.5
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from paw_baseline.cupid import influence
from paw_baseline.cupid.rollouts import read_rollout, rollout_outcomes
from paw_baseline.cupid.select import cupid_keep

TRAIN_FILE = "train_features.npz"
TRAIN_META = "train_features.json"
SCORE_FILE = "cupid_scores.json"
TRAJ_FILE = "trajectory_influence.npz"
# Independent (eps, i) streams for training and rollout pairs; the projection
# itself is shared through --seed.
TRAIN_NOISE_OFFSET = 1_000
SCORE_NOISE_OFFSET = 2_000
# 1: scores used (X^T X)^-2 (Gram matrix squared again; invalid).
# 2: scores use (X^T X)^-1 as in TRAK / paper Eq. 9.
SCORING_VERSION = 2


def checkpoint_digest(checkpoint: Path) -> str:
    return hashlib.sha256((checkpoint / "model.safetensors").read_bytes()).hexdigest()


def episode_source_ids(root: Path, members: list[dict]) -> list[int]:
    """Export episode index -> source episode ID, verified by rating fingerprints.

    The export orders episodes by ascending source ID. Every episode's task
    rating and ordered subtask ratings must agree with that member's metadata.
    """
    members = sorted(members, key=lambda m: m["id"])
    annotations = json.loads((root / "meta/pokeandwiggle_annotations.json").read_text())
    info = json.loads((root / "meta/info.json").read_text())
    if info["total_episodes"] != len(members):
        raise ValueError("member list and export disagree on episode count")
    for index, member in enumerate(members):
        entry = annotations["episodes"][str(index)]
        subtasks = [
            s.get("rating") for s in sorted(entry["subtasks"], key=lambda s: s["start"])
        ]
        task = (entry.get("high_levels") or [{}])[0].get("rating")
        if subtasks != member["subtask_ratings"] or task != member["rating"]:
            raise ValueError(f"export episode {index} does not match source {member['id']}")
    return [m["id"] for m in members]


def strided_indices(root: Path, stride: int) -> list[int]:
    """Global frame indices at ``frame % stride == 0`` within every episode."""
    import pyarrow.parquet as pq

    indices = []
    for path in sorted((root / "meta/episodes").glob("chunk-*/file-*.parquet")):
        for row in pq.read_table(
            path, columns=["episode_index", "length", "dataset_from_index"]
        ).to_pylist():
            indices.extend(
                row["dataset_from_index"] + frame for frame in range(0, row["length"], stride)
            )
    return indices


def training_loader(root: Path, horizon: int, stride: int, jpeg_quality, workers: int, batch: int):
    """Strided anchors decoded as LeRobot training does (uint8 frames, JPEG, padding)."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from torch.utils.data import DataLoader, Subset
    from torchvision.transforms import v2

    from paw_baseline.prepare import VIDEO_BACKEND
    from train.video_timestamps import _install_decoder_patch

    _install_decoder_patch()
    fps = json.loads((root / "meta/info.json").read_text())["fps"]
    dataset = LeRobotDataset(
        repo_id=root.name,
        root=root,
        delta_timestamps={"action": [k / fps for k in range(horizon)]},
        video_backend=VIDEO_BACKEND,
        return_uint8=True,
    )
    if jpeg_quality is not None:
        # Training prepends this JPEG round trip (train/image_transforms.py).
        dataset.set_image_transforms(v2.Compose([v2.JPEG(quality=jpeg_quality)]))
    indices = strided_indices(root, stride)
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch,
        shuffle=False,
        num_workers=workers,
        worker_init_fn=_install_decoder_patch,
    )
    return loader, len(indices)


def run_train(args) -> None:
    from paw_baseline.prepare import prepare_dataset

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    loaded = influence.load_diffusion_policy(args.checkpoint, args.device)
    data_manifest = json.loads((args.checkpoint / ".paw/training_data.json").read_text())
    prepared = prepare_dataset(
        args.export,
        cache_root=args.cache_root,
        visual_stats_profile=data_manifest["normalization"]["visual_stats_profile"],
    )
    stamp = json.loads((prepared.root / ".paw/prepare.json").read_text())
    if stamp["frozen_arm"] != loaded.frozen_arm:
        raise ValueError("prepared view and checkpoint disagree on the frozen arm")
    members = json.loads(Path(args.members).read_text())["members"]
    source_ids = episode_source_ids(args.export, members)
    trained = [row["episode_index"] for row in data_manifest["train_episodes"]]
    if trained != list(range(len(source_ids))):
        raise ValueError("the base checkpoint must be trained on every export episode")
    recorded = [row["source_episode_id"] for row in data_manifest["train_episodes"]]
    if any(value is not None for value in recorded) and recorded != source_ids:
        raise ValueError("verified episode mapping differs from the checkpoint's training manifest")
    jpeg = json.loads((args.checkpoint / ".paw/image_transforms.json").read_text())[
        "jpeg_quality"
    ]
    loader, total = training_loader(
        prepared.root, loaded.horizon, args.stride, jpeg, args.workers, args.batch
    )
    featurizer = influence.ProjectedGradient(
        loaded,
        noise_seed=TRAIN_NOISE_OFFSET + args.seed,
        num_draws=args.num_draws,
        proj_dim=args.proj_dim,
        seed=args.seed,
        device=args.device,
    )
    d = featurizer.sketch.proj_dim
    sums = torch.zeros(len(source_ids), d, dtype=torch.float64, device=args.device)
    counts = np.zeros(len(source_ids), dtype=np.int64)
    xtx = torch.zeros(d, d, dtype=torch.float64, device=args.device)
    task = None
    started, done = time.time(), 0
    for batch in loader:
        task = task or batch["task"][0]
        images = {}
        for key in loaded.camera_keys:
            frames = batch[key]
            images[key] = frames.float() / 255.0 if frames.dtype == torch.uint8 else frames
        state, stacked, action = influence.normalize_pairs(
            loaded, batch["observation.state"], images, batch["action"], batch["task"]
        )
        phis = torch.stack(
            [featurizer(state[i], stacked[i], action[i]) for i in range(state.shape[0])]
        ).double()
        xtx += phis.T @ phis
        episodes = batch["episode_index"].tolist()
        for i, episode in enumerate(episodes):
            sums[episode] += phis[i]
            counts[episode] += 1
        done += len(episodes)
        if done % 1000 < len(episodes):
            rate = done / (time.time() - started)
            print(f"train pairs {done}/{total} ({rate:.1f}/s)", flush=True)
    if (counts == 0).any():
        raise ValueError("every demonstration needs at least one featurized pair")
    np.savez(
        out / TRAIN_FILE,
        source_ids=np.asarray(source_ids, dtype=np.int64),
        sums=sums.cpu().numpy(),
        counts=counts,
        xtx=xtx.cpu().numpy(),
    )
    meta = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_digest(args.checkpoint),
        "export": str(args.export),
        "task": task,
        "stride": args.stride,
        "num_draws": args.num_draws,
        "proj_dim": args.proj_dim,
        "seed": args.seed,
        "train_noise_seed": TRAIN_NOISE_OFFSET + args.seed,
        "num_pairs": int(counts.sum()),
        "num_params": featurizer.num_params,
        "jpeg_quality": jpeg,
        "frozen_arm": loaded.frozen_arm,
        "matmul_precision": args.matmul_precision,
        "seconds": time.time() - started,
    }
    (out / TRAIN_META).write_text(json.dumps(meta, indent=1) + "\n")


def run_score(args) -> None:
    out = Path(args.output)
    meta = json.loads((out / TRAIN_META).read_text())
    if meta["checkpoint_sha256"] != checkpoint_digest(args.checkpoint):
        raise ValueError("rollout features must use the checkpoint of the training features")
    train = np.load(out / TRAIN_FILE)
    loaded = influence.load_diffusion_policy(args.checkpoint, args.device)
    featurizer = influence.ProjectedGradient(
        loaded,
        noise_seed=SCORE_NOISE_OFFSET + meta["seed"],
        num_draws=meta["num_draws"],
        proj_dim=meta["proj_dim"],
        seed=meta["seed"],
        device=args.device,
    )
    manifest = json.loads((Path(args.eval_dir) / "manifest.json").read_text())
    if manifest.get("model") != args.expected_model:
        raise ValueError(
            f"eval was produced by {manifest.get('model')}, not {args.expected_model}"
        )
    rows = rollout_outcomes(Path(args.eval_dir))
    if not rows:
        raise ValueError("the eval has no counted rollouts with a binary outcome")
    rollout_sums, success, episodes, pair_counts, dropped = [], [], [], [], []
    for row in rows:
        pairs = read_rollout(
            row["path"],
            loaded.image_shapes,
            episode_id=row["episode_id"],
            success=row["success"],
            placement_combination_id=row["placement_combination_id"],
            jpeg_quality=meta["jpeg_quality"],
        )
        if len(pairs.capture_ns) == 0:
            raise ValueError(f"rollout {row['episode_id']} has no usable chunk requests")
        if pairs.action.shape[1] != loaded.horizon:
            raise ValueError("recorded chunks must cover the full prediction horizon")
        state, images, action = influence.model_space_rollout(loaded, pairs)
        total = torch.zeros(
            featurizer.sketch.proj_dim, dtype=torch.float64, device=args.device
        )
        for s, im, a in influence.iter_normalized(
            loaded, state, images, action, meta["task"]
        ):
            total += featurizer(s, im, a).double()
        rollout_sums.append(total.cpu().numpy())
        success.append(row["success"])
        episodes.append(row["episode_id"])
        pair_counts.append(len(pairs.capture_ns))
        dropped.append(pairs.dropped)
        print(
            f"rollout {row['episode_id']} success={row['success']} "
            f"pairs={len(pairs.capture_ns)} dropped={pairs.dropped}",
            flush=True,
        )
    write_scores(
        out,
        train,
        meta,
        rollout_sums=np.stack(rollout_sums),
        success=np.asarray(success),
        episodes=np.asarray(episodes),
        rollout_stats={
            "eval_dir": str(args.eval_dir),
            "rollouts": len(rows),
            "rollout_successes": int(np.sum(success)),
            "rollout_pairs": int(np.sum(pair_counts)),
            "rollout_pairs_dropped": int(np.sum(dropped)),
        },
        filter_ratio=args.filter_ratio,
        ridge=args.ridge,
    )


def write_scores(
    out: Path,
    train,
    meta: dict,
    *,
    rollout_sums: np.ndarray,
    success: np.ndarray,
    episodes: np.ndarray,
    rollout_stats: dict,
    filter_ratio: float,
    ridge: float,
) -> None:
    """Score demonstrations from saved X^T X and feature sums; write the selection."""
    if meta["num_pairs"] < meta["proj_dim"] and ridge == 0:
        raise ValueError("X^T X is rank deficient: featurize more training pairs or set --ridge")
    eigenvalues = np.linalg.eigvalsh(train["xtx"])
    print(f"X^T X condition number {eigenvalues[-1] / max(eigenvalues[0], 1e-300):.3g}", flush=True)
    scores, traj = influence.cupid_scores(
        train["xtx"], train["sums"], rollout_sums, success, ridge=ridge
    )
    source_ids = [int(x) for x in train["source_ids"]]
    by_id = dict(zip(source_ids, (float(x) for x in scores), strict=True))
    keep = cupid_keep(by_id, filter_ratio)
    np.savez(
        out / TRAJ_FILE,
        traj=traj,
        rollout_sums=rollout_sums,
        success=success,
        rollout_episode_ids=episodes,
        source_ids=np.asarray(source_ids),
    )
    result = {
        "scoring_version": SCORING_VERSION,
        **rollout_stats,
        "filter_ratio": filter_ratio,
        "ridge": ridge,
        "train": meta,
        "scores": {str(k): v for k, v in by_id.items()},
        "train_episode_ids": keep,
    }
    (out / SCORE_FILE).write_text(json.dumps(result, indent=1) + "\n")
    print(f"kept {len(keep)} of {len(source_ids)} demonstrations", flush=True)


def run_rescore(args) -> None:
    """Recompute scores from saved feature sums without re-featurizing (no GPU).

    Valid because the per-demonstration and per-rollout feature sums do not
    depend on how X^T X is inverted.
    """
    out = Path(args.output)
    meta = json.loads((out / TRAIN_META).read_text())
    train = np.load(out / TRAIN_FILE)
    previous = json.loads((out / SCORE_FILE).read_text())
    saved = np.load(out / TRAJ_FILE)
    if [int(x) for x in saved["source_ids"]] != [int(x) for x in train["source_ids"]]:
        raise ValueError("saved rollout features and training features disagree on demos")
    if previous["train"]["checkpoint_sha256"] != meta["checkpoint_sha256"]:
        raise ValueError("saved rollout features came from a different checkpoint")
    write_scores(
        out,
        train,
        meta,
        rollout_sums=saved["rollout_sums"],
        success=saved["success"],
        episodes=saved["rollout_episode_ids"],
        rollout_stats={
            key: previous[key]
            for key in (
                "eval_dir",
                "rollouts",
                "rollout_successes",
                "rollout_pairs",
                "rollout_pairs_dropped",
            )
        },
        filter_ratio=args.filter_ratio,
        ridge=args.ridge,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m paw_baseline.cupid.score",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    stages = parser.add_subparsers(dest="stage", required=True)
    train = stages.add_parser("train")
    score = stages.add_parser("score")
    rescore = stages.add_parser("rescore")
    for stage in (train, score):
        stage.add_argument("--checkpoint", type=Path, required=True)
        stage.add_argument("--device", default="cuda")
        stage.add_argument("--matmul-precision", default="high", choices=("high", "highest"))
    for stage in (train, score, rescore):
        stage.add_argument("--output", type=Path, required=True)
    rescore.add_argument("--filter-ratio", type=float, required=True)
    rescore.add_argument("--ridge", type=float, default=0.0)
    train.add_argument("--export", type=Path, required=True)
    train.add_argument("--members", type=Path, required=True)
    train.add_argument("--cache-root", type=Path, required=True)
    train.add_argument("--stride", type=int, default=10)
    train.add_argument("--num-draws", type=int, default=influence.NUM_NOISE_DRAWS)
    train.add_argument("--proj-dim", type=int, default=influence.PROJ_DIM)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--workers", type=int, default=12)
    train.add_argument("--batch", type=int, default=32)
    score.add_argument("--eval-dir", type=Path, required=True)
    score.add_argument(
        "--expected-model", required=True, help="model:<id> of the base checkpoint"
    )
    score.add_argument("--filter-ratio", type=float, required=True)
    score.add_argument("--ridge", type=float, default=0.0)
    args = parser.parse_args()
    if args.stage == "rescore":
        run_rescore(args)
        return
    torch.set_float32_matmul_precision(args.matmul_precision)
    torch.backends.cudnn.benchmark = True
    if args.stage == "train":
        run_train(args)
    else:
        run_score(args)


if __name__ == "__main__":
    main()
