"""Train a baseline through the protected local-directory training contract."""

import datetime
import hashlib
import json
import os
import sys
from pathlib import Path

import paw_contract
from paw_baseline.data_selection import (
    TRAINING_DATA_PATH,
    resolve_episode_split,
    training_manifest,
)
from paw_baseline.prepare import (
    VIDEO_BACKEND,
    VISUAL_STATS_DATASET,
    VISUAL_STATS_IMAGENET,
    VISUAL_STATS_OPENAI_CLIP,
    prepare_dataset,
)
from paw_baseline.serving_metadata import (
    load_serving_metadata,
    metadata_from_prepared,
    write_serving_metadata,
)

from train.compilation import training_compilation, training_policy_precision
from train.image_transforms import (
    JPEG_CONFIG_PATH,
    resolve_jpeg_quality,
    training_image_transforms,
)
from train.length import epoch_step_args

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"
HOLDOUT_COUNT = 5
# Below this, holding out a quarter of the episodes costs more than the
# generalization signal is worth — CI fixtures train on everything.
HOLDOUT_MIN_EPISODES = 4 * HOLDOUT_COUNT
TRAIN_STAMP = ".paw/baseline_train.json"


def load_preset(model: str) -> dict:
    path = CONFIGS_DIR / f"{model}.json"
    if not path.exists():
        known = sorted(p.stem for p in CONFIGS_DIR.glob("*.json"))
        raise SystemExit(f"unknown model {model!r}; available presets: {known}")
    return json.loads(path.read_text())


def guard_unpinned_weights(
    model: str, preset_args: list[str], passthrough: list[str]
) -> None:
    """Refuse to train on a preset whose base-weights repo is still the
    PHASE2-TBD placeholder — unless the caller pins it with an explicit
    override for the same key (how Phase 2 GPU verification runs before
    the preset itself is pinned)."""
    overridden = {arg.split("=", 1)[0] for arg in passthrough if "=" in arg}
    for arg in preset_args:
        key = arg.split("=", 1)[0]
        if "PHASE2-TBD" in arg and key not in overridden:
            raise SystemExit(
                f"preset {model!r} carries an unpinned base-weights repo ({arg}); "
                f"it is pinned during Phase 2 GPU verification — pass an explicit "
                f"{key} override to train anyway"
            )


def holdout_episodes(total_episodes: int) -> list[int]:
    """Evenly spaced episode indices to hold out of training for validate's
    probes; empty when the dataset is too small to spare any."""
    if total_episodes < HOLDOUT_MIN_EPISODES:
        return []
    spaced = (
        round(i * (total_episodes - 1) / (HOLDOUT_COUNT - 1))
        for i in range(HOLDOUT_COUNT)
    )
    return sorted(dict.fromkeys(spaced))


def select_holdout_episodes(
    total_episodes: int, preset: dict, caller_picked_episodes: bool
) -> list[int]:
    """Select validation probes unless the caller or preset owns the split."""
    if caller_picked_episodes or preset.get("train_all_episodes", False):
        return []
    return holdout_episodes(total_episodes)


def merge_args(*groups: list[str]) -> list[str]:
    """Merge draccus ``--key=value`` argument groups; later groups win per key."""
    merged: dict[str, str] = {}
    for group in groups:
        for arg in group:
            if not arg.startswith("--") or "=" not in arg:
                raise SystemExit(f"expected --key=value overrides, got {arg!r}")
            key, value = arg.split("=", 1)
            merged[key] = value
    return [f"{key}={value}" for key, value in merged.items()]


def install_checkpoint_metadata_hook(
    lerobot_train, metadata: dict, write_validation, score=None
) -> None:
    """Stamp every LeRobot checkpoint while its pretrained files are saved.

    ``score`` (optional) rates the saved policy before the checkpoint is marked
    complete, as ``score(checkpoint_dir, step, policy, preprocessor, final)``
    where ``final`` marks the run's last save; LeRobot calls save_checkpoint
    with keyword arguments on rank zero.
    """
    original = lerobot_train.save_checkpoint

    def save_checkpoint_with_metadata(*args, **kwargs):
        result = original(*args, **kwargs)
        checkpoint_dir = Path(kwargs.get("checkpoint_dir", args[0] if args else ""))
        write_serving_metadata(checkpoint_dir / "pretrained_model", metadata)
        write_validation(checkpoint_dir / "pretrained_model")
        if score is not None:
            score(
                checkpoint_dir,
                kwargs["step"],
                kwargs["policy"],
                kwargs["preprocessor"],
                kwargs["step"] >= kwargs["cfg"].steps,
            )
        paw_contract.mark_checkpoint_complete(checkpoint_dir)
        return result

    lerobot_train.save_checkpoint = save_checkpoint_with_metadata


def trainer_params(params: dict) -> tuple[str, list[str]]:
    """Interpret the baseline-owned preset and LeRobot override map."""
    if set(params) - {"data", "jpeg_quality", "preset", "overrides"}:
        raise ValueError(
            "baseline params accept only data, jpeg_quality, preset and overrides"
        )
    model = params.get("preset", "act")
    if not isinstance(model, str) or model not in {
        p.stem for p in CONFIGS_DIR.glob("*.json")
    }:
        raise ValueError(f"unknown baseline preset: {model!r}")
    overrides = params.get("overrides", {})
    if not isinstance(overrides, dict):
        raise ValueError("overrides must be an object")
    reserved = {
        "config_path",
        "dataset.eval_split",
        "dataset.repo_id",
        "dataset.root",
        "dataset.streaming",
        "output_dir",
        "policy.push_to_hub",
        "resume",
        "save_checkpoint_to_hub",
        "wandb.mode",
    }
    if reserved.intersection(overrides):
        raise ValueError(
            f"overrides cannot replace contract fields: {sorted(reserved.intersection(overrides))}"
        )
    return model, [
        f"--{key}={value if isinstance(value, str) else json.dumps(value)}"
        for key, value in sorted(overrides.items())
    ]


def effective_visual_stats_profile(
    model: str, preset_args: list[str], overrides: list[str]
) -> str:
    """Match LeRobot's effective fixed-ImageNet flag before fitting any stats."""
    use_imagenet = True  # LeRobot DatasetConfig default, shared by ACT/SmolVLA.
    for arg in merge_args(preset_args, overrides):
        if arg.startswith("--dataset.use_imagenet_stats="):
            value = json.loads(arg.split("=", 1)[1])
            if type(value) is not bool:
                raise ValueError("dataset.use_imagenet_stats must be a boolean")
            use_imagenet = value
    if use_imagenet:
        return VISUAL_STATS_IMAGENET
    return (
        VISUAL_STATS_OPENAI_CLIP if model == "multi_task_dit" else VISUAL_STATS_DATASET
    )


def require_resume_training_data(checkpoint: Path, manifest: dict) -> None:
    """Prevent resumed saved normalizers from claiming a different training fit."""
    path = checkpoint / TRAINING_DATA_PATH
    if not path.is_file():
        raise ValueError(
            "resume checkpoint lacks training_data.json; exact data/normalizer provenance cannot be verified"
        )
    if json.loads(path.read_text()) != manifest:
        raise ValueError(
            "resume checkpoint training data or normalization differs from the requested selection"
        )


def require_requested_device(argv: list[str]) -> None:
    """Fail when explicit CUDA is unavailable instead of accepting CPU fallback."""
    if not any(arg.startswith("--policy.device=cuda") for arg in argv):
        return
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable in this runtime")


def main() -> None:
    args = paw_contract.parse_args(sys.argv[1:])
    model, passthrough = trainer_params(args.params)
    jpeg_quality = resolve_jpeg_quality(args.params, args.resume_from)
    preset = load_preset(model)
    guard_unpinned_weights(model, preset["args"], passthrough)
    output_dir = args.output.resolve()
    if not args.dataset.is_dir():
        raise ValueError("--dataset must be a local export directory")
    if (
        args.resume_from is not None
        and not (args.resume_from / paw_contract.COMPLETE_MARKER).is_file()
    ):
        raise ValueError("--resume-from must name a COMPLETE checkpoint")
    if args.resume_from is not None:
        load_serving_metadata(args.resume_from / "pretrained_model")
    from accelerate import PartialState

    effective_args = merge_args(preset["args"], passthrough)
    distributed_state = PartialState(cpu="--policy.device=cpu" in effective_args)
    # LeRobot refuses an existing fresh output directory. Only remove an empty
    # harness-created directory; rmdir refuses to discard any prior run state.
    if (
        distributed_state.is_main_process
        and args.resume_from is None
        and output_dir.exists()
    ):
        output_dir.rmdir()

    visual_stats_profile = effective_visual_stats_profile(
        model, preset["args"], passthrough
    )
    info = json.loads((args.dataset / "meta/info.json").read_text())
    caller_picked_episodes = any(
        arg.startswith("--dataset.episodes=") for arg in passthrough
    )
    default_holdout = select_holdout_episodes(
        info["total_episodes"], preset, caller_picked_episodes
    )
    override_indices = args.params.get("overrides", {}).get("dataset.episodes")
    if isinstance(override_indices, str):
        override_indices = json.loads(override_indices)
    split = resolve_episode_split(
        args.dataset,
        args.params.get("data", {}),
        default_holdout=default_holdout,
        train_indices=override_indices,
    )
    passthrough.extend(
        epoch_step_args(
            preset,
            merge_args(preset["args"], passthrough),
            args.dataset,
            split.train,
            resume=args.resume_from is not None,
            world_size=int(os.environ.get("WORLD_SIZE", "1")),
        )
    )
    holdout = list(split.holdout)
    # Populate shared conversion caches before other ranks can read their provenance.
    with distributed_state.main_process_first():
        prepared = prepare_dataset(
            args.dataset,
            normalization_episodes=split.train,
            visual_stats_profile=visual_stats_profile,
        )
        if preset.get("model") == "paw_pi05":
            from paw_baseline.pi05.prepare import prepare_pi05

            reserved_pi = {
                "dataset.eval_split",
                "policy.path",
                "policy.paw_representation",
                "policy.pretrained_path",
                "policy.pretrained_revision",
                "policy.rtc_config",
                "policy.type",
                "rename_map",
            }
            bad = reserved_pi.intersection(args.params.get("overrides", {}))
            if bad:
                raise ValueError(
                    f"PI05 overrides cannot replace representation fields: {sorted(bad)}"
                )
            effective = dict(
                arg[2:].split("=", 1) for arg in merge_args(preset["args"], passthrough)
            )
            horizon = int(effective.get("policy.chunk_size", "50"))
            prepared = prepare_pi05(prepared, horizon)
    if preset.get("model") == "attn_diffusion":
        # In-policy min-max statistics from the training episodes (every rank
        # computes the same values from the shared prepared view).
        from paw_baseline.attn_diffusion.stats import override_args, training_statistics

        passthrough.extend(override_args(training_statistics(prepared.root, split.train)))

    data_manifest = training_manifest(
        args.dataset, split, visual_stats_profile=visual_stats_profile
    )
    # A local directory name is display information, not frozen source identity.
    # Metadata/stats fingerprints allow the same export to move across hosts.
    data_manifest["source_repo_id"] = (
        None if prepared.source_revision == "local" else prepared.source_repo_id
    )
    data_manifest["source_revision"] = prepared.source_revision
    if preset.get("model") == "paw_pi05":
        data_manifest["pi05"] = json.loads(
            (prepared.root / ".paw/prepare.json").read_text()
        )["pi05"]
    data_manifest["prepared_stats_sha256"] = hashlib.sha256(
        (prepared.root / "meta/stats.json").read_bytes()
    ).hexdigest()
    if args.resume_from is not None:
        require_resume_training_data(
            args.resume_from / "pretrained_model", data_manifest
        )
    episode_args = [
        "--dataset.episodes=" + json.dumps(split.train, separators=(",", ":"))
    ]
    passthrough = [
        arg for arg in passthrough if not arg.startswith("--dataset.episodes=")
    ]

    lerobot_argv = merge_args(
        preset["args"],
        [
            f"--dataset.repo_id={prepared.root.name}",
            f"--dataset.root={prepared.root}",
            f"--dataset.video_backend={VIDEO_BACKEND}",
            *episode_args,
            f"--output_dir={output_dir}",
            "--policy.push_to_hub=false",
            "--wandb.enable=true",
            # Explicit flag, not WANDB_MODE: lerobot passes an explicit mode
            # to wandb.init, and any explicit mode overrides the env var.
            "--wandb.mode=offline",
            "--wandb.project=ai-scientist",
            "--wandb.disable_artifact=true",
        ],
        passthrough,
    )

    if args.resume_from is not None:
        # A pretrained policy path takes precedence over resume in LeRobot.
        lerobot_argv = [
            arg
            for arg in lerobot_argv
            if not arg.startswith(("--policy.path=", "--reward_model.path="))
        ]
        lerobot_argv.extend(
            [
                f"--config_path={args.resume_from / 'pretrained_model/train_config.json'}",
                "--resume=true",
            ]
        )

    require_requested_device(lerobot_argv)

    from lerobot.scripts import lerobot_train

    from train.video_timestamps import install_video_timestamp_patch

    install_video_timestamp_patch(lerobot_train)

    if preset.get("model") == "paw_pi05":
        from paw_baseline.pi05 import (  # noqa: F401
            configuration_paw_pi05,
            processor_paw_pi05,
        )
    if preset.get("model") == "hist_diffusion":
        import paw_baseline.hist_diffusion  # noqa: F401 — registers the policy type.
    if preset.get("model") == "attn_diffusion":
        import paw_baseline.attn_diffusion  # noqa: F401 — registers the policy type.

    serving_metadata = metadata_from_prepared(prepared.root, model=model)
    validation_bundle = None
    final_checkpoint = args.resume_from

    def write_validation(checkpoint):
        nonlocal final_checkpoint, validation_bundle
        if validation_bundle is None:
            from inference.validate import make_validation_bundle

            config = json.loads((checkpoint / "config.json").read_text())
            validation_bundle = make_validation_bundle(
                prepared.root, config, set(holdout)
            )
        (checkpoint / ".paw/validation_probes.npz").write_bytes(validation_bundle)
        (checkpoint / TRAINING_DATA_PATH).write_text(
            json.dumps(data_manifest, indent=2) + "\n"
        )
        (checkpoint / JPEG_CONFIG_PATH).write_text(
            json.dumps({"jpeg_quality": jpeg_quality}, indent=2) + "\n"
        )

        final_checkpoint = checkpoint.parent

    # hist_diffusion: held-out DDIM MSE logged, final step served.
    # attn_diffusion: keep the 5 lowest-MSE checkpoints and serve the best.
    score = None
    selector = None
    if preset.get("checkpoint_selection") == "val_ddim_mse_top5":
        from paw_baseline.attn_diffusion.selection import TopKCheckpoints

        score = TopKCheckpoints(prepared.root, holdout).score
    elif preset.get("validation_scoring"):
        from paw_baseline.hist_diffusion.checkpoint_selection import (
            ValidationSelector,
        )

        selector = ValidationSelector(prepared.root, holdout)

        def score(checkpoint_dir, step, policy, preprocessor, final):
            selector.score(checkpoint_dir, step, policy, preprocessor)

    install_checkpoint_metadata_hook(
        lerobot_train, serving_metadata, write_validation, score=score
    )

    distributed_state.wait_for_everyone()
    sys.argv = ["lerobot-train", *lerobot_argv]
    with (
        training_compilation(preset.get("model")),
        training_image_transforms(lerobot_train, jpeg_quality),
        training_policy_precision(lerobot_train, preset.get("model")),
    ):
        lerobot_train.main()

    if not distributed_state.is_main_process:
        return
    if final_checkpoint is None:
        raise ValueError("training finished without a complete checkpoint")
    checkpoint = (final_checkpoint / "pretrained_model").resolve()
    selection = None
    if selector is not None:
        selection = selector.write(output_dir, checkpoint.parent)
    paw_contract.write_training_result(
        output_dir,
        checkpoint_path=checkpoint.relative_to(output_dir).as_posix(),
        config={"schema_version": 1, "model": model},
    )
    stamp = {
        "jpeg_quality": jpeg_quality,
        "world_size": distributed_state.num_processes,
        "model": model,
        "dataset_repo_id": prepared.source_repo_id,
        "dataset_revision": prepared.source_revision,
        "prepared_root": str(prepared.root),
        "frozen_arm": serving_metadata["frozen_arm"],
        "holdout_episodes": holdout,
        "training_data": data_manifest,
        "checkpoint_dir": str(checkpoint),
        "validation_scores": None if selection is None else selection,
        "lerobot_argv": lerobot_argv,
        "finished_at": datetime.datetime.now(datetime.UTC).isoformat(),
    }
    stamp_path = output_dir / TRAIN_STAMP
    stamp_path.parent.mkdir(parents=True, exist_ok=True)
    stamp_path.write_text(json.dumps(stamp, indent=2) + "\n")
    print(f"train stamp written: {stamp_path}")


if __name__ == "__main__":
    main()
