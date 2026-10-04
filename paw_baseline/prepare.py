"""Prepare a 20D rotation-6D LeRobot dataset from a structured export.

Fold per-arm state/actions, fit statistics on selected training episodes,
and link source videos. Write provenance last to mark the view complete.
"""

import dataclasses
import datetime
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from paw_baseline.data_selection import episode_indices, source_metadata_hashes
from paw_baseline.layout import (
    ACTION_KEY_DIMS,
    CAMERA_KEYS,
    FPS,
    GRIPPER_MAX_WIDTH_M,
    PAW_DIM,
    STATE_KEY_DIMS,
    TRAIN_GRIP_DIMS,
)
from paw_baseline.representation import (
    MODEL_ARM_DIMS,
    MODEL_DIM,
    MODEL_ROTATION_DIMS,
    ROTATION_CONVENTION,
    encode_model,
    model_names,
)
from paw_baseline.rotations import (
    canonical_flat16,
    canonical_quaternions,
)

VIDEO_BACKEND = "pyav"
"""Decode packed RGB video consistently for training and validation probes."""

PREPARE_VERSION = 6
LAYOUT_NAME = "flat20-rot6d-v5"
PROVENANCE_PATH = ".paw/prepare.json"

GRIPPER_TOL_M = 1e-6

# Pose detection excludes grippers so a spurious closure cannot make a
# parked arm look active. Rotation rows avoid quaternion sign discontinuities.
STATIC_POSE_MAX_RANGE = 2e-2
MOVING_POSE_MIN_RANGE = 5e-2
MIN_POSE_RANGE_RATIO = 10.0
ARM_DIMS = {"left": tuple(range(0, 8)), "right": tuple(range(8, 16))}

VISUAL_STATS_DATASET = "dataset"
VISUAL_STATS_IMAGENET = "imagenet"
VISUAL_STATS_OPENAI_CLIP = "openai_clip"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

_QUANTILES = {"q01": 0.01, "q10": 0.10, "q50": 0.50, "q90": 0.90, "q99": 0.99}
# Copied verbatim from the source export into the prepared view.
_META_COPY = (
    ".paw/episode_mapping.json",
    ".paw/pipeline_config.json",
    "meta/lerobot_annotations.json",
    "meta/pokeandwiggle_annotations.json",
    "meta/robot.urdf",
    "meta/subtasks.parquet",
    "meta/tasks.parquet",
)


@dataclasses.dataclass(frozen=True)
class PreparedDataset:
    root: Path
    source_repo_id: str
    source_revision: str


def default_cache_root() -> Path:
    env = os.environ.get("PAW_BASELINE_CACHE")
    if env:
        return Path(env)
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "paw-baseline"


def prepare_dataset(
    source: str | Path,
    *,
    revision: str = "v3.0",
    cache_root: Path | None = None,
    visual_stats_profile: str = VISUAL_STATS_DATASET,
    normalization_episodes: list[int] | tuple[int, ...] | None = None,
) -> PreparedDataset:
    """Return the prepared flat-20 view of ``source``, building it if absent.

    ``source`` is an HF dataset repo id (downloaded at ``revision``) or a
    local export directory (tests, pre-downloaded data).
    """
    if visual_stats_profile not in (
        VISUAL_STATS_DATASET,
        VISUAL_STATS_IMAGENET,
        VISUAL_STATS_OPENAI_CLIP,
    ):
        raise ValueError(f"unknown visual stats profile {visual_stats_profile!r}")

    source_dir, repo_id, resolved = _resolve_source(source, revision)
    info = json.loads((source_dir / "meta/info.json").read_text())
    _check_source(info)
    source_metadata = source_metadata_hashes(source_dir)
    source_key = {"metadata": source_metadata}
    if resolved == "local":
        source_key["path"] = str(source_dir.resolve())
    metadata_digest = hashlib.sha256(
        json.dumps(source_key, sort_keys=True).encode()
    ).hexdigest()
    fit_episodes = episode_indices(
        list(range(info["total_episodes"]))
        if normalization_episodes is None
        else normalization_episodes,
        info["total_episodes"],
        name="normalization_episodes",
    )
    if not fit_episodes:
        raise ValueError("normalization_episodes must be nonempty")
    fit_all = len(fit_episodes) == info["total_episodes"]
    # Directory name doubles as the lerobot repo_id and leaks into wandb
    # artifact names, which allow only [A-Za-z0-9._-].
    profile_suffix = (
        ""
        if visual_stats_profile == VISUAL_STATS_DATASET
        else f"-{visual_stats_profile}"
    )
    fit_suffix = (
        ""
        if fit_all
        else "-fit-"
        + hashlib.sha256(json.dumps(fit_episodes).encode()).hexdigest()[:12]
    )
    root = (cache_root or default_cache_root()) / (
        f"{repo_id.replace('/', '--')}-{resolved[:12]}-p{PREPARE_VERSION}-{metadata_digest[:12]}-{LAYOUT_NAME}{profile_suffix}{fit_suffix}"
    )
    provenance = root / PROVENANCE_PATH
    if provenance.exists():
        stamp = json.loads(provenance.read_text())
        if (
            stamp["prepare_version"] == PREPARE_VERSION
            and stamp["source_revision"] == resolved
            and stamp.get("visual_stats_profile", VISUAL_STATS_DATASET)
            == visual_stats_profile
            and stamp["normalization_episode_indices"] == list(fit_episodes)
        ):
            return PreparedDataset(root, repo_id, resolved)
        raise RuntimeError(
            f"{root} exists with mismatched provenance {stamp}; delete it to rebuild"
        )
    if root.exists():
        shutil.rmtree(root)  # a previous run died before stamping provenance

    state, action, indices = _rewrite_data(source_dir, root, info)
    fit_mask = np.isin(indices, fit_episodes)
    if set(indices[fit_mask]) != set(fit_episodes):
        raise ValueError("normalization episodes are missing from data parquets")
    _check_conventions(state, action)
    frozen_arm = _resolve_frozen_arm(state[fit_mask], action[fit_mask])
    if _resolve_frozen_arm(state, action) != frozen_arm:
        raise ValueError(
            "normalization selection and full export have incompatible active arms"
        )
    if frozen_arm is not None:
        _freeze_arm_data(root, state, action, frozen_arm, fit_mask=fit_mask)
    state, action = _encode_model_data(root, state, action)
    _write_stats(
        source_dir,
        root,
        state[fit_mask],
        action[fit_mask],
        frozen_arm=frozen_arm,
        visual_stats_profile=visual_stats_profile,
        fit_episodes=fit_episodes,
        fit_all=fit_all,
    )
    _write_info(root, info)
    _copy_meta(source_dir, root)
    _link_videos(source_dir, root)

    provenance.parent.mkdir(parents=True, exist_ok=True)
    provenance.write_text(
        json.dumps(
            {
                "prepare_version": PREPARE_VERSION,
                "layout": LAYOUT_NAME,
                "rotation_convention": ROTATION_CONVENTION,
                "visual_stats_profile": visual_stats_profile,
                "normalization_episode_indices": list(fit_episodes),
                "frozen_arm": frozen_arm,
                "source_repo_id": repo_id,
                "source_revision": resolved,
                "source_path": str(source_dir),
                "source_metadata_sha256": source_metadata,
                "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
            },
            indent=2,
        )
        + "\n"
    )
    return PreparedDataset(root, repo_id, resolved)


def _resolve_source(source: str | Path, revision: str) -> tuple[Path, str, str]:
    if isinstance(source, Path) or Path(source).exists():
        source_dir = Path(source)
        return source_dir, source_dir.name, "local"
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=str(source),
        repo_type="dataset",
        revision=revision,
        allow_patterns=[".paw/**", "data/**", "meta/**", "videos/**"],
        ignore_patterns=["videos/observation.images.depth*/**"],
    )
    source_dir = Path(path)
    # .../<repo>/snapshots/<commit sha>
    return source_dir, str(source), source_dir.name


def _check_source(info: dict) -> None:
    if info["codebase_version"] != "v3.0":
        raise RuntimeError(
            f"expected codebase_version v3.0, got {info['codebase_version']}"
        )
    if info["fps"] != FPS:
        raise RuntimeError(f"expected fps {FPS}, got {info['fps']}")
    features = info["features"]
    missing = [k for k, _ in (*STATE_KEY_DIMS, *ACTION_KEY_DIMS) if k not in features]
    if missing:
        raise RuntimeError(f"source export lacks structured_v1 keys: {missing}")
    absent_cams = [k for k in CAMERA_KEYS if k not in features]
    if absent_cams:
        raise RuntimeError(f"source export lacks cameras: {absent_cams}")


def _fold(table: pa.Table, key_dims: tuple[tuple[str, int], ...]) -> np.ndarray:
    parts = []
    for key, dim in key_dims:
        column = table[key].combine_chunks()
        if dim == 1:
            parts.append(column.to_numpy(zero_copy_only=False).reshape(-1, 1))
        else:
            parts.append(
                column.flatten().to_numpy(zero_copy_only=False).reshape(-1, dim)
            )
    folded = np.concatenate(parts, axis=1).astype(np.float32)
    if not np.isfinite(folded).all():
        raise RuntimeError("non-finite values in folded state/action")
    return canonical_flat16(folded)


def _fixed_list(values: np.ndarray) -> pa.FixedSizeListArray:
    flat = pa.array(values.ravel(), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, values.shape[1])


def _rewrite_data(
    source_dir: Path, root: Path, info: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    states, actions, indices = [], [], []
    files = sorted((source_dir / "data").glob("chunk-*/file-*.parquet"))
    if not files:
        raise RuntimeError(f"no data parquets under {source_dir / 'data'}")
    for path in files:
        table = pq.read_table(path)
        state = _fold(table, STATE_KEY_DIMS)
        action = _fold(table, ACTION_KEY_DIMS)
        states.append(state)
        actions.append(action)
        indices.append(table["episode_index"].to_numpy())
        out = pa.table(
            {
                "observation.state": _fixed_list(state),
                "action": _fixed_list(action),
                "timestamp": table["timestamp"],
                "frame_index": table["frame_index"],
                "episode_index": table["episode_index"],
                "index": table["index"],
                "task_index": table["task_index"],
            }
        )
        dest = root / path.relative_to(source_dir)
        dest.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(out, dest)
    return np.concatenate(states), np.concatenate(actions), np.concatenate(indices)


def _check_conventions(state: np.ndarray, action: np.ndarray) -> None:
    """Require physical gripper widths after quaternion canonicalization."""
    for name, folded in (("observation.state", state), ("action", action)):
        grips = folded[:, list(TRAIN_GRIP_DIMS)].astype(np.float64)
        if (
            grips.min() < -GRIPPER_TOL_M
            or grips.max() > GRIPPER_MAX_WIDTH_M + GRIPPER_TOL_M
        ):
            raise RuntimeError(
                f"{name} gripper widths outside [0, {GRIPPER_MAX_WIDTH_M}] m "
                f"(observed [{grips.min():.4f}, {grips.max():.4f}])"
            )


def _max_pose_range(values: np.ndarray, arm: str) -> float:
    start = 0 if arm == "left" else 10
    values = encode_model(values)
    dims = list(range(start, start + 9))
    spread = np.quantile(values[:, dims], 0.99, axis=0) - np.quantile(
        values[:, dims], 0.01, axis=0
    )
    return float(spread.max())


def _resolve_frozen_arm(state: np.ndarray, action: np.ndarray) -> str | None:
    """Detect a clamped pose from state and action while ignoring grippers.

    A clear single parked side is frozen in the prepared data and statistics.
    Two clearly moving sides need no rewrite; ambiguous data fails before
    training rather than guessing which signal to remove.
    """
    ranges = {
        arm: {
            "observation.state": _max_pose_range(state, arm),
            "action": _max_pose_range(action, arm),
        }
        for arm in ARM_DIMS
    }
    static = [
        arm
        for arm, measured in ranges.items()
        if max(measured.values()) <= STATIC_POSE_MAX_RANGE
    ]
    moving = [
        arm
        for arm, measured in ranges.items()
        if min(measured.values()) >= MOVING_POSE_MIN_RANGE
    ]

    if len(moving) == 2:
        return None
    if len(static) == 1 and len(moving) == 1:
        frozen = static[0]
        driven = moving[0]
        ratio = min(ranges[driven].values()) / max(max(ranges[frozen].values()), 1e-12)
        if ratio >= MIN_POSE_RANGE_RATIO:
            return frozen

    detail = ", ".join(
        f"{arm} state={measured['observation.state']:.3g} action={measured['action']:.3g}"
        for arm, measured in ranges.items()
    )
    raise RuntimeError(
        "cannot determine a parked pose safely "
        f"({detail}; static<={STATIC_POSE_MAX_RANGE:g}, moving>={MOVING_POSE_MIN_RANGE:g}, "
        f"ratio>={MIN_POSE_RANGE_RATIO:g})"
    )


def _frozen_arm_value(values: np.ndarray, arm: str) -> np.ndarray:
    """Robust constant for a frozen arm, including its gripper."""
    dims = list(ARM_DIMS[arm])
    samples = values[:, dims].copy()
    # Detection already established a nearly fixed orientation. Align samples
    # to one observed quaternion before taking its median; dominant-component
    # canonicalization can flip even a parked arm near its sign boundary.
    quats = samples[:, 3:7]
    quats *= np.where(np.sum(quats * quats[0], axis=1, keepdims=True) < 0, -1, 1)
    value = np.median(samples, axis=0).astype(np.float32)
    quat_offset = 3
    quat = value[quat_offset : quat_offset + 4]
    quat_norm = float(np.linalg.norm(quat))
    if quat_norm < 0.5:
        raise RuntimeError(f"{arm} frozen-arm median produced a degenerate quaternion")
    value[quat_offset : quat_offset + 4] = canonical_quaternions(quat / quat_norm)
    return value


def _freeze_arm_data(
    root: Path,
    state: np.ndarray,
    action: np.ndarray,
    arm: str,
    *,
    fit_mask: np.ndarray,
) -> None:
    """Replace every frozen-arm sample with one robust constant in the view."""
    dims = list(ARM_DIMS[arm])
    state_value = _frozen_arm_value(state[fit_mask], arm)
    action_value = _frozen_arm_value(action[fit_mask], arm)
    state[:, dims] = state_value
    action[:, dims] = action_value

    for path in sorted((root / "data").glob("chunk-*/file-*.parquet")):
        table = pq.read_table(path)
        columns = {}
        for key, value in (
            ("observation.state", state_value),
            ("action", action_value),
        ):
            folded = (
                table[key]
                .combine_chunks()
                .flatten()
                .to_numpy(zero_copy_only=False)
                .reshape(-1, PAW_DIM)
                .copy()
            )
            folded[:, dims] = value
            columns[key] = _fixed_list(folded.astype(np.float32))
        for key, column in columns.items():
            table = table.set_column(table.column_names.index(key), key, column)
        pq.write_table(table, path)


def _encode_model_data(
    root: Path, state: np.ndarray, action: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Encode only after physical checks and frozen-arm preparation."""
    for path in sorted((root / "data").glob("chunk-*/file-*.parquet")):
        table = pq.read_table(path)
        for key in ("observation.state", "action"):
            physical = np.asarray(table[key].combine_chunks().flatten()).reshape(
                -1, PAW_DIM
            )
            table = table.set_column(
                table.column_names.index(key), key, _fixed_list(encode_model(physical))
            )
        pq.write_table(table, path)
    return encode_model(state), encode_model(action)


def _vector_stats(values: np.ndarray) -> dict:
    v = values.astype(np.float64)
    stats = {
        "min": v.min(axis=0).tolist(),
        "max": v.max(axis=0).tolist(),
        "mean": v.mean(axis=0).tolist(),
        "std": v.std(axis=0).tolist(),
        "count": [int(v.shape[0])],
    }
    for name, q in _QUANTILES.items():
        stats[name] = np.quantile(v, q, axis=0).tolist()
    return stats


def _squash_arm_stats(stats: dict, key: str, arm: str) -> None:
    """Make a frozen arm normalize to zero and decode zero to its constant."""
    dims = list(MODEL_ARM_DIMS[arm])
    mean = np.asarray(stats[key]["mean"], dtype=np.float64)
    replacements = {
        "min": mean - 0.5,
        "max": mean + 0.5,
        "std": np.ones_like(mean),
        "q01": mean - 0.5,
        "q10": mean - 0.5,
        "q50": mean,
        "q90": mean + 0.5,
        "q99": mean + 0.5,
    }
    for name, replacement in replacements.items():
        values = np.asarray(stats[key][name], dtype=np.float64)
        values[dims] = replacement[dims]
        stats[key][name] = values.tolist()


def _write_stats(
    source_dir: Path,
    root: Path,
    state: np.ndarray,
    action: np.ndarray,
    *,
    frozen_arm: str | None,
    visual_stats_profile: str,
    fit_episodes: tuple[int, ...],
    fit_all: bool,
) -> None:
    source_stats = json.loads((source_dir / "meta/stats.json").read_text())
    kept = {
        key: value
        for key, value in source_stats.items()
        if key in CAMERA_KEYS
        or key in ("timestamp", "frame_index", "episode_index", "index", "task_index")
    }
    stats = {
        "observation.state": _vector_stats(state),
        "action": _vector_stats(action),
        **kept,
    }
    if not fit_all and visual_stats_profile == VISUAL_STATS_DATASET:
        from paw_baseline.visual_stats import selected_visual_stats

        stats.update(selected_visual_stats(source_dir, fit_episodes))
    # Rotation rows have a known domain. Keep them unscaled rather than
    # amplifying almost-constant coordinates with dataset-dependent ranges.
    for key in ("observation.state", "action"):
        for name, value in {
            "mean": 0.0,
            "std": 1.0,
            "min": -1.0,
            "max": 1.0,
            "q01": -1.0,
            "q99": 1.0,
        }.items():
            for dim in MODEL_ROTATION_DIMS:
                if frozen_arm is not None and dim in MODEL_ARM_DIMS[frozen_arm]:
                    continue
                stats[key][name][dim] = value
    if frozen_arm is not None:
        _squash_arm_stats(stats, "observation.state", frozen_arm)
        _squash_arm_stats(stats, "action", frozen_arm)
    if visual_stats_profile != VISUAL_STATS_DATASET:
        mean, std = (
            (IMAGENET_MEAN, IMAGENET_STD)
            if visual_stats_profile == VISUAL_STATS_IMAGENET
            else (OPENAI_CLIP_MEAN, OPENAI_CLIP_STD)
        )
        for key in CAMERA_KEYS:
            stats[key]["mean"] = mean
            stats[key]["std"] = std
    dest = root / "meta/stats.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(stats, indent=2) + "\n")


def _write_info(root: Path, info: dict) -> None:
    source_features = info["features"]
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": [MODEL_DIM],
            "names": model_names(),
        },
        "action": {
            "dtype": "float32",
            "shape": [MODEL_DIM],
            "names": model_names(),
        },
    }
    for key in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
        features[key] = source_features[key]
    for key in CAMERA_KEYS:
        features[key] = source_features[key]
    out = {**info, "features": features}
    (root / "meta/info.json").write_text(json.dumps(out, indent=2) + "\n")


def _copy_meta(source_dir: Path, root: Path) -> None:
    for rel in _META_COPY:
        src = source_dir / rel
        if src.exists():
            dest = root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)
    for src in sorted((source_dir / "meta/episodes").glob("chunk-*/file-*.parquet")):
        dest = root / src.relative_to(source_dir)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)


def _link_videos(source_dir: Path, root: Path) -> None:
    videos = source_dir / "videos"
    for key in CAMERA_KEYS:
        src = videos / key
        if not src.is_dir():
            raise RuntimeError(f"missing video directory {src}")
        dest = root / "videos" / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.symlink_to(src.resolve(), target_is_directory=True)
