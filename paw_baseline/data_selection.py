"""Episode selections and labels from a frozen dataset export.

Poker IDs come only from the export mapping. Annotation offsets are already
remapped into dataset time by the exporter; live source timelines must not be
passed to these helpers.
"""

import dataclasses
import hashlib
import json
import math
from pathlib import Path

EPISODE_MAPPING_PATH = Path(".paw/episode_mapping.json")
TRAINING_DATA_PATH = Path(".paw/training_data.json")
SOURCE_SIDECARS = (
    ".paw/episode_mapping.json",
    ".paw/pipeline_config.json",
    "meta/info.json",
    "meta/lerobot_annotations.json",
    "meta/pokeandwiggle_annotations.json",
    "meta/stats.json",
)


@dataclasses.dataclass(frozen=True)
class EpisodeSplit:
    """Sorted source-export episode indices; training and holdout are disjoint."""

    train: tuple[int, ...]
    holdout: tuple[int, ...]


def episode_indices(values: object, total: int, *, name: str) -> tuple[int, ...]:
    """Parse a unique list of indices without coercion or out-of-range values."""
    if not isinstance(values, (list, tuple)) or any(
        type(value) is not int for value in values
    ):
        raise ValueError(f"{name} must be a list of integer episode indices")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} contains duplicate episode indices")
    if any(value < 0 or value >= total for value in values):
        raise ValueError(f"{name} contains an episode index not in [0, {total})")
    return tuple(sorted(values))


def read_episode_mapping(root: Path, total: int) -> dict[int, int] | None:
    """Read the episode mapping, returning None when the export has no mapping."""
    path = root / EPISODE_MAPPING_PATH
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or set(payload) != {
        str(index) for index in range(total)
    }:
        raise ValueError(
            "episode mapping must cover every export episode index exactly"
        )
    values = list(payload.values())
    if any(type(value) is not int or value <= 0 for value in values):
        raise ValueError("source episode IDs must be positive integers")
    if len(set(values)) != len(values):
        raise ValueError("episode mapping contains duplicate source IDs")
    return {int(key): value for key, value in payload.items()}


def _source_indices(
    values: object, mapping: dict[int, int] | None, *, name: str
) -> tuple[int, ...]:
    if mapping is None:
        raise ValueError(f"{name} requires exported {EPISODE_MAPPING_PATH}")
    if not isinstance(values, list) or any(type(value) is not int for value in values):
        raise ValueError(f"{name} must be a list of integer source episode IDs")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} contains duplicate source IDs")
    indices = {source: index for index, source in mapping.items()}
    missing = sorted(set(values) - indices.keys())
    if missing:
        raise ValueError(f"{name} source IDs not in this export: {missing}")
    return tuple(sorted(indices[source] for source in values))


def resolve_episode_split(
    root: Path,
    data: object,
    *,
    default_holdout: list[int] | None = None,
    train_indices: object = None,
) -> EpisodeSplit:
    """Resolve baseline ``params.data`` and an optional LeRobot index override.

    Explicit training selection disables automatic holdout. Explicit holdout
    alone trains on its complement. Contradictory source and index selections
    are errors, as are overlap and an empty training set.
    """
    if not isinstance(data, dict) or set(data) - {
        "train_episode_ids",
        "holdout_episode_ids",
    }:
        raise ValueError(
            "data has unknown fields; expected train_episode_ids and holdout_episode_ids"
        )
    info = json.loads((root / "meta/info.json").read_text())
    total = info["total_episodes"]
    mapping = read_episode_mapping(root, total)
    train = None
    if "train_episode_ids" in data:
        train = _source_indices(
            data["train_episode_ids"], mapping, name="train_episode_ids"
        )
    if train_indices is not None:
        indices = episode_indices(train_indices, total, name="dataset.episodes")
        if train is not None and train != indices:
            raise ValueError("source IDs conflict with dataset.episodes")
        train = indices
    if "holdout_episode_ids" in data:
        holdout = _source_indices(
            data["holdout_episode_ids"], mapping, name="holdout_episode_ids"
        )
    else:
        holdout = (
            ()
            if train is not None
            else episode_indices(default_holdout or [], total, name="holdout")
        )
    if train is None:
        train = tuple(index for index in range(total) if index not in holdout)
    if not train:
        raise ValueError("training episode selection must be nonempty")
    if set(train).intersection(holdout):
        raise ValueError("training and holdout episodes must be disjoint")
    return EpisodeSplit(train=train, holdout=holdout)


def training_manifest(
    root: Path, split: EpisodeSplit, *, visual_stats_profile: str
) -> dict:
    """Build the portable, exact train/holdout and normalization provenance."""
    info = json.loads((root / "meta/info.json").read_text())
    mapping = read_episode_mapping(root, info["total_episodes"])

    def members(indices):
        rows = []
        for index in indices:
            row = {
                "episode_index": index,
                "source_episode_id": mapping[index] if mapping is not None else None,
            }
            rows.append(row)
        return rows

    return {
        "schema_version": 1,
        "temporal_sampling": {
            "boundaries": "dataset_episodes",
            "out_of_range": "repeat_boundary_frame_with_is_pad",
            "anchor_selection": "unchanged",
        },
        "train_episodes": members(split.train),
        "holdout_episodes": members(split.holdout),
        "excluded_episodes": members(
            tuple(
                index
                for index in range(info["total_episodes"])
                if index not in (*split.train, *split.holdout)
            )
        ),
        "normalization": {
            "episode_indices": list(split.train),
            "state_action": "training_episodes_only",
            "visual_stats_profile": visual_stats_profile,
            "visual_stats_source": "training_episodes_only"
            if visual_stats_profile == "dataset"
            else f"fixed_{visual_stats_profile}",
        },
        "source_mapping_available": mapping is not None,
        "source_sidecar_sha256": source_metadata_hashes(root),
    }


def source_metadata_hashes(root: Path) -> dict[str, str]:
    """Hash frozen mapping, labels, export config and per-episode boundaries."""
    sidecars = [
        *SOURCE_SIDECARS,
        *[
            path.relative_to(root).as_posix()
            for path in sorted((root / "meta/episodes").glob("chunk-*/file-*.parquet"))
        ],
    ]
    return {
        relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
        for relative in sorted(sidecars)
        if (root / relative).is_file()
    }


def aligned_annotations(
    root: Path, *, episode_index: int, timestamps: list[float]
) -> list[dict]:
    """Align existing frozen labels with dataset frames using ``[start, end)``.

    Unannotated frames carry ``None`` rather than a synthetic phase or rating.
    Overlapping segments in a timeline are rejected instead of choosing one.
    Frame timestamps use the exported annotation timeline's millisecond precision.
    """
    if any(not math.isfinite(time) or time < 0 for time in timestamps):
        raise ValueError("timestamps must be finite nonnegative dataset times")
    path = root / "meta/pokeandwiggle_annotations.json"
    entry = {}
    if path.exists():
        payload = json.loads(path.read_text())
        if payload["version"] != 1:
            raise ValueError("unsupported exported annotation version")
        entry = payload["episodes"].get(str(episode_index), {})
    timelines = {}
    for name in ("actors", "high_levels", "subtasks"):
        segments = sorted(entry.get(name, []), key=lambda segment: segment["start"])
        previous_end = 0.0
        for segment in segments:
            start, end = segment["start"], segment["end"]
            if (
                not math.isfinite(start)
                or not math.isfinite(end)
                or start < 0
                or end <= start
            ):
                raise ValueError(f"invalid {name} annotation interval")
            if start < previous_end:
                raise ValueError(f"overlapping {name} annotation intervals")
            previous_end = end
        timelines[name] = segments
    rows = []
    for time in timestamps:
        # Exported boundaries are rounded to milliseconds, unlike float32 frames.
        time = round(time, 3)
        at = {
            name: next(
                (
                    segment
                    for segment in segments
                    if segment["start"] <= time < segment["end"]
                ),
                {},
            )
            for name, segments in timelines.items()
        }
        rows.append(
            {
                "actor": at["actors"].get("actor"),
                "actor_type": at["actors"].get("actor_type"),
                "subtask": at["subtasks"].get("label"),
                "subtask_rating": at["subtasks"].get("rating"),
                "task": at["high_levels"].get("label"),
                "task_rating": at["high_levels"].get("rating"),
            }
        )
    return rows
