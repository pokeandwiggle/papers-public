"""Selections bind source IDs, train-only statistics, and exported label time."""

import json
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from paw_baseline.data_selection import (
    aligned_annotations,
    resolve_episode_split,
    training_manifest,
)
from paw_baseline.prepare import prepare_dataset

from tests.fixture_export import build_export


@pytest.fixture
def mapped_export(tmp_path):
    root = build_export(tmp_path / "export", episodes=3)
    (root / ".paw").mkdir()
    (root / ".paw/episode_mapping.json").write_text('{"0":100,"1":101,"2":102}')
    (root / ".paw/pipeline_config.json").write_text('{"filters":{"task":["target"]}}')
    (root / "meta/pokeandwiggle_annotations.json").write_text(
        json.dumps(
            {
                "version": 1,
                "episodes": {
                    "0": {
                        "subtasks": [
                            {
                                "start": 0,
                                "end": 0.5,
                                "label": "reach",
                                "rating": "GOOD",
                            },
                            {
                                "start": 0.5,
                                "end": 1.0,
                                "label": "grasp",
                                "rating": "PERFECT",
                            },
                        ],
                        "actors": [
                            {
                                "start": 0,
                                "end": 1,
                                "actor": "human",
                                "actor_type": "human",
                            }
                        ],
                        "high_levels": [
                            {
                                "start": 0,
                                "end": 1.6,
                                "label": "target",
                                "rating": "GOOD",
                            }
                        ],
                    }
                },
            }
        )
    )
    return root


def test_source_ids_resolve_disjoint_exact_split(mapped_export):
    split = resolve_episode_split(
        mapped_export, {"train_episode_ids": [102, 100], "holdout_episode_ids": [101]}
    )
    assert split.train == (0, 2)
    assert split.holdout == (1,)
    manifest = training_manifest(
        mapped_export, split, visual_stats_profile="openai_clip"
    )
    assert manifest["train_episodes"] == [
        {"episode_index": 0, "source_episode_id": 100},
        {"episode_index": 2, "source_episode_id": 102},
    ]
    assert manifest["holdout_episodes"] == [
        {"episode_index": 1, "source_episode_id": 101}
    ]
    assert manifest["normalization"]["episode_indices"] == [0, 2]
    assert ".paw/episode_mapping.json" in manifest["source_sidecar_sha256"]


@pytest.mark.parametrize(
    "data, message",
    [
        ({"train_episode_ids": []}, "nonempty"),
        ({"train_episode_ids": [999]}, "not in"),
        ({"train_episode_ids": [100, 100]}, "duplicate"),
        ({"train_episode_ids": [True]}, "integer"),
        ({"train_episode_ids": [100], "holdout_episode_ids": [100]}, "disjoint"),
        ({"holdout_episode_ids": [100, 101, 102]}, "nonempty"),
        ({"typo": 1}, "unknown"),
    ],
)
def test_invalid_source_selection_fails(mapped_export, data, message):
    with pytest.raises(ValueError, match=message):
        resolve_episode_split(mapped_export, data)


def test_source_and_lerobot_index_selection_must_agree(mapped_export):
    with pytest.raises(ValueError, match="conflict"):
        resolve_episode_split(
            mapped_export, {"train_episode_ids": [100]}, train_indices=[1]
        )
    split = resolve_episode_split(
        mapped_export, {"train_episode_ids": [100]}, train_indices=[0]
    )
    assert split.train == (0,)


def test_default_holdout_is_excluded_before_stats_are_fit(mapped_export, tmp_path):
    split = resolve_episode_split(mapped_export, {}, default_holdout=[1, 2])
    assert split.train == (0,)
    prepared = prepare_dataset(
        mapped_export, cache_root=tmp_path / "cache", normalization_episodes=split.train
    )
    stats = json.loads((prepared.root / "meta/stats.json").read_text())
    table = pq.read_table(prepared.root / "data/chunk-000/file-000.parquet")
    values = np.asarray(table["observation.state"].combine_chunks().flatten()).reshape(
        -1, 20
    )
    mask = np.isin(table["episode_index"].to_numpy(), split.train)
    assert stats["observation.state"]["count"] == [48]
    assert stats["observation.state"]["mean"][0] == pytest.approx(
        values[mask, 0].mean()
    )
    assert table.num_rows == 144  # Holdout remains available for validation.
    for relative in (
        ".paw/episode_mapping.json",
        ".paw/pipeline_config.json",
        "meta/pokeandwiggle_annotations.json",
    ):
        assert (prepared.root / relative).read_bytes() == (
            mapped_export / relative
        ).read_bytes()
    other = prepare_dataset(
        mapped_export, cache_root=tmp_path / "cache", normalization_episodes=[1]
    )
    assert other.root != prepared.root
    assert json.loads((other.root / "meta/stats.json").read_text())[
        "observation.state"
    ]["mean"][0] == pytest.approx(
        values[table["episode_index"].to_numpy() == 1, 0].mean()
    )


def test_image_statistics_use_only_selected_episode_frames(mapped_export, tmp_path):
    first = prepare_dataset(
        mapped_export, cache_root=tmp_path / "cache", normalization_episodes=[0]
    )
    last = prepare_dataset(
        mapped_export, cache_root=tmp_path / "cache", normalization_episodes=[2]
    )
    first_stats = json.loads((first.root / "meta/stats.json").read_text())
    last_stats = json.loads((last.root / "meta/stats.json").read_text())
    camera = "observation.images.rgb_static"
    assert first_stats[camera]["count"] == [48]
    assert (
        np.asarray(first_stats[camera]["mean"]).max()
        < np.asarray(last_stats[camera]["mean"]).min()
    )


def test_labels_use_remapped_dataset_timestamps_and_half_open_intervals(mapped_export):
    rows = aligned_annotations(
        mapped_export, episode_index=0, timestamps=[0, 0.499, 0.5, 1.0]
    )
    assert [row["subtask"] for row in rows] == ["reach", "reach", "grasp", None]
    assert [row["subtask_rating"] for row in rows] == ["GOOD", "GOOD", "PERFECT", None]
    assert rows[2]["actor_type"] == "human"
    assert rows[3]["actor_type"] is None
    assert (
        aligned_annotations(mapped_export, episode_index=2, timestamps=[0])[0][
            "subtask"
        ]
        is None
    )


def test_overlapping_exported_labels_fail_instead_of_guessing(mapped_export):
    path = mapped_export / "meta/pokeandwiggle_annotations.json"
    data = json.loads(path.read_text())
    data["episodes"]["0"]["subtasks"][0]["end"] = 0.6
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="overlap"):
        aligned_annotations(mapped_export, episode_index=0, timestamps=[0.55])


@pytest.mark.parametrize(
    "boundary, timestamp", [(0.7, np.float32(0.7)), (0.067, np.float32(2 / 30))]
)
def test_labels_share_export_precision_at_float32_frame_boundaries(
    mapped_export, boundary, timestamp
):
    path = mapped_export / "meta/pokeandwiggle_annotations.json"
    annotations = json.loads(path.read_text())
    subtasks = annotations["episodes"]["0"]["subtasks"]
    subtasks[0]["end"] = boundary
    subtasks[1]["start"] = boundary
    path.write_text(json.dumps(annotations))
    row = aligned_annotations(
        mapped_export, episode_index=0, timestamps=[float(timestamp)]
    )[0]
    assert row["subtask"] == "grasp"


def test_hf_download_includes_canonical_paw_metadata(mapped_export, monkeypatch):
    import huggingface_hub

    def download(**kwargs):
        assert ".paw/**" in kwargs["allow_patterns"]
        return str(mapped_export)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    from paw_baseline.prepare import _resolve_source

    assert _resolve_source("test/example", "frozen-sha")[0] == mapped_export


def test_changed_exported_labels_cannot_reuse_stale_prepared_sidecars(
    mapped_export, tmp_path
):
    first = prepare_dataset(mapped_export, cache_root=tmp_path / "cache")
    path = mapped_export / "meta/pokeandwiggle_annotations.json"
    data = json.loads(path.read_text())
    data["episodes"]["0"]["subtasks"][0]["label"] = "updated exported label"
    path.write_text(json.dumps(data))
    second = prepare_dataset(mapped_export, cache_root=tmp_path / "cache")
    assert first.root != second.root
    assert (
        second.root / "meta/pokeandwiggle_annotations.json"
    ).read_bytes() == path.read_bytes()


def test_frozen_arm_constant_is_fit_only_on_training_episodes(tmp_path):
    export = build_export(tmp_path / "export", episodes=2, frozen_right_arm=True)
    path = export / "data/chunk-000/file-000.parquet"
    table = pq.read_table(path)
    widths = pa.array([0.01] * 48 + [0.08] * 48, type=pa.float32())
    for key in ("observation.state.gripper_width_right", "action.gripper_width_right"):
        table = table.set_column(table.column_names.index(key), key, widths)
    pq.write_table(table, path)
    prepared = prepare_dataset(
        export,
        cache_root=tmp_path / "cache",
        normalization_episodes=[0],
        visual_stats_profile="openai_clip",
    )
    stats = json.loads((prepared.root / "meta/stats.json").read_text())
    assert stats["action"]["mean"][19] == pytest.approx(0.01)
    view = pq.read_table(prepared.root / "data/chunk-000/file-000.parquet")
    values = np.asarray(view["action"].combine_chunks().flatten()).reshape(-1, 20)
    np.testing.assert_allclose(values[:, 19], 0.01)


def test_identical_export_moved_to_a_new_directory_retains_resume_identity(
    mapped_export, tmp_path
):
    from train.__main__ import require_resume_training_data

    moved = tmp_path / "cloud-dataset-download"
    shutil.copytree(mapped_export, moved)
    original = training_manifest(
        mapped_export,
        resolve_episode_split(mapped_export, {}),
        visual_stats_profile="openai_clip",
    )
    relocated = training_manifest(
        moved, resolve_episode_split(moved, {}), visual_stats_profile="openai_clip"
    )
    checkpoint = tmp_path / "checkpoint"
    (checkpoint / ".paw").mkdir(parents=True)
    (checkpoint / ".paw/training_data.json").write_text(json.dumps(original))
    require_resume_training_data(checkpoint, relocated)


def test_multi_task_export_needs_no_extra_identity_or_target_roles(mapped_export):
    path = mapped_export / "meta/info.json"
    info = json.loads(path.read_text())
    info["total_tasks"] = 2
    path.write_text(json.dumps(info))
    split = resolve_episode_split(mapped_export, {"train_episode_ids": [100, 102]})
    assert split.train == (0, 2)
    manifest = training_manifest(
        mapped_export, split, visual_stats_profile="openai_clip"
    )
    assert [row["source_episode_id"] for row in manifest["train_episodes"]] == [
        100,
        102,
    ]
    assert manifest["temporal_sampling"]["boundaries"] == "dataset_episodes"
