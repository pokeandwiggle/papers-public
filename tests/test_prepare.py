import json
from pathlib import Path
from typing import Callable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from paw_baseline.layout import STATE_KEY_DIMS
from paw_baseline.prepare import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    PROVENANCE_PATH,
    VISUAL_STATS_OPENAI_CLIP,
    prepare_dataset,
)
from paw_baseline.representation import MODEL_ARM_DIMS, decode_model, model_names
from tests.fixture_export import CAMERAS, TASK, build_export


@pytest.fixture(scope="module")
def prepared(export_dir, cache_root):
    return prepare_dataset(export_dir, cache_root=cache_root)


def test_fold_round_trip(export_dir, prepared):
    source = pq.read_table(export_dir / "data/chunk-000/file-000.parquet")
    view = pq.read_table(prepared.root / "data/chunk-000/file-000.parquet")
    folded = np.asarray(view["observation.state"].combine_chunks().flatten()).reshape(
        -1, 20
    )
    folded = decode_model(folded)
    offset = 0
    for key, dim in STATE_KEY_DIMS:
        column = source[key].combine_chunks()
        values = np.asarray(column.flatten() if dim > 1 else column).reshape(-1, dim)
        np.testing.assert_allclose(
            folded[:, offset : offset + dim], values.astype(np.float32), atol=1e-7
        )
        offset += dim


def test_view_drops_sub_features_and_keeps_bookkeeping(prepared):
    view = pq.read_table(prepared.root / "data/chunk-000/file-000.parquet")
    assert view.column_names == [
        "observation.state",
        "action",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    ]


def test_info_features(prepared):
    info = json.loads((prepared.root / "meta/info.json").read_text())
    features = info["features"]
    assert features["observation.state"]["shape"] == [20]
    assert features["observation.state"]["names"] == model_names()
    assert "observation.state.tcp_wrench_left" not in features
    for camera in CAMERAS:
        assert features[camera]["dtype"] == "video"


def test_stats_have_quantiles(prepared):
    stats = json.loads((prepared.root / "meta/stats.json").read_text())
    for key in ("observation.state", "action"):
        for name in ("min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99"):
            assert len(stats[key][name]) == 20, f"{key}.{name}"
        assert stats[key]["count"] == [96]
    for camera in CAMERAS:
        assert camera in stats


def test_meta_and_videos(prepared):
    tasks = pq.read_table(prepared.root / "meta/tasks.parquet").to_pylist()
    assert tasks == [{"task_index": 0, "task": TASK}]
    for camera in CAMERAS:
        link = prepared.root / "videos" / camera
        assert link.is_symlink() and (link / "chunk-000/file-000.mp4").exists()


def test_idempotent(export_dir, cache_root, prepared):
    marker = prepared.root / "data/chunk-000/file-000.parquet"
    mtime = marker.stat().st_mtime_ns
    again = prepare_dataset(export_dir, cache_root=cache_root)
    assert again.root == prepared.root
    assert marker.stat().st_mtime_ns == mtime


def test_prepare_version_mismatch_fails(export_dir, cache_root, prepared):
    provenance = prepared.root / PROVENANCE_PATH
    stamp = json.loads(provenance.read_text())
    provenance.write_text(json.dumps({**stamp, "prepare_version": 999}))
    try:
        with pytest.raises(RuntimeError, match="mismatched provenance"):
            prepare_dataset(export_dir, cache_root=cache_root)
    finally:
        provenance.write_text(json.dumps(stamp))


def test_pose_detection_freezes_full_arm_including_gripper(tmp_path, cache_root):
    export = build_export(tmp_path / "frozen", frozen_right_arm=True)

    def add_spurious_gripper_closures(values: np.ndarray) -> None:
        values[:] = 0.085
        values[:2] = 0.0

    _mutate_column(export, "action.gripper_width_right", add_spurious_gripper_closures)
    result = prepare_dataset(export, cache_root=cache_root)
    stats = json.loads((result.root / "meta/stats.json").read_text())
    stamp = json.loads((result.root / PROVENANCE_PATH).read_text())
    view = pq.read_table(result.root / "data/chunk-000/file-000.parquet")
    folded_action = np.asarray(view["action"].combine_chunks().flatten()).reshape(
        -1, 20
    )

    assert stamp["frozen_arm"] == "right"
    dims = list(MODEL_ARM_DIMS["right"])
    assert np.all(folded_action[:, dims] == folded_action[0, dims])
    assert folded_action[0, 19] == pytest.approx(0.085)
    for key in ("observation.state", "action"):
        mean = np.asarray(stats[key]["mean"])
        np.testing.assert_allclose(
            np.asarray(stats[key]["min"])[dims], mean[dims] - 0.5
        )
        np.testing.assert_allclose(
            np.asarray(stats[key]["max"])[dims], mean[dims] + 0.5
        )
        np.testing.assert_allclose(np.asarray(stats[key]["std"])[dims], 1.0)


def test_openai_clip_visual_stats_get_a_distinct_view(export_dir, cache_root):
    result = prepare_dataset(
        export_dir,
        cache_root=cache_root,
        visual_stats_profile=VISUAL_STATS_OPENAI_CLIP,
    )
    stats = json.loads((result.root / "meta/stats.json").read_text())
    stamp = json.loads((result.root / PROVENANCE_PATH).read_text())

    assert result.root.name.endswith("-flat20-rot6d-v5-openai_clip")
    assert stamp["visual_stats_profile"] == VISUAL_STATS_OPENAI_CLIP
    for camera in CAMERAS:
        np.testing.assert_allclose(stats[camera]["mean"], OPENAI_CLIP_MEAN)
        np.testing.assert_allclose(stats[camera]["std"], OPENAI_CLIP_STD)


def _mutate_column(
    export: Path, key: str, mutate: Callable[[np.ndarray], None]
) -> None:
    """Rewrite one column of the fixture's single data parquet in place."""
    path = export / "data/chunk-000/file-000.parquet"
    table = pq.read_table(path)
    column = table[key].combine_chunks()
    if isinstance(column.type, pa.FixedSizeListType):
        values = np.asarray(column.flatten()).reshape(len(table), -1).copy()
        mutate(values)
        flat = pa.array(values.astype(np.float32).ravel(), type=pa.float32())
        rebuilt = pa.FixedSizeListArray.from_arrays(flat, values.shape[1])
    else:
        values = column.to_numpy(zero_copy_only=False).copy()
        mutate(values)
        rebuilt = pa.array(values.astype(np.float32), type=pa.float32())
    index = table.column_names.index(key)
    pq.write_table(table.set_column(index, key, rebuilt), path)


def test_convention_guard_non_unit_quats(tmp_path, cache_root):
    export = build_export(tmp_path / "non-unit")

    def scale_quats(values: np.ndarray) -> None:
        values[:, 3:7] *= 2.0

    _mutate_column(export, "action.tcp_pose_left", scale_quats)
    with pytest.raises(RuntimeError, match="not unit"):
        prepare_dataset(export, cache_root=cache_root)


def test_quaternion_sign_flips_preserve_prepared_data(tmp_path, cache_root):
    export = build_export(tmp_path / "flip")
    reference = prepare_dataset(export, cache_root=tmp_path / "reference")

    def flip_mid_episode(values: np.ndarray) -> None:
        values[10:48, 3:7] *= -1.0  # same rotation, broken sign continuity

    _mutate_column(export, "action.tcp_pose_right", flip_mid_episode)
    result = prepare_dataset(export, cache_root=cache_root)
    for relative in ("data/chunk-000/file-000.parquet", "meta/stats.json"):
        assert (result.root / relative).read_bytes() == (
            reference.root / relative
        ).read_bytes()


def test_convention_guard_gripper_range(tmp_path, cache_root):
    export = build_export(tmp_path / "wide-grip")

    def widen(values: np.ndarray) -> None:
        values[0] = 0.2  # beyond any physical gripper

    _mutate_column(export, "observation.state.gripper_width_left", widen)
    with pytest.raises(RuntimeError, match="gripper widths outside"):
        prepare_dataset(export, cache_root=cache_root)


def test_rotation_stats_do_not_amplify_constant_coordinates(prepared):
    from paw_baseline.representation import MODEL_ROTATION_DIMS

    stats = json.loads((prepared.root / "meta/stats.json").read_text())
    for key in ("observation.state", "action"):
        for name, expected in {
            "mean": 0,
            "std": 1,
            "min": -1,
            "max": 1,
            "q01": -1,
            "q99": 1,
        }.items():
            np.testing.assert_array_equal(
                np.asarray(stats[key][name])[list(MODEL_ROTATION_DIMS)], expected
            )


def test_frozen_arm_at_old_quaternion_boundary(tmp_path):
    from paw_baseline.representation import decode_model

    export = build_export(tmp_path / "frozen-boundary", frozen_right_arm=True)

    def near_boundary(values):
        for i, row in enumerate(values):
            delta = 1e-4 if i % 2 else -1e-4
            q = np.array([0.7 + delta, -0.7 + delta, -0.1, 0.1])
            row[3:7] = q / np.linalg.norm(q)

    for prefix in ("observation.state", "action"):
        _mutate_column(export, f"{prefix}.tcp_pose_right", near_boundary)
    prepared = prepare_dataset(export, cache_root=tmp_path / "cache")
    metadata = json.loads((prepared.root / PROVENANCE_PATH).read_text())
    assert metadata["frozen_arm"] == "right"
    table = pq.read_table(prepared.root / "data/chunk-000/file-000.parquet")
    actions = decode_model(
        np.asarray(table["action"].combine_chunks().flatten()).reshape(-1, 20)
    )
    np.testing.assert_allclose(np.linalg.norm(actions[:, 11:15], axis=-1), 1, atol=1e-7)


def test_task_metadata_matches_lerobot_lookup(prepared):
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    metadata = LeRobotDatasetMetadata(prepared.root.name, root=prepared.root)
    assert metadata.tasks.iloc[0].name == TASK


@pytest.mark.parametrize("source_kind", ["local", "hf", "cached"])
def test_prepares_export_without_conversion_metadata(
    tmp_path, monkeypatch, source_kind
):
    import paw_baseline.prepare as preparation

    export = build_export(tmp_path / "source")
    cache = tmp_path / "cache"
    if source_kind == "cached":
        prepare_dataset(export, cache_root=cache)
    info_path = export / "meta/info.json"
    info = json.loads(info_path.read_text())
    info.pop("episode_convert_version", None)
    info_path.write_text(json.dumps(info))
    source = export
    if source_kind == "hf":
        import huggingface_hub

        monkeypatch.setattr(
            huggingface_hub, "snapshot_download", lambda **kwargs: str(export)
        )
        source = "pokeandwiggle/review-export"
    prepared = preparation.prepare_dataset(source, cache_root=cache)
    assert (prepared.root / "meta/info.json").is_file()
