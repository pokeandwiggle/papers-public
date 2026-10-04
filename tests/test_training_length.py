import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from train.__main__ import load_preset, merge_args, trainer_params
from train.length import epoch_step_args


@pytest.fixture
def dataset(tmp_path):
    path = tmp_path / "meta/episodes/chunk-000/file-000.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(
        pa.table({"episode_index": [0, 1, 2], "length": [100, 10_000, 101]}), path
    )
    return tmp_path


@pytest.mark.parametrize(
    "batch,world_size,expected", [(32, 1, 19), (64, 1, 10), (32, 2, 10)]
)
def test_epochs_count_selected_frames_and_effective_batch(
    dataset, batch, world_size, expected
):
    preset = {"epochs": 3}
    args = [f"--batch_size={batch}"]
    result = epoch_step_args(preset, args, dataset, (0, 2), world_size=world_size)
    assert result == [f"--steps={expected}"]


@pytest.mark.parametrize("name", ["act", "diffusion", "multi_task_dit", "pi05", "smolvla"])
@pytest.mark.parametrize("steps", [7, 50_000])
def test_explicit_steps_override_epochs_without_reading_dataset(tmp_path, steps, name):
    preset = load_preset(name)
    _, overrides = trainer_params({"overrides": {"steps": steps}})
    args = merge_args(preset["args"], overrides)
    assert epoch_step_args(preset, args, tmp_path, (0,)) == []


def test_resume_keeps_checkpoint_target_without_reading_dataset(tmp_path):
    preset = load_preset("pi05")
    assert epoch_step_args(preset, preset["args"], tmp_path, (0,), resume=True) == []


def test_fixed_step_presets_are_unchanged(tmp_path):
    preset = {"args": ["--batch_size=8", "--steps=8000"]}
    assert epoch_step_args(preset, preset["args"], tmp_path, (0,)) == []


def test_missing_selected_episode_is_an_error(dataset):
    preset = load_preset("pi05")
    with pytest.raises(ValueError, match="episode lengths"):
        epoch_step_args(preset, preset["args"], dataset, (3,))


@pytest.mark.parametrize(
    "name,epochs,min_steps,batch",
    [
        ("act", 3, 100_000, 8),
        ("diffusion", 25, 200_000, 64),
        ("multi_task_dit", 25, 20_000, 256),
        ("pi05", 3, 10_000, 64),
        ("smolvla", 3, 20_000, 64),
    ],
)
def test_preset_training_budgets(dataset, name, epochs, min_steps, batch):
    preset = load_preset(name)
    assert preset["epochs"] == epochs
    assert preset["min_steps"] == min_steps
    assert f"--batch_size={batch}" in preset["args"]
    assert not any(arg.startswith("--steps=") for arg in preset["args"])
    assert epoch_step_args(preset, preset["args"], dataset, (0, 2)) == [
        f"--steps={min_steps}"
    ]


@pytest.mark.parametrize("min_steps,expected", [(9, 10), (10, 10), (11, 11)])
def test_floor_and_rounded_epoch_budget(dataset, min_steps, expected):
    assert epoch_step_args(
        {"epochs": 3, "min_steps": min_steps}, ["--batch_size=64"], dataset, (0, 2)
    ) == [f"--steps={expected}"]


@pytest.mark.parametrize("min_steps", [-1, 1.5, True])
def test_invalid_min_steps_fails_before_dataset_read(tmp_path, min_steps):
    with pytest.raises(ValueError, match="min_steps"):
        epoch_step_args(
            {"epochs": 3, "min_steps": min_steps}, ["--batch_size=64"], tmp_path, (0,)
        )
