"""Every shipped preset trains on the full dataset unless a split is explicit."""

import json
from pathlib import Path

import pytest
from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from paw_baseline.data_selection import resolve_episode_split
from train.__main__ import load_preset, select_holdout_episodes

PRESETS = sorted(
    path.stem for path in (Path(__file__).parents[1] / "configs").glob("*.json")
)
# Presets that select checkpoints on held-out episodes (paper replication).
VALIDATING = (
    "attn_dp_to2_exec24",
    "attn_dp_to8_exec6",
    "hist_diffusion_to2_exec24",
    "hist_diffusion_to8_exec6",
)


@pytest.mark.parametrize(
    "preset_name", [name for name in PRESETS if name not in VALIDATING]
)
def test_preset_uses_every_episode_without_validation(tmp_path, preset_name):
    total = 185
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta/info.json").write_text(json.dumps({"total_episodes": total}))
    preset = load_preset(preset_name)
    split = resolve_episode_split(
        tmp_path, {}, default_holdout=select_holdout_episodes(total, preset, False)
    )
    assert split.train == tuple(range(total))
    assert split.holdout == ()
    args = dict(arg[2:].split("=", 1) for arg in preset["args"])
    assert (
        float(
            args.get(
                "dataset.eval_split",
                DatasetConfig.__dataclass_fields__["eval_split"].default,
            )
        )
        == 0
    )
    assert (
        int(
            args.get(
                "eval_steps",
                TrainPipelineConfig.__dataclass_fields__["eval_steps"].default,
            )
        )
        == 0
    )


@pytest.mark.parametrize("preset_name", PRESETS)
def test_preset_preserves_explicit_holdout(tmp_path, preset_name):
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta/info.json").write_text('{"total_episodes": 3}')
    (tmp_path / ".paw").mkdir()
    (tmp_path / ".paw/episode_mapping.json").write_text('{"0":100,"1":101,"2":102}')
    split = resolve_episode_split(
        tmp_path,
        {"holdout_episode_ids": [101]},
        default_holdout=select_holdout_episodes(3, load_preset(preset_name), False),
    )
    assert split.train == (0, 2)
    assert split.holdout == (1,)


@pytest.mark.parametrize("preset_name", VALIDATING)
def test_validating_presets_hold_out_five_spread_episodes(tmp_path, preset_name):
    total = 185
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta/info.json").write_text(json.dumps({"total_episodes": total}))
    preset = load_preset(preset_name)
    assert (
        preset.get("validation_scoring") is True
        or preset.get("checkpoint_selection") == "val_ddim_mse_top5"
    )
    split = resolve_episode_split(
        tmp_path, {}, default_holdout=select_holdout_episodes(total, preset, False)
    )
    assert split.holdout == (0, 46, 92, 138, 184)
    assert set(split.train) == set(range(total)) - set(split.holdout)
