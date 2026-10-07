"""The pyproject contract and the modules it points at must not drift apart."""

import importlib.util
import json
from pathlib import Path

import tomllib

BASELINE = Path(__file__).resolve().parents[1]


def test_contract():
    pyproject = tomllib.loads((BASELINE / "pyproject.toml").read_text())
    paw_ais = pyproject["tool"]["paw-ais"]
    assert paw_ais["contract-version"] == 1
    entrypoints = paw_ais["entrypoints"]
    assert sorted(entrypoints) == ["serve", "validate"]
    import paw_contract

    assert paw_contract.TRAIN_ENTRYPOINT == ("python", "-m", "train")
    for name, command in entrypoints.items():
        assert command[:2] == ["python", "-m"], name
        assert importlib.util.find_spec(command[2]) is not None, (
            f"{name} points at missing module {command[2]}"
        )


def test_python_version_pin():
    assert (BASELINE / ".python-version").read_text().strip() == "3.12"


def test_presets_cover_all_models():
    presets = sorted(p.stem for p in (BASELINE / "configs").glob("*.json"))
    assert presets == [
        "act",
        "diffusion",
        "multi_task_dit",
        "patch_vqbet_cls",
        "patch_vqbet_dense",
        "pi05",
        "smolvla",
    ]


def test_patch_vqbet_presets_differ_only_in_visual_feature():
    dense = json.loads((BASELINE / "configs" / "patch_vqbet_dense.json").read_text())
    cls = json.loads((BASELINE / "configs" / "patch_vqbet_cls.json").read_text())
    assert set(dense["args"]) ^ set(cls["args"]) == {
        "--policy.visual_feature=patch",
        "--policy.visual_feature=cls",
    }
    assert dense["model"] == cls["model"] == "patch_vqbet"
    assert dense["train_all_episodes"] is cls["train_all_episodes"] is True


def test_multi_task_dit_is_paw_like_flow_matching_preset():
    import json

    preset = json.loads((BASELINE / "configs" / "multi_task_dit.json").read_text())
    args = set(preset["args"])
    expected = {
        "--policy.objective=flow_matching",
        "--policy.timestep_sampling_strategy=uniform",
        "--policy.sigma_min=0",
        "--policy.num_integration_steps=10",
        "--policy.integration_method=euler",
        "--policy.n_obs_steps=1",
        "--policy.horizon=32",
        "--policy.n_action_steps=32",
        "--policy.use_rope=false",
        "--policy.use_positional_encoding=true",
        "--policy.use_separate_rgb_encoder_per_camera=true",
        "--policy.use_amp=true",
        "--dataset.image_transforms.enable=true",
        "--dataset.use_imagenet_stats=false",
        "--policy.optimizer_lr=2e-5",
        "--policy.vision_encoder_lr_multiplier=0.1",
        "--policy.scheduler_name=cosine",
        "--policy.optimizer_weight_decay=1e-6",
        "--policy.scheduler_warmup_steps=500",
        "--batch_size=256",
        "--num_workers=32",
    }
    assert expected <= args
    assert preset["train_all_episodes"] is True


def test_train_entrypoint_explicit_contract():
    from pathlib import Path

    import paw_contract
    import pytest

    args = paw_contract.parse_args(
        ["--dataset", "/data", "--output", "/out", "--params", '{"preset":"act"}']
    )
    assert args.dataset == Path("/data")
    assert args.params == {"preset": "act"}
    with pytest.raises(SystemExit):
        paw_contract.parse_args([])


def test_merge_args():
    from train.__main__ import merge_args

    merged = merge_args(["--a=1", "--b=2"], ["--a=3"])
    assert merged == ["--a=3", "--b=2"]

    import pytest

    with pytest.raises(SystemExit, match="--key=value"):
        merge_args(["--flag"])


def test_guard_unpinned_weights():
    import pytest
    from train.__main__ import guard_unpinned_weights

    preset_args = ["--policy.path=PHASE2-TBD:lerobot/pi05_base", "--batch_size=4"]
    with pytest.raises(SystemExit, match="unpinned base-weights"):
        guard_unpinned_weights("pi05", preset_args, [])
    with pytest.raises(SystemExit, match="unpinned base-weights"):
        guard_unpinned_weights("pi05", preset_args, ["--batch_size=2"])
    guard_unpinned_weights("pi05", preset_args, ["--policy.path=lerobot/pi05_base"])


def test_holdout_episodes():
    from train.__main__ import HOLDOUT_COUNT, HOLDOUT_MIN_EPISODES, holdout_episodes

    assert holdout_episodes(HOLDOUT_MIN_EPISODES - 1) == []
    picks = holdout_episodes(132)
    assert len(picks) == HOLDOUT_COUNT
    assert picks == sorted(set(picks))
    assert picks[0] == 0 and picks[-1] == 131


def test_preset_can_train_on_all_episodes():
    from train.__main__ import select_holdout_episodes

    assert select_holdout_episodes(132, {"train_all_episodes": True}, False) == []
    assert len(select_holdout_episodes(132, {}, False)) == 5
    assert select_holdout_episodes(132, {}, True) == []


def test_checkpoint_hook_stamps_every_pretrained_model(tmp_path):
    from types import SimpleNamespace

    from paw_baseline.serving_metadata import SERVING_METADATA_PATH
    from train.__main__ import install_checkpoint_metadata_hook

    calls = []

    def save_checkpoint(*, checkpoint_dir, **kwargs):
        calls.append(checkpoint_dir)
        (checkpoint_dir / "pretrained_model").mkdir(parents=True)

    trainer = SimpleNamespace(save_checkpoint=save_checkpoint)
    metadata = {"frozen_arm": "left"}
    install_checkpoint_metadata_hook(trainer, metadata, lambda checkpoint: None)
    for step in (1000, 2000):
        checkpoint_dir = tmp_path / f"{step:06d}"
        trainer.save_checkpoint(checkpoint_dir=checkpoint_dir)
        assert (
            json.loads(
                (
                    checkpoint_dir / "pretrained_model" / SERVING_METADATA_PATH
                ).read_text()
            )
            == metadata
        )
    assert len(calls) == 2


def test_pi05_preset_has_pinned_pretrained_weights():
    import json

    preset = json.loads((BASELINE / "configs" / "pi05.json").read_text())
    assert (
        "--policy.base_revision=b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba"
        in preset["args"]
    )


def test_smolvla_infers_paw_features_instead_of_pretrained_robot(tmp_path):
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "type": "smolvla",
                "input_features": {
                    "observation.state": {"type": "STATE", "shape": [6]},
                    "observation.images.camera1": {
                        "type": "VISUAL",
                        "shape": [3, 256, 256],
                    },
                },
            }
        )
    )
    preset = json.loads((BASELINE / "configs/smolvla.json").read_text())
    overrides = [
        arg.replace("--policy.", "--", 1)
        for arg in preset["args"]
        if arg.startswith("--policy.") and not arg.startswith("--policy.path=")
    ]
    config = SmolVLAConfig.from_pretrained(tmp_path, cli_overrides=overrides)
    assert config.input_features is None
    assert config.use_amp is False
    assert config.compile_model is True
    assert config.compile_mode == "default"
    assert config.chunk_size == config.n_action_steps == 32


def test_act_and_diffusion_use_native_normalization_and_full_h32(tmp_path):
    from lerobot.configs import NormalizationMode
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig

    for name, config_class, normalization in (
        ("act", ACTConfig, NormalizationMode.MEAN_STD),
        ("diffusion", DiffusionConfig, NormalizationMode.MIN_MAX),
    ):
        checkpoint = tmp_path / name
        checkpoint.mkdir()
        (checkpoint / "config.json").write_text(
            json.dumps(
                {
                    "type": name,
                    "input_features": {
                        "observation.state": {"type": "STATE", "shape": [20]},
                        "observation.images.rgb_static": {
                            "type": "VISUAL",
                            "shape": [3, 256, 256],
                        },
                    },
                    "output_features": {"action": {"type": "ACTION", "shape": [20]}},
                }
            )
        )
        preset = json.loads((BASELINE / "configs" / f"{name}.json").read_text())
        overrides = [
            arg.replace("--policy.", "--", 1)
            for arg in preset["args"]
            if arg.startswith("--policy.") and not arg.startswith("--policy.type=")
        ]
        config = config_class.from_pretrained(checkpoint, cli_overrides=overrides)
        config.validate_features()
        assert config.n_obs_steps == 1
        assert config.use_amp is False
        assert config.n_action_steps == 32
        assert config.action_delta_indices == list(range(32))
        assert config.normalization_mapping["ACTION"] == normalization
        assert config.normalization_mapping["STATE"] == normalization
        if name == "diffusion":
            assert config.compile_model is True
            assert config.compile_mode == "default"
            assert config.noise_scheduler_type == "DDIM"
            assert config.num_inference_steps == 16
            assert config.num_train_timesteps == 100
            assert config.drop_n_last_frames == 0
            assert config.clip_sample is True
