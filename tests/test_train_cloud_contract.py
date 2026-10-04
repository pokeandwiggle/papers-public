"""External training keeps checkpoint completion and resume explicit."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_checkpoint_complete_is_written_after_metadata(tmp_path):
    from train.__main__ import install_checkpoint_metadata_hook

    checkpoint = tmp_path / "checkpoints/1"

    def save_checkpoint(*, checkpoint_dir):
        (checkpoint_dir / "pretrained_model").mkdir(parents=True)
        assert not (checkpoint_dir / "COMPLETE").exists()

    trainer = SimpleNamespace(save_checkpoint=save_checkpoint)
    install_checkpoint_metadata_hook(
        trainer,
        {"model": "act"},
        lambda checkpoint: (checkpoint / ".paw/validation_probes.npz").write_bytes(
            b"probes"
        ),
    )
    trainer.save_checkpoint(checkpoint_dir=checkpoint)
    assert (checkpoint / "COMPLETE").is_file()
    assert json.loads(
        (checkpoint / "pretrained_model/.paw/serving.json").read_text()
    ) == {"model": "act"}


def test_checkpoint_failure_has_no_complete_marker(tmp_path):
    from train.__main__ import install_checkpoint_metadata_hook

    def fail(**kwargs):
        raise RuntimeError("partial save")

    trainer = SimpleNamespace(save_checkpoint=fail)
    install_checkpoint_metadata_hook(trainer, {}, lambda checkpoint: None)
    with pytest.raises(RuntimeError, match="partial"):
        trainer.save_checkpoint(checkpoint_dir=tmp_path)
    assert not (tmp_path / "COMPLETE").exists()


def test_explicit_cuda_never_falls_back_to_cpu(monkeypatch):
    import torch

    from train.__main__ import require_requested_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA.*unavailable"):
        require_requested_device(["--policy.device=cuda"])
    require_requested_device(["--policy.device=cpu"])


def test_visual_normalization_matches_effective_lerobot_flags():
    from train.__main__ import effective_visual_stats_profile, load_preset

    assert (
        effective_visual_stats_profile("act", load_preset("act")["args"], [])
        == "imagenet"
    )
    assert (
        effective_visual_stats_profile(
            "multi_task_dit", load_preset("multi_task_dit")["args"], []
        )
        == "openai_clip"
    )
    assert (
        effective_visual_stats_profile(
            "act", [], ["--dataset.use_imagenet_stats=false"]
        )
        == "dataset"
    )
    assert (
        effective_visual_stats_profile(
            "multi_task_dit", [], ["--dataset.use_imagenet_stats=true"]
        )
        == "imagenet"
    )


def test_secondary_split_cannot_bypass_manifest():
    from train.__main__ import trainer_params

    with pytest.raises(ValueError, match="dataset.eval_split"):
        trainer_params({"overrides": {"dataset.eval_split": 0.2}})


def test_resume_requires_identical_recorded_training_data(tmp_path):
    from train.__main__ import require_resume_training_data

    manifest = {"normalization": {"episode_indices": [0, 2]}}
    with pytest.raises(ValueError, match="lacks training_data"):
        require_resume_training_data(tmp_path, manifest)
    (tmp_path / ".paw").mkdir()
    (tmp_path / ".paw/training_data.json").write_text(json.dumps(manifest))
    require_resume_training_data(tmp_path, manifest)
    with pytest.raises(ValueError, match="differs"):
        require_resume_training_data(
            tmp_path, {"normalization": {"episode_indices": [1]}}
        )


@pytest.mark.parametrize("resume_at_end", [False, True])
@pytest.mark.parametrize("steps_override", [None, 2])
def test_entrypoint_normalizes_and_stamps_the_actual_automatic_train_split(
    tmp_path, monkeypatch, resume_at_end, steps_override
):
    import sys
    from types import ModuleType

    from lerobot.datasets.factory import make_train_eval_datasets

    from tests.fixture_export import build_export
    from train import __main__ as trainer

    export = build_export(tmp_path / "export", episodes=20)
    output = tmp_path / "output"
    args = SimpleNamespace(
        dataset=export, output=output, resume_from=None, params={"preset": "act"}
    )
    if steps_override is not None:
        args.params["overrides"] = {"steps": steps_override}
    monkeypatch.setattr(trainer.paw_contract, "parse_args", lambda argv: args)
    monkeypatch.setenv("PAW_BASELINE_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(
        trainer,
        "load_preset",
        lambda model: {
            "args": ["--batch_size=64", "--policy.type=act", "--policy.device=cpu"],
            "epochs": 3,
            "min_steps": 40,
        },
    )
    monkeypatch.setattr(
        "train.video_timestamps.install_video_timestamp_patch", lambda module: None
    )
    monkeypatch.setattr(
        "inference.validate.make_validation_bundle", lambda *args: b"portable probes"
    )
    native = ModuleType("lerobot.scripts.lerobot_train")
    native.make_train_eval_datasets = make_train_eval_datasets

    def save_checkpoint(*, checkpoint_dir):
        checkpoint = checkpoint_dir / "pretrained_model"
        checkpoint.mkdir(parents=True)
        (checkpoint / "config.json").write_text('{"type":"act"}')

    native.save_checkpoint = save_checkpoint

    def train():
        argv = dict(arg[2:].split("=", 1) for arg in sys.argv[1:])
        assert argv["dataset.video_backend"] == "pyav"
        assert int(argv["steps"]) == (steps_override or 40)
        expected = [
            index for index in range(20) if index not in trainer.holdout_episodes(20)
        ]
        assert json.loads(argv["dataset.episodes"]) == expected
        prepared = Path(argv["dataset.root"])
        stamp = json.loads((prepared / ".paw/prepare.json").read_text())
        assert stamp["normalization_episode_indices"] == expected
        stats = json.loads((prepared / "meta/stats.json").read_text())
        assert stats["action"]["count"] == [len(expected) * 48]
        native.save_checkpoint(checkpoint_dir=output / "checkpoints/2")
        (output / "checkpoints/last").symlink_to("2", target_is_directory=True)

    native.main = train
    monkeypatch.setitem(sys.modules, native.__name__, native)
    import lerobot.scripts

    monkeypatch.setattr(lerobot.scripts, "lerobot_train", native, raising=False)
    monkeypatch.setattr(sys, "argv", ["train"])
    trainer.main()
    checkpoint = output / "checkpoints/last/pretrained_model"
    manifest = json.loads((checkpoint / ".paw/training_data.json").read_text())
    assert len(manifest["train_episodes"]) == 15
    assert [
        row["episode_index"] for row in manifest["holdout_episodes"]
    ] == trainer.holdout_episodes(20)
    assert manifest["normalization"]["visual_stats_source"] == "fixed_imagenet"
    assert manifest["source_repo_id"] is None
    assert (
        checkpoint / ".paw/validation_probes.npz"
    ).read_bytes() == b"portable probes"
    assert (checkpoint.parent / "COMPLETE").is_file()
    assert json.loads((checkpoint / ".paw/image_transforms.json").read_text()) == {
        "jpeg_quality": 85
    }

    if resume_at_end:
        args.resume_from = checkpoint.parent.resolve()
        (output / "checkpoints/last").unlink()
        (output / "training_result.json").unlink()
        monkeypatch.setattr(trainer, "load_serving_metadata", lambda path: {})
        native.main = lambda: None
        trainer.main()
        result = trainer.paw_contract.read_training_result(output)
        assert result["checkpoint_path"] == "checkpoints/2/pretrained_model"


@pytest.mark.parametrize("enabled", [False, True])
def test_pi05_training_defaults_and_explicit_override(enabled):
    from train.__main__ import load_preset, merge_args, trainer_params

    defaults = load_preset("pi05")["args"]
    effective = dict(arg[2:].split("=", 1) for arg in defaults)
    assert effective["policy.chunk_size"] == "32"
    assert effective["policy.n_action_steps"] == "32"
    assert effective["policy.compile_mode"] == "default"
    assert effective["policy.compile_model"] == "true"
    _, overrides = trainer_params(
        {"preset": "pi05", "overrides": {"policy.compile_model": enabled}}
    )
    effective = dict(arg[2:].split("=", 1) for arg in merge_args(defaults, overrides))
    assert effective["policy.compile_model"] == str(enabled).lower()
