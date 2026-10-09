"""SMILE-SmolVLA and its raw-action control differ only in the action interface."""

import json
from pathlib import Path

import pytest
from train.__main__ import load_preset, merge_args, smile_coefficient_args, trainer_params

BASELINE = Path(__file__).resolve().parents[1]
SMILE_ONLY = {
    "type",
    "base_repo",
    "base_revision",
    "num_control_points",
    "spline_degree",
    "coef_mean",
    "coef_std",
    "action_mean",
    "action_std",
    "gripper_min",
    "gripper_max",
    "observability_tol",
    "translation_dims",
    "rotation_dims",
    "gripper_dims",
    "loss_weight_translation",
    "loss_weight_rotation",
    "loss_weight_gripper",
}


def _args(name: str) -> dict[str, str]:
    return dict(arg[2:].split("=", 1) for arg in load_preset(name)["args"])


def test_presets_match_outside_the_action_interface():
    smile = load_preset("smile_smolvla")
    control = load_preset("smolvla_f50_h15")
    for key in ("epochs", "min_steps", "train_all_episodes", "action_representation"):
        assert smile[key] == control[key]
    assert smile["action_representation"] == "delta_rotvec_v1"
    smile_args, control_args = _args("smile_smolvla"), _args("smolvla_f50_h15")
    shared = {
        "batch_size": "64",
        "policy.chunk_size": "50",
        "policy.n_action_steps": "15",
        "policy.compile_model": "true",
        "policy.compile_mode": "default",
        "policy.use_amp": "false",
    }
    for key, value in shared.items():
        assert smile_args[key] == control_args[key] == value, key
    assert smile_args["num_workers"] == control_args["num_workers"]
    assert smile_args["save_freq"] == control_args["save_freq"]
    assert smile_args["policy.base_revision"] == control_args["policy.pretrained_revision"]
    assert smile_args["policy.base_repo"] == control_args["policy.path"]
    assert smile_args["policy.num_control_points"] == "8"
    assert smile_args["policy.spline_degree"] == "3"
    # Only presentation differences remain: the control has no "type" field.
    extra = set(smile_args) - set(control_args)
    assert extra == {
        "policy.type",
        "policy.base_repo",
        "policy.base_revision",
        "policy.num_control_points",
        "policy.spline_degree",
    }


def test_smile_defaults_equal_pinned_smolvla_base_config():
    from huggingface_hub import hf_hub_download
    from paw_baseline.smile.configuration_smile_smolvla import (
        BASE_REVISION,
        SmileSmolVLAConfig,
    )

    base = json.loads(
        Path(
            hf_hub_download("lerobot/smolvla_base", "config.json", revision=BASE_REVISION)
        ).read_text()
    )
    fields = SmileSmolVLAConfig.__dataclass_fields__
    ignored = {"input_features", "output_features", "device", "push_to_hub", "repo_id"}
    for key, value in base.items():
        if key in SMILE_ONLY or key in ignored or key not in fields:
            continue
        default = fields[key].default
        if callable(fields[key].default_factory):
            default = fields[key].default_factory()
        if isinstance(default, tuple):
            default = list(default)
        if isinstance(default, dict):
            default = {k: getattr(v, "value", v) for k, v in default.items()}
        assert default == value, key


def test_smile_coefficient_overrides_are_reserved(tmp_path):
    for key in ("policy.coef_mean", "policy.action_std"):
        model, passthrough = trainer_params(
            {"preset": "smile_smolvla", "overrides": {key: [0.0]}}
        )
        assert model == "smile_smolvla"
        effective = merge_args(load_preset("smile_smolvla")["args"], passthrough)
        with pytest.raises(ValueError, match="cannot replace"):
            smile_coefficient_args(tmp_path, (0,), effective, {key: [0.0]})


def test_smile_policy_type_resolves_through_lerobot_factory():
    from lerobot.policies.factory import _get_policy_cls_from_policy_name
    from paw_baseline.smile import configuration_smile_smolvla  # noqa: F401

    policy = _get_policy_cls_from_policy_name("smile_smolvla")
    assert policy.__name__ == "SmileSmolVLAPolicy"
    from lerobot.policies.factory import _make_processors_from_policy_config  # noqa: F401
    import importlib

    module = importlib.import_module("paw_baseline.smile.processor_smile_smolvla")
    assert hasattr(module, "make_smile_smolvla_pre_post_processors")


def test_smile_samples_every_frame_like_the_control():
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from paw_baseline.smile.configuration_smile_smolvla import SmileSmolVLAConfig

    smile = SmileSmolVLAConfig(chunk_size=50, n_action_steps=15)
    control = SmolVLAConfig(chunk_size=50, n_action_steps=15)
    assert getattr(smile, "drop_n_last_frames", 0) == 0
    assert getattr(control, "drop_n_last_frames", 0) == 0
    assert smile.action_delta_indices == control.action_delta_indices == list(range(50))
