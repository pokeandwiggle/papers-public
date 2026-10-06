"""FACT LN timestep schedule and state-relative PI05 actions, without weights."""

import json
from pathlib import Path

import draccus
import numpy as np
import pytest
import torch
from paw_baseline.pi05.geometry import Pi05Representation, relative_action_mask
from paw_baseline.pi05.prepare import chunk_action_values, prepare_pi05
from paw_baseline.prepare import prepare_dataset
from paw_baseline.representation import MODEL_ROTATION_DIMS

from tests.fixture_export import build_export
from tests.test_pi05_geometry import physical

CONFIGS = Path(__file__).parents[1] / "configs"


def _parse_preset(name):
    from paw_baseline.pi05.configuration_paw_pi05 import PawPI05Config

    settings = json.loads((CONFIGS / f"{name}.json").read_text())
    return settings, draccus.parse(
        PawPI05Config,
        args=[
            "--" + arg.removeprefix("--policy.")
            for arg in settings["args"]
            if arg.startswith("--policy.") and not arg.startswith("--policy.type=")
        ],
    )


def test_logit_normal_matches_fact_reference_and_beta_control():
    from lerobot.policies.common.flow_matching import sample_time_beta
    from paw_baseline.pi05.modeling_paw_pi05 import sample_time_logit_normal

    torch.manual_seed(0)
    n = 400_000
    ln = sample_time_logit_normal(n, "cpu", location=1.5, scale=1.0)
    beta = sample_time_beta(n, "cpu", alpha=1.5, beta=1.0, scale=0.999, offset=0.001)
    assert ln.dtype == beta.dtype == torch.float32
    assert float(ln.min()) >= 0.001 and float(ln.max()) <= 0.999
    # Paper Sec. 4.1/5.1: Beta(1.5, 1) puts 8.9% below 0.2; LN(m=1.5) ~6x more.
    ln_low = float((ln < 0.2).float().mean())
    beta_low = float((beta < 0.2).float().mean())
    assert beta_low == pytest.approx(0.2**1.5, abs=0.003)
    assert ln_low == pytest.approx(0.5453, abs=0.003)
    assert ln_low / beta_low == pytest.approx(6.1, abs=0.2)
    # Reference: clip(sigmoid(sigma * z - alpha), 0.001, 0.999).
    z = torch.tensor([-10.0, 0.0, 1.5, 10.0])
    expected = torch.sigmoid(z - 1.5).clamp(0.001, 0.999)
    torch.manual_seed(1)
    drawn = torch.randn(4)
    torch.manual_seed(1)
    np.testing.assert_allclose(
        sample_time_logit_normal(4, "cpu", location=1.5, scale=1.0),
        torch.sigmoid(drawn - 1.5).clamp(0.001, 0.999),
    )
    assert float(expected[1]) == pytest.approx(1 / (1 + np.exp(1.5)))


@pytest.mark.parametrize("schedule", ["beta", "logit_normal"])
def test_policy_installs_only_the_configured_training_sampler(monkeypatch, schedule):
    from types import SimpleNamespace

    from paw_baseline.pi05 import modeling_paw_pi05 as model

    native = object()

    def native_init(self, config, **kwargs):
        torch.nn.Module.__init__(self)
        self.model = torch.nn.Module()
        self.model.sample_time = native

    monkeypatch.setattr(model.PI05Policy, "__init__", native_init)
    monkeypatch.setattr(model, "load_strict_weights", lambda *args: None)
    policy = model.PawPI05Policy(
        SimpleNamespace(
            compile_model=False,
            paw_time_sampling=schedule,
            paw_ln_location=1.5,
            paw_ln_scale=1.0,
        ),
        _weights_path=Path("unused"),
    )
    if schedule == "beta":
        assert policy.model.sample_time is native
        return
    torch.manual_seed(3)
    drawn = policy.model.sample_time(200_000, "cpu")
    assert drawn.shape == (200_000,)
    assert float((drawn < 0.2).float().mean()) == pytest.approx(0.5453, abs=0.005)


def test_presets_differ_only_in_timestep_schedule():
    control, control_config = _parse_preset("pi05_delta")
    treatment, ln_config = _parse_preset("pi05_delta_ln")
    base, _ = _parse_preset("pi05")
    shared = [arg for arg in control["args"] if "paw_time_sampling" not in arg]
    assert shared == [
        arg
        for arg in treatment["args"]
        if "paw_time_sampling" not in arg and "paw_ln_" not in arg
    ]
    assert shared[: len(base["args"])] == base["args"]
    for key in ("model", "epochs", "min_steps", "train_all_episodes"):
        assert control[key] == treatment[key] == base[key]
    for config in (control_config, ln_config):
        assert config.use_relative_actions
        assert config.relative_exclude_joints == ["closed_fraction"]
        assert config.chunk_size == config.n_action_steps == 32
        assert config.num_inference_steps == 10
    assert control_config.paw_time_sampling == "beta"
    assert (
        control_config.time_sampling_beta_alpha,
        control_config.time_sampling_beta_beta,
        control_config.time_sampling_scale,
        control_config.time_sampling_offset,
    ) == (1.5, 1.0, 0.999, 0.001)
    assert ln_config.paw_time_sampling == "logit_normal"
    assert (ln_config.paw_ln_location, ln_config.paw_ln_scale) == (1.5, 1.0)


def test_relative_mask_keeps_closed_fractions_absolute():
    mask = relative_action_mask()
    assert len(mask) == 20
    assert [i for i, value in enumerate(mask) if not value] == [9, 19]


def test_relative_chunk_targets_subtract_anchor_state():
    rep = Pi05Representation()
    action = np.stack([physical()] * 4)
    action[:, 0] = [0.0, 0.1, 0.2, 0.3]
    action[:, 7] = [0.085, 0.04, 0.02, 0.0]
    state = action.copy()
    state[:, 0] -= 0.01
    values = chunk_action_values(
        action, np.zeros(4), {0}, 2, state=state, relative=True
    ).reshape(4, 2, 20)
    encoded_state = rep.encode(state)
    encoded_action = rep.encode(action)
    # anchor 1 predicts actions 1 and 2 relative to state 1
    np.testing.assert_allclose(values[1, :, 0], [0.01, 0.11], atol=1e-6)
    np.testing.assert_allclose(
        values[1, :, 9], encoded_action[[1, 2], 9], atol=1e-6
    )  # closed fraction stays absolute
    np.testing.assert_allclose(
        values[3, :, 3:9], encoded_action[[3, 3], 3:9] - encoded_state[3, 3:9]
    )
    with pytest.raises(ValueError, match="anchor states"):
        chunk_action_values(action, np.zeros(4), {0}, 2, relative=True)


def test_relative_prepared_view_has_distinct_contract_and_stats(tmp_path):
    source = build_export(tmp_path / "source", episodes=2)
    base = prepare_dataset(source, cache_root=tmp_path / "cache")
    absolute = prepare_pi05(base, 8)
    relative = prepare_pi05(base, 8, relative=True)
    assert relative.root != absolute.root
    assert prepare_pi05(base, 8, relative=True) == relative
    stamp = json.loads((relative.root / ".paw/prepare.json").read_text())["pi05"]
    assert stamp["relative_actions"] == {
        "mask": list(relative_action_mask()),
        "reference": "anchor_observation_state",
    }
    assert "relative_actions" not in json.loads(
        (absolute.root / ".paw/prepare.json").read_text()
    )["pi05"]
    rel_stats = json.loads((relative.root / "meta/stats.json").read_text())
    abs_stats = json.loads((absolute.root / "meta/stats.json").read_text())
    # absolute states unchanged; relative action stats differ on moving dims
    assert rel_stats["observation.state"] == abs_stats["observation.state"]
    assert rel_stats["action"]["q99"][0] < abs_stats["action"]["q99"][0]
    for dim in (9, 19):
        assert rel_stats["action"]["q01"][dim] == 0.0
        assert rel_stats["action"]["q99"][dim] == 1.0
    for dim in MODEL_ROTATION_DIMS:
        assert rel_stats["observation.state"]["q01"][dim] == -1.0
    # identity rotations everywhere: relative rotation offsets are fitted (zero)
    for dim in MODEL_ROTATION_DIMS:
        assert abs(rel_stats["action"]["q99"][dim]) < 1e-6


def _relative_checkpoint(tmp_path, monkeypatch, relative=True):
    from lerobot.configs import FeatureType, PolicyFeature
    from paw_baseline.layout import CAMERA_KEYS
    from paw_baseline.pi05.configuration_paw_pi05 import PawPI05Config
    from paw_baseline.pi05.processor_paw_pi05 import (
        make_paw_pi05_pre_post_processors,
    )
    from paw_baseline.serving_metadata import (
        metadata_from_prepared,
        write_serving_metadata,
    )

    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained", lambda *a, **k: object()
    )
    source = build_export(tmp_path / "source", episodes=2)
    base = prepare_dataset(source, cache_root=tmp_path / "cache")
    prepared = prepare_pi05(base, 8, relative=relative)
    info = json.loads((prepared.root / "meta/info.json").read_text())
    config = PawPI05Config(
        device="cpu",
        chunk_size=8,
        n_action_steps=8,
        use_relative_actions=relative,
        relative_exclude_joints=["closed_fraction"],
        action_feature_names=info["features"]["action"]["names"],
        input_features={
            "observation.state": PolicyFeature(FeatureType.STATE, (20,)),
            **{
                key: PolicyFeature(FeatureType.VISUAL, (3, 224, 224))
                for key in CAMERA_KEYS
            },
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (20,))},
    )
    config.validate_features()
    stats = json.loads((prepared.root / "meta/stats.json").read_text())
    pre, post = make_paw_pi05_pre_post_processors(config, stats)
    checkpoint = tmp_path / "checkpoint"
    config.save_pretrained(checkpoint)
    pre.save_pretrained(checkpoint, config_filename="policy_preprocessor.json")
    post.save_pretrained(checkpoint, config_filename="policy_postprocessor.json")
    metadata = metadata_from_prepared(prepared.root, model="pi05_delta")
    write_serving_metadata(checkpoint, metadata)
    return checkpoint, config, stats, metadata


def test_saved_relative_processors_roundtrip_to_absolute(tmp_path, monkeypatch):
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.processor import NormalizerProcessorStep
    from paw_baseline.serving_metadata import load_serving_metadata

    checkpoint, config, stats, metadata = _relative_checkpoint(tmp_path, monkeypatch)
    assert load_serving_metadata(checkpoint) == metadata
    pre, post = make_pre_post_processors(config, pretrained_path=checkpoint)
    rep = Pi05Representation()
    state = physical((0.3, -0.2))
    target = np.stack([physical((0.35, -0.25))] * 8)
    target[:, 0] += np.linspace(0.0, 0.03, 8)
    target[:, 7] = 0.03
    state_t = torch.tensor(rep.encode(state)[None])
    action_t = torch.tensor(rep.encode(target)[None])
    # Training-time view: stop before tokenization, after normalization.
    index = next(
        i for i, step in enumerate(pre.steps) if isinstance(step, NormalizerProcessorStep)
    )
    transition = pre.to_transition(
        {"observation.state": state_t, "action": action_t, "task": "t"}
    )
    for step in pre.steps[: index + 1]:
        transition = step(transition)
    from lerobot.lerobot_types import TransitionKey

    normalized = transition[TransitionKey.ACTION]
    q01 = torch.tensor(stats["action"]["q01"])
    q99 = torch.tensor(stats["action"]["q99"])
    mask = torch.tensor(relative_action_mask())
    relative = action_t.clone()
    relative[..., mask] -= state_t[:, None, mask]
    expected = 2 * (relative - q01) / (q99 - q01).clamp_min(1e-8) - 1
    torch.testing.assert_close(normalized, expected, atol=1e-5, rtol=1e-5)
    # Serving view: the postprocessor unnormalizes and restores absolute poses.
    restored = torch.stack(
        [post(normalized[:, i]) for i in range(normalized.shape[1])], dim=1
    )
    torch.testing.assert_close(restored, action_t, atol=1e-5, rtol=1e-5)


def test_serving_rejects_relative_checkpoint_without_absolute_restore(
    tmp_path, monkeypatch
):
    from paw_baseline.serving_metadata import load_serving_metadata

    checkpoint, *_ = _relative_checkpoint(tmp_path, monkeypatch)
    path = checkpoint / "policy_postprocessor.json"
    saved = json.loads(path.read_text())
    for step in saved["steps"]:
        if step.get("registry_name") == "absolute_actions_processor":
            step["config"]["enabled"] = False
    path.write_text(json.dumps(saved))
    with pytest.raises(RuntimeError, match="back to absolute"):
        load_serving_metadata(checkpoint)
    native = json.loads((checkpoint / "config.json").read_text())
    native["use_relative_actions"] = False
    (checkpoint / "config.json").write_text(json.dumps(native))
    with pytest.raises(RuntimeError, match="disagree"):
        load_serving_metadata(checkpoint)


def test_relative_config_requires_named_closed_fractions():
    from lerobot.configs import FeatureType, PolicyFeature
    from paw_baseline.layout import CAMERA_KEYS
    from paw_baseline.pi05.configuration_paw_pi05 import PawPI05Config

    with pytest.raises(ValueError, match="closed fractions"):
        PawPI05Config(device="cpu", use_relative_actions=True)
    with pytest.raises(ValueError, match="time sampling"):
        PawPI05Config(device="cpu", paw_time_sampling="uniform")
    config = PawPI05Config(
        device="cpu",
        use_relative_actions=True,
        relative_exclude_joints=["closed_fraction"],
        action_feature_names=[f"a{i}" for i in range(20)],
        input_features={
            "observation.state": PolicyFeature(FeatureType.STATE, (20,)),
            **{
                key: PolicyFeature(FeatureType.VISUAL, (3, 224, 224))
                for key in CAMERA_KEYS
            },
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (20,))},
    )
    with pytest.raises(ValueError, match="closed-fraction"):
        config.validate_features()
