"""Inference variants must change effective config without changing weights."""

import json
from types import SimpleNamespace

import pytest
from paw_baseline.variant import (
    build_variant,
    load_policy_config,
    parse_inference_config,
)


def test_variant_preserves_package_and_resets_local_identity(tmp_path):
    source = tmp_path / "source"
    (source / ".paw").mkdir(parents=True)
    (source / "config.json").write_text(
        json.dumps({"type": "multi_task_dit", "num_inference_steps": 10})
    )
    (source / "model.safetensors").write_bytes(b"same weights")
    (source / ".paw/validation_probes.npz").write_bytes(b"probes")
    (source / ".paw/validation_receipt.json").write_text("old receipt")
    (source / ".paw/model_registration.json").write_text("old key")
    (source / "teacher").mkdir()
    (source / "teacher/stats.json").write_text('{"auxiliary": true}')
    output = tmp_path / "variant"
    build_variant(source, output, {"schema_version": 1, "num_inference_steps": 5})
    assert (output / "model.safetensors").read_bytes() == (
        source / "model.safetensors"
    ).read_bytes()
    assert (output / "config.json").read_bytes() == (
        source / "config.json"
    ).read_bytes()
    assert (output / ".paw/validation_probes.npz").read_bytes() == b"probes"
    assert (output / "teacher/stats.json").read_text() == '{"auxiliary": true}'
    assert not (output / ".paw/validation_receipt.json").exists()
    assert not (output / ".paw/model_registration.json").exists()
    assert (source / ".paw/model_registration.json").read_text() == "old key"
    assert json.loads((output / ".paw/inference.json").read_text()) == {
        "schema_version": 1,
        "num_inference_steps": 5,
    }
    with pytest.raises(FileExistsError):
        build_variant(source, output, {"schema_version": 1, "num_inference_steps": 2})


@pytest.mark.parametrize(
    "settings",
    [
        {"schema_version": 2, "num_inference_steps": 5},
        {"schema_version": 1, "num_inference_steps": True},
        {"schema_version": 1, "num_inference_steps": 0},
        {"schema_version": 1, "num_inference_steps": 1.5},
        {"schema_version": 1, "num_inference_steps": 5, "typo": 1},
        {"schema_version": 1},
    ],
)
def test_invalid_settings_fail_before_package_creation(settings):
    with pytest.raises(ValueError):
        parse_inference_config(settings)


def test_unsupported_policy_cannot_silently_ignore_settings(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text('{"type":"act"}')
    with pytest.raises(
        ValueError, match="multi_task_dit.*diffusion|diffusion.*multi_task_dit"
    ):
        build_variant(
            source, tmp_path / "output", {"schema_version": 1, "num_inference_steps": 5}
        )
    assert not (tmp_path / "output").exists()


def test_loader_applies_settings_before_policy_construction(tmp_path, monkeypatch):
    from lerobot.configs.policies import PreTrainedConfig

    (tmp_path / ".paw").mkdir()
    (tmp_path / "config.json").write_text(
        '{"type":"multi_task_dit","num_inference_steps":10}'
    )
    (tmp_path / ".paw/inference.json").write_text(
        '{"schema_version":1,"num_inference_steps":5}'
    )
    native_config = SimpleNamespace(type="multi_task_dit", num_inference_steps=10)
    monkeypatch.setattr(PreTrainedConfig, "from_pretrained", lambda path: native_config)
    assert load_policy_config(tmp_path).num_inference_steps == 5


def test_original_package_loads_native_configuration(tmp_path, monkeypatch):
    from lerobot.configs.policies import PreTrainedConfig

    (tmp_path / "config.json").write_text('{"type":"act"}')
    config = SimpleNamespace(type="act")
    monkeypatch.setattr(PreTrainedConfig, "from_pretrained", lambda path: config)
    assert load_policy_config(tmp_path) is config


def test_flow_matching_uses_integration_steps_and_rejects_ignored_diffusion_field(
    tmp_path, monkeypatch
):
    from lerobot.configs.policies import PreTrainedConfig

    (tmp_path / ".paw").mkdir()
    (tmp_path / "config.json").write_text(
        '{"type":"multi_task_dit","objective":"flow_matching","num_integration_steps":10}'
    )
    config = SimpleNamespace(type="multi_task_dit", num_integration_steps=10)
    monkeypatch.setattr(PreTrainedConfig, "from_pretrained", lambda path: config)
    sidecar = tmp_path / ".paw/inference.json"
    sidecar.write_text('{"schema_version":1,"num_inference_steps":5}')
    with pytest.raises(ValueError, match="requires num_integration_steps"):
        load_policy_config(tmp_path)
    sidecar.write_text('{"schema_version":1,"num_integration_steps":5}')
    assert load_policy_config(tmp_path).num_integration_steps == 5


@pytest.mark.parametrize("steps", [1, 5])
def test_variant_controls_real_lerobot_euler_evaluations(tmp_path, steps):
    import torch
    from lerobot.policies.multi_task_dit.modeling_multi_task_dit import (
        FlowMatchingObjective,
    )

    (tmp_path / ".paw").mkdir()
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "type": "multi_task_dit",
                "objective": "flow_matching",
                "device": "cpu",
                "num_integration_steps": 10,
                "integration_method": "euler",
            }
        )
    )
    (tmp_path / ".paw/inference.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "num_integration_steps": steps,
            }
        )
    )

    class ConstantVelocity(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.calls = 0

        def forward(self, x, t, *, conditioning_vec):
            self.calls += 1
            return torch.zeros_like(x)

    model = ConstantVelocity()
    objective = FlowMatchingObjective(
        load_policy_config(tmp_path), action_dim=20, horizon=2
    )
    result = objective.conditional_sample(model, 1, torch.zeros(1, 2))
    assert result.shape == (1, 2, 20)
    assert model.calls == steps


@pytest.mark.parametrize("policy_type", ["act", "diffusion", "multi_task_dit"])
@pytest.mark.parametrize("steps", [1, 8, 32])
def test_action_steps_override_keeps_training_shape(tmp_path, policy_type, steps):
    source = tmp_path / "source"
    source.mkdir()
    native = {"type": policy_type, "n_action_steps": 32, "device": "cpu"}
    native["chunk_size" if policy_type == "act" else "horizon"] = 32
    (source / "config.json").write_text(json.dumps(native))
    (source / "model.safetensors").write_bytes(b"same weights")
    output = build_variant(
        source, tmp_path / "variant", {"schema_version": 1, "n_action_steps": steps}
    )
    effective = load_policy_config(output)
    assert effective.n_action_steps == steps
    assert getattr(effective, "chunk_size" if policy_type == "act" else "horizon") == 32
    assert json.loads((output / "config.json").read_text()) == native
    assert (output / "model.safetensors").read_bytes() == b"same weights"


@pytest.mark.parametrize("steps", [0, -1, True, 1.5, "8", None, 33])
def test_invalid_action_steps_refused_before_copy(tmp_path, steps):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps({"type": "multi_task_dit", "n_action_steps": 32})
    )
    (source / "model.safetensors").write_bytes(b"weights")
    with pytest.raises(ValueError, match="n_action_steps"):
        build_variant(
            source, tmp_path / "out", {"schema_version": 1, "n_action_steps": steps}
        )
    assert not (tmp_path / "out").exists()


def test_action_steps_combine_with_sampler_and_replace_variant_settings(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps(
            {
                "type": "multi_task_dit",
                "objective": "flow_matching",
                "device": "cpu",
                "n_action_steps": 32,
                "num_integration_steps": 10,
            }
        )
    )
    (source / "model.safetensors").write_bytes(b"weights")
    first = build_variant(
        source,
        tmp_path / "first",
        {
            "schema_version": 1,
            "n_action_steps": 8,
            "num_integration_steps": 5,
        },
    )
    assert load_policy_config(first).n_action_steps == 8
    assert load_policy_config(first).num_integration_steps == 5
    second = build_variant(
        first, tmp_path / "second", {"schema_version": 1, "n_action_steps": 16}
    )
    assert load_policy_config(second).n_action_steps == 16
    assert load_policy_config(second).num_integration_steps == 10


@pytest.mark.parametrize("steps", [1, 3])
def test_dit_native_prefix_consumes_queue_and_replans_from_next_observation(
    tmp_path, steps
):
    import torch
    from lerobot.policies.multi_task_dit.modeling_multi_task_dit import (
        MultiTaskDiTPolicy,
    )

    (tmp_path / ".paw").mkdir()
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "type": "multi_task_dit",
                "device": "cpu",
                "horizon": 8,
                "n_action_steps": 8,
                "n_obs_steps": 1,
                "input_features": {
                    "observation.state": {"type": "STATE", "shape": [1]}
                },
                "output_features": {"action": {"type": "ACTION", "shape": [1]}},
            }
        )
    )
    (tmp_path / ".paw/inference.json").write_text(
        json.dumps({"schema_version": 1, "n_action_steps": steps})
    )
    policy = MultiTaskDiTPolicy.__new__(MultiTaskDiTPolicy)
    torch.nn.Module.__init__(policy)
    policy.config = load_policy_config(tmp_path)
    policy.observation_encoder = SimpleNamespace(
        encode=lambda batch: batch["observation.state"]
    )
    calls = []

    def sample(model, batch_size, conditioning):
        calls.append(conditioning.clone())
        return conditioning[:, :1, :] + torch.arange(8).reshape(1, 8, 1)

    policy.objective = SimpleNamespace(conditional_sample=sample)
    policy.noise_predictor = None
    policy.reset()
    for observation in (10.0, 100.0):
        actual = torch.stack(
            [
                policy.select_action(
                    {"observation.state": torch.tensor([[observation]])}
                )
                for _ in range(steps)
            ],
            dim=1,
        )
        torch.testing.assert_close(
            actual,
            observation + torch.arange(steps, dtype=torch.float32).reshape(1, steps, 1),
        )
    assert len(calls) == 2
