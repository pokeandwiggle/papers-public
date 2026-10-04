"""Causal Spectral Policy: spectral algebra, causal stop-gradient and serving API."""

import numpy as np
import pytest
import torch
from lerobot.configs import FeatureType, PolicyFeature

from paw_baseline.csp.configuration_csp import CSPConfig
from paw_baseline.csp.modeling_csp import CSPPolicy, dct_matrix
from paw_baseline.representation import model_names

CAMERAS = (
    "observation.images.rgb_static",
    "observation.images.rgb_left",
    "observation.images.rgb_right",
)
HORIZON = 8


def _config(**overrides) -> CSPConfig:
    kwargs = dict(
        device="cpu",
        horizon=HORIZON,
        n_action_steps=HORIZON,
        n_low_freq=3,
        pretrained_backbone_weights=None,
        n_layer=2,
        dropout=0.0,
        input_features={
            "observation.state": PolicyFeature(FeatureType.STATE, (20,)),
            **{key: PolicyFeature(FeatureType.VISUAL, (3, 64, 64)) for key in CAMERAS},
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (20,))},
        action_feature_names=model_names(),
    )
    kwargs.update(overrides)
    return CSPConfig(**kwargs)


def _batch(batch_size=2, seed=0):
    generator = torch.Generator().manual_seed(seed)
    batch = {
        "observation.state": torch.rand(batch_size, 20, generator=generator) * 2 - 1,
        "action": torch.rand(batch_size, HORIZON, 20, generator=generator) * 2 - 1,
        "action_is_pad": torch.zeros(batch_size, HORIZON, dtype=torch.bool),
    }
    for key in CAMERAS:
        batch[key] = torch.rand(batch_size, 3, 64, 64, generator=generator)
    return batch


def test_dct_is_orthonormal_and_matches_type_ii():
    basis = dct_matrix(32).double()
    assert torch.allclose(basis @ basis.T, torch.eye(32, dtype=torch.float64), atol=1e-6)
    x = np.random.default_rng(0).normal(size=32)
    n = np.arange(32)
    reference = np.array(
        [np.sum(x * np.cos(np.pi / 32 * (n + 0.5) * k)) for k in range(32)]
    )
    reference[0] *= np.sqrt(1 / 32)
    reference[1:] *= np.sqrt(2 / 32)
    assert np.allclose(basis.numpy() @ x, reference, atol=1e-5)


def test_gripper_dims_must_be_gripper_widths():
    with pytest.raises(ValueError, match="gripper"):
        CSPPolicy(_config(time_domain_action_dims=(8, 19)))


def test_loss_equals_time_domain_mse_of_reconstruction():
    policy = CSPPolicy(_config())
    policy.eval()
    batch = _batch()
    loss, metrics = policy.forward(batch)
    c_low, c_high, time_part = policy.model.predict_coefficients(batch)
    reconstruction = policy.model.assemble(c_low, c_high, time_part)
    mse = torch.mean((reconstruction - batch["action"]) ** 2)
    assert torch.allclose(loss, mse, atol=1e-5)
    assert set(metrics) == {"loss_low", "loss_high", "loss_gripper", "high_rel_err"}


def test_low_and_high_targets_split_dct_coefficients():
    policy = CSPPolicy(_config())
    actions = torch.randn(2, HORIZON, 20)
    model = policy.model
    coefficients = model.to_coefficients(actions[..., model.spec_dims])
    rebuilt = model.assemble(
        coefficients[:, : model.n_low],
        coefficients[:, model.n_low :],
        actions[..., model.time_dims],
    )
    assert torch.allclose(rebuilt, actions, atol=1e-5)
    assert model.spec_dims == [d for d in range(20) if d not in (9, 19)]


def test_high_frequency_loss_does_not_reach_low_stage():
    policy = CSPPolicy(_config())
    batch = _batch()
    model = policy.model
    target = model.to_coefficients(batch["action"][..., model.spec_dims])
    c_low, c_high, _ = model.predict_coefficients(batch)
    ((c_high - target[:, model.n_low :]) ** 2).sum().backward()
    low_only = [
        *model.actor["low_trunk"].parameters(),
        *model.actor["low_head"].parameters(),
        model.low_query,
    ]
    assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in low_only)
    assert any(
        p.grad is not None and torch.count_nonzero(p.grad) > 0
        for p in model.actor["coef_proj"].parameters()
    )


def test_high_stage_is_supervised_only_by_high_coefficients():
    policy = CSPPolicy(_config())
    policy.eval()
    batch = _batch()
    model = policy.model
    c_low, _, gripper = model.predict_coefficients(batch)
    with torch.no_grad():
        for parameter in model.actor["high_head"].parameters():
            parameter.add_(1.0)
    c_low_after, _, gripper_after = model.predict_coefficients(batch)
    assert torch.equal(gripper, gripper_after)
    assert torch.equal(c_low, c_low_after)


def test_preset_matches_diffusion_chunking_and_budget():
    import json
    from pathlib import Path

    configs = Path(__file__).resolve().parents[1] / "configs"
    csp = json.loads((configs / "csp.json").read_text())
    diffusion = json.loads((configs / "diffusion.json").read_text())
    for key in ("epochs", "min_steps", "train_all_episodes"):
        assert csp[key] == diffusion[key]
    shared = {
        "--batch_size=64",
        "--num_workers=32",
        "--policy.drop_n_last_frames=0",
        "--policy.horizon=32",
        "--policy.n_action_steps=32",
        "--policy.n_obs_steps=1",
        "--save_freq=5000",
    }
    assert shared <= set(csp["args"]) and shared <= set(diffusion["args"])
    assert "--policy.n_low_freq=8" in csp["args"]


def test_high_stage_conditions_on_low_prediction():
    policy = CSPPolicy(_config())
    policy.eval()
    batch = _batch()
    model = policy.model
    _, c_high, _ = model.predict_coefficients(batch)
    with torch.no_grad():
        model.actor["low_head"][-1].bias.add_(1.0)
    _, c_high_shifted, _ = model.predict_coefficients(batch)
    assert not torch.allclose(c_high, c_high_shifted)


def test_select_action_drains_one_chunk_per_plan(tmp_path):
    policy = CSPPolicy(_config(n_action_steps=5))
    policy.eval()
    batch = {k: v[:1] for k, v in _batch().items() if not k.startswith("action")}
    chunk = policy.predict_action_chunk(batch)
    assert chunk.shape == (1, 5, 20)
    policy.reset()
    steps = [policy.select_action(batch) for _ in range(5)]
    assert torch.allclose(torch.stack(steps, dim=1), chunk)
    assert len(policy._action_queue) == 0

    policy.save_pretrained(tmp_path)
    restored = CSPPolicy.from_pretrained(tmp_path, config=policy.config)
    restored.eval()
    assert torch.allclose(restored.predict_action_chunk(batch), chunk, atol=1e-6)
