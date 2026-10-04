"""Training compiler assumptions must match native AMP and remain scoped."""

import pytest
from torch._functorch import config
from torch._inductor import config as inductor
from train.compilation import training_compilation


def test_pi05_backward_matches_native_amp_and_restores_on_failure():
    previous = config.backward_pass_autocast
    previous_casts = inductor.emulate_precision_casts
    with (
        pytest.raises(RuntimeError, match="training failed"),
        training_compilation("paw_pi05"),
    ):
        assert config.backward_pass_autocast == "off"
        assert inductor.emulate_precision_casts
        raise RuntimeError("training failed")
    assert config.backward_pass_autocast == previous
    assert inductor.emulate_precision_casts == previous_casts


@pytest.mark.parametrize("model", [None, "act", "multi_task_dit", "smolvla"])
def test_other_families_retain_their_compiler_configuration(model):
    with (
        config.patch(backward_pass_autocast="same_as_forward"),
        inductor.patch(emulate_precision_casts=False),
        training_compilation(model),
    ):
        assert config.backward_pass_autocast == "same_as_forward"
        assert not inductor.emulate_precision_casts


@pytest.mark.parametrize("compile_model", [False, True])
def test_policy_load_preserves_conservative_compile_precision(
    monkeypatch, compile_model
):
    from pathlib import Path
    from types import SimpleNamespace

    import torch
    from paw_baseline.pi05 import modeling_paw_pi05 as model

    def native_init(self, config, **kwargs):
        torch.nn.Module.__init__(self)
        if config.compile_model:
            torch.set_float32_matmul_precision("high")

    monkeypatch.setattr(model.PI05Policy, "__init__", native_init)
    monkeypatch.setattr(model, "load_strict_weights", lambda *args: None)
    previous = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        model.PawPI05Policy(
            SimpleNamespace(compile_model=compile_model), _weights_path=Path("unused")
        )
        assert torch.get_float32_matmul_precision() == "highest"
    finally:
        torch.set_float32_matmul_precision(previous)


def test_diffusion_precision_applies_after_trainer_setup_and_restores_on_failure():
    from types import SimpleNamespace

    import torch
    from train.compilation import training_policy_precision

    policy = object()

    def factory(**kwargs):
        return policy

    trainer = SimpleNamespace(make_policy=factory)
    precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("high")
        with (
            pytest.raises(RuntimeError, match="training failed"),
            training_policy_precision(trainer, "diffusion"),
        ):
            # LeRobot enables TF32 during initialization.
            torch.set_float32_matmul_precision("high")
            assert trainer.make_policy(cfg=None) is policy
            assert torch.get_float32_matmul_precision() == "highest"
            raise RuntimeError("training failed")
        assert trainer.make_policy is factory
        assert torch.get_float32_matmul_precision() == "high"
    finally:
        torch.set_float32_matmul_precision(precision)
