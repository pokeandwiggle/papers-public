import json
from pathlib import Path

import draccus
import pytest
import torch
from paw_baseline.pi05.configuration_paw_pi05 import PawPI05Config


@pytest.mark.parametrize("steps", [10_000, 50_000])
def test_training_preset_keeps_full_warmup_then_constant_lr(steps):
    settings = json.loads(
        (Path(__file__).parents[1] / "configs" / "pi05.json").read_text()
    )
    config = draccus.parse(
        PawPI05Config,
        args=[
            "--" + arg.removeprefix("--policy.")
            for arg in settings["args"]
            if arg.startswith("--policy.") and not arg.startswith("--policy.type=")
        ],
    )
    assert "--batch_size=64" in settings["args"]
    assert config.chunk_size == config.n_action_steps == 32
    assert config.compile_model
    assert config.freeze_vision_encoder
    optimizer = config.get_optimizer_preset().build(
        [torch.nn.Parameter(torch.zeros(()))]
    )
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(1e-10, abs=0)
    scheduler = config.get_scheduler_preset().build(optimizer, steps)
    rates = {}
    for step in range(steps + 1):
        if step in (0, 250, 499, 500, 501, steps // 2, steps):
            rates[step] = scheduler.get_last_lr()[0]
        optimizer.step()
        scheduler.step()
    assert rates[0] == 0
    assert rates[250] == pytest.approx(2.5e-5)
    assert rates[499] == pytest.approx(4.99e-5)
    for step in (500, 501, steps // 2, steps):
        assert rates[step] == pytest.approx(5e-5)
