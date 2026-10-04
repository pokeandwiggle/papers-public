import numpy as np
import pytest
import torch
from paw_baseline.pi05.geometry import Pi05Representation

from tests.test_pi05_geometry import physical


def test_loader_requires_pinned_revision_without_allocating_model():
    from paw_baseline.pi05.modeling_paw_pi05 import PawPI05Policy

    with pytest.raises(ValueError, match="immutable"):
        PawPI05Policy.from_pretrained("lerobot/pi05_base", revision="main")


def test_weight_load_errors_propagate(tmp_path):
    from paw_baseline.pi05.modeling_paw_pi05 import load_strict_weights

    with pytest.raises(FileNotFoundError):
        load_strict_weights(object(), tmp_path / "model.safetensors")


def test_padding_survives_processor_serialization(tmp_path):
    from lerobot.processor import PolicyProcessorPipeline
    from paw_baseline.pi05.processor_paw_pi05 import PadNormalizedState

    rep = Pi05Representation()
    pipeline = PolicyProcessorPipeline(steps=[PadNormalizedState()])
    pipeline.save_pretrained(tmp_path, config_filename="processor.json")
    loaded = PolicyProcessorPipeline.from_pretrained(
        tmp_path, config_filename="processor.json"
    )
    state = physical()[None]
    target = state[:, None].copy()
    target[..., 0] += 0.02
    batch = {
        "observation.state": torch.tensor(rep.encode(state)),
        "action": torch.tensor(rep.encode(target)),
    }
    output = loaded(batch)
    assert output["observation.state"].shape == (1, 32)
    np.testing.assert_allclose(output["observation.state"][..., 20:], 0)
    np.testing.assert_allclose(output["action"], rep.encode(target), atol=1e-6)
    assert batch["observation.state"].shape == (1, 20)


def test_incompatible_weights_fail_strictly(tmp_path):
    from types import SimpleNamespace

    from paw_baseline.pi05.modeling_paw_pi05 import load_strict_weights
    from safetensors.torch import save_file

    model = torch.nn.Linear(2, 2)
    model.config = SimpleNamespace()
    model._fix_pytorch_state_dict_keys = lambda weights, config: weights
    path = tmp_path / "model.safetensors"
    save_file({"wrong": torch.ones(2)}, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_strict_weights(model, path)
