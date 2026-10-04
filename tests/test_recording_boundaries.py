"""The native reader uses standard exported episode boundaries."""

import json

import numpy as np
import pytest
import torch
from lerobot.datasets import lerobot_dataset
from paw_baseline.prepare import prepare_dataset

from tests.fixture_export import build_export


@pytest.mark.parametrize("objective_name", ["diffusion", "flow_matching"])
def test_dit_preset_excludes_episode_padding_from_native_objective(objective_name):
    from lerobot.policies.multi_task_dit.configuration_multi_task_dit import (
        MultiTaskDiTConfig,
    )
    from lerobot.policies.multi_task_dit.modeling_multi_task_dit import (
        DiffusionObjective,
        FlowMatchingObjective,
    )
    from train.__main__ import load_preset

    flag = next(
        (
            arg.split("=", 1)[1]
            for arg in load_preset("multi_task_dit")["args"]
            if arg.startswith("--policy.do_mask_loss_for_padding=")
        ),
        "false",
    )
    config = MultiTaskDiTConfig(
        n_obs_steps=1, horizon=3, n_action_steps=3, timestep_sampling_strategy="uniform"
    )
    objective_class = {
        "diffusion": DiffusionObjective,
        "flow_matching": FlowMatchingObjective,
    }[objective_name]
    objective = objective_class(
        config, action_dim=1, horizon=3, do_mask_loss_for_padding=json.loads(flag)
    )
    predicted = torch.zeros(1, 3, 1, requires_grad=True)
    torch.manual_seed(0)
    loss = objective.compute_loss(
        lambda *args, **kwargs: predicted,
        {
            "action": torch.zeros(1, 3, 1),
            "action_is_pad": torch.tensor([[False, True, True]]),
        },
        torch.zeros(1, 1),
    )
    loss.backward()
    assert predicted.grad[0, 0, 0] != 0
    assert torch.count_nonzero(predicted.grad[0, 1:]) == 0


@pytest.fixture
def episode_dataset(tmp_path):
    export = build_export(tmp_path / "export", episodes=2)
    prepared = prepare_dataset(export, cache_root=tmp_path / "cache")
    return lerobot_dataset.LeRobotDataset(
        prepared.root.name,
        root=prepared.root,
        episodes=[1],
        delta_timestamps={
            "action": [0, 1 / 30, 2 / 30],
            "observation.state": [-1 / 30, 0],
        },
        video_backend="pyav",
    )


def test_native_reader_crosses_interior_frames_without_fragment_metadata(
    episode_dataset,
):
    item = episode_dataset[23]
    assert item["action_is_pad"].tolist() == [False, False, False]
    assert episode_dataset[24]["observation.state_is_pad"].tolist() == [False, False]


def test_native_reader_pads_at_selected_episode_boundaries(episode_dataset):
    item = episode_dataset[47]
    assert item["action_is_pad"].tolist() == [False, True, True]
    np.testing.assert_array_equal(item["action"][0], item["action"][2])
    assert episode_dataset[0]["observation.state_is_pad"].tolist() == [True, False]


def test_validation_uses_whole_episode_without_fragment_metadata(episode_dataset):
    from inference.validate import dataset_probes

    samples, _ = dataset_probes(
        episode_dataset.root, [], horizon=30, count=1, holdout={1}
    )
    assert samples
    assert all(sample["probe"].episode_index == 1 for sample in samples)
    assert all(sample["probe"].frame + len(sample["truth"]) <= 48 for sample in samples)
