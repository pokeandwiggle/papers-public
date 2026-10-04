"""Differential tests of the OFT/ActionMap heads against the pinned upstream code.

Upstream sources are read from the workspace reference clone
(showlab/ActionMap @ 58721300b42f8ab9b6fd45921259a8203e2009fe) and are not
vendored; the tests skip when the clone is absent.
"""

import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest
import torch

from paw_baseline.oft.heads import HeatmapActionHead, L1RegressionActionHead

REFERENCE = Path(
    os.environ.get(
        "ACTIONMAP_REFERENCE",
        Path(__file__).resolve().parents[2] / "tmp/references/actionmap/actionmap_oft",
    )
)
pytestmark = pytest.mark.skipif(
    not (REFERENCE / "actionmap/heatmap_action_head.py").is_file(),
    reason="ActionMap reference clone not available",
)


def _load(name: str, path: Path, stubs: dict[str, types.ModuleType] | None = None):
    saved = {key: sys.modules.get(key) for key in (stubs or {})}
    sys.modules.update(stubs or {})
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value


def _reference_heatmap():
    return _load("ref_heatmap", REFERENCE / "actionmap/heatmap_action_head.py").HeatmapActionHead


def _reference_l1(action_dim: int, chunk: int):
    constants = types.ModuleType("prismatic.vla.constants")
    constants.ACTION_DIM, constants.NUM_ACTIONS_CHUNK = action_dim, chunk
    ddim = types.ModuleType("diffusers.schedulers.scheduling_ddim")
    ddim.DDIMScheduler = object
    heatmap = types.ModuleType("actionmap.heatmap_action_head")
    heatmap.HeatmapActionHead = object
    stubs = {
        "prismatic": types.ModuleType("prismatic"),
        "prismatic.vla": types.ModuleType("prismatic.vla"),
        "prismatic.vla.constants": constants,
        "diffusers": types.ModuleType("diffusers"),
        "diffusers.schedulers": types.ModuleType("diffusers.schedulers"),
        "diffusers.schedulers.scheduling_ddim": ddim,
        "actionmap": types.ModuleType("actionmap"),
        "actionmap.heatmap_action_head": heatmap,
    }
    module = _load("ref_heads", REFERENCE / "prismatic/models/action_heads.py", stubs)
    return module.L1RegressionActionHead


def test_l1_head_matches_openvla_oft():
    torch.manual_seed(0)
    dim, hidden, action_dim, chunk = 8, 16, 14, 4
    reference = _reference_l1(action_dim, chunk)(dim, hidden, action_dim)
    ours = L1RegressionActionHead(dim, hidden, action_dim, chunk)
    ours.load_state_dict(reference.state_dict(), strict=True)
    states = torch.randn(3, chunk * action_dim, dim)
    torch.testing.assert_close(ours.predict_action(states), reference.predict_action(states))


@pytest.mark.parametrize("mode", ["topk", "soft_argmax", "hard_argmax"])
def test_single_arm_heatmap_matches_actionmap(mode):
    torch.manual_seed(0)
    dim, chunk = 8, 4
    kwargs = dict(trans_grid=(6, 5, 4), rot_grid=(4, 4, 3), trans_sigma=0.1, rot_sigma=0.1)
    reference = _reference_heatmap()(
        input_dim=dim, hidden_dim=dim, action_dim=7, num_actions_chunk=chunk,
        loss_type="ce", decode_mode=mode, decode_top_k=10, decode_temperature=1.0,
        grid_pad=0.2, **kwargs,
    )
    ours = HeatmapActionHead(
        dim, dim, 7, chunk, num_arms=1, grid_pad=0.2,
        decode_mode=mode, decode_top_k=10, decode_temperature=1.0, **kwargs,
    )
    state = reference.state_dict()
    renamed = {}
    for key, value in state.items():
        for old, new in (("trans_head.", "trans_heads.0."), ("rot_head.", "rot_heads.0."),
                         ("grip_head.", "grip_heads.0.")):
            if key.startswith(old):
                key = new + key[len(old):]
        renamed[key] = value
    ours.load_state_dict(renamed, strict=True)
    hidden = torch.randn(3, chunk * 7, dim)
    truth = torch.rand(3, chunk, 7) * 2 - 1
    truth[..., 6] = torch.where(truth[..., 6] > 0, 1.0, -1.0)
    _, ref_loss = reference.predict_action_with_loss(hidden, truth)
    _, loss, _ = ours.predict_action_with_loss(hidden, truth)
    torch.testing.assert_close(loss, ref_loss)
    torch.testing.assert_close(ours.predict_action(hidden), reference.predict_action(hidden))


def test_bimanual_heatmap_sums_arm_losses_over_shared_trunk():
    torch.manual_seed(0)
    dim, chunk = 8, 3
    head = HeatmapActionHead(dim, dim, 14, chunk, trans_grid=(4, 4, 4), rot_grid=(3, 3, 3))
    hidden = torch.randn(2, chunk * 14, dim)
    truth = torch.rand(2, chunk, 14) * 2 - 1
    decoded, loss, metrics = head.predict_action_with_loss(hidden, truth)
    parts = sum(value for key, value in metrics.items() if key.startswith("arm"))
    torch.testing.assert_close(loss, parts)
    assert decoded.shape == (2, chunk, 14)
    torch.testing.assert_close(decoded, head.predict_action(hidden))
