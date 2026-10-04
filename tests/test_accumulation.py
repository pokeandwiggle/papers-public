"""Gradient accumulation must reproduce the full-batch gradient and step once."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from train.accumulation import make_accumulating_update


class _Accelerator:
    def no_sync(self, model):
        return nullcontext()

    def autocast(self):
        return nullcontext()

    def backward(self, loss):
        loss.backward()


class _Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 1)

    def forward(self, batch):
        pred = self.linear(batch["observation.state"]).squeeze(-1)
        assert len(batch["task"]) == pred.shape[0]
        return ((pred - batch["action"]) ** 2).mean(), {"metric": 1.0}


class _Metrics(SimpleNamespace):
    def update_metrics(self, values):
        self.values = values


@pytest.mark.parametrize("size,steps", [(8, 2), (7, 2), (5, 4)])
def test_accumulated_gradient_equals_full_batch(size, steps):
    torch.manual_seed(0)
    batch = {
        "observation.state": torch.randn(size, 3),
        "action": torch.randn(size),
        "task": ["t"] * size,
    }
    reference = _Policy()
    policy = _Policy()
    policy.load_state_dict(reference.state_dict())
    loss, _ = reference(batch)
    loss.backward()
    optimizer = torch.optim.SGD(policy.parameters(), lr=1.0)
    before = [p.detach().clone() for p in policy.parameters()]
    metrics, outputs = make_accumulating_update(steps)(
        _Metrics(), policy, batch, optimizer, 0.0, _Accelerator()
    )
    for p0, p, ref in zip(before, policy.parameters(), reference.parameters()):
        assert torch.allclose(p0 - p, ref.grad, atol=1e-6)
    assert metrics.loss == pytest.approx(loss.item(), rel=1e-5)
    assert outputs["metric"] == pytest.approx(1.0)
