"""Exponential moving average of policy weights, shared by ManiFlow and Diffusion Policy.

Replicates ``EMAModel`` from real-stanford/diffusion_policy@5ba07ac and
geyan21/ManiFlow_Policy@ef2f116 (byte-identical in both repositories):
warmup decay ``1 - (1 + step / inv_gamma) ** -power`` clipped to
[min_value, max_value], zero for the first two steps; BatchNorm and frozen
parameters are copied rather than averaged.

Adaptation: the references use GroupNorm encoders and never touch buffers.
The shared ImageNet ResNet-18 keeps BatchNorm, so its running statistics are
copied from the online network; otherwise the served EMA encoder would keep
its initial statistics.
"""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm


@dataclass(frozen=True)
class EmaSchedule:
    update_after_step: int = 0
    inv_gamma: float = 1.0
    power: float = 0.75
    min_value: float = 0.0
    max_value: float = 0.9999

    def decay(self, optimization_step: int) -> float:
        step = max(0, optimization_step - self.update_after_step - 1)
        value = 1 - (1 + step / self.inv_gamma) ** -self.power
        if step <= 0:
            return 0.0
        return max(self.min_value, min(value, self.max_value))


def _canonical(name: str) -> str:
    """Ignore torch.compile wrappers so compiled and eager copies match by name."""
    return name.replace("._orig_mod", "").replace("_orig_mod.", "").replace("_orig_mod", "")


def _modules_by_name(model: nn.Module) -> dict[str, nn.Module]:
    modules = {}
    for name, module in model.named_modules():
        if hasattr(module, "_orig_mod"):  # OptimizedModule wrapper owns no parameters
            continue
        modules[_canonical(name)] = module
    return modules


@torch.no_grad()
def ema_step(averaged: nn.Module, model: nn.Module, decay: float) -> None:
    """One reference EMA update of ``averaged`` towards ``model``."""
    online = _modules_by_name(model)
    for name, ema_module in _modules_by_name(averaged).items():
        module = online[name]
        ema_params = list(ema_module.named_parameters(recurse=False))
        params = list(module.named_parameters(recurse=False))
        if [n for n, _ in ema_params] != [n for n, _ in params]:
            raise RuntimeError(f"EMA parameter layout differs at module {name!r}")
        for (_, ema_param), (_, param) in zip(ema_params, params, strict=True):
            if isinstance(module, _BatchNorm) or not param.requires_grad:
                ema_param.copy_(param.detach().to(dtype=ema_param.dtype))
            else:
                ema_param.mul_(decay)
                ema_param.add_(param.detach().to(dtype=ema_param.dtype), alpha=1 - decay)
        buffers = dict(module.named_buffers(recurse=False))
        for buffer_name, ema_buffer in ema_module.named_buffers(recurse=False):
            ema_buffer.copy_(buffers[buffer_name])
