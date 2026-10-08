"""LeRobot Diffusion Policy plus the original EMA weights used for evaluation.

Training is unchanged (LeRobot DiffusionModel, optionally with a compiled
U-Net). After every optimizer step the trainer calls ``update()``, which moves
an eager copy of the whole network (encoders and U-Net) towards the online
weights with the original Diffusion Policy EMA schedule. Inference uses the EMA
copy, as the original workspace evaluates ``ema_model``.
"""

import copy

import torch
from lerobot.policies.diffusion.modeling_diffusion import DiffusionModel, DiffusionPolicy
from lerobot.utils.constants import OBS_IMAGES
from torch import Tensor

from paw_baseline.diffusion_ema.configuration_diffusion_ema import DiffusionEmaConfig
from paw_baseline.ema import EmaSchedule, ema_step


def eager_state_dict(model: torch.nn.Module) -> dict[str, Tensor]:
    """State dict with torch.compile wrapper prefixes removed."""
    return {
        key.replace("._orig_mod", "").replace("_orig_mod.", ""): value
        for key, value in model.state_dict().items()
    }


class DiffusionEmaPolicy(DiffusionPolicy):
    config_class = DiffusionEmaConfig
    name = "diffusion_ema"

    def __init__(self, config: DiffusionEmaConfig, **kwargs):
        super().__init__(config, **kwargs)
        # Original workspace: ema_model = copy.deepcopy(model). Build the copy
        # eagerly so its weights never depend on torch.compile wrappers.
        ema_config = copy.copy(config)
        ema_config.compile_model = False
        self.ema_diffusion = DiffusionModel(ema_config)
        self.ema_diffusion.load_state_dict(eager_state_dict(self.diffusion), strict=True)
        self.ema_diffusion.requires_grad_(False)
        self.register_buffer("ema_optimization_step", torch.zeros((), dtype=torch.long))

    @classmethod
    def from_pretrained(cls, *args, strict: bool = True, **kwargs):
        """Load strictly: silently missing online or EMA weights would invalidate serving."""
        return super().from_pretrained(*args, strict=strict, **kwargs)

    def ema_schedule(self) -> EmaSchedule:
        cfg = self.config
        return EmaSchedule(
            update_after_step=cfg.ema_update_after_step,
            inv_gamma=cfg.ema_inv_gamma,
            power=cfg.ema_power,
            min_value=cfg.ema_min_value,
            max_value=cfg.ema_max_value,
        )

    def update(self) -> None:
        """Called by the LeRobot trainer after every optimizer step."""
        decay = self.ema_schedule().decay(int(self.ema_optimization_step.item()))
        ema_step(self.ema_diffusion, self.diffusion, decay)
        self.ema_optimization_step += 1

    def inference_diffusion(self) -> DiffusionModel:
        return self.ema_diffusion if self.config.use_ema_for_inference else self.diffusion

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """DiffusionPolicy.predict_action_chunk, sampling with the inference (EMA) network."""
        queues_populated = any(len(q) > 0 for q in self._queues.values())
        if queues_populated:
            batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
        else:
            batch = dict(batch)
            if self.config.image_features:
                for key in self.config.image_features:
                    if batch[key].ndim == 4:
                        batch[key] = batch[key].unsqueeze(1)
                batch[OBS_IMAGES] = torch.stack(
                    [batch[key] for key in self.config.image_features], dim=-4
                )
        return self.inference_diffusion().generate_actions(batch, noise=noise)
