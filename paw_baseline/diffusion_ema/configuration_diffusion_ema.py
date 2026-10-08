"""Diffusion Policy configuration with an EMA copy of the network for inference.

EMA settings follow real-stanford/diffusion_policy@5ba07ac
(train_diffusion_unet_real_hybrid_workspace.yaml: update_after_step 0,
inv_gamma 1.0, power 0.75, min_value 0.0, max_value 0.9999; evaluation uses
the EMA policy). All other fields are LeRobot's DiffusionConfig.
"""

from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig


@PreTrainedConfig.register_subclass("diffusion_ema")
@dataclass
class DiffusionEmaConfig(DiffusionConfig):
    ema_update_after_step: int = 0
    ema_inv_gamma: float = 1.0
    ema_power: float = 0.75
    ema_min_value: float = 0.0
    ema_max_value: float = 0.9999
    use_ema_for_inference: bool = True
