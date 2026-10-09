"""Configure Diffusion Policy with a strided observation history.

Timing follows the paper's real-robot recipe on the 30 Hz grid:

* ``n_obs_steps`` context frames spaced ``obs_stride`` grid steps apart
  (stride 3 = 100 ms, the paper's 10 Hz observations);
* ``prediction_steps`` current-and-future actions (45 = 1.5 s, the paper's
  T_p = 15 at 10 Hz), held fixed across context lengths;
* past-token prediction: the denoised sequence also covers every 30 Hz action
  inside the context window, so ``horizon`` (LeRobot's total sequence length)
  is ``(n_obs_steps - 1) * obs_stride + prediction_steps``;
* ``replan_steps`` grid steps executed open loop before the server plans from
  a new observation (the paper's execution horizon T_exec).

``n_action_steps`` is the number of actions emitted per plan and equals
``prediction_steps``: the server supersedes each chunk after ``replan_steps``.
"""

import math
from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig


@PreTrainedConfig.register_subclass("hist_diffusion")
@dataclass
class HistDiffusionConfig(DiffusionConfig):
    obs_stride: int = 3
    prediction_steps: int = 45
    replan_steps: int = 24
    # Linear crossfade, in grid steps, from the robot's current plan into a new
    # chunk at the first replaceable slot (the paper's ~180 ms seam interpolation).
    seam_blend_steps: int = 4

    def __post_init__(self):
        for name in ("obs_stride", "prediction_steps", "replan_steps"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if type(self.seam_blend_steps) is not int or self.seam_blend_steps < 0:
            raise ValueError("seam_blend_steps must be a nonnegative integer")
        if self.replan_steps > self.prediction_steps:
            raise ValueError("replan_steps cannot exceed prediction_steps")
        horizon = self.past_steps + self.prediction_steps
        # The U-Net crops after each upsampling, so any length works. The parent
        # check requires a multiple of 2**len(down_dims); satisfy it transiently.
        factor = 2 ** len(self.down_dims)
        self.horizon = math.ceil(horizon / factor) * factor
        super().__post_init__()
        self.horizon = horizon
        self.n_action_steps = self.prediction_steps

    @property
    def past_steps(self) -> int:
        """Grid steps of past actions denoised alongside the prediction."""
        return (self.n_obs_steps - 1) * self.obs_stride

    @property
    def observation_delta_indices(self) -> list:
        return list(range(-self.past_steps, 1, self.obs_stride))

    @property
    def action_delta_indices(self) -> list:
        return list(range(-self.past_steps, self.prediction_steps))
