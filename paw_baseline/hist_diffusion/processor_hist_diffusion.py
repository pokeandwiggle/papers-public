"""Use LeRobot's Diffusion pre/post-processing unchanged.

Normalization broadcasts over a leading context axis, so strided histories
(B, n_obs_steps, ...) pass through the same saved pipelines as single frames.
"""

from lerobot.policies.diffusion.processor_diffusion import (
    make_diffusion_pre_post_processors,
)


def make_hist_diffusion_pre_post_processors(config, dataset_stats=None):
    return make_diffusion_pre_post_processors(config, dataset_stats)
