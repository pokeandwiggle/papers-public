"""LeRobot pre/post-processing for attn_diffusion: device moves only.

All normalization happens inside the policy (images by the encoders' ImageNet
statistics, state and delta actions by in-policy min-max statistics), so the
configured normalization mapping is IDENTITY for every feature type.
"""

from lerobot.processor import make_default_pre_post_processors


def make_attn_diffusion_pre_post_processors(config, dataset_stats=None):
    return make_default_pre_post_processors(config, dataset_stats)
