"""CSP uses the same normalize/unnormalize pipeline as LeRobot Diffusion Policy."""

from lerobot.processor import make_default_pre_post_processors


def make_csp_pre_post_processors(config, dataset_stats=None):
    return make_default_pre_post_processors(config, dataset_stats)
