"""ManiFlow uses LeRobot's default normalization pipeline, like Diffusion Policy."""

from lerobot.processor import make_default_pre_post_processors


def make_maniflow_pre_post_processors(config, dataset_stats=None):
    return make_default_pre_post_processors(config, dataset_stats)
