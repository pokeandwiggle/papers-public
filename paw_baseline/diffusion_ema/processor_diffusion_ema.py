"""Diffusion Policy with EMA uses LeRobot's default normalization pipeline."""

from lerobot.processor import make_default_pre_post_processors


def make_diffusion_ema_pre_post_processors(config, dataset_stats=None):
    return make_default_pre_post_processors(config, dataset_stats)
