"""Default LeRobot pre/post-processing: identity images, MIN_MAX actions."""

from lerobot.processor import make_default_pre_post_processors


def make_patch_vqbet_pre_post_processors(config, dataset_stats=None):
    return make_default_pre_post_processors(config, dataset_stats)
