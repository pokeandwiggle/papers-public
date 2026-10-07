"""Default LeRobot processors: MIN_MAX normalization in, unnormalization out."""

from lerobot.processor import make_default_pre_post_processors

from paw_baseline.artok.configuration_artok import ARTokConfig


def make_artok_pre_post_processors(config: ARTokConfig, dataset_stats=None):
    return make_default_pre_post_processors(config, dataset_stats)
