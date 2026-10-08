"""Default LeRobot processors; all FlorenceVLA features use IDENTITY normalization."""

from lerobot.processor import make_default_pre_post_processors

from paw_baseline.florence_vla.configuration_florence_vla import FlorenceVLAConfig


def make_florence_vla_pre_post_processors(config: FlorenceVLAConfig, dataset_stats=None):
    return make_default_pre_post_processors(
        config, dataset_stats, normalizer_device=config.device
    )
