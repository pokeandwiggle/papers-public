"""LeRobot processors for OFT: batching and device placement; the policy normalizes."""

from lerobot.processor import make_default_pre_post_processors


def make_paw_oft_pre_post_processors(config, dataset_stats=None):
    """Identity normalization keeps saved statistics for serving metadata checks."""
    return make_default_pre_post_processors(
        config, dataset_stats, normalizer_device=config.device
    )
