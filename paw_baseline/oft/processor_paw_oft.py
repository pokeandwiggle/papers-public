"""Pass absolute TCP state/actions through; the OFT policy normalizes internally.

All features use IDENTITY normalization, so the default pipeline only batches
and places tensors. The saved unnormalizer still records the prepared dataset
statistics that serving uses to verify frozen-arm metadata.
"""

from lerobot.processor import make_default_pre_post_processors


def make_paw_oft_pre_post_processors(config, dataset_stats=None):
    return make_default_pre_post_processors(
        config, dataset_stats, normalizer_device=config.device
    )
