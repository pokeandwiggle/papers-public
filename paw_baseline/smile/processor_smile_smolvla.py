"""SMILE leaves observation and action processing identical to SmolVLA."""

from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors


def make_smile_smolvla_pre_post_processors(config, dataset_stats=None):
    """Use SmolVLA's pipeline; the coefficient interface lives inside the policy."""
    return make_smolvla_pre_post_processors(config, dataset_stats)
