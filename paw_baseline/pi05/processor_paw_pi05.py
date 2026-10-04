"""Pad normalized physical state for the native PI05 tokenizer."""

from dataclasses import dataclass

import torch
from lerobot.lerobot_types import TransitionKey
from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors
from lerobot.processor import ProcessorStep, ProcessorStepRegistry


@ProcessorStepRegistry.register(name="paw_pi05_pad_normalized_state_v1")
@dataclass
class PadNormalizedState(ProcessorStep):
    """The normalizer fits 20 physical values; PI's state tokenizer sees 32."""

    def __call__(self, transition):
        result = transition.copy()
        observation = transition[TransitionKey.OBSERVATION].copy()
        state = observation["observation.state"]
        if state.shape[-1] != 20:
            raise ValueError(
                "PI05 normalized state must have 20 dimensions before padding"
            )
        observation["observation.state"] = torch.nn.functional.pad(state, (0, 12))
        result[TransitionKey.OBSERVATION] = observation
        return result

    def transform_features(self, features):
        return features


def make_paw_pi05_pre_post_processors(config, dataset_stats=None):
    """Pad normalized state before LeRobot tokenizes it."""
    pre, post = make_pi05_pre_post_processors(config, dataset_stats)
    from lerobot.policies.pi05.processor_pi05 import (
        Pi05PrepareStateTokenizerProcessorStep,
    )

    tokenizers = [
        i
        for i, step in enumerate(pre.steps)
        if isinstance(step, Pi05PrepareStateTokenizerProcessorStep)
    ]
    if len(tokenizers) != 1:
        raise ValueError("PI05 pipeline must have exactly one state tokenizer")
    pre.steps.insert(tokenizers[0], PadNormalizedState())
    return pre, post
