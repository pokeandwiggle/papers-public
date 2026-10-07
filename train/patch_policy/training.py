"""Fit the VQ tokenizer on the selected training chunks before policy updates.

The reference collects every training action chunk, keeps the unique ones,
fits the residual VQ-VAE, freezes it, and only then trains the policy. Here
chunks come from the selected training episodes, are clamped at episode ends
exactly like LeRobot's delta-timestamp queries, and are normalized by the
saved preprocessor so the tokenizer sees the training targets' scale.
"""

import logging
from contextlib import contextmanager

import numpy as np
import torch
from lerobot.configs.types import FeatureType
from lerobot.processor.normalize_processor import NormalizerProcessorStep

MODEL = "patch_vqbet"


def episode_action_chunks(actions: np.ndarray, episode_index: np.ndarray, chunk: int) -> np.ndarray:
    """(N, A) frames -> (N, chunk, A): the chunk starting at every frame, end-clamped."""
    out = []
    for episode in np.unique(episode_index):
        rows = np.flatnonzero(episode_index == episode)
        if np.any(np.diff(rows) != 1):
            raise ValueError("episode frames must be contiguous")
        x = actions[rows]
        idx = np.minimum(np.arange(len(x))[:, None] + np.arange(chunk)[None], len(x) - 1)
        out.append(x[idx])
    return np.concatenate(out)


def dataset_action_chunks(dataset, chunk: int) -> np.ndarray:
    table = dataset.hf_dataset.data
    frames = np.asarray(table.column("frame_index").to_numpy())
    episodes = np.asarray(table.column("episode_index").to_numpy())
    actions = np.asarray(table.column("action").combine_chunks().flatten()).reshape(len(table), -1)
    order = np.lexsort((frames, episodes))
    if dataset.episodes is not None and set(np.unique(episodes).tolist()) != set(dataset.episodes):
        raise ValueError("loaded action rows do not match the selected training episodes")
    return episode_action_chunks(actions[order], episodes[order], chunk)


def normalize_actions(preprocessor, actions: torch.Tensor) -> torch.Tensor:
    steps = [step for step in preprocessor.steps if isinstance(step, NormalizerProcessorStep)]
    if len(steps) != 1:
        raise ValueError("expected exactly one normalizer step in the preprocessor")
    return steps[0]._apply_transform(actions, "action", FeatureType.ACTION)


@contextmanager
def tokenizer_fit(trainer, model: str | None):
    """Fit a fresh policy's tokenizer once its preprocessor exists; resumes keep theirs."""
    if model != MODEL:
        yield
        return
    captured = {}
    make_datasets = trainer.make_train_eval_datasets
    make_policy = trainer.make_policy
    make_processors = trainer.make_pre_post_processors

    def capture_datasets(cfg):
        dataset, eval_dataset = make_datasets(cfg)
        captured["dataset"] = dataset
        return dataset, eval_dataset

    def capture_policy(*args, **kwargs):
        policy = make_policy(*args, **kwargs)
        captured["policy"] = policy
        return policy

    def processors_then_fit(*args, **kwargs):
        preprocessor, postprocessor = make_processors(*args, **kwargs)
        policy = captured["policy"]
        if not bool(policy.vqvae_fitted):
            device = next(policy.gpt.parameters()).device
            chunks = dataset_action_chunks(captured["dataset"], policy.config.chunk_size)
            normalized = normalize_actions(
                preprocessor, torch.from_numpy(chunks).float().to(device)
            )
            logging.info("fitting VQ tokenizer on %d chunks", len(normalized))
            info = policy.fit_tokenizer(normalized, log=logging.info)
            logging.info("VQ tokenizer fitted: %s", info)
        return preprocessor, postprocessor

    trainer.make_train_eval_datasets = capture_datasets
    trainer.make_policy = capture_policy
    trainer.make_pre_post_processors = processors_then_fit
    try:
        yield
    finally:
        trainer.make_train_eval_datasets = make_datasets
        trainer.make_policy = make_policy
        trainer.make_pre_post_processors = make_processors
