"""Gradient accumulation for LeRobot's single-pass ``update_policy``.

OpenVLA-OFT trains with a fixed effective batch (per-device batch x devices x
accumulation). LeRobot steps the optimizer once per loader batch, so this
splits each per-device batch into equal micro-batches, skips DDP gradient
synchronization until the last one, and then steps exactly once.
"""

import time
from contextlib import contextmanager, nullcontext

import torch


def _slice(batch: dict, size: int, start: int, stop: int) -> dict:
    part = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == size:
            part[key] = value[start:stop]
        elif isinstance(value, list) and len(value) == size:
            part[key] = value[start:stop]
        else:
            part[key] = value
    return part


def make_accumulating_update(steps: int):
    def update_policy(
        train_metrics,
        policy,
        batch,
        optimizer,
        grad_clip_norm,
        accelerator,
        lr_scheduler=None,
        lock=None,
        sample_weighter=None,
    ):
        if sample_weighter is not None:
            raise ValueError("gradient accumulation does not support sample weighting")
        start_time = time.perf_counter()
        policy.train()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        size = batch["observation.state"].shape[0]
        # Near-equal micro-batches; a short final loader batch (drop_last=False)
        # weights each per-sample-mean loss by its share of the batch.
        bounds = [round(i * size / steps) for i in range(steps + 1)]
        parts = [(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
        loss_total = 0.0
        outputs: dict = {}
        for i, (start, stop) in enumerate(parts):
            weight = (stop - start) / size
            part = _slice(batch, size, start, stop)
            sync = accelerator.no_sync(policy) if i < len(parts) - 1 else nullcontext()
            with sync:
                with accelerator.autocast():
                    loss, output = policy.forward(part)
                accelerator.backward(loss * weight)
            loss_total += loss.item() * weight
            for key, value in (output or {}).items():
                if isinstance(value, (int, float)):
                    outputs[key] = outputs.get(key, 0.0) + value * weight
        if grad_clip_norm > 0:
            grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                policy.parameters(), float("inf"), error_if_nonfinite=False
            )
        with lock if lock is not None else nullcontext():
            optimizer.step()
        optimizer.zero_grad()
        if lr_scheduler is not None:
            lr_scheduler.step()
        train_metrics.loss = loss_total
        train_metrics.grad_norm = grad_norm.item()
        train_metrics.lr = optimizer.param_groups[0]["lr"]
        train_metrics.update_s = time.perf_counter() - start_time
        if torch.cuda.is_available():
            train_metrics.gpu_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
        if outputs:
            train_metrics.update_metrics(outputs)
        return train_metrics, outputs

    return update_policy


@contextmanager
def gradient_accumulation(trainer, steps: int):
    """Replace the trainer's update step while LeRobot's training loop runs."""
    if steps <= 1:
        yield
        return
    original = trainer.update_policy
    trainer.update_policy = make_accumulating_update(steps)
    try:
        yield
    finally:
        trainer.update_policy = original
