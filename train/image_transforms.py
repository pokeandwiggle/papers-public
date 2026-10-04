"""Match serving JPEG compression at the shared training dataset boundary."""

import json
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

from torchvision.transforms import v2

# Inference uses JPEG85; retain this default to limit train/serve shift and instability.
DEFAULT_JPEG_QUALITY = 85
JPEG_CONFIG_PATH = ".paw/image_transforms.json"


def _parse_quality(value: object) -> int | None:
    if value is None or (type(value) is int and 1 <= value <= 100):
        return value
    raise ValueError("jpeg_quality must be an integer in [1, 100] or null")


def resolve_jpeg_quality(params: dict, resume_from: Path | None) -> int | None:
    """Use the fresh-run default or preserve the checkpoint's compression recipe.

    ``jpeg_quality=null`` disables compression. Checkpoints predating this metadata
    trained without the shared JPEG step, so resuming them keeps it disabled.
    An explicit override must agree with the saved recipe on resume.
    """
    if resume_from is None:
        return _parse_quality(params.get("jpeg_quality", DEFAULT_JPEG_QUALITY))

    path = resume_from / "pretrained_model" / JPEG_CONFIG_PATH
    saved = json.loads(path.read_text())["jpeg_quality"] if path.exists() else None
    quality = _parse_quality(saved)
    if "jpeg_quality" in params and _parse_quality(params["jpeg_quality"]) != quality:
        raise ValueError(
            "jpeg_quality cannot change the saved training recipe on resume"
        )
    return quality


@contextmanager
def training_image_transforms(trainer, jpeg_quality: int | None):
    """Prepend JPEG to training RGB transforms without altering eval or serving.

    The dataset's existing transforms retain their order and selection weights.
    The composed torchvision transform is serializable to DataLoader workers.
    """
    if jpeg_quality is None:
        yield
        return

    original = trainer.make_train_eval_datasets

    @wraps(original)
    def make_datasets(cfg):
        dataset, eval_dataset = original(cfg)
        transforms = [v2.JPEG(quality=jpeg_quality)]
        if dataset.image_transforms is not None:
            transforms.append(dataset.image_transforms)
        dataset.set_image_transforms(v2.Compose(transforms))
        return dataset, eval_dataset

    trainer.make_train_eval_datasets = make_datasets
    try:
        yield
    finally:
        trainer.make_train_eval_datasets = original
