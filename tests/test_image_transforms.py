"""Training inputs match serving compression without changing policy code."""

import json
from types import SimpleNamespace

import cv2
import pytest
import torch
from lerobot.configs.default import DatasetConfig
from lerobot.datasets.factory import make_train_eval_datasets
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.transforms import ImageTransformsConfig
from torchvision.transforms import v2

from paw_baseline.prepare import prepare_dataset
from train.__main__ import CONFIGS_DIR, load_preset, trainer_params
from train.image_transforms import (
    JPEG_CONFIG_PATH,
    resolve_jpeg_quality,
    training_image_transforms,
)


def _opencv_jpeg(image):
    frames = image.reshape(-1, *image.shape[-3:])
    decoded = []
    for frame in frames:
        rgb = frame.permute(1, 2, 0).numpy()
        ok, encoded = cv2.imencode(
            ".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85]
        )
        assert ok
        rgb = cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        decoded.append(torch.from_numpy(rgb).permute(2, 0, 1))
    return torch.stack(decoded).reshape(image.shape)


@pytest.fixture
def dataset_config(export_dir, cache_root, monkeypatch):
    monkeypatch.setenv("PAW_BASELINE_CACHE", str(cache_root))
    prepared = prepare_dataset(export_dir)
    return SimpleNamespace(
        dataset=DatasetConfig(
            repo_id=prepared.root.name,
            root=prepared.root,
            video_backend="pyav",
            use_imagenet_stats=False,
        ),
        trainable_config=ACTConfig(chunk_size=2, n_action_steps=2, n_obs_steps=1),
        tolerance_s=1e-4,
    )


@pytest.mark.parametrize("preset", sorted(p.stem for p in CONFIGS_DIR.glob("*.json")))
def test_all_presets_compress_before_existing_transforms(dataset_config, preset):
    args = dict(arg[2:].split("=", 1) for arg in load_preset(preset)["args"])
    enabled = json.loads(args.get("dataset.image_transforms.enable", "false"))
    dataset_config.dataset.image_transforms = ImageTransformsConfig(enable=enabled)
    trainer = SimpleNamespace(make_train_eval_datasets=make_train_eval_datasets)
    with training_image_transforms(trainer, resolve_jpeg_quality({}, None)):
        dataset, _ = trainer.make_train_eval_datasets(dataset_config)
    assert trainer.make_train_eval_datasets is make_train_eval_datasets
    clean, _ = make_train_eval_datasets(dataset_config)
    existing = clean.image_transforms
    clean.clear_image_transforms()
    original = clean[0]
    torch.manual_seed(31)
    augmented = dataset[0]
    torch.manual_seed(31)
    for key in dataset.meta.camera_keys:
        expected = v2.JPEG(85)(original[key])
        assert not torch.equal(expected, original[key])
        torch.testing.assert_close(
            expected, _opencv_jpeg(original[key]), rtol=0, atol=0
        )
        if existing is not None:
            expected = existing(expected)
        torch.testing.assert_close(augmented[key], expected, rtol=0, atol=0)
        assert augmented[key].dtype == original[key].dtype == torch.uint8
    for key in original.keys() - set(dataset.meta.camera_keys):
        if isinstance(original[key], torch.Tensor):
            torch.testing.assert_close(augmented[key], original[key], rtol=0, atol=0)
        else:
            assert augmented[key] == original[key]


def test_validation_dataset_and_disabled_training_remain_uncompressed(dataset_config):
    dataset_config.dataset.eval_split = 0.5
    trainer = SimpleNamespace(make_train_eval_datasets=make_train_eval_datasets)
    with training_image_transforms(trainer, 85):
        train, evaluation = trainer.make_train_eval_datasets(dataset_config)
    assert train.image_transforms is not None
    assert evaluation.image_transforms is None
    with training_image_transforms(trainer, None):
        train, _ = trainer.make_train_eval_datasets(dataset_config)
    assert train.image_transforms is None


def test_transform_survives_spawn_workers(dataset_config):
    trainer = SimpleNamespace(make_train_eval_datasets=make_train_eval_datasets)
    with training_image_transforms(trainer, 85):
        dataset, _ = trainer.make_train_eval_datasets(dataset_config)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, [0]),
        num_workers=1,
        multiprocessing_context="spawn",
    )
    batch = next(iter(loader))
    for key in dataset.meta.camera_keys:
        torch.testing.assert_close(batch[key][0], dataset[0][key], rtol=0, atol=0)


@pytest.mark.parametrize("quality", [None, 85, 95])
def test_quality_is_explicit_and_resumed_without_changing_recipe(tmp_path, quality):
    params = {"jpeg_quality": quality}
    trainer_params(params)
    assert resolve_jpeg_quality(params, None) == quality
    path = tmp_path / "pretrained_model" / JPEG_CONFIG_PATH
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(params))
    assert resolve_jpeg_quality({}, tmp_path) == quality
    assert resolve_jpeg_quality(params, tmp_path) == quality
    with pytest.raises(ValueError, match="resume"):
        resolve_jpeg_quality({"jpeg_quality": 90}, tmp_path)


def test_legacy_resume_preserves_uncompressed_training(tmp_path):
    assert resolve_jpeg_quality({}, None) == 85
    assert resolve_jpeg_quality({}, tmp_path) is None
    with pytest.raises(ValueError, match="resume"):
        resolve_jpeg_quality({"jpeg_quality": 85}, tmp_path)


@pytest.mark.parametrize("quality", [True, 0, 101, 85.5, "85"])
def test_invalid_quality_fails_before_training(quality):
    with pytest.raises(ValueError, match="jpeg_quality"):
        resolve_jpeg_quality({"jpeg_quality": quality}, None)
