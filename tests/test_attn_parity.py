"""Parity with the authors' DiffusionUnetTimmAttentionPolicy on the same batch and noise.

Builds the reference policy from a checkout of diffusion-policy-experiments
(commit 6645bd1; path from PAW_REFERENCE_DP or ../tmp/references/dp_exp) and
our AttnDiffusionPolicy with identical weights, then compares the training
loss (same RNG: random crops, short-range dropout, attention dropout, noise,
input perturbation, timesteps) and DDIM samples (same initial noise) on one
normalized batch. A small timm ViT keeps the test offline and fast; the
DINOv3 configuration is checked with the same code on the GPU before launch.
"""

import os
import sys
import types
from pathlib import Path

import pytest
import torch
from lerobot.configs import FeatureType, PolicyFeature

import paw_baseline.attn_diffusion  # noqa: F401
from paw_baseline.attn_diffusion.configuration_attn_diffusion import AttnDiffusionConfig
from paw_baseline.attn_diffusion.deltas import DELTA_DIM
from paw_baseline.attn_diffusion.modeling_attn_diffusion import (
    STATE_PARTS,
    AttnDiffusionPolicy,
)

REFERENCE = Path(
    os.environ.get(
        "PAW_REFERENCE_DP", Path(__file__).resolve().parents[2] / "tmp/references/dp_exp"
    )
)
CAMERA_KEYS = (
    "observation.images.rgb_left",
    "observation.images.rgb_right",
    "observation.images.rgb_static",
)
CAMERAS = tuple(key.removeprefix("observation.images.") for key in CAMERA_KEYS)
pytestmark = pytest.mark.skipif(
    not (REFERENCE / "diffusion_policy").is_dir(), reason="reference checkout unavailable"
)


def _reference():
    if "zarr" not in sys.modules:  # the normalizer imports zarr for type hints only
        stub = types.ModuleType("zarr")
        stub.Array = type("Array", (), {})
        sys.modules["zarr"] = stub
    if str(REFERENCE) not in sys.path:
        sys.path.insert(0, str(REFERENCE))
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    from diffusion_policy.model.common.normalizer import (
        LinearNormalizer,
        SingleFieldLinearNormalizer,
    )
    from diffusion_policy.model.vision.timm_obs_encoder import TimmObsEncoder
    from diffusion_policy.policy.diffusion_unet_timm_attention_policy import (
        DiffusionUnetTimmAttentionPolicy,
    )

    return DDPMScheduler, LinearNormalizer, SingleFieldLinearNormalizer, TimmObsEncoder, DiffusionUnetTimmAttentionPolicy


def build_pair(config: AttnDiffusionConfig):
    DDPM, LinearNormalizer, SingleField, TimmObsEncoder, ReferencePolicy = _reference()
    size = config.image_resize
    shape_meta = {
        "obs": {
            **{key: {"shape": [3, size, size], "type": "rgb"} for key in CAMERAS},
            **{name: {"shape": [sl.stop - sl.start], "type": "low_dim"} for name, sl in STATE_PARTS},
        },
        "action": {"shape": [DELTA_DIM]},
    }

    def encoder(steps):
        return TimmObsEncoder(
            shape_meta=shape_meta,
            model_name=config.vision_backbone,
            pretrained=config.pretrained_backbone,
            n_obs_steps=steps,
            feature_aggregation=config.feature_aggregation,
            crop_ratio=config.crop_ratio,
            imagenet_norm=config.imagenet_norm,
        )

    reference = ReferencePolicy(
        shape_meta=shape_meta,
        noise_scheduler=DDPM(
            num_train_timesteps=config.num_train_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule=config.beta_schedule,
            clip_sample=config.clip_sample,
            prediction_type=config.prediction_type,
            variance_type=config.variance_type,
        ),
        obs_encoder=encoder(config.n_obs_steps),
        horizon=config.horizon,
        short_range_encoder=encoder(config.short_range_obs_horizon),
        short_range_obs_horizon=config.short_range_obs_horizon,
        short_range_dropout=config.short_range_dropout,
        num_ddim_inference_steps=config.num_inference_steps,
        diffusion_step_embed_dim=config.diffusion_step_embed_dim,
        down_dims=config.down_dims,
        kernel_size=config.kernel_size,
        n_groups=config.n_groups,
        num_attention_heads=config.num_attention_heads,
        attention_dropout=config.attention_dropout,
        input_pertub=config.input_pertub,
    )
    normalizer = LinearNormalizer()
    for key in [*shape_meta["obs"], "action"]:
        normalizer[key] = SingleField.create_identity()
    reference.set_normalizer(normalizer)

    config.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(20,)),
        **{key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, size, size)) for key in CAMERA_KEYS},
    }
    config.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(20,))}
    ours = AttnDiffusionPolicy(config)
    weights = {
        k: v
        for k, v in reference.state_dict().items()
        if k.split(".")[0] in ("obs_encoder", "short_range_encoder", "model")
    }
    missing, unexpected = ours.core.load_state_dict(weights, strict=True)
    assert not missing and not unexpected
    return reference, ours


def batch(config: AttnDiffusionConfig, size: int = 3, seed: int = 7):
    generator = torch.Generator().manual_seed(seed)
    nobs = {
        key: torch.rand(size, config.n_obs_steps, 3, config.image_resize, config.image_resize, generator=generator)
        for key in CAMERAS
    }
    for name, sl in STATE_PARTS:
        nobs[name] = torch.rand(size, config.n_obs_steps, sl.stop - sl.start, generator=generator) * 2 - 1
    nactions = torch.rand(size, config.horizon, DELTA_DIM, generator=generator) * 2 - 1
    return nobs, nactions


def tiny_config(n_obs_steps: int) -> AttnDiffusionConfig:
    return AttnDiffusionConfig(
        n_obs_steps=n_obs_steps,
        replan_steps=6,
        vision_backbone="vit_tiny_patch16_224",
        pretrained_backbone=False,
        imagenet_norm=False,
        diffusion_step_embed_dim=32,
        down_dims=(32, 64, 128),
        num_attention_heads=4,
        num_inference_steps=4,
        use_ema=False,
    )


@pytest.mark.parametrize("n_obs_steps", [2, 8])
def test_loss_matches_reference(n_obs_steps):
    config = tiny_config(n_obs_steps)
    reference, ours = build_pair(config)
    nobs, nactions = batch(config)
    reference.train()
    ours.train()
    torch.manual_seed(123)
    expected = reference.compute_loss({"obs": nobs, "action": nactions})
    torch.manual_seed(123)
    actual = ours._loss(ours.core, nobs, nactions)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("n_obs_steps", [2, 8])
def test_ddim_samples_match_reference(n_obs_steps):
    config = tiny_config(n_obs_steps)
    reference, ours = build_pair(config)
    nobs, _ = batch(config)
    reference.eval()
    ours.eval()
    torch.manual_seed(321)
    expected = reference.predict_action(nobs, use_DDIM=True)["action_pred"]
    torch.manual_seed(321)
    with torch.no_grad():
        actual = ours._conditional_sample(ours.core, ours._encode_obs(ours.core, nobs))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_optimizer_groups_scale_encoder_lr():
    config = tiny_config(2)
    _, ours = build_pair(config)
    groups = ours.get_optim_params()
    assert groups[0]["lr"] == pytest.approx(config.optimizer_lr * 0.1)
    encoder_ids = {id(p) for p in ours.core.obs_encoder.parameters()} | {
        id(p) for p in ours.core.short_range_encoder.parameters()
    }
    assert {id(p) for p in groups[0]["params"]} == encoder_ids
    trainable = {id(p) for p in ours.core.parameters()}
    assert {id(p) for g in groups for p in g["params"]} == trainable
