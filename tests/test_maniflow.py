"""ManiFlow and Diffusion-EMA behavior, with parity against the official reference.

The reference files are vendored at the pinned commit under
tests/reference/maniflow_ef2f116 (MANIFEST.json records upstream paths and
hashes), so these parity tests run in a clean checkout. timm, zarr and r3m are
stubbed: the tests exercise only the modules that import them.
"""

import copy
import hashlib
import importlib
import json
import sys
import types
from pathlib import Path

import pytest
import torch
import torchvision
from torch import nn

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE
from paw_baseline.diffusion_ema.configuration_diffusion_ema import DiffusionEmaConfig
from paw_baseline.diffusion_ema.modeling_diffusion_ema import DiffusionEmaPolicy
from paw_baseline.ema import EmaSchedule, ema_step
from paw_baseline.maniflow.configuration_maniflow import ManiFlowConfig
from paw_baseline.maniflow.ditx import DiTX
from paw_baseline.maniflow.modeling_maniflow import ManiFlowObsEncoder, ManiFlowPolicy

REFERENCE = Path(__file__).resolve().parent / "reference/maniflow_ef2f116"
CAMERAS = (
    "observation.images.rgb_static",
    "observation.images.rgb_left",
    "observation.images.rgb_right",
)


def features(image_hw=(64, 64)):
    return {
        "input_features": {
            "observation.state": PolicyFeature(FeatureType.STATE, (20,)),
            **{key: PolicyFeature(FeatureType.VISUAL, (3, *image_hw)) for key in CAMERAS},
        },
        "output_features": {"action": PolicyFeature(FeatureType.ACTION, (20,))},
    }


def small_config(image_hw=(64, 64), **overrides) -> ManiFlowConfig:
    kwargs = {
        "horizon": 8,
        "n_action_steps": 4,
        "n_layer": 2,
        "n_head": 4,
        "n_emb": 64,
        "pretrained_backbone_weights": None,
        "crop_shape": (60, 60),
        "device": "cpu",
        **overrides,
    }
    return ManiFlowConfig(**features(image_hw), **kwargs)


def batch(size: int, horizon: int = 8, image_hw=(64, 64)) -> dict:
    return {
        "observation.state": torch.randn(size, 1, 20),
        **{key: torch.rand(size, 1, 3, *image_hw) for key in CAMERAS},
        "action": torch.randn(size, horizon, 20),
        "action_is_pad": torch.zeros(size, horizon, dtype=torch.bool),
    }


def observation(size: int = 1, image_hw=(64, 64)) -> dict:
    return {k: v[:size] for k, v in batch(size, image_hw=image_hw).items() if k not in ("action", "action_is_pad")}


# ------------------------------- ManiFlow --------------------------------


def test_multi_token_conditioning_shapes_for_station_images():
    """3 cameras at 256x256 cropped to 243: an 8x8 layer-4 map gives 64 tokens each, plus state."""
    config = small_config(image_hw=(256, 256), crop_shape=(243, 243), n_layer=1)
    policy = ManiFlowPolicy(config)
    encoder = policy.model.obs_encoder
    assert encoder.feature_map_shape == (512, 8, 8)
    assert encoder.num_tokens == 3 * 64 + 1
    tokens = encoder({OBS_IMAGES: torch.rand(2, 1, 3, 3, 256, 256), OBS_STATE: torch.randn(2, 1, 20)})
    assert tokens.shape == (2, 193, 64)
    assert policy.model.dit.vis_cond_pos_embed.shape == (1, 193, 64)


def test_training_step_updates_only_online_network_and_ema_tracks_it():
    torch.manual_seed(0)
    policy = ManiFlowPolicy(small_config())
    policy.train()
    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=1e-3)
    ema_ids = {id(p) for p in policy.ema_model.parameters()}
    assert not ema_ids & {id(p) for p in policy.get_optim_params()}
    loss, metrics = policy.forward(batch(8))
    assert set(metrics) == {"loss_flow", "loss_ct"}
    loss.backward()
    assert all(p.grad is None for p in policy.ema_model.parameters())
    optimizer.step()
    policy.update()  # step 0: decay 0 -> copy
    for ema, online in zip(policy.ema_model.parameters(), policy.model.parameters(), strict=True):
        torch.testing.assert_close(ema, online)
    policy.update()
    assert int(policy.ema_optimization_step) == 2


def test_inference_samples_with_ema_network():
    torch.manual_seed(1)
    policy = ManiFlowPolicy(small_config())
    policy.eval()
    obs = observation()
    noise = torch.randn(1, 8, 20)
    reference = policy.predict_action_chunk(obs, noise=noise.clone())
    with torch.no_grad():
        for p in policy.model.parameters():
            p.add_(torch.randn_like(p))
    policy.reset()
    torch.testing.assert_close(policy.predict_action_chunk(obs, noise=noise.clone()), reference)
    policy.config.use_ema_for_inference = False
    policy.reset()
    assert not torch.allclose(policy.predict_action_chunk(obs, noise=noise.clone()), reference)


@pytest.mark.parametrize("size", [1, 3, 9])
def test_short_final_batches_keep_a_finite_flow_loss(size):
    torch.manual_seed(4)
    policy = ManiFlowPolicy(small_config())
    policy.train()
    loss, metrics = policy.forward(batch(size))
    assert torch.isfinite(loss)
    assert ("loss_ct" in metrics) == (size >= 4)


def test_select_action_emits_chunk_from_one_observation_and_round_trips(tmp_path):
    torch.manual_seed(5)
    policy = ManiFlowPolicy(small_config())
    policy.train()
    policy.forward(batch(4))[0].backward()
    policy.update()
    policy.update()
    policy.update()
    policy.eval()
    obs = observation()
    first = policy.select_action({k: v[:, 0] for k, v in obs.items()})
    assert first.shape == (1, 20)
    assert len(policy._queues["action"]) == 3
    policy.save_pretrained(tmp_path)
    loaded = ManiFlowPolicy.from_pretrained(tmp_path, config=policy.config)
    for (name, a), (_, b) in zip(policy.state_dict().items(), loaded.state_dict().items(), strict=True):
        torch.testing.assert_close(a, b, msg=name)
    assert int(loaded.ema_optimization_step) == 3
    noise = torch.randn(1, 8, 20)
    policy.reset()
    loaded.reset()
    loaded.eval()
    torch.testing.assert_close(
        policy.predict_action_chunk(obs, noise=noise.clone()),
        loaded.predict_action_chunk(obs, noise=noise.clone()),
    )


@pytest.mark.parametrize("steps", [2, 10])
def test_euler_sampler_uses_relative_step_size(steps):
    torch.manual_seed(2)
    policy = ManiFlowPolicy(small_config(num_inference_steps=steps))
    policy.eval()
    calls = []
    model = policy.inference_model()
    original = model.dit.forward

    def spy(sample, timestep, target_t, vis_cond):
        calls.append((round(float(timestep[0]), 6), round(float(target_t[0]), 6)))
        return original(sample, timestep, target_t, vis_cond)

    model.dit.forward = spy
    policy.predict_action_chunk(observation())
    assert calls == [(round(i / steps, 6), round(1 / steps, 6)) for i in range(steps)]


# ---------------------------------- EMA ----------------------------------


def test_ema_schedule_matches_reference_formula():
    schedule = EmaSchedule()
    assert schedule.decay(0) == 0.0 and schedule.decay(1) == 0.0
    assert schedule.decay(11) == pytest.approx(1 - 11**-0.75)
    assert schedule.decay(10**9) == 0.9999


# ------------------------------ Diffusion EMA -----------------------------


def dp_config(**overrides) -> DiffusionEmaConfig:
    kwargs = {
        "n_obs_steps": 1,
        "horizon": 16,
        "n_action_steps": 8,
        "down_dims": (32, 64, 128),
        "pretrained_backbone_weights": None,
        "crop_shape": (60, 60),
        "noise_scheduler_type": "DDIM",
        "num_inference_steps": 2,
        "drop_n_last_frames": 0,
        "device": "cpu",
        **overrides,
    }
    return DiffusionEmaConfig(**features(), **kwargs)


def test_diffusion_ema_updates_with_original_schedule_and_samples_with_ema(tmp_path):
    torch.manual_seed(6)
    policy = DiffusionEmaPolicy(dp_config())
    assert {id(p) for p in policy.get_optim_params()}.isdisjoint(id(p) for p in policy.ema_diffusion.parameters())
    policy.train()
    optimizer = torch.optim.Adam(policy.get_optim_params(), lr=1e-3)
    loss, _ = policy.forward(batch(4, horizon=16))
    loss.backward()
    optimizer.step()
    policy.update()
    for ema, online in zip(policy.ema_diffusion.parameters(), policy.diffusion.parameters(), strict=True):
        torch.testing.assert_close(ema, online)
    for _ in range(10):
        policy.update()
    assert int(policy.ema_optimization_step) == 11

    policy.eval()
    obs = observation()  # LeRobot's offline Diffusion branch expects (B, T, D) state
    noise = torch.randn(1, 16, 20)
    reference = policy.predict_action_chunk(dict(obs), noise=noise.clone())
    with torch.no_grad():
        for p in policy.diffusion.parameters():
            p.add_(torch.randn_like(p))
    policy.reset()
    torch.testing.assert_close(policy.predict_action_chunk(dict(obs), noise=noise.clone()), reference)

    policy.save_pretrained(tmp_path)
    loaded = DiffusionEmaPolicy.from_pretrained(tmp_path, config=policy.config)
    loaded.eval()
    torch.testing.assert_close(loaded.predict_action_chunk(dict(obs), noise=noise.clone()), reference)
    assert int(loaded.ema_optimization_step) == 11


def test_diffusion_ema_copy_is_eager_when_training_unet_is_compiled(tmp_path):
    torch.manual_seed(7)
    policy = DiffusionEmaPolicy(dp_config(compile_model=True))
    assert hasattr(policy.diffusion.unet, "_orig_mod")
    assert not hasattr(policy.ema_diffusion.unet, "_orig_mod")
    with torch.no_grad():
        for p in policy.diffusion.parameters():
            p.add_(1.0)
    policy.update()  # decay 0: exact copy across the compile wrapper
    eager = {k.replace("._orig_mod", ""): v for k, v in policy.diffusion.state_dict().items()}
    for key, value in policy.ema_diffusion.state_dict().items():
        torch.testing.assert_close(value, eager[key], msg=key)
    policy.save_pretrained(tmp_path)
    DiffusionEmaPolicy.from_pretrained(tmp_path, config=policy.config)  # strict


# ----------------------------- reference parity ----------------------------


def test_vendored_reference_matches_manifest():
    manifest = json.loads((REFERENCE / "MANIFEST.json").read_text())
    assert manifest["commit"] == "ef2f116f1f90163ed36e657b8c5503740bb468af"
    for relative, entry in manifest["files"].items():
        assert hashlib.sha256((REFERENCE / relative).read_bytes()).hexdigest() == entry["sha256"], relative
    assert manifest["diffusion_policy_ema_model"]["sha256"] == manifest["files"]["maniflow/model/diffusion/ema_model.py"]["sha256"]


def _install_reference_stubs():
    """Provide the timm symbols, zarr and r3m modules the reference imports."""

    class TimmMlp(nn.Module):
        def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
            super().__init__()
            hidden_features = hidden_features or in_features  # timm defaults
            out_features = out_features or in_features
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.act = act_layer()
            self.fc2 = nn.Linear(hidden_features, out_features)

        def forward(self, x):
            return self.fc2(self.act(self.fc1(x)))

    class TimmRmsNorm(nn.Module):
        def __init__(self, channels, eps=1e-6):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(channels))

        def forward(self, x):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

    def create_model(model_name, pretrained=False, global_pool="", num_classes=0):
        # timm resnet18 has the same child layout; [:-2] drops pooling and head.
        assert model_name == "resnet18" and not pretrained and global_pool == "" and num_classes == 0
        return torchvision.models.resnet18(weights=None)

    timm = types.ModuleType("timm")
    models = types.ModuleType("timm.models")
    vit = types.ModuleType("timm.models.vision_transformer")
    vit.Mlp, vit.RmsNorm, vit.use_fused_attn = TimmMlp, TimmRmsNorm, lambda: True
    timm.models, timm.create_model = models, create_model
    models.vision_transformer = vit
    sys.modules["timm"] = timm
    sys.modules["timm.models"] = models
    sys.modules["timm.models.vision_transformer"] = vit
    zarr = types.ModuleType("zarr")
    zarr.Array = type("Array", (), {})
    sys.modules.setdefault("zarr", zarr)
    sys.modules.setdefault("r3m", types.ModuleType("r3m"))
    if str(REFERENCE) not in sys.path:
        sys.path.insert(0, str(REFERENCE))


def _paired_ditx(cond_len=9):
    _install_reference_stubs()
    ref_ditx = importlib.import_module("maniflow.model.diffusion.ditx")
    torch.manual_seed(3)
    reference = ref_ditx.DiTX(
        input_dim=20, output_dim=20, horizon=8, n_obs_steps=1, cond_dim=64,
        visual_cond_len=cond_len, diffusion_timestep_embed_dim=128, diffusion_target_t_embed_dim=128,
        n_layer=2, n_head=4, n_emb=64, qkv_bias=True, qk_norm=True,
    )
    ours = DiTX(
        action_dim=20, horizon=8, cond_dim=64, cond_len=cond_len, n_layer=2, n_head=4, n_emb=64,
        mlp_ratio=4.0, p_drop_attn=0.1, qkv_bias=True, qk_norm=True,
        timestep_embed_dim=128, target_t_embed_dim=128,
    )
    assert set(reference.state_dict()) == set(ours.state_dict())
    with torch.no_grad():  # randomize zero-initialized layers so the comparison is nontrivial
        for p in reference.parameters():
            p.add_(0.05 * torch.randn_like(p))
    ours.load_state_dict(reference.state_dict(), strict=True)
    return reference.eval(), ours.eval()


def test_ditx_forward_with_multiple_tokens_matches_reference():
    reference, ours = _paired_ditx(cond_len=9)
    x = torch.randn(5, 8, 20)
    t, dt = torch.rand(5), torch.rand(5)
    cond = torch.randn(5, 9, 64)
    torch.testing.assert_close(ours(x, t, dt, cond), reference(x, t, dt, cond), rtol=1e-5, atol=1e-5)


def test_obs_encoder_matches_reference_transformer_obs_encoder():
    """Official TransformerObsEncoder(resnet18, feature_aggregation=None) token layout and projections."""
    _install_reference_stubs()
    ref_module = importlib.import_module("maniflow.model.vision_2d.transformer_obs_encoder")
    torch.manual_seed(8)
    shape_meta = {
        "obs": {
            **{name: {"shape": [3, 64, 64], "type": "rgb", "horizon": 1} for name in ("rgb_static", "rgb_left", "rgb_right")},
            "agent_pos": {"shape": [20], "type": "low_dim", "horizon": 1},
        }
    }
    reference = ref_module.TransformerObsEncoder(
        shape_meta=shape_meta, model_name="resnet18", global_pool="", transforms=None, n_emb=64,
        pretrained=False, frozen=False, use_group_norm=False, share_rgb_model=False,
        feature_aggregation=None, downsample_ratio=32,
    ).eval()
    config = small_config(crop_shape=None, n_emb=64)
    ours = ManiFlowObsEncoder(config).eval()
    for index, key in enumerate(ours.camera_keys):
        name = key.removeprefix("observation.images.")
        ours.backbones[index].load_state_dict(reference.key_model_map[name].state_dict(), strict=True)
        ours.rgb_projections[index].load_state_dict(reference.key_projection_map[name].state_dict(), strict=True)
    ours.state_projection.load_state_dict(reference.key_projection_map["agent_pos"].state_dict(), strict=True)

    images = {key: 0.99 * torch.rand(2, 1, 3, 64, 64) for key in CAMERAS}
    state = torch.randn(2, 1, 20)
    expected = reference({**{k.removeprefix("observation.images."): v for k, v in images.items()}, "agent_pos": state})
    actual = ours({OBS_IMAGES: torch.stack([images[k] for k in CAMERAS], dim=2), OBS_STATE: state})
    assert actual.shape == expected.shape == (2, 3 * 4 + 1, 64)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_flow_and_consistency_targets_match_reference():
    _install_reference_stubs()
    cls = importlib.import_module("maniflow.policy.maniflow_image_policy").ManiFlowTransformerImagePolicy
    reference_ditx, ours_ditx = _paired_ditx()
    model = ManiFlowPolicy(small_config()).model

    ref = types.SimpleNamespace(
        device=torch.device("cpu"),
        sample_t_mode_flow="beta",
        sample_t_mode_consistency="discrete",
        sample_dt_mode_consistency="uniform",
        sample_target_t_mode="relative",
        denoise_timesteps=10,
    )
    ref.sample_t = lambda size, mode: cls.sample_t(ref, size, mode)
    ref.sample_dt = lambda size, sample_dt_mode: cls.sample_dt(ref, size, sample_dt_mode)
    ref.linear_interpolate = lambda *a, **k: cls.linear_interpolate(ref, *a, **k)
    ema = types.SimpleNamespace(model=lambda **kw: reference_ditx(kw["sample"], kw["timestep"], kw["target_t"], kw["vis_cond"]))

    actions = torch.randn(6, 8, 20)
    cond = torch.randn(6, 9, 64)

    torch.manual_seed(10)
    expected = cls.get_flow_velocity(ref, actions, vis_cond=cond)
    torch.manual_seed(10)
    x_t, t, target_t, v_target = model.flow_targets(actions)
    torch.testing.assert_close(x_t, expected["x_t"])
    torch.testing.assert_close(t, expected["t"].reshape(-1))
    torch.testing.assert_close(target_t, expected["target_t"].reshape(-1))
    torch.testing.assert_close(v_target, expected["v_target"])

    torch.manual_seed(11)
    expected = cls.get_consistency_velocity(ref, actions, vis_cond=cond, ema_model=ema)
    torch.manual_seed(11)
    x_t, t, target_t, v_target = model.consistency_targets(actions, cond, ours_ditx)
    torch.testing.assert_close(x_t, expected["x_t"])
    torch.testing.assert_close(t, expected["t"].reshape(-1))
    torch.testing.assert_close(target_t, expected["target_t"].reshape(-1))
    torch.testing.assert_close(v_target, expected["v_target"], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("steps", [2, 10])
def test_sampler_matches_reference_sample_ode(steps):
    _install_reference_stubs()
    cls = importlib.import_module("maniflow.policy.maniflow_image_policy").ManiFlowTransformerImagePolicy
    reference_ditx, ours_ditx = _paired_ditx()
    cond = torch.randn(2, 9, 64)
    noise = torch.randn(2, 8, 20)
    ref = types.SimpleNamespace(
        device=torch.device("cpu"), num_inference_steps=steps, sample_target_t_mode="relative",
        model=lambda x, t, target_t, vis_cond: reference_ditx(x, t, target_t, vis_cond),
    )
    expected = cls.sample_ode(ref, x0=noise, N=steps, vis_cond=cond)[-1]

    policy = ManiFlowPolicy(small_config(horizon=8, n_action_steps=8, num_inference_steps=steps))
    model = policy.model
    model.dit = ours_ditx
    model._vis_cond = lambda batch: cond
    actual = model.generate_actions({OBS_STATE: torch.zeros(2, 1, 20)}, noise=copy.deepcopy(noise))
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_ema_step_matches_reference_ema_model_on_parameters():
    """Shared EMA equals the official EMAModel (identical in ManiFlow and Diffusion Policy)."""
    _install_reference_stubs()
    EMAModel = importlib.import_module("maniflow.model.diffusion.ema_model").EMAModel
    torch.manual_seed(12)
    online = nn.Sequential(nn.Linear(4, 8), nn.BatchNorm1d(8), nn.Linear(8, 2))
    online[2].bias.requires_grad_(False)
    ours = copy.deepcopy(online).requires_grad_(False)
    reference_copy = copy.deepcopy(online)
    reference = EMAModel(reference_copy, update_after_step=0, inv_gamma=1.0, power=0.75, min_value=0.0, max_value=0.9999)
    schedule = EmaSchedule()
    for step in range(25):
        with torch.no_grad():
            for p in online.parameters():
                p.add_(0.1 * torch.randn_like(p))
        online.train()(torch.randn(16, 4))  # moves BatchNorm running statistics
        reference.step(online)
        ema_step(ours, online, schedule.decay(step))
    for (name, a), (_, b) in zip(ours.named_parameters(), reference_copy.named_parameters(), strict=True):
        torch.testing.assert_close(a, b, msg=name)
    # Adaptation: BatchNorm statistics follow the online network.
    torch.testing.assert_close(ours[1].running_mean, online[1].running_mean)
    torch.testing.assert_close(ours[1].running_var, online[1].running_var)
