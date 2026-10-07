"""OAT/QueST autoregressive policy: reference parity, tokenizer stage, policy API."""

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from paw_baseline.artok.ar_model import AutoregressiveModel
from paw_baseline.artok.oat_tokenizer import OATTokenizer
from paw_baseline.artok.quest_tokenizer import QueSTTokenizer

REFERENCE = Path(__file__).resolve().parents[2] / "tmp/references/oat"


@pytest.fixture(scope="module")
def reference():
    """Import the pinned reference implementation, stubbing unrelated imports."""
    if not (REFERENCE / "oat").is_dir():
        pytest.skip("reference OAT checkout not available")
    for name in ("dill", "hydra", "zarr"):
        if name not in sys.modules:
            try:
                __import__(name)
            except ImportError:
                stub = types.ModuleType(name)
                stub.Array = type("Array", (), {})  # only used in type hints
                sys.modules[name] = stub
    sys.path.insert(0, str(REFERENCE))
    try:
        from oat.model.autoregressive.transformer_cache import AutoregressiveModel as RefAR
        from oat.tokenizer.oat.decoder.single_pass_decoder import SinglePassDecoder
        from oat.tokenizer.oat.encoder.register_encoder import RegisterEncoder
        from oat.tokenizer.oat.quantizer.fsq import FSQ
        from oat.tokenizer.quest.tokenizer import QueSTTok
    finally:
        sys.path.remove(str(REFERENCE))
    return types.SimpleNamespace(
        RegisterEncoder=RegisterEncoder,
        SinglePassDecoder=SinglePassDecoder,
        FSQ=FSQ,
        QueSTTok=QueSTTok,
        AR=RefAR,
    )


def _oat_key(key: str) -> str:
    return key.replace("sample_emb.patch_proj.", "sample_emb.").replace(
        "latent_proj.proj.", "latent_proj."
    )


def test_oat_tokenizer_matches_reference(reference):
    torch.manual_seed(0)
    ref_encoder = reference.RegisterEncoder(
        sample_dim=20, sample_horizon=32, emb_dim=256, head_dim=64, depth=2,
        pdropout=0.1, latent_dim=4, num_registers=8,
    )
    ref_decoder = reference.SinglePassDecoder(
        sample_dim=20, sample_horizon=32, emb_dim=256, head_dim=64, depth=4,
        pdropout=0.1, token_dropout_mode="pow2", use_causal_decoder=True,
        latent_dim=4, latent_horizon=8,
    )
    ref_fsq = reference.FSQ(levels=[8, 5, 5, 5])
    for module in (ref_encoder, ref_decoder):
        for param in module.parameters():
            if param.requires_grad:
                param.data.normal_(0, 0.05)
    ours = OATTokenizer(action_dim=20, horizon=32)
    state = {
        **{"encoder." + _oat_key(k): v for k, v in ref_encoder.state_dict().items()},
        **{"decoder." + _oat_key(k): v for k, v in ref_decoder.state_dict().items()},
    }
    ours.load_state_dict(state, strict=True)
    for module in (ref_encoder, ref_decoder, ref_fsq, ours):
        module.eval()

    x = torch.rand(6, 32, 20) * 2 - 1
    with torch.no_grad():
        ref_latents, ref_tokens = ref_fsq(ref_encoder(x))
        tokens = ours.tokenize(x)
        assert torch.equal(tokens, ref_tokens)
        assert ref_tokens.unique().numel() > 1
        for keep in (1, 2, 4, 8):
            ref = ref_decoder(
                ref_fsq.indices_to_embedding(
                    torch.cat([ref_tokens[:, :keep], torch.zeros(6, 8 - keep, dtype=torch.long)], 1)
                ),
                eval_keep_k=[keep] * 6,
            )
            torch.testing.assert_close(ours.detokenize(tokens[:, :keep]), ref)
        torch.testing.assert_close(ours.autoencode(x), ref_decoder(ref_latents, eval_keep_k=[8] * 6))


def test_quest_tokenizer_matches_reference(reference):
    torch.manual_seed(1)
    ref = reference.QueSTTok(action_dim=20, horizon=32)
    torch.manual_seed(1)
    ours = QueSTTokenizer(action_dim=20, horizon=32)
    # Fixed sin-cos frequencies and the mixin's device marker are not weights.
    ref_state = {
        k: v
        for k, v in ref.state_dict().items()
        if not k.startswith(("normalizer", "_dummy_variable", "add_pos_emb", "fix_pos_emb"))
    }
    mapped = {
        k.replace("conv_block.blocks.", "conv_block.blocks.")
        .replace(".block.0.conv.", ".conv.conv.")
        .replace(".block.2.", ".norm."): v
        for k, v in ref_state.items()
    }
    ours.load_state_dict(mapped, strict=True)
    # Identity reference normalizer: ours receives already-normalized actions.
    ref.normalizer = _IdentityNormalizer()
    ref.eval()
    ours.eval()
    x = torch.rand(5, 32, 20) * 2 - 1
    with torch.no_grad():
        ref_tokens = ref.tokenize(x)
        tokens = ours.tokenize(x)
        assert torch.equal(tokens, ref_tokens.long())
        assert tokens.shape == (5, 8)
        torch.testing.assert_close(ours.detokenize(tokens), ref.detokenize(ref_tokens))
        torch.testing.assert_close(ours.autoencode(x), ref.autoencode(x))


class _IdentityNormalizer(torch.nn.Module):
    class _Field:
        def normalize(self, x):
            return x

        def unnormalize(self, x):
            return x

    def __getitem__(self, key):
        return self._Field()


def test_ar_model_matches_reference(reference):
    torch.manual_seed(2)
    kwargs = dict(
        vocab_size=1001, max_seq_len=9, max_cond_len=1, cond_dim=212,
        n_layer=4, n_head=4, n_emb=256, p_drop_emb=0.1, p_drop_attn=0.1,
    )
    ref = reference.AR(**kwargs)
    ours = AutoregressiveModel(**kwargs)
    ours.load_state_dict(
        {k: v for k, v in ref.state_dict().items() if k != "_dummy_variable"}, strict=True
    )
    ref.eval()
    ours.eval()
    tokens = torch.randint(0, 1001, (3, 9))
    cond = torch.randn(3, 1, 212)
    with torch.no_grad():
        torch.testing.assert_close(ours(tokens, cond), ref(tokens, cond))
    prefix = torch.full((3, 1), 1000)
    torch.manual_seed(5)
    ref_out = ref.generate(prefix, cond, max_new_tokens=8, temperature=1.0, top_k=10)
    torch.manual_seed(5)
    out = ours.generate(prefix, cond, max_new_tokens=8, temperature=1.0, top_k=10)
    assert torch.equal(out, ref_out)
    assert out.shape == (3, 9)


def test_oat_nested_dropout_masks_tail_in_training():
    tokenizer = OATTokenizer(action_dim=4, horizon=32)
    tokenizer.train()
    latents = torch.randn(64, 8, 256)
    out = tokenizer.decoder.nested_dropout(latents.clone())
    masked = (out == tokenizer.decoder.nested_dropout.dropout_mask_token).all(-1)
    kept = (~masked).sum(1)
    assert set(kept.tolist()) <= {1, 2, 4, 8}
    # Masks are suffixes: once a token is masked, every later token is masked.
    assert torch.all(masked[:, 1:] >= masked[:, :-1])


def test_min_max_matches_lerobot_normalizer():
    from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.processor.normalize_processor import NormalizerProcessorStep

    from train.tokenizer_stage import normalize_min_max

    rng = np.random.default_rng(0)
    values = rng.normal(size=(50, 20)).astype(np.float32)
    stats = {"min": values.min(0).tolist(), "max": values.max(0).tolist()}
    stats["min"][3] = stats["max"][3] = 0.5  # constant dimension
    step = NormalizerProcessorStep(
        features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(20,))},
        norm_map={FeatureType.ACTION: NormalizationMode.MIN_MAX},
        stats={"action": {k: np.asarray(v) for k, v in stats.items()}},
    )
    expected = step._normalize_action(torch.from_numpy(values), inverse=False)
    np.testing.assert_allclose(normalize_min_max(values, stats), expected.numpy(), rtol=1e-5, atol=1e-5)


def _write_prepared(root: Path, episodes: list[int], length: int) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = len(episodes) * length
    actions = np.random.default_rng(0).uniform(-1, 1, size=(rows, 20)).astype(np.float32)
    (root / "data/chunk-000").mkdir(parents=True)
    (root / "meta").mkdir()
    table = pa.table(
        {
            "action": pa.FixedSizeListArray.from_arrays(pa.array(actions.ravel()), 20),
            "episode_index": np.repeat(episodes, length),
            "frame_index": np.tile(np.arange(length), len(episodes)),
        }
    )
    pq.write_table(table, root / "data/chunk-000/file-000.parquet")
    (root / "meta/stats.json").write_text(
        json.dumps({"action": {"min": [-1.0] * 20, "max": [1.0] * 20}})
    )


def test_chunks_repeat_episode_end(tmp_path):
    from train.tokenizer_stage import _chunks, load_action_chunks

    _write_prepared(tmp_path, [0, 1, 2], 40)
    actions, anchors, ends = load_action_chunks(tmp_path, [0, 2])
    assert actions.shape == (80, 20)
    chunk = _chunks(
        torch.from_numpy(actions), torch.from_numpy(anchors), torch.from_numpy(ends),
        torch.tensor([30, 79]), 32,
    )
    np.testing.assert_array_equal(chunk[0, :10].numpy(), actions[30:40])
    np.testing.assert_array_equal(chunk[0, 10:].numpy(), np.repeat(actions[39:40], 22, 0))
    np.testing.assert_array_equal(chunk[1].numpy(), np.repeat(actions[79:80], 32, 0))


@pytest.mark.parametrize("tokenizer_type", ["oat", "quest"])
def test_tokenizer_stage_and_policy_roundtrip(tmp_path, tokenizer_type):
    from lerobot.configs import FeatureType, PolicyFeature
    from safetensors.torch import load_file

    from paw_baseline.artok.configuration_artok import ARTokConfig
    from paw_baseline.artok.modeling_artok import ARTokPolicy
    from train.tokenizer_stage import artok_config_from_args, train_tokenizer

    _write_prepared(tmp_path / "prepared", [0, 1], 40)
    config = artok_config_from_args(
        [
            "--policy.type=artok",
            f"--policy.tokenizer_type={tokenizer_type}",
            "--policy.tokenizer_steps=6",
            "--policy.tokenizer_batch_size=16",
            "--policy.device=cpu",
            "--batch_size=8",
        ]
    )
    assert config.tokenizer_type == tokenizer_type and config.tokenizer_steps == 6
    path = train_tokenizer(config, tmp_path / "prepared", [0, 1], tmp_path / "tok", "cpu")
    report = json.loads((tmp_path / "tok/tokenizer_report.json").read_text())
    assert report["steps"] == 6
    assert len(report["ema"]["train_mode_token_agreement_per_position"]) == 8

    from train.tokenizer_stage import scratch_dir

    other = artok_config_from_args(["--policy.type=artok", "--policy.tokenizer_type=oat" if tokenizer_type == "quest" else "--policy.tokenizer_type=quest"])
    base = scratch_dir(tmp_path / "out", config, tmp_path / "prepared", [0, 1])
    assert base == scratch_dir(tmp_path / "out", config, tmp_path / "prepared", [0, 1])
    assert base != scratch_dir(tmp_path / "out", other, tmp_path / "prepared", [0, 1])
    assert base != scratch_dir(tmp_path / "out", config, tmp_path / "prepared", [0])

    policy_config = ARTokConfig(
        tokenizer_type=tokenizer_type,
        device="cpu",
        input_features={
            "observation.images.rgb_static": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
            "observation.images.rgb_left": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(20,)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(20,))},
        resize_shape=(32, 32),
        crop_shape=(28, 28),
    )
    policy = ARTokPolicy(policy_config)
    policy.load_tokenizer(path)
    for key, value in load_file(str(path)).items():
        torch.testing.assert_close(policy.tokenizer.state_dict()[key], value)

    batch = {
        "observation.images.rgb_static": torch.rand(4, 3, 64, 64) * 2 - 1,
        "observation.images.rgb_left": torch.rand(4, 3, 64, 64) * 2 - 1,
        "observation.state": torch.rand(4, 20) * 2 - 1,
        "action": torch.rand(4, 32, 20) * 2 - 1,
    }
    policy.train()
    loss, info = policy.forward(batch)
    loss.backward()
    assert torch.isfinite(loss) and 0 <= info["token_accuracy"] <= 1
    assert all(p.grad is None for p in policy.tokenizer.parameters())
    assert any(p.grad is not None for p in policy.model.parameters())
    assert any(p.grad is not None for p in policy.obs_encoder.parameters())
    groups = policy.get_optim_params()
    assert groups[1]["lr"] == policy_config.obs_encoder_lr
    optim_ids = {id(p) for group in groups for p in group["params"]}
    assert not optim_ids & {id(p) for p in policy.ema_model.parameters()}
    assert not optim_ids & {id(p) for p in policy.tokenizer.parameters()}

    # First EMA update copies the online weights; later ones average.
    with torch.no_grad():
        for p in policy.model.parameters():
            p.add_(1.0)
    policy.update()
    for p, e in zip(policy.model.parameters(), policy.ema_model.parameters(), strict=True):
        torch.testing.assert_close(p, e)
    assert int(policy.ema_step) == 1

    policy.eval()
    policy.reset()
    actions = [policy.select_action({k: v[:1] for k, v in batch.items() if k != "action"}) for _ in range(16)]
    assert len(policy._action_queue) == 0
    assert torch.stack(actions, 1).shape == (1, 16, 20)

    policy.save_pretrained(tmp_path / "saved")
    restored = ARTokPolicy.from_pretrained(tmp_path / "saved", config=policy_config)
    for key, value in policy.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value)


PRETRAINED = "ResNet18_Weights.IMAGENET1K_V1"


def _pretrained_config(**overrides):
    from lerobot.configs import FeatureType, PolicyFeature

    from paw_baseline.artok.configuration_artok import ARTokConfig

    return ARTokConfig(
        device="cpu",
        input_features={
            "observation.images.rgb_static": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
            "observation.images.rgb_left": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(20,)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(20,))},
        resize_shape=(32, 32),
        crop_shape=(28, 28),
        pretrained_backbone_weights=PRETRAINED,
        use_group_norm=False,
        **overrides,
    )


def test_pretrained_config_requires_batchnorm_and_imagenet_normalization():
    from lerobot.configs import NormalizationMode

    from paw_baseline.artok.configuration_artok import ARTokConfig

    config = _pretrained_config()
    assert config.normalization_mapping["VISUAL"] == NormalizationMode.MEAN_STD
    assert config.normalization_mapping["ACTION"] == NormalizationMode.MIN_MAX
    assert ARTokConfig(device="cpu").normalization_mapping["VISUAL"] == NormalizationMode.MIN_MAX
    with pytest.raises(ValueError, match="BatchNorm"):
        ARTokConfig(device="cpu", pretrained_backbone_weights=PRETRAINED)


def test_pretrained_backbones_load_into_online_and_ema_trunks(tmp_path):
    from paw_baseline.artok.modeling_artok import ARTokPolicy, resnet_trunk

    config = _pretrained_config()
    policy = ARTokPolicy(config)
    trunk = policy.obs_encoder.cameras[0].backbone
    assert any(isinstance(m, torch.nn.BatchNorm2d) for m in trunk.modules())
    assert not any(isinstance(m, torch.nn.GroupNorm) for m in trunk.modules())
    reference = resnet_trunk("resnet18", PRETRAINED).state_dict()
    assert not torch.equal(trunk.state_dict()["0.weight"], reference["0.weight"])

    policy.load_pretrained_backbones()
    for encoder in (policy.obs_encoder, policy.ema_obs_encoder):
        for camera in encoder.cameras:
            for key, value in reference.items():
                torch.testing.assert_close(camera.backbone.state_dict()[key], value)

    # A training step moves BatchNorm running statistics; EMA must follow them.
    batch = {
        "observation.images.rgb_static": torch.randn(4, 3, 64, 64),
        "observation.images.rgb_left": torch.randn(4, 3, 64, 64),
        "observation.state": torch.rand(4, 20) * 2 - 1,
        "action": torch.rand(4, 32, 20) * 2 - 1,
    }
    policy.train()
    policy.forward(batch)[0].backward()
    online_bn = policy.obs_encoder.cameras[0].backbone[1]
    ema_bn = policy.ema_obs_encoder.cameras[0].backbone[1]
    assert not torch.equal(online_bn.running_mean, ema_bn.running_mean)
    policy.update()
    policy.update()
    torch.testing.assert_close(online_bn.running_mean, ema_bn.running_mean)
    torch.testing.assert_close(online_bn.running_var, ema_bn.running_var)
    assert torch.equal(online_bn.num_batches_tracked, ema_bn.num_batches_tracked)
    # Tied embedding/head weights are averaged once, like other parameters.
    torch.testing.assert_close(policy.ema_model.head.weight, policy.ema_model.tok_emb.weight)

    # Serving restores the trunks strictly from the checkpoint, without torchvision weights.
    policy.save_pretrained(tmp_path / "saved")
    restored = ARTokPolicy.from_pretrained(tmp_path / "saved", config=config)
    for key, value in policy.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value)


def test_train_hook_loads_tokenizer_and_pretrained_trunks(tmp_path):
    from safetensors.torch import save_file

    from paw_baseline.artok.modeling_artok import ARTokPolicy, resnet_trunk
    from train.tokenizer_stage import install_tokenizer_hook

    config = _pretrained_config()
    donor = ARTokPolicy(config)
    save_file({k: v.contiguous() for k, v in donor.tokenizer.state_dict().items()}, str(tmp_path / "tok.safetensors"))
    fake = types.SimpleNamespace(make_policy=lambda: ARTokPolicy(config))
    install_tokenizer_hook(fake, tmp_path / "tok.safetensors")
    policy = fake.make_policy()
    reference = resnet_trunk("resnet18", PRETRAINED).state_dict()
    torch.testing.assert_close(policy.obs_encoder.cameras[1].backbone.state_dict()["0.weight"], reference["0.weight"])
    torch.testing.assert_close(policy.ema_obs_encoder.cameras[0].backbone.state_dict()["1.running_var"], reference["1.running_var"])
    for key, value in donor.tokenizer.state_dict().items():
        torch.testing.assert_close(policy.tokenizer.state_dict()[key], value)


def test_imagenet_presets_resolve_and_differ_only_in_tokenizer():
    from lerobot.configs import NormalizationMode

    from train.tokenizer_stage import artok_config_from_args

    configs = {}
    for name in ("oat_imagenet", "quest_imagenet", "oat"):
        preset = json.loads((Path(__file__).resolve().parents[1] / f"configs/{name}.json").read_text())
        configs[name] = artok_config_from_args(preset["args"])
    assert configs["oat_imagenet"].pretrained_backbone_weights == PRETRAINED
    assert configs["oat_imagenet"].use_group_norm is False
    assert configs["oat_imagenet"].normalization_mapping["VISUAL"] == NormalizationMode.MEAN_STD
    assert configs["oat"].pretrained_backbone_weights is None and configs["oat"].use_group_norm
    a, b = vars(configs["oat_imagenet"]), vars(configs["quest_imagenet"])
    assert {k for k in a if a[k] != b[k]} == {"tokenizer_type"}
