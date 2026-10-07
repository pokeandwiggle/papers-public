"""Patch Policy: reference parity of the trunk, train/serve history alignment, tokenizer."""

import math

import numpy as np
import pytest
import torch
from lerobot.configs.types import FeatureType, PolicyFeature

from paw_baseline.layout import CAMERA_KEYS
from train.patch_policy.configuration_patch_vqbet import PatchVQBeTConfig
from train.patch_policy.encoder import FrozenDinoV2, hub_interpolate_pos_encoding
from train.patch_policy.gpt import GPT, GPTConfig
from train.patch_policy.modeling_patch_vqbet import PatchVQBeTPolicy, batch_idx, history_indices
from train.patch_policy.training import episode_action_chunks


def reference_mask(npatch: int, nwindow: int) -> torch.Tensor:
    """generate_mask_matrix from the reference gpt.py."""
    zeros, ones = torch.zeros(npatch, npatch), torch.ones(npatch, npatch)
    rows = [torch.cat([ones] * (i + 1) + [zeros] * (nwindow - i - 1), dim=1) for i in range(nwindow)]
    return torch.cat(rows, dim=0)


def reference_attention(attn, x, mask):
    """CausalSelfAttention.forward from the reference, on the same weights."""
    b, t, c = x.size()
    q, k, v = attn.c_attn(x).split(attn.n_embd, dim=2)
    q, k, v = (z.view(b, t, attn.n_head, c // attn.n_head).transpose(1, 2) for z in (q, k, v))
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    att = att.masked_fill(mask[:t, :t] == 0, float("-inf")).softmax(dim=-1)
    y = (att @ v).transpose(1, 2).contiguous().view(b, t, c)
    return attn.c_proj(y)


@pytest.mark.parametrize("frames,tokens", [(2, 6), (3, 4), (1, 5)])
def test_block_causal_attention_matches_reference_mask(frames, tokens):
    torch.manual_seed(0)
    gpt = GPT(GPTConfig(n_frames=frames, tokens_per_frame=tokens, input_dim=8, n_layer=1, n_head=2, n_embd=16))
    attn = gpt.transformer.h[0].attn.eval()
    x = torch.randn(3, frames * tokens, 16)
    expected = reference_attention(attn, x, reference_mask(tokens, frames))
    torch.testing.assert_close(attn(x), expected, atol=1e-5, rtol=1e-5)


def test_fused_gelu_matches_reference_formula():
    from train.patch_policy.gpt import new_gelu

    x = torch.linspace(-6, 6, 1001)
    reference = 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
    torch.testing.assert_close(new_gelu(x), reference, atol=1e-6, rtol=1e-6)


def test_gpt_reads_last_token_and_is_causal_over_frames():
    torch.manual_seed(0)
    gpt = GPT(GPTConfig(n_frames=2, tokens_per_frame=3, input_dim=4, n_layer=2, n_head=2, n_embd=8)).eval()
    x = torch.randn(1, 2, 3, 4)
    out = gpt(x)
    assert out.shape == (1, 2, 256)
    later = x.clone()
    later[:, 1] += 1.0  # a newer frame cannot change an earlier frame's readout
    torch.testing.assert_close(gpt(later)[:, 0], out[:, 0])
    assert not torch.allclose(gpt(later)[:, 1], out[:, 1])


def test_hub_interpolation_matches_hub_formula():
    class Embeddings:
        patch_size = 14

    emb = Embeddings()
    emb.position_embeddings = torch.randn(1, 1 + 37 * 37, 8)
    out = hub_interpolate_pos_encoding(emb, torch.zeros(1, 257, 8), 224, 224)
    grid = emb.position_embeddings[:, 1:].reshape(1, 37, 37, 8).permute(0, 3, 1, 2)
    expected = torch.nn.functional.interpolate(
        grid, scale_factor=(16.1 / 37, 16.1 / 37), mode="bicubic", antialias=False
    ).permute(0, 2, 3, 1).reshape(1, 256, 8)
    torch.testing.assert_close(out[:, 1:], expected)
    torch.testing.assert_close(out[:, :1], emb.position_embeddings[:, :1])
    same = hub_interpolate_pos_encoding(emb, torch.zeros(1, 1 + 37 * 37, 8), 518, 518)
    assert same is emb.position_embeddings


def test_history_indices_match_lerobot_clamping():
    offsets = [-3, 0]
    for t in range(8):
        buffer_length = min(t + 1, 4)  # ring buffer of (n_obs_steps - 1) * stride + 1 frames
        newest_frame = t
        frames = [newest_frame - buffer_length + 1 + i for i in range(buffer_length)]
        picked = [frames[i] for i in history_indices(buffer_length, offsets)]
        assert picked == [max(t + o, 0) for o in offsets]


def test_episode_chunks_match_lerobot_end_clamping():
    actions = np.arange(10, dtype=np.float32)[:, None]
    episodes = np.array([0] * 6 + [1] * 4)
    chunks = episode_action_chunks(actions, episodes, 3)
    assert chunks.shape == (10, 3, 1)
    np.testing.assert_array_equal(chunks[4, :, 0], [4, 5, 5])
    np.testing.assert_array_equal(chunks[6, :, 0], [6, 7, 8])
    np.testing.assert_array_equal(chunks[9, :, 0], [9, 9, 9])


def test_batch_idx_selects_code_offsets():
    x = torch.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5).float()
    idx = torch.tensor([[1, 0, 3], [2, 2, 0]])
    out = batch_idx(x, idx)
    assert out.shape == (2, 3, 5)
    torch.testing.assert_close(out[1, 0], x[1, 0, 2])


def tiny_config(**overrides) -> PatchVQBeTConfig:
    features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(20,)),
        **{key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)) for key in CAMERA_KEYS},
    }
    values = dict(
        input_features=features,
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(20,))},
        device="cpu",
        encoder_pretrained=False,
        image_size=28,
        chunk_size=5,
        n_action_steps=5,
        n_layer=1,
        n_head=2,
        n_embd=32,
        vqvae_steps=3,
        vqvae_batch_size=16,
        trunk_dtype="float32",
    )
    values.update(overrides)
    return PatchVQBeTConfig(**values)


def tiny_batch(config, batch=2):
    span = (config.n_obs_steps - 1) * config.obs_stride + config.chunk_size
    out = {key: torch.rand(batch, config.n_obs_steps, 3, 32, 32) for key in CAMERA_KEYS}
    out["action"] = torch.rand(batch, span, 20) * 2 - 1
    return out


def test_config_delta_indices_and_unpacking():
    config = tiny_config()
    assert config.observation_delta_indices == [-3, 0]
    assert config.action_delta_indices == list(range(-3, 5))
    policy = PatchVQBeTPolicy(config)
    actions = torch.arange(8).float().view(1, 8, 1).expand(1, 8, 20)
    unpacked = policy.unpack_actions(actions)
    assert unpacked.shape == (1, 2, 5, 20)
    torch.testing.assert_close(unpacked[0, 0, :, 0], torch.arange(0, 5).float())  # frame t-3
    torch.testing.assert_close(unpacked[0, 1, :, 0], torch.arange(3, 8).float())  # frame t


@pytest.mark.parametrize("feature,tokens", [("patch", 4 * 3), ("cls", 3)])
def test_policy_trains_only_trunk_and_heads(feature, tokens):
    torch.manual_seed(0)
    config = tiny_config(visual_feature=feature)
    policy = PatchVQBeTPolicy(config)
    assert policy.gpt.config.tokens_per_frame == tokens
    with pytest.raises(RuntimeError, match="fit the VQ tokenizer"):
        policy.forward(tiny_batch(config))
    info = policy.fit_tokenizer(torch.rand(64, 5, 20) * 2 - 1, log=lambda *_: None)
    assert info["unique_chunks"] == 64
    policy.train()
    assert not policy.vqvae.training and not policy.encoder.model.training
    loss, metrics = policy.forward(tiny_batch(config))
    assert torch.isfinite(loss) and "offset_loss" in metrics
    loss.backward()
    optimized = {id(p) for group in policy.get_optim_params() for p in group["params"]}
    for name, param in policy.named_parameters():
        trainable = name.startswith(("gpt.", "bin_head.", "offset_head."))
        assert (id(param) in optimized) == trainable, name
        if trainable:
            assert param.grad is not None, name
        else:
            assert param.grad is None, name
    decay = {g["weight_decay"] for g in policy.get_optim_params()}
    assert decay == {config.optimizer_weight_decay, 0.0, config.bin_head_weight_decay}


def test_dense_and_cls_policies_share_one_tokenizer():
    chunks = torch.rand(64, 5, 20) * 2 - 1
    states = []
    for feature in ("patch", "cls"):
        torch.manual_seed(123 if feature == "patch" else 7)  # unrelated global RNG state
        policy = PatchVQBeTPolicy(tiny_config(visual_feature=feature))
        torch.rand(5)
        policy.fit_tokenizer(chunks, log=lambda *_: None)
        states.append(policy.vqvae.state_dict())
    assert states[0].keys() == states[1].keys()
    for key in states[0]:
        torch.testing.assert_close(states[0][key], states[1][key], msg=key)


def test_save_load_roundtrip_is_strict_and_identical(tmp_path):
    torch.manual_seed(0)
    config = tiny_config()
    policy = PatchVQBeTPolicy(config)
    policy.fit_tokenizer(torch.rand(64, 5, 20) * 2 - 1, log=lambda *_: None)
    policy.save_pretrained(tmp_path)
    restored = PatchVQBeTPolicy.from_pretrained(tmp_path)
    assert bool(restored.vqvae_fitted)
    batch = {key: value[:, :, :] for key, value in tiny_batch(config).items() if key != "action"}
    policy.eval()
    torch.manual_seed(1)
    expected = policy.predict_action_chunk(batch)
    torch.manual_seed(1)
    actual = restored.predict_action_chunk(batch)
    assert actual.shape == (2, 5, 20)
    torch.testing.assert_close(actual, expected)


def test_select_action_uses_strided_history():
    torch.manual_seed(0)
    config = tiny_config(n_action_steps=1)
    policy = PatchVQBeTPolicy(config)
    policy.fit_tokenizer(torch.rand(64, 5, 20) * 2 - 1, log=lambda *_: None)
    seen = []
    original = policy.predict_action_chunk

    def record(batch):
        seen.append(batch[CAMERA_KEYS[0]][0, :, 0, 0, 0].tolist())
        return original(batch)

    policy.predict_action_chunk = record
    policy.reset()
    for t in range(6):
        frame = {key: torch.full((1, 3, 32, 32), float(t)) for key in CAMERA_KEYS}
        policy.select_action(frame)
    assert seen == [[0, 0], [0, 1], [0, 2], [0, 3], [1, 4], [2, 5]]


def test_encoder_tokens_shape():
    encoder = FrozenDinoV2("patch", 28, pretrained=None)
    assert encoder(torch.rand(2, 3, 32, 32)).shape == (2, 4, 384)
    encoder = FrozenDinoV2("cls", 28, pretrained=None)
    assert encoder(torch.rand(2, 3, 32, 32)).shape == (2, 1, 384)
