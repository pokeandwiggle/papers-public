import torch
import torch.nn as nn
from transformers import LlamaConfig

from paw_baseline.oft.prismatic import EMPTY_ID, OpenVLABackbone
from paw_baseline.oft.modeling_paw_oft import openvla_prompt


class _TinyVision(nn.Module):
    embed_dim = 6

    def forward(self, x):
        return x.flatten(2).transpose(1, 2)[:, :4, :6]


def _backbone():
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=32064, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=256,
    )
    config._attn_implementation = "sdpa"
    return OpenVLABackbone(vision_backbone=_TinyVision(), config=config).eval()


def test_attention_is_bidirectional_and_reads_shifted_positions():
    model = _backbone()
    patches = torch.randn(1, 5, 32)
    prompt = torch.tensor([1, 512, 29901, 3744, EMPTY_ID])
    base = model(patches, [prompt], num_action_tokens=6)
    assert base.shape == (1, 6, 32)
    # A later prompt token is not visible to earlier patches under causal attention;
    # with OFT's bidirectional attention every readout changes.
    changed = model(patches, [torch.tensor([1, 512, 29901, 3745, EMPTY_ID])], 6)
    assert not torch.allclose(base, changed)
    # The first readout sits at the last prompt token. Under causal attention it
    # cannot see the action tokens and </s> that follow it.
    longer = model(patches, [prompt], num_action_tokens=9)
    assert not torch.allclose(base[0, 0], longer[0, 0])


def test_padding_mask_matches_unpadded_samples():
    model = _backbone()
    patches = torch.randn(2, 5, 32)
    short = torch.tensor([1, 512, 29901, EMPTY_ID])
    long = torch.tensor([1, 512, 29901, 3744, 13, EMPTY_ID])
    batched = model(patches, [short, long], 4)
    torch.testing.assert_close(batched[0], model(patches[:1], [short], 4)[0], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(batched[1], model(patches[1:], [long], 4)[0], atol=1e-5, rtol=1e-5)


def test_prompt_matches_openvla_builder():
    assert openvla_prompt("Pick Up The Cup") == (
        "In: What action should the robot take to pick up the cup?\nOut:"
    )
