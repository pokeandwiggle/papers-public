"""OpenVLA-7B (Prismatic DINOv2+SigLIP / Llama-2) backbone for OpenVLA-OFT.

This mirrors openvla-oft ``prismatic/extern/hf/modeling_prismatic.py`` for
current timm/transformers: each fused featurizer returns the output of its
second-to-last block without prefix tokens, the fused-GELU MLP projector maps
the concatenated patches into Llama-2, and the language model attends
bidirectionally over all non-padding tokens (OpenVLA-OFT's transformers fork).
"""

import json
from pathlib import Path

import timm
import torch
import torch.nn as nn
from safetensors import safe_open
from torch.utils.checkpoint import checkpoint
from transformers import LlamaConfig, LlamaModel

TIMM_MODEL_IDS = ("vit_large_patch14_reg4_dinov2.lvd142m", "vit_so400m_patch14_siglip_224")
IMAGE_SIZE = 224
#: Prismatic image processor normalization for the DINOv2 and SigLIP featurizers.
DINO_MEAN = (0.484375, 0.455078125, 0.40625)
DINO_STD = (0.228515625, 0.2236328125, 0.224609375)
SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)
LLM_DIM = 4096
VOCAB_SIZE = 32064
PAD_TOKEN_ID = 32000
STOP_TOKEN_ID = 2
EMPTY_TOKEN_ID = 29871
ACTION_PLACEHOLDER_ID = 31744


def llama2_7b_config() -> LlamaConfig:
    """OpenVLA's text config: LlamaConfig defaults of transformers 4.40.1."""
    return LlamaConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=LLM_DIM,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        hidden_act="silu",
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_bias=False,
        mlp_bias=False,
        tie_word_embeddings=False,
        pad_token_id=PAD_TOKEN_ID,
        bos_token_id=1,
        eos_token_id=2,
        attn_implementation="sdpa",
    )


class Featurizer(nn.Module):
    """timm ViT returning patch tokens from its second-to-last block."""

    def __init__(self, model_id: str):
        super().__init__()
        self.vit = timm.create_model(
            model_id, pretrained=False, num_classes=0, img_size=IMAGE_SIZE
        )
        self.embed_dim = self.vit.embed_dim
        self.gradient_checkpointing = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        vit = self.vit
        x = vit.norm_pre(vit.patch_drop(vit._pos_embed(vit.patch_embed(x))))
        for block in vit.blocks[: len(vit.blocks) - 1]:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return x[:, vit.num_prefix_tokens :]


class PrismaticProjector(nn.Module):
    """Prismatic fused-GELU MLP projector (``no-align+fused-gelu-mlp``)."""

    def __init__(self, vision_dim: int, llm_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(vision_dim, 4 * vision_dim, bias=True)
        self.fc2 = nn.Linear(4 * vision_dim, llm_dim, bias=True)
        self.fc3 = nn.Linear(llm_dim, llm_dim, bias=True)
        self.act_fn1 = nn.GELU()
        self.act_fn2 = nn.GELU()

    def forward(self, patches):
        return self.fc3(self.act_fn2(self.fc2(self.act_fn1(self.fc1(patches)))))


class ProprioProjector(nn.Module):
    """OpenVLA-OFT proprio projector: one token in the LLM embedding space."""

    def __init__(self, llm_dim: int, proprio_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(proprio_dim, llm_dim, bias=True)
        self.fc2 = nn.Linear(llm_dim, llm_dim, bias=True)
        self.act_fn1 = nn.GELU()

    def forward(self, proprio):
        return self.fc2(self.act_fn1(self.fc1(proprio)))


class PrismaticVLA(nn.Module):
    """Fused vision backbone, projector and Llama-2 decoder (no LM head)."""

    def __init__(self):
        super().__init__()
        self.featurizer = Featurizer(TIMM_MODEL_IDS[0])
        self.fused_featurizer = Featurizer(TIMM_MODEL_IDS[1])
        vision_dim = self.featurizer.embed_dim + self.fused_featurizer.embed_dim
        self.projector = PrismaticProjector(vision_dim, LLM_DIM)
        self.llm = LlamaModel(llama2_7b_config())
        self.num_patches = self.featurizer.vit.patch_embed.num_patches

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        self.featurizer.gradient_checkpointing = enabled
        self.fused_featurizer.gradient_checkpointing = enabled
        if enabled:
            self.llm.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        else:
            self.llm.gradient_checkpointing_disable()

    def embed_images(self, dino_pixels: torch.Tensor, siglip_pixels: torch.Tensor):
        """(N, 3, 224, 224) pixels per featurizer to projected patches (N, 256, D)."""
        patches = torch.cat(
            [self.featurizer(dino_pixels), self.fused_featurizer(siglip_pixels)], dim=2
        )
        return self.projector(patches)

    def bidirectional_hidden_states(
        self, embeddings: torch.Tensor, key_mask: torch.Tensor
    ) -> torch.Tensor:
        """Run Llama with full attention over non-padding keys (OFT fork semantics)."""
        batch, length = key_mask.shape
        mask = key_mask[:, None, None, :].expand(batch, 1, length, length)
        output = self.llm(
            inputs_embeds=embeddings, attention_mask=mask, use_cache=False
        )
        return output.last_hidden_state


def openvla_weight_files(snapshot: Path) -> list[Path]:
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())
    return sorted({snapshot / name for name in index["weight_map"].values()})


def map_openvla_key(key: str) -> str | None:
    """OpenVLA HF checkpoint key to ``PrismaticVLA`` key; None drops the LM head."""
    if key.startswith("language_model.lm_head."):
        return None
    if key.startswith("language_model.model."):
        return "llm." + key.removeprefix("language_model.model.")
    for prefix in ("vision_backbone.featurizer.", "vision_backbone.fused_featurizer."):
        if key.startswith(prefix):
            name = prefix.removeprefix("vision_backbone.") + "vit."
            key = name + key.removeprefix(prefix)
            # Prismatic renames timm LayerScale.gamma to scale_factor for HF loading.
            return key.replace(".scale_factor", ".gamma")
    if key.startswith("projector."):
        return key
    raise KeyError(f"unexpected OpenVLA checkpoint key {key!r}")


def load_openvla_weights(model: PrismaticVLA, snapshot: Path) -> None:
    """Strictly load OpenVLA-7B weights; only the unused LM head is dropped."""
    target = model.state_dict()
    seen = set()
    with torch.no_grad():
        for file in openvla_weight_files(snapshot):
            with safe_open(str(file), framework="pt") as handle:
                for key in handle.keys():
                    mapped = map_openvla_key(key)
                    if mapped is None:
                        continue
                    if mapped not in target:
                        raise KeyError(f"{key} maps to unknown parameter {mapped}")
                    tensor = handle.get_tensor(key)
                    if tensor.shape != target[mapped].shape:
                        raise ValueError(
                            f"{key}: shape {tuple(tensor.shape)} != {tuple(target[mapped].shape)}"
                        )
                    target[mapped].copy_(tensor)
                    seen.add(mapped)
    # Rotary inverse frequencies are buffers computed from the config.
    missing = [k for k in target if k not in seen and "rotary_emb" not in k]
    if missing:
        raise KeyError(f"OpenVLA weights missing for {len(missing)} keys, e.g. {missing[:5]}")
