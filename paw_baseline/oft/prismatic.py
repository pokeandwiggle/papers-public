"""OpenVLA-7B (Prismatic) backbone for parallel decoding, without remote code.

Mirrors ``modeling_prismatic.py`` of OpenVLA-OFT: fused DINOv2 + SigLIP patch
features from each tower's second-to-last block, a three-layer GELU projector and
Llama-2-7B. OpenVLA-OFT patches Llama attention to be fully bidirectional
(``moojink/transformers-openvla-oft``); here every decoder layer runs with
``is_causal=False`` and, only when prompts differ in length, a key-padding mask.
"""

import json
from pathlib import Path

import timm
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import LlamaConfig, LlamaModel
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

DINO_ID = "vit_large_patch14_reg4_dinov2.lvd142m"
SIGLIP_ID = "vit_so400m_patch14_siglip_224"
IMAGE_SIZE = 224
# OpenVLA preprocessor_config.json tvf_normalize_params (DINOv2 first, SigLIP second).
DINO_MEAN = (0.484375, 0.455078125, 0.40625)
DINO_STD = (0.228515625, 0.2236328125, 0.224609375)
SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)
BOS_ID, STOP_ID, EMPTY_ID = 1, 2, 29871


def llama_config() -> LlamaConfig:
    """OpenVLA's ``text_config``: LlamaConfig defaults with the padded 32064 vocabulary."""
    config = LlamaConfig(
        vocab_size=32064,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        hidden_act="silu",
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_bias=False,
        tie_word_embeddings=False,
        pad_token_id=32000,
        bos_token_id=BOS_ID,
        eos_token_id=STOP_ID,
    )
    config._attn_implementation = "sdpa"
    return config


class FusedVisionBackbone(nn.Module):
    """DINOv2 (featurizer) and SigLIP (fused_featurizer) on channel-stacked inputs."""

    def __init__(self):
        super().__init__()
        # OpenVLA passes act_layer=None (timm_override_act_layers), keeping timm's default.
        self.featurizer = timm.create_model(
            DINO_ID, pretrained=False, num_classes=0, img_size=IMAGE_SIZE, act_layer=None
        )
        self.fused_featurizer = timm.create_model(
            SIGLIP_ID, pretrained=False, num_classes=0, img_size=IMAGE_SIZE, act_layer=None
        )
        self.embed_dim = self.featurizer.embed_dim + self.fused_featurizer.embed_dim

    @staticmethod
    def _second_to_last(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        index = len(model.blocks) - 2
        return model.forward_intermediates(
            x, indices=[index], norm=False, stop_early=True,
            output_fmt="NLC", intermediates_only=True,
        )[0]

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """``(N, 6, 224, 224)`` to ``(N, 256, 2176)`` patch features."""
        dino, siglip = torch.split(pixel_values, [3, 3], dim=1)
        return torch.cat(
            (
                self._second_to_last(self.featurizer, dino),
                self._second_to_last(self.fused_featurizer, siglip),
            ),
            dim=2,
        )


class Projector(nn.Module):
    """Prismatic fused-backbone projector (``fused-gelu-mlp``)."""

    def __init__(self, vision_dim: int, llm_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(vision_dim, 4 * vision_dim)
        self.fc2 = nn.Linear(4 * vision_dim, llm_dim)
        self.fc3 = nn.Linear(llm_dim, llm_dim)
        self.act_fn1 = nn.GELU()
        self.act_fn2 = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc3(self.act_fn2(self.fc2(self.act_fn1(self.fc1(x)))))


class ProprioProjector(nn.Module):
    """OFT proprio projector: one appended token after the vision patches."""

    def __init__(self, llm_dim: int, proprio_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(proprio_dim, llm_dim)
        self.fc2 = nn.Linear(llm_dim, llm_dim)
        self.act_fn1 = nn.GELU()

    def forward(self, proprio: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act_fn1(self.fc1(proprio)))


class OpenVLABackbone(nn.Module):
    """Vision towers, projector and Llama decoder (no LM head; only hidden states are used)."""

    def __init__(self, vision_backbone: nn.Module | None = None, config: LlamaConfig | None = None):
        super().__init__()
        config = config or llama_config()
        self.vision_backbone = vision_backbone or FusedVisionBackbone()
        self.projector = Projector(self.vision_backbone.embed_dim, config.hidden_size)
        self.language_model = LlamaModel(config)
        self.llm_dim = config.hidden_size

    def reset_rotary(self, device) -> None:
        """Rebuild non-persistent RoPE buffers after meta-device construction."""
        self.language_model.rotary_emb = LlamaRotaryEmbedding(
            self.language_model.config, device=device
        )

    def enable_gradient_checkpointing(self) -> None:
        self.language_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        self.vision_backbone.featurizer.set_grad_checkpointing(True)
        self.vision_backbone.fused_featurizer.set_grad_checkpointing(True)

    def patch_embeddings(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """``(B, cams, 6, H, W)`` to ``(B, cams * 256, llm_dim)``; cameras in input order."""
        batch, cams = pixel_values.shape[:2]
        features = self.vision_backbone(pixel_values.flatten(0, 1))
        features = features.reshape(batch, cams * features.shape[1], features.shape[2])
        return self.projector(features)

    def forward(
        self,
        patches: torch.Tensor,
        prompts: list[torch.Tensor],
        num_action_tokens: int,
    ) -> torch.Tensor:
        """Return last-layer states at the OFT action-token readout positions.

        Sequence per sample (OFT): ``<s>``, vision patches and proprio token, prompt
        tokens after ``<s>`` (ending in the empty token 29871), ``num_action_tokens``
        zeroed action embeddings, ``</s>``. As in OFT, the readout for action ``j`` is
        the state at the position preceding it (labels are shifted by one).
        """
        embed = self.language_model.embed_tokens
        stop = embed(torch.tensor([STOP_ID], device=patches.device))
        sequences, starts = [], []
        for i, ids in enumerate(prompts):
            if int(ids[0]) != BOS_ID or int(ids[-1]) != EMPTY_ID:
                raise ValueError("prompt must start with <s> and end with token 29871")
            text = embed(ids.to(patches.device)).to(patches.dtype)
            actions = patches.new_zeros(num_action_tokens, patches.shape[-1])
            sequence = torch.cat((text[:1], patches[i], text[1:], actions, stop.to(patches.dtype)))
            starts.append(1 + patches.shape[1] + len(ids) - 1)
            sequences.append(sequence)
        lengths = [len(s) for s in sequences]
        width = max(lengths)
        hidden = torch.stack(
            [torch.cat((s, s.new_zeros(width - len(s), s.shape[-1]))) for s in sequences]
        )
        mask = None
        if len(set(lengths)) > 1:
            valid = torch.arange(width, device=hidden.device)[None] < torch.tensor(
                lengths, device=hidden.device
            )[:, None]
            mask = valid[:, None, None, :].expand(-1, 1, width, -1)
        position_ids = torch.arange(width, device=hidden.device)[None]
        position_embeddings = self.language_model.rotary_emb(hidden, position_ids=position_ids)
        for layer in self.language_model.layers:
            hidden = layer(
                hidden,
                attention_mask=mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                is_causal=False,
            )
        hidden = self.language_model.norm(hidden)
        return torch.stack(
            [hidden[i, start - 1 : start - 1 + num_action_tokens] for i, start in enumerate(starts)]
        )


def checkpoint_files(repo: str, revision: str) -> list[Path]:
    """Download the pinned OpenVLA safetensors shards."""
    index = Path(hf_hub_download(repo, "model.safetensors.index.json", revision=revision))
    names = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    return [Path(hf_hub_download(repo, name, revision=revision)) for name in names]


def openvla_state_dict(
    files: list[Path], dtype: torch.dtype, device="cpu"
) -> dict[str, torch.Tensor]:
    """Map OpenVLA HF keys to this module; drop the unused LM head."""
    state = {}
    for path in files:
        with safe_open(path, framework="pt", device=str(device)) as handle:
            for key in handle.keys():
                if key.startswith("language_model.lm_head."):
                    continue
                name = key.replace("language_model.model.", "language_model.", 1)
                # OpenVLA renames timm LayerScale `gamma` to `scale_factor` for HF.
                if name.endswith(".scale_factor"):
                    name = name[: -len("scale_factor")] + "gamma"
                state[name] = handle.get_tensor(key).to(dtype)
    return state


def load_openvla_backbone(repo: str, revision: str, device, dtype=torch.bfloat16) -> OpenVLABackbone:
    """Build the backbone on the meta device and assign pinned OpenVLA weights strictly."""
    with torch.device("meta"):
        backbone = OpenVLABackbone()
    state = openvla_state_dict(checkpoint_files(repo, revision), dtype, device)
    backbone.load_state_dict(state, strict=True, assign=True)
    backbone.reset_rotary(device)
    return backbone
