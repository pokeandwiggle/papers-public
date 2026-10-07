"""Frozen DINOv2 ViT-S/14 features matching the reference torch.hub encoder.

The reference loads ``dinov2_vits14`` from ``facebookresearch/dinov2:b48308a``
and reads ``x_norm_patchtokens`` or ``x_norm_clstoken`` after ImageNet
normalization of 224x224 inputs. This module uses the identical pretrained
weights from the pinned Hugging Face ``facebook/dinov2-small`` revision, and
reproduces torch.hub's positional-embedding interpolation (scale factor with
a 0.1 offset, bicubic, no antialiasing) so features match the reference.
"""

import math
import types

import torch
import torch.nn.functional as F
from torch import nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# config.json of facebook/dinov2-small at the pinned revision; building from it
# lets a saved policy restore encoder weights from its own checkpoint offline.
DINOV2_SMALL_CONFIG = {
    "architectures": ["Dinov2Model"],
    "attention_probs_dropout_prob": 0.0,
    "drop_path_rate": 0.0,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.0,
    "hidden_size": 384,
    "image_size": 518,
    "initializer_range": 0.02,
    "layer_norm_eps": 1e-06,
    "layerscale_value": 1.0,
    "mlp_ratio": 4,
    "model_type": "dinov2",
    "num_attention_heads": 6,
    "num_channels": 3,
    "num_hidden_layers": 12,
    "patch_size": 14,
    "qkv_bias": True,
    "use_swiglu_ffn": False,
}
HUB_INTERPOLATE_OFFSET = 0.1


def hub_interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int):
    """torch.hub DINOv2 ``interpolate_pos_encoding`` on HF parameters.

    hub calls it with ``(w, h) = x.shape[-2:]``; for the square inputs used here
    the order is immaterial.
    """
    pos_embed = self.position_embeddings
    npatch = embeddings.shape[1] - 1
    n = pos_embed.shape[1] - 1
    if npatch == n and height == width:
        return pos_embed
    previous_dtype = embeddings.dtype
    pos_embed = pos_embed.float()
    class_pos_embed = pos_embed[:, 0]
    patch_pos_embed = pos_embed[:, 1:]
    dim = embeddings.shape[-1]
    w0, h0 = height // self.patch_size, width // self.patch_size
    m = int(math.sqrt(n))
    assert n == m * m
    sx = float(w0 + HUB_INTERPOLATE_OFFSET) / m
    sy = float(h0 + HUB_INTERPOLATE_OFFSET) / m
    patch_pos_embed = F.interpolate(
        patch_pos_embed.reshape(1, m, m, dim).permute(0, 3, 1, 2),
        scale_factor=(sx, sy),
        mode="bicubic",
        antialias=False,
    )
    assert (w0, h0) == patch_pos_embed.shape[-2:]
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim)
    return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)


class FrozenDinoV2(nn.Module):
    """(B, 3, H, W) images in [0, 1] -> (B, P, 384) patch tokens or (B, 1, 384) CLS."""

    def __init__(
        self,
        feature: str,
        image_size: int,
        *,
        pretrained: tuple[str, str] | None,
    ):
        super().__init__()
        from transformers import Dinov2Config, Dinov2Model

        if pretrained is None:
            self.model = Dinov2Model(Dinov2Config(**DINOV2_SMALL_CONFIG))
        else:
            repo, revision = pretrained
            self.model = Dinov2Model.from_pretrained(
                repo, revision=revision, dtype=torch.float32
            )
            expected = Dinov2Config(**DINOV2_SMALL_CONFIG).to_dict()
            loaded = self.model.config.to_dict()
            for key in DINOV2_SMALL_CONFIG:
                if key != "architectures" and loaded.get(key) != expected.get(key):
                    raise ValueError(f"pinned DINOv2 config mismatch for {key!r}")
        self.model.embeddings.interpolate_pos_encoding = types.MethodType(
            hub_interpolate_pos_encoding, self.model.embeddings
        )
        self.model.requires_grad_(False)
        self.model.eval()
        self.feature = feature
        self.image_size = image_size
        self.register_buffer(
            "mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.eval()  # frozen: never enable dropout/drop-path
        return self

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        if images.shape[-2:] != (self.image_size, self.image_size):
            images = F.interpolate(
                images,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        return (images - self.mean) / self.std

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        hidden = self.model(pixel_values=self.preprocess(images)).last_hidden_state
        if self.feature == "patch":
            return hidden[:, 1:]
        return hidden[:, :1]
