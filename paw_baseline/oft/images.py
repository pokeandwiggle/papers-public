"""OpenVLA-OFT image pipeline on GPU tensors.

Training (OFT ``image_aug``): resize to 224, random square crop of 90% area resized
back to 224, then brightness (+-0.2), contrast [0.8, 1.2], saturation [0.8, 1.2]
and hue (+-0.05). Inference: the matching 90%-area center crop. Each image is then
normalized twice and channel-stacked for the DINOv2 and SigLIP towers.
"""

import math

import torch
import torch.nn.functional as F
from torchvision.transforms.v2 import functional as TF

from paw_baseline.oft.prismatic import DINO_MEAN, DINO_STD, IMAGE_SIZE, SIGLIP_MEAN, SIGLIP_STD


def crop_and_resize(images: torch.Tensor, centers: torch.Tensor, side: float) -> torch.Tensor:
    """Bilinear crop of square boxes (side fraction, normalized centers) to the input size.

    ``align_corners=True`` matches ``tf.image.crop_and_resize`` sampling.
    """
    count = images.shape[0]
    theta = images.new_zeros(count, 2, 3)
    theta[:, 0, 0] = side
    theta[:, 1, 1] = side
    theta[:, :, 2] = centers
    grid = F.affine_grid(theta, list(images.shape), align_corners=True)
    return F.grid_sample(images, grid, mode="bilinear", padding_mode="border", align_corners=True)


def _color_jitter(images: torch.Tensor) -> torch.Tensor:
    count = images.shape[0]
    shape = (count, 1, 1, 1)
    brightness = torch.empty(shape, device=images.device).uniform_(-0.2, 0.2)
    images = (images + brightness).clamp(0, 1)
    contrast = torch.empty(shape, device=images.device).uniform_(0.8, 1.2)
    mean = images.mean(dim=(-2, -1), keepdim=True)
    images = ((images - mean) * contrast + mean).clamp(0, 1)
    saturation = torch.empty(count, device=images.device).uniform_(0.8, 1.2)
    hue = torch.empty(count, device=images.device).uniform_(-0.05, 0.05)
    return torch.stack(
        [
            TF.adjust_hue(TF.adjust_saturation(image, float(s)).clamp(0, 1), float(h))
            for image, s, h in zip(images, saturation, hue)
        ]
    ).clamp(0, 1)


def preprocess_images(images: torch.Tensor, *, train: bool, crop_scale: float) -> torch.Tensor:
    """``(B, cams, 3, H, W)`` RGB in [0, 1] to ``(B, cams, 6, 224, 224)`` model pixels."""
    batch, cams = images.shape[:2]
    x = images.flatten(0, 1).float()
    if x.shape[-2:] != (IMAGE_SIZE, IMAGE_SIZE):
        x = F.interpolate(
            x, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bicubic", antialias=True, align_corners=False
        ).clamp(0, 1)
    side = math.sqrt(crop_scale)
    if train:
        centers = (torch.rand(x.shape[0], 2, device=x.device) * 2 - 1) * (1 - side)
    else:
        centers = x.new_zeros(x.shape[0], 2)
    x = crop_and_resize(x, centers, side)
    if train:
        x = _color_jitter(x)
    dino = TF.normalize(x, list(DINO_MEAN), list(DINO_STD))
    siglip = TF.normalize(x, list(SIGLIP_MEAN), list(SIGLIP_STD))
    stacked = torch.cat((dino, siglip), dim=1)
    return stacked.reshape(batch, cams, 6, IMAGE_SIZE, IMAGE_SIZE)
