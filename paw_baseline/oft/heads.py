"""OpenVLA-OFT L1 regression head and the ActionMap voxel heatmap head.

Both heads read the backbone's last-layer hidden states at the ``chunk * action_dim``
action-token positions, concatenate the ``action_dim`` states of each chunk step
and decode one action per step. The L1 head follows OpenVLA-OFT
``prismatic/models/action_heads.py``. The heatmap head follows ActionMap
``actionmap/heatmap_action_head.py`` (showlab/ActionMap @ 5872130); this robot has
two arms, so the shared trunk feeds one translation, rotation and gripper branch
per arm instead of one of each.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPResNetBlock(nn.Module):
    """One pre-LayerNorm MLP block with a residual connection."""

    def __init__(self, dim: int):
        super().__init__()
        self.ffn = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ffn(x)


class MLPResNet(nn.Module):
    """OFT ``MLPResNet``: LayerNorm, Linear, ReLU, residual blocks, LayerNorm, Linear."""

    def __init__(self, num_blocks: int, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList(
            MLPResNetBlock(hidden_dim) for _ in range(num_blocks)
        )
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(self.layer_norm1(x)))
        for block in self.mlp_resnet_blocks:
            x = block(x)
        return self.fc2(self.layer_norm2(x))


class L1RegressionActionHead(nn.Module):
    """OpenVLA-OFT continuous head trained with L1 loss on normalized actions."""

    def __init__(self, input_dim: int, hidden_dim: int, action_dim: int, chunk_size: int):
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.model = MLPResNet(
            num_blocks=2,
            input_dim=input_dim * action_dim,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
        )

    def predict_action(self, hidden: torch.Tensor) -> torch.Tensor:
        """``(B, T * D, H)`` action-token states to ``(B, T, D)`` normalized actions."""
        return self.model(hidden.reshape(hidden.shape[0], self.chunk_size, -1))

    def predict_action_with_loss(self, hidden: torch.Tensor, target: torch.Tensor):
        predicted = self.predict_action(hidden)
        loss = F.l1_loss(predicted.float(), target.float())
        return predicted, loss, {"l1_loss": loss.detach()}

    def decode(self, predicted: torch.Tensor, gripper_dims) -> torch.Tensor:
        """Binarize regressed gripper commands, as OFT does at inference."""
        decoded = predicted.float().clone()
        dims = list(gripper_dims)
        decoded[..., dims] = torch.where(decoded[..., dims] > 0, 1.0, -1.0)
        return decoded


def make_grid(shape, pad: float = 0.0) -> torch.Tensor:
    """Uniform voxel centers in ``[-(1 + pad), 1 + pad]^3``, shape ``(N, 3)``."""
    lo, hi = -1.0 - pad, 1.0 + pad
    axes = [torch.linspace(lo, hi, n) for n in shape]
    grid = torch.meshgrid(*axes, indexing="ij")
    return torch.stack(grid, dim=-1).reshape(-1, 3)


def gaussian_target(truth: torch.Tensor, centers: torch.Tensor, sigma: float) -> torch.Tensor:
    """Softmax-normalized Gaussian blob over voxel centers, ``(B, T, N)``."""
    diff = truth.unsqueeze(-2) - centers
    return torch.softmax(-0.5 * (diff * diff).sum(-1) / sigma**2, dim=-1)


def decode_logits(logits, centers, mode: str, top_k: int, temperature: float):
    """Recover ``(B, T, 3)`` coordinates from voxel logits (ActionMap decoders)."""
    logits = logits.float()
    if mode == "hard_argmax":
        return centers[logits.argmax(dim=-1)]
    if mode == "topk":
        values, index = logits.topk(min(top_k, logits.shape[-1]), dim=-1)
        weights = torch.softmax(values / temperature, dim=-1)
        return (weights.unsqueeze(-1) * centers[index]).sum(dim=-2)
    if mode == "soft_argmax":
        return torch.softmax(logits / temperature, dim=-1) @ centers
    raise ValueError(f"unknown heatmap decode mode {mode!r}")


class HeatmapActionHead(nn.Module):
    """ActionMap head: per-arm translation/rotation voxel heatmaps and a gripper logit.

    Per arm the 7 normalized dims are ``[dx, dy, dz, rx, ry, rz, grip]`` in ``[-1, 1]``.
    Training minimizes soft-label cross-entropy against Gaussian blobs (and BCE for
    the gripper); inference decodes by top-k soft-argmax.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        action_dim: int,
        chunk_size: int,
        num_arms: int = 2,
        trans_grid=(48, 48, 48),
        rot_grid=(16, 16, 16),
        trans_sigma: float = 0.1,
        rot_sigma: float = 0.1,
        grid_pad: float = 0.2,
        decode_mode: str = "topk",
        decode_top_k: int = 10,
        decode_temperature: float = 1.0,
    ):
        super().__init__()
        if action_dim != 7 * num_arms:
            raise ValueError("heatmap head expects 7 action dims per arm")
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.num_arms = num_arms
        self.trans_sigma, self.rot_sigma = trans_sigma, rot_sigma
        self.decode_mode, self.decode_top_k = decode_mode, decode_top_k
        self.decode_temperature = decode_temperature
        trans_bins, rot_bins = math.prod(trans_grid), math.prod(rot_grid)
        self.trunk = nn.Sequential(
            nn.LayerNorm(input_dim * action_dim),
            nn.Linear(input_dim * action_dim, hidden_dim),
            nn.ReLU(),
            MLPResNetBlock(hidden_dim),
            MLPResNetBlock(hidden_dim),
        )
        self.trans_heads = nn.ModuleList(
            nn.Sequential(nn.Linear(hidden_dim, 1024), nn.ReLU(), nn.Linear(1024, trans_bins))
            for _ in range(num_arms)
        )
        self.rot_heads = nn.ModuleList(
            nn.Sequential(nn.Linear(hidden_dim, 512), nn.ReLU(), nn.Linear(512, rot_bins))
            for _ in range(num_arms)
        )
        self.grip_heads = nn.ModuleList(nn.Linear(hidden_dim, 1) for _ in range(num_arms))
        # Translation is padded beyond the clipped [-1, 1] range; rotation is not.
        self.register_buffer("trans_centers", make_grid(trans_grid, pad=grid_pad))
        self.register_buffer("rot_centers", make_grid(rot_grid))

    def _trunk(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.trunk(hidden.reshape(hidden.shape[0], self.chunk_size, -1))

    def _arm_logits(self, h: torch.Tensor, arm: int):
        return (
            self.trans_heads[arm](h),
            self.rot_heads[arm](h),
            self.grip_heads[arm](h).squeeze(-1),
        )

    def _decode(self, trans_logits, rot_logits, grip_logits) -> torch.Tensor:
        centers_t = self.trans_centers.float()
        centers_r = self.rot_centers.float()
        args = (self.decode_mode, self.decode_top_k, self.decode_temperature)
        trans = decode_logits(trans_logits, centers_t, *args)
        rot = decode_logits(rot_logits, centers_r, *args)
        grip = torch.where(grip_logits.float() > 0, 1.0, -1.0)
        return torch.cat((trans, rot, grip.unsqueeze(-1)), dim=-1)

    def predict_action(self, hidden: torch.Tensor) -> torch.Tensor:
        """Decode ``(B, T, 7 * arms)`` normalized actions from action-token states."""
        h = self._trunk(hidden)
        return torch.cat(
            [self._decode(*self._arm_logits(h, arm)) for arm in range(self.num_arms)],
            dim=-1,
        )

    def predict_action_with_loss(self, hidden: torch.Tensor, target: torch.Tensor):
        """Sum of per-arm translation CE, rotation CE and gripper BCE (ActionMap Eq. 2)."""
        h = self._trunk(hidden)
        target = target.float()
        loss = h.new_zeros((), dtype=torch.float32)
        metrics, decoded = {}, []
        for arm in range(self.num_arms):
            truth = target[..., 7 * arm : 7 * arm + 7]
            trans_logits, rot_logits, grip_logits = self._arm_logits(h, arm)
            with torch.no_grad():
                trans_target = gaussian_target(
                    truth[..., :3], self.trans_centers.float(), self.trans_sigma
                )
                rot_target = gaussian_target(
                    truth[..., 3:6], self.rot_centers.float(), self.rot_sigma
                )
            loss_trans = -(trans_target * F.log_softmax(trans_logits.float(), dim=-1)).sum(-1).mean()
            loss_rot = -(rot_target * F.log_softmax(rot_logits.float(), dim=-1)).sum(-1).mean()
            loss_grip = F.binary_cross_entropy_with_logits(
                grip_logits.float(), (truth[..., 6] > 0).float()
            )
            loss = loss + loss_trans + loss_rot + loss_grip
            metrics[f"arm{arm}_trans_ce"] = loss_trans.detach()
            metrics[f"arm{arm}_rot_ce"] = loss_rot.detach()
            metrics[f"arm{arm}_grip_bce"] = loss_grip.detach()
            with torch.no_grad():
                decoded.append(self._decode(trans_logits, rot_logits, grip_logits))
        return torch.cat(decoded, dim=-1), loss, metrics

    def decode(self, predicted: torch.Tensor, gripper_dims) -> torch.Tensor:
        return predicted.float()
