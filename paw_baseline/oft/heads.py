"""OpenVLA-OFT's L1 regression head and the ActionMap voxel heatmap head.

Both read the backbone's last hidden states at the chunk's action-token
positions, shaped (B, T * A, D), and regroup them per action slot as
(B, T, A * D). The L1 head follows openvla-oft ``L1RegressionActionHead``; the
heatmap head follows showlab/ActionMap ``actionmap/heatmap_action_head.py`` and
gives each arm its own translation, rotation and gripper branches on the shared
trunk. Actions are normalized to [-1, 1]; grippers are binary {0, 1} targets.
"""

import torch
import torch.nn as nn

from paw_baseline.oft.actions import ARM_DELTA_DIM


class MLPResNetBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.ffn = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.ReLU())

    def forward(self, x):
        return x + self.ffn(x)


class MLPResNet(nn.Module):
    def __init__(self, num_blocks: int, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList(
            [MLPResNetBlock(hidden_dim) for _ in range(num_blocks)]
        )
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.relu(self.fc1(self.layer_norm1(x)))
        for block in self.mlp_resnet_blocks:
            x = block(x)
        return self.fc2(self.layer_norm2(x))


class L1RegressionActionHead(nn.Module):
    """OpenVLA-OFT continuous head trained with an L1 loss on normalized actions."""

    def __init__(self, input_dim: int, hidden_dim: int, action_dim: int, chunk_size: int):
        super().__init__()
        self.chunk_size = chunk_size
        self.model = MLPResNet(
            num_blocks=2,
            input_dim=input_dim * action_dim,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
        )

    def predict_action(self, actions_hidden_states: torch.Tensor) -> torch.Tensor:
        batch = actions_hidden_states.shape[0]
        return self.model(actions_hidden_states.reshape(batch, self.chunk_size, -1))

    def predict_action_with_loss(self, actions_hidden_states, ground_truth_actions):
        predicted = self.predict_action(actions_hidden_states).float()
        return predicted, nn.functional.l1_loss(predicted, ground_truth_actions.float())


def make_grid(grid_shape, pad: float) -> torch.Tensor:
    """Uniform voxel centers over [-(1 + pad), 1 + pad]^3 in 'ij' order, shape (N, 3)."""
    axes = [torch.linspace(-1.0 - pad, 1.0 + pad, n) for n in grid_shape]
    mesh = torch.meshgrid(*axes, indexing="ij")
    return torch.stack(mesh, dim=-1).reshape(-1, 3)


class HeatmapActionHead(nn.Module):
    """ActionMap: per-slot translation/rotation voxel heatmaps and a binary gripper."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        action_dim: int,
        chunk_size: int,
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
        if action_dim % ARM_DELTA_DIM:
            raise ValueError("heatmap head expects 7 action dims per arm")
        if decode_mode not in ("topk", "soft_argmax", "hard_argmax", "mean"):
            raise ValueError(f"unknown decode mode {decode_mode!r}")
        self.chunk_size = chunk_size
        self.num_arms = action_dim // ARM_DELTA_DIM
        self.trans_grid = tuple(trans_grid)
        self.rot_grid = tuple(rot_grid)
        self.trans_sigma = trans_sigma
        self.rot_sigma = rot_sigma
        self.grid_pad = grid_pad
        self.decode_mode = decode_mode
        self.decode_top_k = decode_top_k
        self.decode_temperature = decode_temperature
        trans_bins = trans_grid[0] * trans_grid[1] * trans_grid[2]
        rot_bins = rot_grid[0] * rot_grid[1] * rot_grid[2]

        self.trunk = nn.Sequential(
            nn.LayerNorm(input_dim * action_dim),
            nn.Linear(input_dim * action_dim, hidden_dim),
            nn.ReLU(),
            MLPResNetBlock(hidden_dim),
            MLPResNetBlock(hidden_dim),
        )
        self.trans_heads = nn.ModuleList(
            nn.Sequential(nn.Linear(hidden_dim, 1024), nn.ReLU(), nn.Linear(1024, trans_bins))
            for _ in range(self.num_arms)
        )
        self.rot_heads = nn.ModuleList(
            nn.Sequential(nn.Linear(hidden_dim, 512), nn.ReLU(), nn.Linear(512, rot_bins))
            for _ in range(self.num_arms)
        )
        self.grip_heads = nn.ModuleList(nn.Linear(hidden_dim, 1) for _ in range(self.num_arms))
        self.register_buffer("trans_centers", make_grid(trans_grid, grid_pad), persistent=False)
        self.register_buffer("rot_centers", make_grid(rot_grid, grid_pad), persistent=False)

    @staticmethod
    def gaussian_log_target(true_xyz, grid_shape, pad: float, sigma: float) -> torch.Tensor:
        """log of the softmax-normalized Gaussian blob centered on ``true_xyz`` (B, T, N).

        Equal to log softmax(-|b - a|^2 / 2 sigma^2) over the 'ij'-flattened grid;
        the squared distance is separable, which avoids an (N, 3) difference tensor.
        """
        axes = [
            torch.linspace(-1.0 - pad, 1.0 + pad, n, device=true_xyz.device)
            for n in grid_shape
        ]
        sq = [(true_xyz[..., i, None] - axes[i]) ** 2 for i in range(3)]
        dist = (
            sq[0][..., :, None, None] + sq[1][..., None, :, None] + sq[2][..., None, None, :]
        )
        logits = -0.5 * dist.flatten(-3) / sigma**2
        return torch.log_softmax(logits, dim=-1)

    @staticmethod
    def soft_label_ce(logits: torch.Tensor, log_target: torch.Tensor) -> torch.Tensor:
        return -(log_target.exp() * torch.log_softmax(logits, dim=-1)).sum(-1).mean()

    def _decode(self, logits: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        if self.decode_mode == "hard_argmax":
            return centers[logits.argmax(dim=-1)]
        if self.decode_mode == "soft_argmax":
            return torch.softmax(logits / self.decode_temperature, dim=-1) @ centers
        k = min(self.decode_top_k, logits.shape[-1])
        top_logits, top_idx = logits.topk(k, dim=-1)
        top_centers = centers[top_idx]
        if self.decode_mode == "mean":
            return top_centers.mean(dim=-2)
        weights = torch.softmax(top_logits / self.decode_temperature, dim=-1)
        return (weights.unsqueeze(-1) * top_centers).sum(dim=-2)

    def _trunk(self, actions_hidden_states):
        batch = actions_hidden_states.shape[0]
        return self.trunk(actions_hidden_states.reshape(batch, self.chunk_size, -1))

    def predict_action(self, actions_hidden_states: torch.Tensor) -> torch.Tensor:
        h = self._trunk(actions_hidden_states)
        parts = []
        for arm in range(self.num_arms):
            trans = self._decode(self.trans_heads[arm](h).float(), self.trans_centers)
            rot = self._decode(self.rot_heads[arm](h).float(), self.rot_centers)
            grip = (self.grip_heads[arm](h).float() > 0).float()
            parts += [trans, rot, grip]
        return torch.cat(parts, dim=-1)

    def predict_action_with_loss(self, actions_hidden_states, ground_truth_actions):
        h = self._trunk(actions_hidden_states)
        gt = ground_truth_actions.float()
        loss = 0.0
        parts = []
        for arm in range(self.num_arms):
            g = gt[..., arm * ARM_DELTA_DIM : (arm + 1) * ARM_DELTA_DIM]
            trans_logits = self.trans_heads[arm](h).float()
            rot_logits = self.rot_heads[arm](h).float()
            grip_logits = self.grip_heads[arm](h).float().squeeze(-1)
            trans_target = self.gaussian_log_target(
                g[..., 0:3], self.trans_grid, self.grid_pad, self.trans_sigma
            )
            rot_target = self.gaussian_log_target(
                g[..., 3:6], self.rot_grid, self.grid_pad, self.rot_sigma
            )
            loss = (
                loss
                + self.soft_label_ce(trans_logits, trans_target)
                + self.soft_label_ce(rot_logits, rot_target)
                + nn.functional.binary_cross_entropy_with_logits(
                    grip_logits, (g[..., 6] > 0.5).float()
                )
            )
            with torch.no_grad():
                parts += [
                    torch.softmax(trans_logits, -1) @ self.trans_centers,
                    torch.softmax(rot_logits, -1) @ self.rot_centers,
                    (grip_logits > 0).float().unsqueeze(-1),
                ]
        return torch.cat(parts, dim=-1), loss
