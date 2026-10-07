"""Residual VQ-VAE action tokenizer from the Patch Policy reference (VQ-BeT).

Adapted from ``models/vq_behavior_transformer/vqvae.py`` and ``bet.py``
(https://github.com/gaoyuezhou/patch_policy, MIT). The encoder/decoder width
and depth are configurable so tokenizer capacity can follow the action chunk
size; with ``hidden_dim=128`` and ``hidden_layers=1`` this is the reference.
"""

import einops
import torch
import torch.nn.functional as F
from torch import nn

from train.patch_policy.vector_quantize_pytorch import ResidualVQ


def weights_init_encoder(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)


class EncoderMLP(nn.Module):
    def __init__(self, input_dim, output_dim=16, hidden_dim=128, layer_num=1):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(layer_num):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        self.encoder = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.apply(weights_init_encoder)

    def forward(self, x):
        return self.fc(self.encoder(x))


class VqVae(nn.Module):
    """Encode a (chunk, action_dim) window to G residual codes from C-entry books."""

    def __init__(
        self,
        chunk_size: int,
        action_dim: int,
        latent_dim: int = 512,
        n_embed: int = 16,
        groups: int = 2,
        hidden_dim: int = 128,
        hidden_layers: int = 1,
        encoder_loss_multiplier: float = 1.0,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.n_embed = n_embed
        self.groups = groups
        self.encoder_loss_multiplier = encoder_loss_multiplier
        # sync_codebook=False: the tokenizer is fit by one process before any
        # distributed wrapping; DDP then broadcasts rank zero's codebooks.
        self.vq_layer = ResidualVQ(
            dim=latent_dim,
            num_quantizers=groups,
            codebook_size=n_embed,
            sync_codebook=False,
        )
        self.encoder = EncoderMLP(
            chunk_size * action_dim, latent_dim, hidden_dim, hidden_layers
        )
        self.decoder = EncoderMLP(
            latent_dim, chunk_size * action_dim, hidden_dim, hidden_layers
        )

    def preprocess(self, actions: torch.Tensor) -> torch.Tensor:
        return einops.rearrange(actions, "... T A -> ... (T A)")

    def draw_code_forward(self, indices: torch.Tensor) -> torch.Tensor:
        """Sum of the selected residual code vectors; indices (..., G)."""
        with torch.no_grad():
            shape = indices.shape[:-1]
            flat = indices.reshape(-1, self.groups)
            z = self.vq_layer.get_codes_from_indices(flat).sum(dim=0)
        return z.reshape(*shape, self.latent_dim)

    def get_action_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        out = self.decoder(latent)
        return einops.rearrange(out, "... (T A) -> ... T A", A=self.action_dim)

    def get_code(self, actions: torch.Tensor):
        """Quantize (..., T, A) chunks; returns (quantized latent, codes (..., G))."""
        shape = actions.shape[:-2]
        state = self.preprocess(actions).reshape(-1, 1, self.chunk_size * self.action_dim)
        with torch.no_grad():
            rep = self.encoder(state)
            quantized, codes, _ = self.vq_layer(rep)
        return (
            quantized.reshape(*shape, self.latent_dim),
            codes.reshape(*shape, self.groups),
        )

    def forward(self, actions: torch.Tensor):
        """Training loss for (N, T, A) chunks, as in the reference VqVae.forward."""
        state = self.preprocess(actions)
        rep = self.encoder(state)
        rep_flat = rep.view(rep.size(0), -1, rep.size(1))
        rep_flat, codes, vq_loss = self.vq_layer(rep_flat)
        quantized = rep_flat.view(rep.shape[0], -1)
        codes = codes.view(rep.shape[0], -1)
        decoded = self.decoder(quantized)
        vq_loss = vq_loss.sum()
        encoder_loss = (state - decoded).abs().mean()
        loss = encoder_loss * self.encoder_loss_multiplier + vq_loss * 5
        return loss, codes, {
            "vq_loss": vq_loss.detach(),
            "recon_l1": encoder_loss.detach(),
            "recon_mse": F.mse_loss(state, decoded).detach(),
        }


def fit_vqvae(
    vqvae: VqVae,
    chunks: torch.Tensor,
    *,
    steps: int,
    batch_size: int,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 0,
    log_every: int = 0,
    log=print,
) -> dict:
    """Fit on unique (N, T, A) chunks by reshuffled epochs, like the reference.

    The reference iterates a fixed number of epochs; this uses a fixed number
    of optimizer steps so tasks of different sizes get the same tokenizer
    budget. Returns final code usage and losses.
    """
    device = next(vqvae.parameters()).device
    flat = torch.unique(chunks.reshape(len(chunks), -1), dim=0)
    data = flat.reshape(-1, vqvae.chunk_size, vqvae.action_dim).to(device)
    optimizer = torch.optim.Adam(vqvae.parameters(), lr=lr, weight_decay=weight_decay)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    vqvae.train()
    step = 0
    loss_dict = {}
    while step < steps:
        order = torch.randperm(len(data), generator=generator).to(device)
        for start in range(0, len(data), batch_size):
            batch = data[order[start : start + batch_size]]
            loss, _, loss_dict = vqvae(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            if log_every and step % log_every == 0:
                log(
                    f"vqvae step {step}/{steps}: "
                    + ", ".join(f"{k}={float(v):.5f}" for k, v in loss_dict.items())
                )
            if step >= steps:
                break
    vqvae.zero_grad(set_to_none=True)
    vqvae.eval()
    with torch.no_grad():
        codes = torch.cat(
            [vqvae.get_code(data[i : i + 8192])[1] for i in range(0, len(data), 8192)]
        )
    return {
        "unique_chunks": len(data),
        "codes_used_per_group": [
            int(torch.unique(codes[:, g]).numel()) for g in range(vqvae.groups)
        ],
        "code_combinations": int(torch.unique(codes, dim=0).shape[0]),
        **{k: float(v) for k, v in loss_dict.items()},
    }
