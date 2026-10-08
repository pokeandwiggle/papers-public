"""DiT-X transformer, ported from the official ManiFlow reference.

Source: geyan21/ManiFlow_Policy@ef2f116, maniflow/model/diffusion/ditx.py and
ditx_block.py. Language conditioning and the unused pre-norm option are
omitted; timm's Mlp/RmsNorm are replaced by equivalent torch modules.
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class Mlp(nn.Module):
    """timm.layers.Mlp with tanh-approximate GELU and no dropout."""

    def __init__(self, in_features: int, hidden_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool, qk_norm: bool):
        super().__init__()
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        batch, n, dim = x.shape
        length = c.shape[1]
        q = self.q(x).reshape(batch, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv(c).reshape(batch, length, 2, self.num_heads, self.head_dim)
        k, v = kv.permute(2, 0, 3, 1, 4).unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.permute(0, 2, 1, 3).reshape(batch, n, dim))


class DiTXBlock(nn.Module):
    """Self-attention, AdaLN-modulated cross-attention and MLP, each gated."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        p_drop_attn: float,
        qkv_bias: bool,
        qk_norm: bool,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True, dropout=p_drop_attn
        )
        self.cross_attn = CrossAttention(hidden_size, num_heads, qkv_bias, qk_norm)
        self.mlp = Mlp(hidden_size, int(hidden_size * mlp_ratio), hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 9 * hidden_size, bias=True)
        )

    def forward(self, x: Tensor, time_c: Tensor, context_c: Tensor) -> Tensor:
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_cross,
            scale_cross,
            gate_cross,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(time_c).chunk(9, dim=-1)
        normed = modulate(self.norm1(x), shift_msa, scale_msa)
        attn, _ = self.self_attn(normed, normed, normed, need_weights=False)
        x = x + gate_msa.unsqueeze(1) * attn
        normed = modulate(self.norm2(x), shift_cross, scale_cross)
        x = x + gate_cross.unsqueeze(1) * self.cross_attn(normed, context_c)
        normed = modulate(self.norm3(x), shift_mlp, scale_mlp)
        return x + gate_mlp.unsqueeze(1) * self.mlp(normed)


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.RMSNorm(hidden_size, eps=1e-6)
        self.ffn_final = Mlp(hidden_size, hidden_size, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.ffn_final(self.norm_final(x))


class DiTX(nn.Module):
    """Predict velocity v(x_t, t, Δt | observation tokens) for a whole action chunk."""

    def __init__(
        self,
        action_dim: int,
        horizon: int,
        cond_dim: int,
        cond_len: int,
        n_layer: int,
        n_head: int,
        n_emb: int,
        mlp_ratio: float,
        p_drop_attn: float,
        qkv_bias: bool,
        qk_norm: bool,
        timestep_embed_dim: int,
        target_t_embed_dim: int,
    ):
        super().__init__()
        self.horizon = horizon
        self.input_emb = nn.Linear(action_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, horizon, n_emb))
        self.vis_cond_obs_emb = nn.Linear(cond_dim, n_emb)
        self.vis_cond_pos_embed = nn.Parameter(torch.zeros(1, cond_len, n_emb))
        self.flow_timestep_encoder = nn.Sequential(
            SinusoidalPosEmb(timestep_embed_dim),
            nn.Linear(timestep_embed_dim, timestep_embed_dim * 4),
            nn.Mish(),
            nn.Linear(timestep_embed_dim * 4, n_emb),
        )
        self.flow_target_t_encoder = nn.Sequential(
            SinusoidalPosEmb(target_t_embed_dim),
            nn.Linear(target_t_embed_dim, target_t_embed_dim * 4),
            nn.Mish(),
            nn.Linear(target_t_embed_dim * 4, n_emb),
        )
        self.timestep_target_t_adaptor = nn.Linear(n_emb * 2, n_emb)
        self.blocks = nn.ModuleList(
            [
                DiTXBlock(n_emb, n_head, mlp_ratio, p_drop_attn, qkv_bias, qk_norm)
                for _ in range(n_layer)
            ]
        )
        self.final_layer = FinalLayer(n_emb, action_dim)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        """Replicate the reference initialization order exactly."""
        for block in self.blocks:
            nn.init.xavier_uniform_(block.self_attn.in_proj_weight)
            if block.self_attn.in_proj_bias is not None:
                nn.init.zeros_(block.self_attn.in_proj_bias)
            nn.init.xavier_uniform_(block.self_attn.out_proj.weight)
            if block.self_attn.out_proj.bias is not None:
                nn.init.zeros_(block.self_attn.out_proj.bias)

        def basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(basic_init)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.normal_(self.input_emb.weight, std=0.02)
        nn.init.constant_(self.input_emb.bias, 0)
        nn.init.normal_(self.pos_emb, std=0.02)
        for encoder in (self.flow_timestep_encoder, self.flow_target_t_encoder):
            for layer in encoder:
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, std=0.02)
                    nn.init.constant_(layer.bias, 0)
        nn.init.normal_(self.vis_cond_obs_emb.weight, std=0.02)
        nn.init.constant_(self.vis_cond_obs_emb.bias, 0)
        nn.init.normal_(self.timestep_target_t_adaptor.weight, std=0.02)
        nn.init.constant_(self.timestep_target_t_adaptor.bias, 0)
        nn.init.constant_(self.final_layer.ffn_final.fc2.weight, 0)
        nn.init.constant_(self.final_layer.ffn_final.fc2.bias, 0)

    def forward(
        self, sample: Tensor, timestep: Tensor, target_t: Tensor, vis_cond: Tensor
    ) -> Tensor:
        """sample (B, T, A); timestep and target_t (B,); vis_cond (B, L, cond_dim)."""
        batch = sample.shape[0]
        x = self.input_emb(sample) + self.pos_emb
        timestep = torch.as_tensor(timestep, dtype=sample.dtype, device=sample.device)
        target_t = torch.as_tensor(target_t, dtype=sample.dtype, device=sample.device)
        timestep = timestep.reshape(-1).expand(batch)
        target_t = target_t.reshape(-1).expand(batch)
        time_c = self.timestep_target_t_adaptor(
            torch.cat(
                [
                    self.flow_timestep_encoder(timestep),
                    self.flow_target_t_encoder(target_t),
                ],
                dim=-1,
            )
        )
        context_c = (
            self.vis_cond_obs_emb(vis_cond)
            + self.vis_cond_pos_embed[:, : vis_cond.shape[1]]
        )
        for block in self.blocks:
            x = block(x, time_c, context_c)
        return self.final_layer(x)[:, -self.horizon :]
