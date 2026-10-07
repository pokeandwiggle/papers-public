# --------------------------------------------------------------------
# Ported from https://github.com/Chaoqi-LIU/oat (revision 1da9269):
#   oat/tokenizer/oat/{tokenizer,encoder,decoder,quantizer,model}/*.py
# Several of those files are adapted by the OAT authors from EPFL & Apple
# FlexTok code (EPFL-Apple Sample Code License, non-commercial research use).
# Only the modules and code paths used by the OAT tokenizer are kept; the
# behaviour of the kept paths is unchanged.
# --------------------------------------------------------------------
"""Ordered Action Tokenizer: register encoder, FSQ, nested dropout, decoder.

Inputs are action chunks already normalized to [-1, 1] by the LeRobot
preprocessor (the reference fits a min/max "limits" normalizer inside the
tokenizer; LeRobot MIN_MAX normalization is the same affine map).
"""

from functools import lru_cache, partial

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_


class Fp32LayerNorm(nn.LayerNorm):
    """Mixed-precision friendly LayerNorm (torchtune)."""

    def forward(self, input):
        output = F.layer_norm(
            input.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)


_NORM = partial(Fp32LayerNorm, bias=False, elementwise_affine=False)


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        proj_bias=False,
        proj_drop=0.0,
        qk_norm=True,
        norm_layer=_NORM,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        if qk_norm:
            self.q_norm = norm_layer(self.head_dim)
            self.k_norm = norm_layer(self.head_dim)
        else:
            self.q_norm = self.k_norm = nn.Identity()

    def forward(self, x, block_mask=None):
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        q = einops.rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k = einops.rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
        v = einops.rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)
        q = self.q_norm(q)
        k = self.k_norm(k)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=block_mask)
        x = einops.rearrange(x, "b h n d -> b n (h d)")
        return self.proj_drop(self.proj(x))


class GatedMlp(nn.Module):
    """SwiGLU-style gated MLP."""

    def __init__(self, in_features, hidden_features, act_layer=nn.SiLU, drop=0.0, bias=True):
        super().__init__()
        hidden_features = int(2 * hidden_features / 3)
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features, bias=bias)
        self.fc3 = nn.Linear(in_features, hidden_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x)) * self.fc3(x)))


class Block(nn.Module):
    """Pre-norm transformer block (reference defaults of the OAT Transformer)."""

    def __init__(self, dim, head_dim, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.norm1 = _NORM(dim)
        self.attn = SelfAttention(
            dim,
            num_heads=dim // head_dim,
            qkv_bias=False,
            proj_bias=False,
            proj_drop=drop,
            qk_norm=True,
            norm_layer=_NORM,
        )
        self.norm2 = _NORM(dim)
        self.mlp = GatedMlp(
            dim, int(dim * mlp_ratio), act_layer=nn.SiLU, drop=drop, bias=False
        )

    def forward(self, x, block_mask=None):
        x = x + self.attn(self.norm1(x), block_mask=block_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, dim, depth, head_dim, drop):
        super().__init__()
        self.blocks = nn.ModuleList(
            [Block(dim=dim, head_dim=head_dim, drop=drop) for _ in range(depth)]
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, block_mask=None):
        for block in self.blocks:
            x = block(x, block_mask=block_mask)
        return x


class LinearHead(nn.Module):
    """LayerNorm (no affine) followed by a zero-initialized linear projection."""

    def __init__(self, dim, dim_out):
        super().__init__()
        self.norm = _NORM(dim)
        self.proj = nn.Linear(dim, dim_out, bias=True)
        nn.init.constant_(self.proj.weight, 0)
        nn.init.constant_(self.proj.bias, 0)

    def forward(self, x):
        return self.proj(self.norm(x))


def build_1d_sincos_posemb(length, embed_dim, temperature=10000.0):
    """Sine-cosine embeddings of shape (1, D, L)."""
    pos = torch.arange(length, dtype=torch.float32)
    dim_t = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega = 1.0 / (temperature ** (dim_t / (embed_dim // 2)))
    out = torch.einsum("n,d->nd", pos, omega)
    pos_emb = torch.cat([torch.sin(out), torch.cos(out)], dim=1)
    return pos_emb.transpose(0, 1).unsqueeze(0)


class SinCosPosEmb(nn.Module):
    """Fixed 1D sin-cos embedding stored as a frozen parameter, as in the reference."""

    def __init__(self, dim, max_len):
        super().__init__()
        self.posembs = nn.Parameter(build_1d_sincos_posemb(max_len, dim), requires_grad=False)

    def table(self, length):
        """(1, L, D) embedding table."""
        return einops.rearrange(self.posembs[:, :, :length], "b d l -> b l d")

    def forward(self, x):
        return x + self.table(x.shape[1])


@lru_cache(maxsize=32)
def create_causal_last_mask(action_len: int, register_len: int, device: str) -> torch.Tensor:
    """Boolean mask (True = attend) over [actions, registers].

    Actions attend to all actions but not registers; registers attend to all
    actions and causally to previous registers.
    """
    total_len = action_len + register_len
    mask = torch.zeros((total_len, total_len), dtype=torch.bool, device=device)
    causal_reg_mask = torch.triu(
        torch.ones((register_len, register_len), dtype=torch.bool, device=device),
        diagonal=1,
    )
    mask[action_len:, action_len:] = causal_reg_mask
    mask[:action_len, action_len:] = True
    return ~mask


class RegisterEncoder(nn.Module):
    def __init__(
        self, sample_dim, sample_horizon, emb_dim, head_dim, depth, pdropout, latent_dim, num_registers
    ):
        super().__init__()
        self.sample_emb = nn.Linear(sample_dim, emb_dim, bias=True)
        nn.init.xavier_uniform_(self.sample_emb.weight)
        nn.init.constant_(self.sample_emb.bias, 0)
        self.pos_emb = SinCosPosEmb(emb_dim, sample_horizon)
        self.registers = nn.Parameter(torch.randn(num_registers, emb_dim))
        self.transformer = Transformer(dim=emb_dim, depth=depth, head_dim=head_dim, drop=pdropout)
        self.head = LinearHead(emb_dim, latent_dim)
        self.num_registers = num_registers

    def forward(self, sample):
        batch, horizon, _ = sample.shape
        x = self.pos_emb(self.sample_emb(sample))
        x = torch.cat([x, self.registers.unsqueeze(0).expand(batch, -1, -1)], dim=1)
        mask = create_causal_last_mask(horizon, self.num_registers, str(sample.device))
        x = self.transformer(x, block_mask=mask)
        return self.head(x[:, horizon:])


def _powers_of_two(min_val, max_val):
    return [2**i for i in range(int(min_val).bit_length() - 1, int(max_val).bit_length())]


class MaskedNestedDropout(nn.Module):
    """Replace a random tail of tokens with a learnable mask token (training) or a
    requested tail (eval, when ``eval_keep_k`` is given)."""

    def __init__(self, dim, size_sampling_mode="pow2"):
        super().__init__()
        self.size_sampling_mode = size_sampling_mode
        if size_sampling_mode != "disable":
            self.dropout_mask_token = nn.Parameter(torch.randn(dim), requires_grad=True)
            trunc_normal_(self.dropout_mask_token, std=0.02)

    def sample_keep_k(self, batch_size, horizon, device):
        if self.size_sampling_mode == "uniform":
            return torch.randint(1, horizon + 1, (batch_size,), device=device)
        if self.size_sampling_mode == "pow2":
            if horizon <= 0 or horizon & (horizon - 1):
                raise ValueError("pow2 nested dropout requires a power-of-two token count")
            values = torch.tensor(_powers_of_two(1, horizon), device=device)
            return values[torch.randint(0, len(values), (batch_size,), device=device)]
        raise ValueError(f"unsupported nested dropout mode {self.size_sampling_mode!r}")

    def forward(self, x, eval_keep_k=None):
        if self.size_sampling_mode == "disable":
            return x
        batch, n, _ = x.shape
        if self.training:
            keep = self.sample_keep_k(batch, n, x.device)
        elif eval_keep_k is not None:
            keep = torch.as_tensor(eval_keep_k, device=x.device)
        else:
            return x
        mask = keep.unsqueeze(1) <= torch.arange(n, device=x.device).unsqueeze(0)
        x = x.clone()
        x[mask] = self.dropout_mask_token.to(x.dtype)
        return x


class SinglePassDecoder(nn.Module):
    def __init__(
        self,
        sample_dim,
        sample_horizon,
        emb_dim,
        head_dim,
        depth,
        pdropout,
        token_dropout_mode,
        use_causal_decoder,
        latent_dim,
        latent_horizon,
    ):
        super().__init__()
        self.sample_pos_emb = SinCosPosEmb(emb_dim, sample_horizon)
        self.latent_pos_emb = SinCosPosEmb(emb_dim, latent_horizon)
        self.nested_dropout = MaskedNestedDropout(emb_dim, size_sampling_mode=token_dropout_mode)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=emb_dim,
                nhead=emb_dim // head_dim,
                dim_feedforward=4 * emb_dim,
                dropout=pdropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=depth,
        )
        self.latent_proj = nn.Linear(latent_dim, emb_dim, bias=True)
        nn.init.xavier_uniform_(self.latent_proj.weight)
        nn.init.constant_(self.latent_proj.bias, 0)
        self.head = LinearHead(emb_dim, sample_dim)
        self.sample_horizon = sample_horizon
        self.latent_horizon = latent_horizon
        self.use_causal_decoder = use_causal_decoder

    def forward(self, latents, eval_keep_k=None):
        x = self.sample_pos_emb.table(self.sample_horizon).expand(latents.shape[0], -1, -1)
        memory = self.latent_pos_emb(self.latent_proj(latents))
        memory = self.nested_dropout(memory, eval_keep_k=eval_keep_k)
        if self.use_causal_decoder:
            mask = nn.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device)
            x = self.decoder(x, memory, tgt_mask=mask, tgt_is_causal=True)
        else:
            x = self.decoder(x, memory)
        return self.head(x)


class FSQ(nn.Module):
    """Finite scalar quantization with straight-through rounding (no projections)."""

    def __init__(self, levels):
        super().__init__()
        levels = list(levels)
        self.register_buffer("_levels", torch.tensor(levels, dtype=torch.int32), persistent=False)
        basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=torch.int32)
        self.register_buffer("_basis", basis, persistent=False)
        self.dim = len(levels)
        self.codebook_size = int(torch.tensor(levels).prod().item())

    def bound(self, z, eps=1e-3):
        half_l = (self._levels - 1) * (1 + eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).atanh()
        return (z + shift).tanh() * half_l - offset

    def quantize(self, z):
        bounded = self.bound(z)
        quantized = bounded + (bounded.round() - bounded).detach()
        return quantized / (self._levels // 2)

    def codes_to_indices(self, zhat):
        half_width = self._levels // 2
        zhat = zhat * half_width + half_width
        return (zhat * self._basis).sum(dim=-1).to(torch.int32)

    def indices_to_embedding(self, indices):
        indices = indices.unsqueeze(-1)
        codes_non_centered = (indices // self._basis) % self._levels
        half_width = self._levels // 2
        return (codes_non_centered - half_width) / half_width

    def forward(self, z):
        with torch.autocast(device_type=z.device.type, enabled=False):
            quant = self.quantize(z.float())
            tokens = self.codes_to_indices(quant)
        return quant, tokens.long()


class OATTokenizer(nn.Module):
    """Encode normalized (B, H_a, D_a) chunks into H_l ordered tokens and back."""

    def __init__(
        self,
        action_dim,
        horizon,
        emb_dim=256,
        head_dim=64,
        encoder_depth=2,
        decoder_depth=4,
        pdropout=0.1,
        num_registers=8,
        fsq_levels=(8, 5, 5, 5),
        token_dropout_mode="pow2",
        use_causal_decoder=True,
    ):
        super().__init__()
        latent_dim = len(fsq_levels)
        self.encoder = RegisterEncoder(
            sample_dim=action_dim,
            sample_horizon=horizon,
            emb_dim=emb_dim,
            head_dim=head_dim,
            depth=encoder_depth,
            pdropout=pdropout,
            latent_dim=latent_dim,
            num_registers=num_registers,
        )
        self.decoder = SinglePassDecoder(
            sample_dim=action_dim,
            sample_horizon=horizon,
            emb_dim=emb_dim,
            head_dim=head_dim,
            depth=decoder_depth,
            pdropout=pdropout,
            token_dropout_mode=token_dropout_mode,
            use_causal_decoder=use_causal_decoder,
            latent_dim=latent_dim,
            latent_horizon=num_registers,
        )
        self.quantizer = FSQ(fsq_levels)
        self.codebook_size = self.quantizer.codebook_size
        self.num_tokens = num_registers

    def reconstruction_loss(self, nactions):
        latents, _ = self.quantizer(self.encoder(nactions))
        return F.mse_loss(self.decoder(latents), nactions)

    def tokenize(self, nactions):
        _, tokens = self.quantizer(self.encoder(nactions))
        return tokens

    def detokenize(self, tokens):
        """Decode a (B, K) token prefix, K <= num_tokens; the tail is masked."""
        keep = tokens.shape[1]
        if keep > self.num_tokens:
            raise ValueError(f"at most {self.num_tokens} tokens can be decoded, got {keep}")
        if keep < self.num_tokens:
            tokens = F.pad(tokens, (0, self.num_tokens - keep))
        latents = self.quantizer.indices_to_embedding(tokens)
        return self.decoder(latents, eval_keep_k=[keep] * tokens.shape[0])

    def autoencode(self, nactions, keep=None):
        latents, _ = self.quantizer(self.encoder(nactions))
        keep = self.num_tokens if keep is None else keep
        return self.decoder(latents, eval_keep_k=[keep] * nactions.shape[0])
