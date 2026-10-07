"""Block-causal GPT trunk from the Patch Policy reference (nanoGPT-derived, MIT).

Adapted from ``models/vq_behavior_transformer/gpt.py``: tokens of one frame
attend bidirectionally to each other and causally to earlier frames; a learned
1D position embedding indexes the flattened (frame, token) sequence; the trunk
reads out the last token of each frame. Attention is evaluated frame by frame
with fused scaled-dot-product kernels, which is exactly the reference's
block-causal mask without materializing it.
"""

import math
from dataclasses import dataclass

import einops
import torch
import torch.nn.functional as F
from torch import nn


def new_gelu(x):
    """The reference's tanh-approximate GELU as one fused op.

    Writing the formula out saves every elementwise intermediate for backward,
    which dominates activation memory at ~200k tokens per batch.
    """
    return F.gelu(x, approximate="tanh")


@dataclass
class GPTConfig:
    n_frames: int
    tokens_per_frame: int
    input_dim: int
    output_dim: int = 256
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 384
    dropout: float = 0.0


class BlockCausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.tokens_per_frame = config.tokens_per_frame

    def forward(self, x):
        b, n, c = x.size()
        p = self.tokens_per_frame
        assert n % p == 0
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q, k, v = (t.view(b, n, self.n_head, c // self.n_head).transpose(1, 2) for t in (q, k, v))
        dropout = self.attn_dropout if self.training else 0.0
        frames = [
            F.scaled_dot_product_attention(
                q[:, :, f * p : (f + 1) * p],
                k[:, :, : (f + 1) * p],
                v[:, :, : (f + 1) * p],
                dropout_p=dropout,
            )
            for f in range(n // p)
        ]
        y = torch.cat(frames, dim=2).transpose(1, 2).contiguous().view(b, n, c)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(new_gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = BlockCausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Linear(config.input_dim, config.n_embd),
                wpe=nn.Embedding(config.n_frames * config.tokens_per_frame, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.output_dim, bias=False)
        self.apply(self._init_weights)
        for name, param in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """(B, T, P, D) frame tokens -> (B, T, output_dim) from each frame's last token."""
        b, t, p, _ = tokens.shape
        if t != self.config.n_frames or p != self.config.tokens_per_frame:
            raise ValueError(
                f"expected {self.config.n_frames} frames x {self.config.tokens_per_frame} tokens, got {t} x {p}"
            )
        pos = torch.arange(t * p, device=tokens.device)
        x = self.transformer.wte(tokens).reshape(b, t * p, -1) + self.transformer.wpe(pos)
        x = self.transformer.drop(x)
        for block in self.transformer.h:
            x = block(x)
        x = self.lm_head(self.transformer.ln_f(x))
        return einops.rearrange(x, "b (t p) d -> b t p d", t=t)[:, :, -1]

    def param_groups(self, weight_decay: float) -> list[dict]:
        """Reference split: decay Linear weights only; biases, LayerNorm, Embedding not."""
        decay, no_decay = [], []
        for module_name, module in self.named_modules():
            for name, param in module.named_parameters(recurse=False):
                if name.endswith("bias"):
                    no_decay.append(param)
                elif name.endswith("weight") and isinstance(module, nn.Linear):
                    decay.append(param)
                elif name.endswith("weight") and isinstance(module, (nn.LayerNorm, nn.Embedding)):
                    no_decay.append(param)
                else:
                    raise ValueError(f"unclassified GPT parameter {module_name}.{name}")
        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
