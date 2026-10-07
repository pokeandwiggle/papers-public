# --------------------------------------------------------------------
# Ported from https://github.com/Chaoqi-LIU/oat (revision 1da9269):
#   oat/model/autoregressive/transformer_cache.py (used by OATPolicy and
#   QueSTPolicy). Architecture, initialization and KV-cached sampling are
#   unchanged.
# --------------------------------------------------------------------
"""Autoregressive token decoder conditioned on observation features."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, n_emb: int, p_drop: float):
        super().__init__()
        self.c_fc = nn.Linear(n_emb, 4 * n_emb)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_emb, n_emb)
        self.dropout = nn.Dropout(p_drop)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xf = x.float()
        output = (xf * torch.rsqrt(torch.mean(xf * xf, dim=-1, keepdim=True) + self.eps)).type_as(x)
        return output * self.weight


class CausalSelfAttention(nn.Module):
    def __init__(self, n_head: int, n_emb: int, p_drop_attn: float):
        super().__init__()
        self.c_attn = nn.Linear(n_emb, 3 * n_emb, bias=False)
        self.c_proj = nn.Linear(n_emb, n_emb, bias=False)
        self.resid_dropout = nn.Dropout(p_drop_attn)
        self.p_attn_dropout = p_drop_attn
        self.n_head = n_head
        self.n_emb = n_emb
        self.head_dim = n_emb // n_head

    def forward(self, x, layer_past=None):
        batch, length, channels = x.size()
        q, k, v = self.c_attn(x).split(self.n_emb, dim=-1)
        q, k, v = (
            t.view(batch, length, self.n_head, self.head_dim).transpose(1, 2) for t in (q, k, v)
        )
        if layer_past is not None:
            past_k, past_v = layer_past
            k = torch.cat((past_k, k), dim=-2)
            v = torch.cat((past_v, v), dim=-2)
        is_causal = layer_past is None and length > 1
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=is_causal,
            dropout_p=self.p_attn_dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(batch, length, channels)
        return self.resid_dropout(self.c_proj(y)), (k, v)


class CrossAttention(nn.Module):
    def __init__(self, n_head: int, n_emb: int, p_drop_attn: float):
        super().__init__()
        self.q_proj = nn.Linear(n_emb, n_emb, bias=False)
        self.kv_proj = nn.Linear(n_emb, 2 * n_emb, bias=False)
        self.c_proj = nn.Linear(n_emb, n_emb, bias=False)
        self.resid_dropout = nn.Dropout(p_drop_attn)
        self.p_attn_dropout = p_drop_attn
        self.n_head = n_head
        self.n_emb = n_emb
        self.head_dim = n_emb // n_head

    def memory_kv(self, memory):
        batch, length, _ = memory.size()
        k, v = self.kv_proj(memory).split(self.n_emb, dim=2)
        return tuple(
            t.view(batch, length, self.n_head, self.head_dim).transpose(1, 2) for t in (k, v)
        )

    def forward(self, x, memory, memory_kv_cache=None):
        batch, length, channels = x.size()
        q = self.q_proj(x).view(batch, length, self.n_head, self.head_dim).transpose(1, 2)
        k, v = memory_kv_cache if memory_kv_cache is not None else self.memory_kv(memory)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.p_attn_dropout if self.training else 0.0,
            is_causal=False,
        )
        y = y.transpose(1, 2).contiguous().view(batch, length, channels)
        return self.resid_dropout(self.c_proj(y))


class Block(nn.Module):
    def __init__(self, n_head: int, n_emb: int, p_drop_attn: float):
        super().__init__()
        self.ln_1 = RMSNorm(n_emb)
        self.attn = CausalSelfAttention(n_head, n_emb, p_drop_attn)
        self.ln_2 = RMSNorm(n_emb)
        self.cross_attn = CrossAttention(n_head, n_emb, p_drop_attn)
        self.ln_3 = RMSNorm(n_emb)
        self.mlp = MLP(n_emb, p_drop_attn)

    def forward(self, x, memory, layer_past=None, memory_kv_cache=None):
        attn_output, present = self.attn(self.ln_1(x), layer_past=layer_past)
        x = x + attn_output
        x = x + self.cross_attn(self.ln_2(x), memory, memory_kv_cache)
        x = x + self.mlp(self.ln_3(x))
        return x, present


class AutoregressiveModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        max_cond_len: int,
        cond_dim: int,
        n_layer: int,
        n_head: int,
        n_emb: int,
        p_drop_emb: float,
        p_drop_attn: float,
    ):
        super().__init__()
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_emb = n_emb
        self.tok_emb = nn.Embedding(vocab_size, n_emb)
        self.tok_pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, n_emb))
        self.cond_emb = nn.Linear(cond_dim, n_emb)
        self.cond_pos_emb = nn.Parameter(torch.zeros(1, max_cond_len, n_emb))
        self.drop = nn.Dropout(p_drop_emb)
        self.encoder = nn.Sequential(
            nn.Linear(n_emb, 4 * n_emb), nn.Mish(), nn.Linear(4 * n_emb, n_emb)
        )
        self.blocks = nn.ModuleList([Block(n_head, n_emb, p_drop_attn) for _ in range(n_layer)])
        self.ln_f = RMSNorm(n_emb)
        self.head = nn.Linear(n_emb, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.xavier_uniform_(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, RMSNorm):
                nn.init.constant_(m.weight, 1.0)

    def encode_condition(self, cond):
        memory = self.drop(self.cond_emb(cond) + self.cond_pos_emb[:, : cond.shape[1], :])
        return self.encoder(memory)

    def forward(self, tokens, cond):
        """Teacher-forced logits: tokens (B, T), cond (B, T_cond, cond_dim)."""
        x = self.drop(self.tok_emb(tokens) + self.tok_pos_emb[:, : tokens.shape[1], :])
        memory = self.encode_condition(cond)
        for block in self.blocks:
            x, _ = block(x, memory)
        return self.head(self.ln_f(x))

    @torch.inference_mode()
    def generate(self, prefix, cond, max_new_tokens, temperature=1.0, top_k=None):
        """Sample ``max_new_tokens`` tokens after ``prefix`` with KV caching."""
        memory = self.encode_condition(cond)
        memory_kv_cache = [block.cross_attn.memory_kv(memory) for block in self.blocks]

        prefix_len = prefix.shape[1]
        x = self.drop(self.tok_emb(prefix) + self.tok_pos_emb[:, :prefix_len, :])
        past = [None] * self.n_layer
        for i, block in enumerate(self.blocks):
            x, past[i] = block(x, memory, layer_past=None, memory_kv_cache=memory_kv_cache[i])
        logits = self.head(self.ln_f(x[:, -1:, :]))

        out_tokens = prefix
        for i in range(max_new_tokens):
            if temperature > 0:
                logits = logits.squeeze(1) / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")
                next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            else:
                next_token = torch.argmax(logits.squeeze(1), dim=-1, keepdim=True)
            out_tokens = torch.cat((out_tokens, next_token), dim=1)
            if out_tokens.shape[1] >= prefix_len + max_new_tokens:
                break
            position = prefix_len + i
            x = self.drop(self.tok_emb(next_token) + self.tok_pos_emb[:, position : position + 1, :])
            for layer, block in enumerate(self.blocks):
                x, past[layer] = block(
                    x, memory, layer_past=past[layer], memory_kv_cache=memory_kv_cache[layer]
                )
            logits = self.head(self.ln_f(x))
        return out_tokens
