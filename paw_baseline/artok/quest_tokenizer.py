# --------------------------------------------------------------------
# Ported from https://github.com/Chaoqi-LIU/oat (revision 1da9269):
#   oat/tokenizer/quest/{tokenizer.py,model/conv.py,model/pos_emb.py}
# which the OAT authors adapted from https://github.com/pairlab/QueST.
# Only the FSQ code path used by the paper's QueST baseline is kept.
# --------------------------------------------------------------------
"""QueST tokenizer: causal temporal convolution + causal transformer + FSQ.

Like ``OATTokenizer`` it operates on LeRobot-normalized [-1, 1] actions. The
reference QueST computes its reconstruction loss after unnormalizing; on the
paper's benchmarks raw actions already lie in about [-1, 1], so the loss here
uses the normalized space shared with OAT (absolute TCP metres would otherwise
underweight position and gripper errors by orders of magnitude).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch import FSQ


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = dilation * (kernel_size - 1)
        self.stride = stride
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation,
            stride=stride,
        )

    def forward(self, x):
        x = self.conv(x)
        last_n = (2 * self.padding - self.kernel_size) // self.stride + 1
        return x[:, :, :-last_n] if last_n > 0 else x


class Conv1dBlock(nn.Module):
    """Causal Conv1d -> GroupNorm -> Mish."""

    def __init__(self, inp_channels, out_channels, kernel_size, stride, n_groups):
        super().__init__()
        self.conv = CausalConv1d(inp_channels, out_channels, kernel_size, dilation=1, stride=stride)
        self.norm = nn.GroupNorm(n_groups, out_channels)
        self.act = nn.Mish()

    def forward(self, x):
        # GroupNorm over (B, C, 1, T) equals GroupNorm over (B, C, T).
        return self.act(self.norm(self.conv(x)))


class ResidualTemporalBlock(nn.Module):
    """Reference block with residual=False and no pooling (the QueST configuration)."""

    def __init__(self, channels, kernel_size, stride, n_groups=8):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(channels, channels, kernel_size[i], stride[i], n_groups)
                for i in range(len(kernel_size))
            ]
        )

    def forward(self, x):
        out = x.transpose(1, 2)
        for block in self.blocks:
            out = block(out)
        return out.transpose(1, 2)


def sinusoidal_encoding_1d(length, channels, device, dtype):
    """tatp22 PositionalEncoding1D: interleaved sin/cos, shape (1, L, C)."""
    padded = int(np.ceil(channels / 2) * 2)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, padded, 2, device=device).float() / padded))
    pos = torch.arange(length, device=device, dtype=inv_freq.dtype)
    sin_inp = torch.einsum("i,j->ij", pos, inv_freq)
    emb = torch.flatten(torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1), -2, -1)
    return emb[None, :, :channels].to(dtype)


class QueSTTokenizer(nn.Module):
    def __init__(
        self,
        action_dim,
        horizon,
        encoder_dim=256,
        encoder_nhead=4,
        encoder_nlayer=2,
        use_causal_encoder=True,
        decoder_dim=256,
        decoder_nhead=4,
        decoder_nlayer=4,
        use_causal_decoder=True,
        downsample_factor=4,
        dropout=0.1,
        fsq_levels=(8, 5, 5, 5),
    ):
        super().__init__()
        if 2 ** int(np.log2(downsample_factor)) != downsample_factor:
            raise ValueError("downsample_factor must be a power of 2")
        if not use_causal_encoder:
            raise ValueError("only the causal QueST encoder is supported")
        strides = [2] * int(np.log2(downsample_factor)) + [1]
        kernel_sizes = [5] + [3] * int(np.log2(downsample_factor))
        if len(strides) == 1:
            kernel_sizes, strides = [3, 2], [1, 1]

        self.vq = FSQ(dim=encoder_dim, levels=list(fsq_levels))
        self.codebook_size = int(np.prod(fsq_levels))
        self.action_proj = nn.Linear(action_dim, encoder_dim)
        self.action_head = nn.Linear(decoder_dim, action_dim)
        self.conv_block = ResidualTemporalBlock(encoder_dim, kernel_sizes, strides)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=encoder_dim,
                nhead=encoder_nhead,
                dim_feedforward=4 * encoder_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=encoder_nlayer,
            enable_nested_tensor=False,
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=decoder_dim,
                nhead=decoder_nhead,
                dim_feedforward=4 * decoder_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=decoder_nlayer,
        )
        self.horizon = horizon
        self.num_tokens = horizon // downsample_factor
        self.decoder_dim = decoder_dim
        self.use_causal_decoder = use_causal_decoder
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.xavier_uniform_(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def encode(self, nactions):
        x = self.conv_block(self.action_proj(nactions))
        x = x + sinusoidal_encoding_1d(x.shape[1], x.shape[2], x.device, x.dtype)
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device)
        x = self.encoder(x, mask=mask)
        return self.vq(x)

    def decode(self, quants):
        queries = sinusoidal_encoding_1d(
            self.horizon, self.decoder_dim, quants.device, quants.dtype
        ).expand(quants.shape[0], -1, -1)
        if self.use_causal_decoder:
            mask = nn.Transformer.generate_square_subsequent_mask(self.horizon, device=quants.device)
            x = self.decoder(queries, quants, tgt_mask=mask, tgt_is_causal=True)
        else:
            x = self.decoder(queries, quants)
        return self.action_head(x)

    def reconstruction_loss(self, nactions):
        quants, _ = self.encode(nactions)
        return F.mse_loss(self.decode(quants), nactions)

    def tokenize(self, nactions):
        _, indices = self.encode(nactions)
        return indices.long()

    def detokenize(self, tokens):
        if tokens.shape[1] != self.num_tokens:
            raise ValueError(f"QueST decodes exactly {self.num_tokens} tokens, got {tokens.shape[1]}")
        return self.decode(self.vq.indices_to_codes(tokens))

    def autoencode(self, nactions, keep=None):
        if keep not in (None, self.num_tokens):
            raise ValueError("QueST has no prefix decoding")
        quants, _ = self.encode(nactions)
        return self.decode(quants)
