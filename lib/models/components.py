"""Reusable building blocks shared by every model architecture."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """Layer normalization with learnable scale and shift."""

    def __init__(self, embed_dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(embed_dim))
        self.shift = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encodings added to the token embeddings."""

    def __init__(self, embed_dim: int, max_seq_length: int = 512):
        super().__init__()

        pe = torch.zeros(max_seq_length, embed_dim)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class MultiHeadAttention(nn.Module):
    """Scaled dot-product multi-head self-attention."""

    def __init__(self, embed_dim: int, num_heads: int, qkv_bias: bool = False):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.o_layer = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None):
        B = x.shape[0]

        q = self.q_proj(x).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attention = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if mask is not None:
            attention = attention.masked_fill(mask == 0, float("-inf"))
        attention = torch.softmax(attention, dim=-1)

        out = torch.matmul(attention, v)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.embed_dim)
        out = self.o_layer(out)

        return out


class FeedForward(nn.Module):
    """Two-layer GELU MLP used as the dense FFN and as each MoE expert."""

    def __init__(self, embed_dim: int, hidden_size: int):
        super().__init__()
        self.in_linear = nn.Linear(embed_dim, hidden_size)
        self.out_linear = nn.Linear(hidden_size, embed_dim)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.gelu(self.in_linear(x))
        x = self.out_linear(x)
        return x
