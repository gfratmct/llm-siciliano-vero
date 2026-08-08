import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_seq_length: int = 512):
        super(PositionalEncoding, self).__init__()

        # empty pe tensor
        pe = torch.zeros(max_seq_length, embed_dim) # shape: [max_seq_length, embed_dim]

        # create a tensor for positions 
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
    

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, qkv_bias: bool = False):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)

        self.o_layer = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask = None):
        B = x.shape[0]

        q = self.q_proj(x).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attention = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if mask != None:
            attention = attention.masked_fill(mask == 0, float('-inf'))
        attention = torch.softmax(attention, dim=-1)

        out = torch.matmul(attention, v)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.embed_dim)
        out = self.o_layer(out)

        return out

class LayerNorm(nn.Module):
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
    

class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, hidden_size: int):
        super().__init__()
        self.in_linear = nn.Linear(embed_dim, hidden_size)
        self.out_linear = nn.Linear(hidden_size, embed_dim)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.in_linear(x)
        x = self.gelu(x)
        x = self.out_linear(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, hidden_size: int, dropout: float = 0.1, qkv_bias = False):
        super().__init__()

        self.mha = MultiHeadAttention(embed_dim, num_heads, qkv_bias)
        self.layer_norm_1 = LayerNorm(embed_dim)
        self.layer_norm_2 = LayerNorm(embed_dim)
        self.feed_forward = FeedForward(embed_dim, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask = None):
        # Pre-LayerNorm: more stable for deep transformers trained from scratch.
        attention_out = x + self.dropout(self.mha(self.layer_norm_1(x), mask))
        ff_out = attention_out + self.dropout(self.feed_forward(self.layer_norm_2(attention_out)))

        return ff_out
    

class LLM(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, num_layers: int, hidden_size: int, vocab_size: int, max_seq_length: int = 512, dropout: float = 0.1, qkv_bias = False):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_length)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, hidden_size, dropout, qkv_bias)
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        # Tie the output projection to the input embedding, standard for LLMs.
        self.fc_out.weight = self.embedding.weight
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_seq_length, max_seq_length, dtype=torch.bool)),
        )

        # Apply GPT-2-style weight initialization for stable training.
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights following the GPT-2 / NanoGPT recipe."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        if mask is None:
            mask = self.causal_mask[: x.size(1), : x.size(1)]

        for block in self.transformer_blocks:
            x = block(x, mask)

        x = self.fc_out(x)
        return x