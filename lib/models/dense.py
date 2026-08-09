"""Standard (dense) transformer: attention + full FeedForward per block."""

import torch.nn as nn

from ..config import ModelConfig
from .base import BaseLLM
from .components import FeedForward, LayerNorm, MultiHeadAttention


class DenseBlock(nn.Module):
    """Pre-LayerNorm transformer block with a dense FeedForward."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        hidden_size: int,
        dropout: float = 0.1,
        qkv_bias: bool = False,
    ):
        super().__init__()
        self.mha = MultiHeadAttention(embed_dim, num_heads, qkv_bias)
        self.layer_norm_1 = LayerNorm(embed_dim)
        self.layer_norm_2 = LayerNorm(embed_dim)
        self.feed_forward = FeedForward(embed_dim, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Pre-LayerNorm: more stable for deep transformers trained from scratch.
        attention_out = x + self.dropout(self.mha(self.layer_norm_1(x), mask))
        ff_out = attention_out + self.dropout(
            self.feed_forward(self.layer_norm_2(attention_out))
        )
        return ff_out


class DenseLLM(BaseLLM):
    """Decoder-only transformer language model with dense FFNs."""

    def build_blocks(self, config: ModelConfig) -> nn.ModuleList:
        return nn.ModuleList(
            [
                DenseBlock(
                    config.embedding_dim,
                    config.num_heads,
                    config.hidden_size,
                    config.drop_rate,
                    config.qkv_bias,
                )
                for _ in range(config.num_layers)
            ]
        )
