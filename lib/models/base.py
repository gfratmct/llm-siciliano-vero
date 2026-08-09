"""Base decoder-only transformer shared by the dense and MoE models."""

import torch
import torch.nn as nn

from ..config import ModelConfig
from .components import PositionalEncoding


class BaseLLM(nn.Module):
    """Token embeddings + positional encodings + stacked blocks + lm head.

    Subclasses only need to provide ``build_blocks`` returning an
    ``nn.ModuleList`` of transformer blocks; everything else (embedding,
    positional encoding, tied output projection, weight init, forward pass)
    is shared, so dense and MoE share the same state-dict layout for the
    non-block components.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.positional_encoding = PositionalEncoding(
            config.embedding_dim, config.max_sequence_length
        )
        self.transformer_blocks = self.build_blocks(config)
        self.fc_out = nn.Linear(config.embedding_dim, config.vocab_size)
        # Tie the output projection to the input embedding, standard for LLMs.
        self.fc_out.weight = self.embedding.weight
        self.dropout = nn.Dropout(config.drop_rate)
        self.register_buffer(
            "causal_mask",
            torch.tril(
                torch.ones(
                    config.max_sequence_length,
                    config.max_sequence_length,
                    dtype=torch.bool,
                )
            ),
        )

        # Apply GPT-2-style weight initialization for stable training.
        self.apply(self._init_weights)

    def build_blocks(self, config: ModelConfig) -> nn.ModuleList:
        raise NotImplementedError

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
