"""Mixture-of-Experts transformer: dense attention + expert-routed FFN.

The FeedForward of every block is replaced by a MoE layer: a noisy top-k
router dispatches each token to ``num_experts_per_tok`` of ``num_experts``
parallel FFN experts and combines their outputs. A load-balancing auxiliary
loss is accumulated across blocks and exposed as ``model.moe_aux_loss`` so the
training loop can add it to the main loss without changing the forward
signature.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from .base import BaseLLM
from .components import FeedForward, LayerNorm, MultiHeadAttention


class MoELayer(nn.Module):
    """Noisy top-k router over ``num_experts`` parallel FeedForwards."""

    def __init__(
        self,
        embed_dim: int,
        hidden_size: int,
        num_experts: int,
        num_experts_per_tok: int,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok

        self.gate = nn.Linear(embed_dim, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [FeedForward(embed_dim, hidden_size) for _ in range(num_experts)]
        )
        self.aux_loss: torch.Tensor = torch.zeros(())

    def _gating_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Router logits with Gaussian noise during training for better balance."""
        logits = self.gate(x)
        if self.training:
            noise = torch.randn_like(logits) * F.softplus(logits)
            logits = logits + noise
        return logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, embed_dim] where N = B * T
        gating_logits = self._gating_logits(x)
        top_logits, top_indices = torch.topk(
            gating_logits, self.num_experts_per_tok, dim=-1
        )
        probs = F.softmax(top_logits, dim=-1)

        # Load-balancing aux loss (Shazeer et al. 2017): encourage the router
        # to distribute tokens evenly across experts.
        router_probs = F.softmax(gating_logits, dim=-1)
        tokens_per_expert = torch.zeros(
            self.num_experts, dtype=x.dtype, device=x.device
        ).scatter_add_(0, top_indices.reshape(-1), torch.ones_like(top_indices.reshape(-1), dtype=x.dtype))
        fraction_per_expert = tokens_per_expert / x.size(0)
        mean_router_prob = router_probs.mean(dim=0)
        self.aux_loss = self.num_experts * torch.sum(
            fraction_per_expert * mean_router_prob
        )

        # Dispatch each token to its top-k experts and weight-sum the outputs.
        num_tokens = x.size(0)
        token_ids = (
            torch.arange(num_tokens, device=x.device).repeat_interleave(
                self.num_experts_per_tok
            )
        )
        flat_x = x.repeat_interleave(self.num_experts_per_tok, dim=0)
        flat_experts = top_indices.reshape(-1)
        flat_weights = probs.reshape(-1, 1)

        out = torch.zeros_like(x)
        for expert_idx in range(self.num_experts):
            mask = flat_experts == expert_idx
            if not mask.any():
                continue
            expert_in = flat_x[mask]
            expert_out = self.experts[expert_idx](expert_in) * flat_weights[mask]
            out.index_add_(0, token_ids[mask], expert_out)

        return out


class MoEBlock(nn.Module):
    """Pre-LayerNorm transformer block whose FeedForward is a MoE layer."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        hidden_size: int,
        dropout: float = 0.1,
        qkv_bias: bool = False,
        num_experts: int = 8,
        num_experts_per_tok: int = 2,
    ):
        super().__init__()
        self.mha = MultiHeadAttention(embed_dim, num_heads, qkv_bias)
        self.layer_norm_1 = LayerNorm(embed_dim)
        self.layer_norm_2 = LayerNorm(embed_dim)
        self.moe = MoELayer(embed_dim, hidden_size, num_experts, num_experts_per_tok)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attention_out = x + self.dropout(self.mha(self.layer_norm_1(x), mask))
        ff_in = self.layer_norm_2(attention_out)
        ff_out = self.moe(ff_in.reshape(-1, ff_in.size(-1)))
        ff_out = ff_out.reshape(attention_out.shape)
        return attention_out + self.dropout(ff_out)


class MoELLM(BaseLLM):
    """Decoder-only transformer language model with MoE FFN layers."""

    def __init__(self, config: ModelConfig):
        self.moe_aux_loss: torch.Tensor = torch.zeros(())
        super().__init__(config)

    def build_blocks(self, config: ModelConfig) -> nn.ModuleList:
        return nn.ModuleList(
            [
                MoEBlock(
                    config.embedding_dim,
                    config.num_heads,
                    config.hidden_size,
                    config.drop_rate,
                    config.qkv_bias,
                    config.num_experts,
                    config.num_experts_per_tok,
                )
                for _ in range(config.num_layers)
            ]
        )

    def forward(self, x, mask=None):
        logits = super().forward(x, mask)
        self.moe_aux_loss = sum(block.moe.aux_loss for block in self.transformer_blocks)
        return logits
