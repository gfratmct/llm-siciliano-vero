"""Model architectures: dense transformer and Mixture-of-Experts.

Use :func:`build_model` with a :class:`lib.config.ModelConfig` to construct
either architecture. ``arch`` is stored in config.json, so dense checkpoints
trained before MoE existed (no ``arch`` field) load as dense automatically.
"""

from .base import BaseLLM
from .dense import DenseBlock, DenseLLM
from .factory import (
    build_model,
    load_model_from_checkpoint,
    resize_token_embeddings,
)
from .moe import MoEBlock, MoELayer, MoELLM

# Backward-compatible alias: code importing ``LLM`` from lib.models keeps working.
LLM = DenseLLM

__all__ = [
    "BaseLLM",
    "DenseBlock",
    "DenseLLM",
    "LLM",
    "MoEBlock",
    "MoELayer",
    "MoELLM",
    "build_model",
    "load_model_from_checkpoint",
    "resize_token_embeddings",
]
