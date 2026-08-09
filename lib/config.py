"""Typed configuration objects for model, training, and generation.

All persisted configs (``config.json``) round-trip through these dataclasses
so there is a single source of truth for every hyperparameter in the project.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, fields
from typing import Any, Literal

Architecture = Literal["dense", "moe"]


@dataclass
class ModelConfig:
    """Architecture hyperparameters needed to rebuild a model from disk.

    MoE-only fields (``num_experts``, ``num_experts_per_tok``,
    ``moe_aux_loss_coeff``) are ignored when ``arch == "dense"`` so a single
    config can describe either architecture without extra bookkeeping.
    """

    arch: Architecture = "dense"
    max_sequence_length: int = 4096
    vocab_size: int = 50257
    embedding_dim: int = 1024
    hidden_size: int = 4096
    num_heads: int = 16
    num_layers: int = 24
    drop_rate: float = 0.1
    qkv_bias: bool = False

    # MoE-only (ignored when arch == "dense").
    num_experts: int = 8
    num_experts_per_tok: int = 2
    moe_aux_loss_coeff: float = 0.01

    @property
    def is_moe(self) -> bool:
        return self.arch == "moe"

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ModelConfig":
        """Build a config from a dict, tolerating missing MoE fields.

        A config.json produced before MoE existed simply omits ``arch`` and the
        MoE fields, so this defaults them and still loads old dense checkpoints.
        """
        if not data:
            return cls()
        return cls(**{k: v for k, v in data.items() if v is not None})

    def to_dict(self) -> dict[str, Any]:
        """Flat dict form used for persistence in config.json."""
        return dataclass_dict(self)

    @classmethod
    def from_json(cls, path: str | None) -> "ModelConfig":
        """Load a config.json file, defaulting when missing or invalid."""
        if path is None or not os.path.exists(path):
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


@dataclass
class TrainingConfig:
    """Hyperparameters controlling the training loop."""

    batch_size: int = 32
    block_size: int = 256
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 8
    test_ratio: float = 0.1
    grad_clip_norm: float = 1.0
    num_workers: int = 4
    pin_memory: bool = True
    seed: int = 42

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "TrainingConfig":
        if not data:
            return cls()
        return cls(**{k: v for k, v in data.items() if v is not None})

    def to_dict(self) -> dict[str, Any]:
        return dataclass_dict(self)


@dataclass
class GenerationConfig:
    """Hyperparameters used for text generation at inference time."""

    generate_max_tokens: int = 64
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
    repetition_penalty: float = 1.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "GenerationConfig":
        if not data:
            return cls()
        return cls(**{k: v for k, v in data.items() if v is not None})

    def to_dict(self) -> dict[str, Any]:
        return dataclass_dict(self)


def dataclass_dict(instance: Any) -> dict[str, Any]:
    """Serialize a dataclass to a dict, skipping fields left at their defaults.

    Keeps persisted config.json minimal (no trailing ``null`` values and no
    MoE fields in dense configs) while remaining lossless on reload.
    """
    return {
        f.name: getattr(instance, f.name)
        for f in fields(instance)
        if getattr(instance, f.name) is not None
    }
