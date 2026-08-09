"""Model factory: build the right architecture from a ModelConfig."""

import os

import torch
import torch.nn as nn

from ..config import ModelConfig
from .dense import DenseLLM
from .moe import MoELLM

MODEL_ARCHITECTURES = {
    "dense": DenseLLM,
    "moe": MoELLM,
}


def build_model(config: ModelConfig) -> nn.Module:
    """Instantiate the model matching ``config.arch`` (defaults to dense)."""
    cls = MODEL_ARCHITECTURES.get(config.arch, DenseLLM)
    return cls(config)


def resize_token_embeddings(model: nn.Module, new_vocab_size: int) -> None:
    """Resize the model's input embedding and re-tie the output projection.

    Works for both dense and MoE models since embedding and ``fc_out`` are
    shared, architecture-independent components. Initializes new rows with
    GPT-2-style normal noise so added special tokens start from a sane prior.
    """
    old_vocab_size = model.embedding.num_embeddings
    embed_dim = model.embedding.embedding_dim

    if old_vocab_size == new_vocab_size:
        return

    new_embedding = nn.Embedding(new_vocab_size, embed_dim)
    with torch.no_grad():
        new_embedding.weight[:old_vocab_size] = model.embedding.weight
        nn.init.normal_(new_embedding.weight[old_vocab_size:], mean=0.0, std=0.02)

    model.embedding = new_embedding
    model.fc_out.weight = model.embedding.weight


def load_model_from_checkpoint(
    config: ModelConfig,
    checkpoint_path: str | None,
    device: torch.device,
    tokenizer_vocab_size: int | None = None,
) -> nn.Module:
    """Build a model from config, load weights, and align the vocabulary.

    The vocab size is inferred from the checkpoint's embedding (so the state
    dict always loads) and then resized to match the tokenizer. Returns a model
    on ``device`` in eval mode.
    """
    from safetensors.torch import load_file

    state_dict = None
    if checkpoint_path and os.path.exists(checkpoint_path):
        state_dict = load_file(checkpoint_path)
        print(f"Loaded model weights from {checkpoint_path}")

    ckpt_vocab_size = (
        state_dict["embedding.weight"].shape[0] if state_dict is not None else None
    )
    if ckpt_vocab_size is not None:
        config = ModelConfig(**{**config.to_dict(), "vocab_size": ckpt_vocab_size})

    model = build_model(config).to(device)
    if state_dict is not None:
        model.load_state_dict(state_dict)
        model.eval()

    if tokenizer_vocab_size is not None:
        resize_token_embeddings(model, tokenizer_vocab_size)
        model = model.to(device)

    return model
