"""Inference-only entrypoint: load a saved model (dense or MoE) and generate text.

The architecture and all model hyperparameters come from ``config.json`` next
to the checkpoint; generation settings come from CLI args (with defaults).
"""

import argparse
import os

import torch

from lib.config import GenerationConfig, ModelConfig
from lib.models import load_model_from_checkpoint
from lib.tokenizer import Tokenizer
from lib.training import generate_text
from lib.utils import get_device

# Defaults used only when a config.json is missing or a field is absent.
DEFAULTS = {
    "max_sequence_length": 4096,
    "vocab_size": 50257,
    "embedding_dim": 1024,
    "num_heads": 16,
    "num_layers": 24,
    "drop_rate": 0.1,
    "qkv_bias": False,
    "model_checkpoint": "models/best_model.safetensors",
}


def parse_args() -> argparse.Namespace:
    """Parse CLI args; explicit args override config.json values."""
    parser = argparse.ArgumentParser(
        description="Inference for the Italian transformer LLM (dense or MoE)."
    )

    parser.add_argument(
        "--config", type=str, default=None, help="Path to config.json from training."
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to the .safetensors checkpoint."
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Path to tokenizer.json (default: models/tokenizer.json).",
    )
    parser.add_argument("--prompt", type=str, default=None, help="Prompt text for generation.")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)

    # Model architecture overrides (rarely needed; config.json is the source of truth)
    parser.add_argument("--embedding-dim", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--max-sequence-length", type=int, default=None)
    parser.add_argument("--drop-rate", type=float, default=None)
    parser.add_argument("--qkv-bias", action=argparse.BooleanOptionalAction, default=None)

    return parser.parse_args()


def resolve_config_path(args: argparse.Namespace) -> str | None:
    """Prefer --config, then config.json next to the checkpoint."""
    if args.config is not None:
        return args.config
    checkpoint = args.checkpoint or DEFAULTS["model_checkpoint"]
    candidate = os.path.join(os.path.dirname(os.path.abspath(checkpoint)), "config.json")
    return candidate if os.path.exists(candidate) else None


def main() -> None:
    """Inference-only entrypoint for generating text from a saved model."""
    args = parse_args()
    config_path = resolve_config_path(args)

    model_config = ModelConfig.from_json(config_path)
    print(f"Loaded config from: {config_path or 'defaults'}")
    print(f"Architecture: {model_config.arch}")

    # Apply explicit CLI overrides on top of the config.
    for key, value in (
        ("embedding_dim", args.embedding_dim),
        ("num_heads", args.num_heads),
        ("num_layers", args.num_layers),
        ("max_sequence_length", args.max_sequence_length),
        ("drop_rate", args.drop_rate),
        ("qkv_bias", args.qkv_bias),
    ):
        if value is not None:
            setattr(model_config, key, value)

    generation_config = GenerationConfig.from_dict(
        {
            "generate_max_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
        }
    )

    device = get_device()
    print(f"Using device: {device}")

    tokenizer = Tokenizer(tokenizer_path=args.tokenizer)
    print(f"Loaded tokenizer with vocab size {tokenizer.get_vocab_size()}")

    checkpoint = args.checkpoint or DEFAULTS["model_checkpoint"]
    model = load_model_from_checkpoint(
        model_config,
        checkpoint,
        device,
        tokenizer_vocab_size=tokenizer.vocab_size,
    )
    print(f"Model vocab size: {model.embedding.num_embeddings}")

    prompt = args.prompt or "Ciao sono Gabriele e nella vita faccio lo scritto, sta mattina "
    print("Prompt:", prompt)

    generated_text = generate_text(model, tokenizer, prompt, device, generation_config)
    print("Generated:", generated_text)


if __name__ == "__main__":
    main()
