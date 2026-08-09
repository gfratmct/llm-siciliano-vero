import argparse
import json
import os

import torch
from lib.models import LLM, resize_token_embeddings
from lib.tokenizer import Tokenizer
from safetensors.torch import load_file

# Default fallbacks used only when config.json is missing or a field is null.
DEFAULTS = {
    "MAX_SEQUENCE_LENGTH": 4096,
    "VOCAB_SIZE": 50257,
    "EMBEDDING_DIM": 1024,
    "HIDDEN_SIZE": 4096,
    "NUM_HEADS": 16,
    "NUM_LAYERS": 24,
    "DROP_RATE": 0.1,
    "QKV_BIAS": False,
    # Inference hyperparameters
    "GENERATE_MAX_TOKENS": 128,
    "TEMPERATURE": 0.7,
    "TOP_K": 50,
    "TOP_P": 0.9,
    "REPETITION_PENALTY": 1.2,
    "MODEL_CHECKPOINT": "models/checkpoint_epoch_2.safetensors",
    "CONFIG_PATH": None,
}

# Model hyperparameters (fallback when config.json is missing or null)
MAX_SEQUENCE_LENGTH = DEFAULTS["MAX_SEQUENCE_LENGTH"]
VOCAB_SIZE = DEFAULTS["VOCAB_SIZE"]
EMBEDDING_DIM = DEFAULTS["EMBEDDING_DIM"]
NUM_HEADS = DEFAULTS["NUM_HEADS"]
NUM_LAYERS = DEFAULTS["NUM_LAYERS"]
DROP_RATE = DEFAULTS["DROP_RATE"]
QKV_BIAS = DEFAULTS["QKV_BIAS"]

# Inference hyperparameters
GENERATE_MAX_TOKENS = DEFAULTS["GENERATE_MAX_TOKENS"]
TEMPERATURE = DEFAULTS["TEMPERATURE"]
TOP_K = DEFAULTS["TOP_K"]
TOP_P = DEFAULTS["TOP_P"]
REPETITION_PENALTY = DEFAULTS["REPETITION_PENALTY"]
MODEL_CHECKPOINT = DEFAULTS["MODEL_CHECKPOINT"]


def get_device() -> torch.device:
    """Return CUDA if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(config_path: str | None) -> dict:
    """Load a model architecture config JSON.

    Only architecture fields are read from the file; inference/generation
    hyperparameters always come from the DEFAULTS (or CLI overrides) since
    they are not part of the persisted config.
    """
    cfg = dict(DEFAULTS)

    if config_path is None:
        return cfg

    if not os.path.exists(config_path):
        print(f"Config not found at {config_path}. Using default constants.")
        return cfg

    with open(config_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    # Only architecture fields are stored in config.json.
    for key, value in loaded.items():
        if value is not None:
            cfg[key.upper()] = value
    if "HIDDEN_SIZE" not in cfg:
        cfg["HIDDEN_SIZE"] = cfg["EMBEDDING_DIM"] * 4
    return cfg


def build_model(device: torch.device, cfg: dict) -> LLM:
    """Construct the transformer model and move it to the compute device."""
    model = LLM(
        embed_dim=cfg["EMBEDDING_DIM"],
        num_heads=cfg["NUM_HEADS"],
        num_layers=cfg["NUM_LAYERS"],
        hidden_size=cfg["HIDDEN_SIZE"],
        vocab_size=cfg["VOCAB_SIZE"],
        max_seq_length=cfg["MAX_SEQUENCE_LENGTH"],
        dropout=cfg["DROP_RATE"],
        qkv_bias=cfg["QKV_BIAS"],
    )
    return model.to(device)


def load_state_dict(checkpoint_path: str) -> dict | None:
    """Load a checkpoint state dict, returning None if the file is missing."""
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}. Using randomly initialized model.")
        return None
    if checkpoint_path.endswith(".safetensors"):
        state_dict = load_file(checkpoint_path)
    else:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
    print(f"Loaded model weights from {checkpoint_path}")
    return state_dict


def generate_text(
    model: LLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    device: torch.device,
) -> str:
    """Generate text from a prompt using top-k, top-p, and repetition penalty."""
    model.eval()
    encoded = tokenizer.encode(prompt)
    input_ids = torch.tensor(encoded.ids, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)
            next_token_logits = logits[:, -1, :] / max(temperature, 1e-6)

            if repetition_penalty is not None and repetition_penalty != 1.0:
                for token_id in set(input_ids.squeeze(0).tolist()):
                    if next_token_logits[0, token_id] > 0:
                        next_token_logits[0, token_id] /= repetition_penalty
                    else:
                        next_token_logits[0, token_id] *= repetition_penalty

            probs = torch.softmax(next_token_logits, dim=-1)

            if top_k is not None and top_k > 0:
                probs, top_indices = torch.topk(probs, min(top_k, probs.size(-1)), dim=-1)
            else:
                top_indices = torch.arange(probs.size(-1), device=device).unsqueeze(0)

            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                probs = probs.masked_fill(sorted_indices_to_remove, 0.0)

            probs = probs / probs.sum(dim=-1, keepdim=True)
            next_token_idx = torch.multinomial(probs.squeeze(0), num_samples=1)
            next_token = top_indices.squeeze(0)[next_token_idx]

            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

    generated_ids = input_ids.squeeze(0).tolist()
    return tokenizer.decode(generated_ids)


def parse_args() -> argparse.Namespace:
    """Parse CLI args; explicit args override config.json values."""
    parser = argparse.ArgumentParser(description="Inference for the Italian transformer LLM.")

    parser.add_argument("--config", type=str, default=None, help="Path to config.json from training.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to the .safetensors checkpoint.")
    parser.add_argument("--tokenizer", type=str, default=None, help="Path to tokenizer.json (default: models/tokenizer.json).")
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


def main() -> None:
    """Inference-only entrypoint for generating text from a saved model."""
    args = parse_args()

    # Resolve the config path: explicit --config, then next to the checkpoint,
    # then next to the default checkpoint path.
    config_path = args.config
    if config_path is None:
        checkpoint = args.checkpoint or MODEL_CHECKPOINT
        candidate = os.path.join(os.path.dirname(os.path.abspath(checkpoint)), "config.json")
        if os.path.exists(candidate):
            config_path = candidate

    cfg = load_config(config_path)
    print(f"Loaded config from: {config_path or 'defaults'}")

    # Apply explicit CLI overrides on top of the config.
    cli_overrides = {
        "MODEL_CHECKPOINT": args.checkpoint,
        "GENERATE_MAX_TOKENS": args.max_new_tokens,
        "TEMPERATURE": args.temperature,
        "TOP_K": args.top_k,
        "TOP_P": args.top_p,
        "REPETITION_PENALTY": args.repetition_penalty,
        "EMBEDDING_DIM": args.embedding_dim,
        "NUM_HEADS": args.num_heads,
        "NUM_LAYERS": args.num_layers,
        "MAX_SEQUENCE_LENGTH": args.max_sequence_length,
        "DROP_RATE": args.drop_rate,
        "QKV_BIAS": args.qkv_bias,
    }
    for key, value in cli_overrides.items():
        if value is not None:
            cfg[key] = value

    device = get_device()
    print(f"Using device: {device}")

    tokenizer = Tokenizer(tokenizer_path=args.tokenizer)
    print(f"Loaded tokenizer with vocab size {tokenizer.get_vocab_size()}")

    # Load the checkpoint (if any) and read its embedding size so the model is
    # built with a matching vocabulary. This works for both old GPT-2-sized
    # checkpoints and for new ones trained with the custom tokenizer.
    state_dict = load_state_dict(cfg["MODEL_CHECKPOINT"])
    ckpt_vocab_size = (
        state_dict["embedding.weight"].shape[0] if state_dict is not None else tokenizer.vocab_size
    )

    # Build the model from the config; the vocab size comes from the checkpoint
    # so load_state_dict always succeeds, then we resize to the tokenizer's vocab.
    cfg["VOCAB_SIZE"] = ckpt_vocab_size
    model = build_model(device, cfg)
    if state_dict is not None:
        model.load_state_dict(state_dict)
        model.eval()

    # Align the model vocabulary with the tokenizer: resize is a no-op when the
    # checkpoint was already trained with the current tokenizer.
    resize_token_embeddings(model, tokenizer.vocab_size)
    print(f"Model vocab size: {model.embedding.num_embeddings}")

    prompt = args.prompt or "Ciao sono Gabriele e nella vita faccio lo scritto, sta mattina "
    print("Prompt:", prompt)

    generated_text = generate_text(
        model,
        tokenizer,
        prompt,
        max_new_tokens=cfg["GENERATE_MAX_TOKENS"],
        temperature=cfg["TEMPERATURE"],
        top_k=cfg["TOP_K"],
        top_p=cfg["TOP_P"],
        repetition_penalty=cfg["REPETITION_PENALTY"],
        device=device,
    )
    print("Generated:", generated_text)


if __name__ == "__main__":
    main()