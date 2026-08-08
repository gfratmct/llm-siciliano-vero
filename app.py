import os

import torch
from lib.models import LLM, resize_token_embeddings
from lib.tokenizer import Tokenizer
from safetensors.torch import load_file

# Model hyperparameters
MAX_SEQUENCE_LENGTH = 4096
# Fallback vocab size used only when no checkpoint exists; the actual size
# always comes from the trained tokenizer (see lib/tokenizer.json).
VOCAB_SIZE = 50257
EMBEDDING_DIM = 1024
NUM_HEADS = 16
NUM_LAYERS = 24
DROP_RATE = 0.1
QKV_BIAS = False

# Inference hyperparameters
GENERATE_MAX_TOKENS = 128
TEMPERATURE = 0.7
TOP_K = 50
TOP_P = 0.9
REPETITION_PENALTY = 1.2
MODEL_CHECKPOINT = "models/checkpoint_epoch_1.safetensors"


def get_device() -> torch.device:
    """Return CUDA if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(device: torch.device, vocab_size: int) -> LLM:
    """Construct the transformer model and move it to the compute device."""
    model = LLM(
        embed_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        hidden_size=EMBEDDING_DIM * 4,
        vocab_size=vocab_size,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        dropout=DROP_RATE,
        qkv_bias=QKV_BIAS,
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
    max_new_tokens: int = GENERATE_MAX_TOKENS,
    temperature: float = TEMPERATURE,
    top_k: int = TOP_K,
    top_p: float = TOP_P,
    repetition_penalty: float = REPETITION_PENALTY,
    device: torch.device = torch.device("cpu"),
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


def main() -> None:
    """Inference-only entrypoint for generating text from a saved model."""
    device = get_device()
    print(f"Using device: {device}")

    tokenizer = Tokenizer()
    print(f"Loaded tokenizer with vocab size {tokenizer.get_vocab_size()}")

    # Load the checkpoint (if any) and read its embedding size so the model is
    # built with a matching vocabulary. This works for both old GPT-2-sized
    # checkpoints and for new ones trained with the custom tokenizer.
    state_dict = load_state_dict(MODEL_CHECKPOINT)
    ckpt_vocab_size = (
        state_dict["embedding.weight"].shape[0] if state_dict is not None else tokenizer.vocab_size
    )

    model = build_model(device, vocab_size=ckpt_vocab_size)
    if state_dict is not None:
        model.load_state_dict(state_dict)
        model.eval()

    # Align the model vocabulary with the tokenizer: resize is a no-op when the
    # checkpoint was already trained with the current tokenizer.
    resize_token_embeddings(model, tokenizer.vocab_size)
    print(f"Model vocab size: {model.embedding.num_embeddings}")

    prompt = "Ciao sono Gabriele e nella vita faccio lo scritto, sta mattina "
    print("Prompt:", prompt)

    generated_text = generate_text(
        model,
        tokenizer,
        prompt,
        max_new_tokens=GENERATE_MAX_TOKENS,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
        repetition_penalty=REPETITION_PENALTY,
        device=device,
    )
    print("Generated:", generated_text)


if __name__ == "__main__":
    main()
