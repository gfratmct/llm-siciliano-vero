import os

import torch
from lib.models import LLM
from safetensors.torch import load_file
from tokenizers import Tokenizer

# Model hyperparameters
MAX_SEQUENCE_LENGTH = 1024
VOCAB_SIZE = 50257
EMBEDDING_DIM = 768
NUM_HEADS = 12
NUM_LAYERS = 12
DROP_RATE = 0.1
QKV_BIAS = False

# Inference hyperparameters
GENERATE_MAX_TOKENS = 64
TEMPERATURE = 1.0
TOP_K = 50
MODEL_CHECKPOINT = "runs/best_model.safetensors"


def get_device() -> torch.device:
    """Return CUDA if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(device: torch.device) -> LLM:
    """Construct the transformer model and move it to the compute device."""
    model = LLM(
        embed_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        hidden_size=EMBEDDING_DIM * 4,
        vocab_size=VOCAB_SIZE,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        dropout=DROP_RATE,
        qkv_bias=QKV_BIAS,
    )
    return model.to(device)


def load_checkpoint(model: LLM, checkpoint_path: str):
    """Load saved weights if the checkpoint exists."""
    if os.path.exists(checkpoint_path):
        if checkpoint_path.endswith(".safetensors"):
            state_dict = load_file(checkpoint_path)
            model.load_state_dict(state_dict)
        else:
            model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        model.eval()
        print(f"Loaded model weights from {checkpoint_path}")
    else:
        print(f"Checkpoint not found at {checkpoint_path}. Using randomly initialized model.")


def generate_text(
    model: LLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = GENERATE_MAX_TOKENS,
    temperature: float = TEMPERATURE,
    top_k: int = TOP_K,
    device: torch.device = torch.device("cpu"),
) -> str:
    """Generate text from a prompt using top-k sampling."""
    model.eval()
    encoded = tokenizer.encode(prompt)
    input_ids = torch.tensor(encoded.ids, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)
            next_token_logits = logits[:, -1, :] / max(temperature, 1e-6)

            if top_k is not None and top_k > 0:
                top_logits, top_indices = torch.topk(next_token_logits, top_k, dim=-1)
                probs = torch.softmax(top_logits, dim=-1)
                next_token = top_indices.squeeze(0)[torch.multinomial(probs.squeeze(0), num_samples=1)]
            else:
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs.squeeze(0), num_samples=1)

            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

    generated_ids = input_ids.squeeze(0).tolist()
    return tokenizer.decode(generated_ids)


def main() -> None:
    """Inference-only entrypoint for generating text from a saved model."""
    device = get_device()
    print(f"Using device: {device}")

    tokenizer = Tokenizer.from_pretrained("gpt2")
    print(f"Loaded GPT-2 tokenizer with vocab size {tokenizer.get_vocab_size()}")

    model = build_model(device)
    load_checkpoint(model, MODEL_CHECKPOINT)

    prompt = "The science of machine learning"
    print("Prompt:", prompt)

    generated_text = generate_text(model, tokenizer, prompt, max_new_tokens=GENERATE_MAX_TOKENS, temperature=TEMPERATURE, top_k=TOP_K, device=device)
    print("Generated:", generated_text)


if __name__ == "__main__":
    main()
