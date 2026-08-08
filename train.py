import os
import time

import torch
from safetensors.torch import save_file
from torch import nn
from torch.optim import AdamW

from lib.dataset import DatasetReader, load_text_datasets, create_dataloaders
from lib.models import LLM
from tokenizers import Tokenizer

# Model hyperparameters
# MAX_SEQUENCE_LENGTH: maximum input length the model can handle in one forward pass.
#   If your corpus contains longer sequences, they will be truncated or split.
MAX_SEQUENCE_LENGTH = 2048
# VOCAB_SIZE: GPT-2 tokenizer vocabulary size used by the model embedding layer.
#   The embedding matrix must match the tokenizer vocabulary.
VOCAB_SIZE = 50257
# EMBEDDING_DIM: hidden dimension size for token embeddings and transformer layers.
#   Larger dims allow more expressive representations but increase compute.
EMBEDDING_DIM = 1024
# NUM_HEADS: number of attention heads in multi-head self-attention.
#   More heads let the model learn multiple attention patterns in parallel.
NUM_HEADS = 26
# NUM_LAYERS: number of transformer blocks in the model.
#   More layers generally improve capacity at the cost of training time.
NUM_LAYERS = 48
# DROP_RATE: dropout probability for regularization.
#   Dropout helps reduce overfitting by randomly dropping network activations.
DROP_RATE = 0.1
# QKV_BIAS: whether query/key/value projection layers include a bias term.
#   Most transformer variants disable QKV bias for stable scaling behavior.
QKV_BIAS = False

# Training hyperparameters
# BATCH_SIZE: number of examples processed before each optimizer update.
#   Larger batches use memory faster but give smoother gradient estimates.
BATCH_SIZE = 8
# BLOCK_SIZE: sequence length of each training example in tokens.
#   The model learns from blocks of this length at a time.
BLOCK_SIZE = 32
# LEARNING_RATE: step size used by the optimizer to update weights.
#   If too high, training can diverge; if too low, convergence is slow.
LEARNING_RATE = 5e-5
# WEIGHT_DECAY: L2 regularization strength to prevent overfitting.
#   Small weight decay helps keep model weights from growing too large.
WEIGHT_DECAY = 0.01
# NUM_EPOCHS: number of full passes over the training data.
#   More epochs allow the model to fit the data better, up to a point.
NUM_EPOCHS = 1000
# TEST_RATIO: fraction of the dataset held out for validation.
#   Validation loss measures generalization and prevents overfitting.
TEST_RATIO = 0.1
# GRAD_CLIP_NORM: maximum gradient norm for gradient clipping.
#   Clipping keeps gradient updates stable on deep transformer models.
GRAD_CLIP_NORM = 1.0
# NUM_WORKERS: number of subprocesses used by DataLoader.
#   Set to 0 for compatibility; increase for faster loading on large datasets.
NUM_WORKERS = 0

# Inference hyperparameters
# GENERATE_MAX_TOKENS: maximum number of new tokens generated at inference time.
GENERATE_MAX_TOKENS = 64
# TEMPERATURE: controls randomness of sampling; lower values make output more deterministic,
#   while higher values increase diversity.
TEMPERATURE = 1.0
# TOP_K: restricts sampling to the top K probable tokens.
#   This discards low-probability tokens and makes generation more coherent.
TOP_K = 50
# SEED: random seed for reproducibility across runs.
SEED = 42
# CHECKPOINT_DIR: directory where model checkpoints are saved.
CHECKPOINT_DIR = "runs"
# SAVE_EVERY_EPOCHS: save a checkpoint every N epochs during training.
SAVE_EVERY_EPOCHS = 10


def set_seed(seed: int = 42):
    """Set random seed for reproducible training and evaluation."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Return the device to use for training and inference."""
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")


def shift_logits_and_labels(logits: torch.Tensor, labels: torch.Tensor):
    """Prepare logits and labels for causal language modeling loss."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return shift_logits, shift_labels


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    tokenizer: Tokenizer,
) -> float:
    """Train the model for one epoch and return the average loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        batch = batch.to(device)

        # Forward pass: compute raw logits for each token in each sequence.
        optimizer.zero_grad()
        logits = model(batch)

        # Shift the logits and labels so each input token predicts the next token.
        shift_logits, shift_labels = shift_logits_and_labels(logits, batch)

        # Flatten the predictions and labels to compute cross-entropy loss.
        loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # Backpropagate gradients and update model weights.
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # Print detailed debug information for the first batch of each epoch.
        if batch_idx == 0:
            first_input_ids = batch[0].tolist()
            first_target_ids = shift_labels[0].tolist()
            print("\n--- Debug batch 0 ---")
            print(f"Batch shape: {batch.shape}")
            print(f"First example input ids: {first_input_ids}")
            print(f"First example input text: {tokenizer.decode(first_input_ids)}")
            print(f"First example target ids: {first_target_ids}")
            print(f"First example target text: {tokenizer.decode(first_target_ids)}")
            print(f"Loss for first batch: {loss.item():.4f}")
            print("---------------------\n")

        if (batch_idx + 1) % 20 == 0:
            print(f"Epoch {epoch} | Batch {batch_idx + 1}/{len(dataloader)} | Loss {loss.item():.4f}")

    return total_loss / max(1, num_batches)


def evaluate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Evaluate the model on the validation set and return the average loss."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            logits = model(batch)

            shift_logits, shift_labels = shift_logits_and_labels(logits, batch)
            loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            total_loss += loss.item()
            num_batches += 1

    return total_loss / max(1, num_batches)


def generate_text(
    model: nn.Module,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = GENERATE_MAX_TOKENS,
    temperature: float = TEMPERATURE,
    top_k: int = TOP_K,
    device: torch.device = torch.device("cpu"),
) -> str:
    """Generate text from a prompt using simple top-k sampling."""
    model.eval()
    encoded = tokenizer.encode(prompt)
    input_ids = torch.tensor(encoded.ids, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)
            next_token_logits = logits[:, -1, :] / max(temperature, 1e-6)

            if top_k is not None and top_k > 0:
                top_probs, top_indices = torch.topk(next_token_logits, top_k, dim=-1)
                probs = torch.softmax(top_probs, dim=-1)
                next_token = top_indices.squeeze(0)[torch.multinomial(probs.squeeze(0), num_samples=1)]
            else:
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs.squeeze(0), num_samples=1)

            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

    generated_ids = input_ids.squeeze(0).tolist()
    return tokenizer.decode(generated_ids)


def build_model(device: torch.device) -> nn.Module:
    """Construct the transformer language model and move it to the target device."""
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


def main() -> None:
    """Full training entrypoint with dataset preparation, training, validation, and inference."""
    set_seed(SEED)
    device = get_device()
    print(f"Using device: {device}")

    reader = DatasetReader("data/")
    sentences = reader.read()
    corpus = " ".join(sentences)

    if not corpus:
        print("No text data was found. Using a fallback sample corpus.")
        corpus = "This is a sample sentence for dataset loader testing. " * 40

    tokenizer = Tokenizer.from_pretrained("gpt2")
    print(f"Loaded GPT-2 tokenizer with vocab size {tokenizer.get_vocab_size()}")

    # Print an example of how raw text is tokenized by GPT-2 tokenizer.
    sample_text = "The quick brown fox jumps over the lazy dog."
    sample_encoding = tokenizer.encode(sample_text)
    print("\n--- Tokenizer example ---")
    print(f"Sample text: {sample_text}")
    print(f"Token ids: {sample_encoding.ids}")
    print(f"Decoded text: {tokenizer.decode(sample_encoding.ids)}")
    print("-------------------------\n")

    # Create a dataset of fixed-length token blocks for training and testing.
    # The split keeps a small validation set to monitor generalization.
    train_dataset, test_dataset = load_text_datasets(
        corpus,
        tokenizer,
        block_size=BLOCK_SIZE,
        test_ratio=TEST_RATIO,
        seed=SEED,
    )

    # Wrap the datasets in DataLoader objects for batching and shuffling.
    train_loader, test_loader = create_dataloaders(
        train_dataset,
        test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    print(f"Train examples: {len(train_dataset)}")
    print(f"Test examples: {len(test_dataset)}")

    model = build_model(device)
    # AdamW is the standard optimizer for transformer training.
    # It decouples weight decay from the gradient update step.
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # CrossEntropyLoss combines LogSoftmax + NLLLoss in one function.
    # This is the standard loss for next-token prediction.
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch, tokenizer)
        val_loss = evaluate_model(model, test_loader, criterion, device)
        epoch_time = time.time() - start_time

        print(f"Epoch {epoch}/{NUM_EPOCHS} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | time={epoch_time:.1f}s")

        # Save periodic checkpoints to track progress across training.
        if epoch % SAVE_EVERY_EPOCHS == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}.safetensors")
            save_file(model.state_dict(), checkpoint_path)
            print(f"Saved periodic checkpoint: {checkpoint_path}")

        # Save the best model by validation loss.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(CHECKPOINT_DIR, "best_model.safetensors")
            save_file(model.state_dict(), best_path)
            print(f"Saved best validation checkpoint: {best_path}")

    # Save final checkpoint after all epochs are complete.
    final_path = os.path.join(CHECKPOINT_DIR, "final_model.safetensors")
    save_file(model.state_dict(), final_path)
    print(f"Saved final model checkpoint: {final_path}")

    prompt = "Ciao come stai"
    generated_text = generate_text(model, tokenizer, prompt, max_new_tokens=GENERATE_MAX_TOKENS, device=device)
    print(f"Prompt: {prompt}")
    print(generated_text)


if __name__ == "__main__":
    main()
