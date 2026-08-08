import math
import os
import time

import torch
from safetensors.torch import save_file
from torch import nn
from torch.optim import AdamW

import wandb

from lib.dataset import DatasetReader, load_text_datasets, create_dataloaders
from lib.models import LLM
from tokenizers import Tokenizer


def safe_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return a state-dict with shared tensors cloned so safetensors can save it.

    We tie ``embedding.weight`` to ``fc_out.weight`` for stability, but
    ``safetensors.save_file`` refuses to serialize tensors that share memory.
    This helper clones only the duplicates.
    """
    state_dict = model.state_dict()
    seen_ptrs: set[int] = set()
    result: dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        ptr = tensor.data_ptr()
        if ptr in seen_ptrs:
            result[key] = tensor.clone()
        else:
            seen_ptrs.add(ptr)
            result[key] = tensor
    return result


# Model hyperparameters
# MAX_SEQUENCE_LENGTH: maximum input length the model can handle in one forward pass.
#   If your corpus contains longer sequences, they will be truncated or split.
#   This is the positional-encoding ceiling and must be >= BLOCK_SIZE.
MAX_SEQUENCE_LENGTH = 4096
# VOCAB_SIZE: GPT-2 tokenizer vocabulary size used by the model embedding layer.
#   The embedding matrix must match the tokenizer vocabulary.
VOCAB_SIZE = 50257
# EMBEDDING_DIM: hidden dimension size for token embeddings and transformer layers.
#   Larger dims allow more expressive representations but increase compute.
EMBEDDING_DIM = 1024
# NUM_HEADS: number of attention heads in multi-head self-attention.
#   More heads let the model learn multiple attention patterns in parallel.
#   Must divide EMBEDDING_DIM evenly (1024 / 16 = 64).
NUM_HEADS = 16
# NUM_LAYERS: number of transformer blocks in the model.
#   More layers generally improve capacity at the cost of training time.
NUM_LAYERS = 24
# DROP_RATE: dropout probability for regularization.
#   Dropout helps reduce overfitting by randomly dropping network activations.
DROP_RATE = 0.1
# QKV_BIAS: whether query/key/value projection layers include a bias term.
#   Most transformer variants disable QKV bias for stable scaling behavior.
QKV_BIAS = False

# Training hyperparameters
# BATCH_SIZE: number of examples processed before each optimizer update.
#   Larger batches use memory faster but give smoother gradient estimates.
#   32 x 128 tokens/step = 4096 tokens per update.
BATCH_SIZE = 32
# BLOCK_SIZE: sequence length of each training example in tokens.
#   The model learns from blocks of this length at a time.
BLOCK_SIZE = 256
# LEARNING_RATE: step size used by the optimizer to update weights.
#   Lowered from 3e-4 to reduce the risk of loss spikes/divergence on
#   a 24-layer model trained from scratch with batch size 32.
LEARNING_RATE = 1e-4
# WEIGHT_DECAY: L2 regularization strength to prevent overfitting.
#   Small weight decay helps keep model weights from growing too large.
WEIGHT_DECAY = 0.01
# NUM_EPOCHS: number of full passes over the training data.
#   The corpus yields ~58k steps/epoch, so even 2 epochs is ~116k optimizer steps.
NUM_EPOCHS = 2
# TEST_RATIO: fraction of the dataset held out for validation.
#   Validation loss measures generalization and prevents overfitting.
TEST_RATIO = 0.1
# GRAD_CLIP_NORM: maximum gradient norm for gradient clipping.
#   Clipping keeps gradient updates stable on deep transformer models.
GRAD_CLIP_NORM = 1.0
# NUM_WORKERS: number of subprocesses used by DataLoader for prefetching.
#   On Linux/CUDA this parallelizes data loading; set 0 if you hit issues.
NUM_WORKERS = 4
# PIN_MEMORY: use page-locked host memory for faster GPU transfers (CUDA only).
PIN_MEMORY = True

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
CHECKPOINT_DIR = "/workspace/runs"
# CACHE_DIR: directory where the pre-tokenized corpus is cached to disk.
#   Tokenizing the full corpus takes minutes, so a cache makes reruns instant.
CACHE_DIR = "cache"
# SAVE_EVERY_EPOCHS: save a checkpoint every N epochs during training.
SAVE_EVERY_EPOCHS = 1
# WANDB_PROJECT: Weights & Biases project name for experiment tracking.
WANDB_PROJECT = "llm-siciliano-vero"
# WANDB_NAME: run name shown in the W&B dashboard (set None to auto-generate).
WANDB_NAME = None
# WANDB_ENTITY
WANDB_ENTITY = "gfratmct-personal"

def build_wandb_config() -> dict:
    """Collect every hyperparameter into a nested config for W&B."""
    return {
        "model": {
            "max_sequence_length": MAX_SEQUENCE_LENGTH,
            "vocab_size": VOCAB_SIZE,
            "embedding_dim": EMBEDDING_DIM,
            "num_heads": NUM_HEADS,
            "num_layers": NUM_LAYERS,
            "drop_rate": DROP_RATE,
            "qkv_bias": QKV_BIAS,
        },
        "training": {
            "batch_size": BATCH_SIZE,
            "block_size": BLOCK_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "num_epochs": NUM_EPOCHS,
            "test_ratio": TEST_RATIO,
            "grad_clip_norm": GRAD_CLIP_NORM,
            "num_workers": NUM_WORKERS,
            "seed": SEED,
        },
        "inference": {
            "generate_max_tokens": GENERATE_MAX_TOKENS,
            "temperature": TEMPERATURE,
            "top_k": TOP_K,
        },
        "logging": {
            "save_every_epochs": SAVE_EVERY_EPOCHS,
        },
    }


def set_seed(seed: int = 42):
    """Set random seed for reproducible training and evaluation."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    """Return the device to use for training and inference."""
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")


def get_amp_config(device: torch.device) -> tuple[torch.dtype, torch.amp.GradScaler | None]:
    """Return (autocast dtype, grad scaler) for mixed-precision training."""
    if device.type == "cuda":
        # bfloat16 is fast on RTX 30xx+ GPUs and needs no gradient scaling.
        return torch.bfloat16, None
    if device.type == "mps":
        # Apple Silicon: float16 with gradient scaling.
        return torch.float16, torch.amp.GradScaler("mps")
    return torch.float32, None


def make_lr_lambda(warmup_steps: int, total_steps: int):
    """Linear warmup, then cosine decay to 5% of the peak learning rate."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0))) + 0.05
    return lr_lambda


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    tokenizer: Tokenizer,
    amp_dtype: torch.dtype,
    scaler: torch.amp.GradScaler | None,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None = None,
    wandb_run=None,
    step_counter: list[int] | None = None,
) -> float:
    """Train the model for one epoch and return the average loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    use_amp = amp_dtype != torch.float32

    for batch_idx, batch in enumerate(dataloader):
        x, y = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # Forward pass: compute raw logits for each token in each sequence.
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits = model(x)
            # Targets are pre-shifted in the dataset, so each logit row directly
            # predicts the next token without any extra copies in the loop.
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))

        # Backpropagate gradients and update model weights.
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        num_batches += 1
        if step_counter is not None:
            step_counter[0] += 1

        # Print detailed debug information for the first batch of each epoch.
        if batch_idx == 0:
            first_input_ids = x[0].tolist()
            first_target_ids = y[0].tolist()
            print("\n--- Debug batch 0 ---")
            print(f"Batch shape: {x.shape}")
            print(f"First example input ids: {first_input_ids}")
            print(f"First example input text: {tokenizer.decode(first_input_ids)}")
            print(f"First example target ids: {first_target_ids}")
            print(f"First example target text: {tokenizer.decode(first_target_ids)}")
            print(f"Loss for first batch: {loss.item():.4f}")
            print("---------------------\n")

        if (batch_idx + 1) % 20 == 0:
            print(f"Epoch {epoch} | Batch {batch_idx + 1}/{len(dataloader)} | Loss {loss.item():.4f} | GradNorm {grad_norm:.4f}")
            if wandb_run is not None:
                wandb_run.log({
                    "train/batch_loss": loss.item(),
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/grad_norm": grad_norm,
                    "step": step_counter[0] if step_counter is not None else batch_idx,
                })

    return total_loss / max(1, num_batches)


def evaluate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> float:
    """Evaluate the model on the validation set and return the average loss."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    use_amp = amp_dtype != torch.float32

    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(x)
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))

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

    amp_dtype, scaler = get_amp_config(device)
    print(f"Mixed precision: {'enabled (' + str(amp_dtype) + ')' if amp_dtype != torch.float32 else 'off'}")

    reader = DatasetReader("data/")
    cache_path = os.path.join(CACHE_DIR, f"tokens_{reader.fingerprint()}.pt")

    # Skip the slow read + clean + tokenize steps entirely when the cache is warm.
    if os.path.exists(cache_path):
        corpus = None
        print(f"Token cache hit: {cache_path}")
    else:
        corpus = reader.read()

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
        cache_path=cache_path,
    )

    # Wrap the datasets in DataLoader objects for batching and shuffling.
    train_loader, test_loader = create_dataloaders(
        train_dataset,
        test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY and device.type == "cuda",
    )

    print(f"Train examples: {len(train_dataset)}")
    print(f"Test examples: {len(test_dataset)}")

    # Free the raw corpus string (only needed to build the token cache).
    del corpus

    model = build_model(device)
    # AdamW is the standard optimizer for transformer training.
    # It decouples weight decay from the gradient update step.
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # Warmup + cosine decay: 5% linear warmup, then cosine down to 5% of peak LR.
    # Longer warmup helps stabilize early training on deep transformers.
    total_steps = len(train_loader) * NUM_EPOCHS
    warmup_steps = max(1, int(0.05 * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, make_lr_lambda(warmup_steps, total_steps))
    # CrossEntropyLoss combines LogSoftmax + NLLLoss in one function.
    # This is the standard loss for next-token prediction.
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Weights & Biases: log all hyperparameters + train/val loss + LR.
    wandb_run = None
    try:
        wandb_run = wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT, name=WANDB_NAME, config=build_wandb_config())
        wandb_run.config["num_train_examples"] = len(train_dataset)
        wandb_run.config["num_test_examples"] = len(test_dataset)
        wandb_run.config["num_parameters"] = sum(p.numel() for p in model.parameters())
        wandb.watch(model, log="all", log_freq=100)
        print("W&B logging enabled")
    except Exception as exc:  # keep training running even without W&B auth/network
        print(f"W&B init failed, continuing without logging: {exc}")
        wandb_run = None

    step_counter = [0]

    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch, tokenizer, amp_dtype, scaler, scheduler, wandb_run, step_counter)
        val_loss = evaluate_model(model, test_loader, criterion, device, amp_dtype)
        epoch_time = time.time() - start_time
        lr_now = scheduler.get_last_lr()[0] if scheduler is not None else LEARNING_RATE

        print(f"Epoch {epoch}/{NUM_EPOCHS} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | time={epoch_time:.1f}s")

        if wandb_run is not None:
            wandb_run.log({
                "train/loss": train_loss,
                "val/loss": val_loss,
                "train/lr": lr_now,
                "epoch": epoch,
                "epoch_time_s": epoch_time,
            })

        # Save periodic checkpoints to track progress across training.
        if epoch % SAVE_EVERY_EPOCHS == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}.safetensors")
            save_file(safe_state_dict(model), checkpoint_path)
            print(f"Saved periodic checkpoint: {checkpoint_path}")

        # Save the best model by validation loss.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(CHECKPOINT_DIR, "best_model.safetensors")
            save_file(safe_state_dict(model), best_path)
            print(f"Saved best validation checkpoint: {best_path}")
            if wandb_run is not None:
                wandb_run.log({"val/best_loss": best_val_loss, "epoch": epoch})

    # Save final checkpoint after all epochs are complete.
    final_path = os.path.join(CHECKPOINT_DIR, "final_model.safetensors")
    save_file(safe_state_dict(model), final_path)
    print(f"Saved final model checkpoint: {final_path}")

    prompt = "Ciao come stai"
    generated_text = generate_text(model, tokenizer, prompt, max_new_tokens=GENERATE_MAX_TOKENS, device=device)
    print(f"Prompt: {prompt}")
    print(generated_text)

    if wandb_run is not None:
        wandb_run.log({"val/final_loss": val_loss, "generated_text": generated_text})
        wandb_run.finish()


if __name__ == "__main__":
    main()
