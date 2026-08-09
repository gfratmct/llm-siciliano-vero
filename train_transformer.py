"""Train the transformer language model (dense or Mixture-of-Experts).

The CLI controls every hyperparameter. Model architecture (``--arch dense|moe``
plus MoE options), training, and generation settings each live in their own
typed config in :mod:`lib.config`, and the architecture config is persisted as
``config.json`` so inference can rebuild the exact same model.
"""

import argparse
import json
import os
import time

import torch
from safetensors.torch import save_file
from torch import nn
from torch.optim import AdamW

import wandb

from lib.config import GenerationConfig, ModelConfig, TrainingConfig
from lib.dataset import DatasetReader, create_dataloaders, load_text_datasets
from lib.models import build_model
from lib.tokenizer import Tokenizer
from lib.training import evaluate_model, generate_text, train_epoch
from lib.utils import (
    get_amp_config,
    get_device,
    make_lr_lambda,
    safe_state_dict,
    set_seed,
)

# ---------------------------------------------------------------------------
# Defaults (overridable via CLI; also the fallbacks for config.json missing fields)
# ---------------------------------------------------------------------------
DEFAULTS = {
    # Model
    "max_sequence_length": 4096,
    "vocab_size": 50257,
    "embedding_dim": 1024,
    "num_heads": 16,
    "num_layers": 24,
    "drop_rate": 0.1,
    "qkv_bias": False,
    # MoE (used only when arch == "moe")
    "num_experts": 8,
    "num_experts_per_tok": 2,
    "moe_aux_loss_coeff": 0.01,
    # Training
    "batch_size": 32,
    "block_size": 256,
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
    "num_epochs": 8,
    "test_ratio": 0.1,
    "grad_clip_norm": 1.0,
    "num_workers": 4,
    "pin_memory": True,
    "seed": 42,
    # Inference (used for the end-of-training sample generation)
    "generate_max_tokens": 64,
    "temperature": 1.0,
    "top_k": 50,
    # Paths & logging
    "data_dir": "data/",
    "checkpoint_dir": "models",
    "cache_dir": "cache",
    "save_every_epochs": 1,
    "wandb_project": "llm-siciliano-vero",
    "wandb_name": None,
    "wandb_entity": "gfratmct-personal",
}


def build_configs(args: argparse.Namespace) -> tuple[ModelConfig, TrainingConfig, GenerationConfig]:
    """Construct typed configs from parsed CLI arguments."""
    model_config = ModelConfig.from_dict(
        {
            "arch": args.arch,
            "max_sequence_length": args.max_sequence_length,
            "vocab_size": args.vocab_size,
            "embedding_dim": args.embedding_dim,
            "hidden_size": args.embedding_dim * 4,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
            "drop_rate": args.drop_rate,
            "qkv_bias": args.qkv_bias,
            "num_experts": args.num_experts,
            "num_experts_per_tok": args.num_experts_per_tok,
            "moe_aux_loss_coeff": args.moe_aux_loss_coeff,
        }
    )
    training_config = TrainingConfig.from_dict(
        {
            "batch_size": args.batch_size,
            "block_size": args.block_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "num_epochs": args.num_epochs,
            "test_ratio": args.test_ratio,
            "grad_clip_norm": args.grad_clip_norm,
            "num_workers": args.num_workers,
            "pin_memory": args.pin_memory,
            "seed": args.seed,
        }
    )
    generation_config = GenerationConfig.from_dict(
        {
            "generate_max_tokens": args.generate_max_tokens,
            "temperature": args.temperature,
            "top_k": args.top_k,
        }
    )
    return model_config, training_config, generation_config


def save_model_config(model_config: ModelConfig, checkpoint_dir: str) -> str:
    """Save config.json into the checkpoint directory and return its path."""
    path = os.path.join(checkpoint_dir, "config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(model_config.to_dict(), f, indent=2, ensure_ascii=False)
    print(f"Saved model config: {path}")
    return path


def build_wandb_config(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    generation_config: GenerationConfig,
    save_every_epochs: int,
) -> dict:
    """Collect every hyperparameter into a nested config for W&B."""
    return {
        "model": model_config.to_dict(),
        "training": training_config.to_dict(),
        "inference": generation_config.to_dict(),
        "logging": {"save_every_epochs": save_every_epochs},
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI hyperparameters, defaulting to the module-level constants."""
    parser = argparse.ArgumentParser(
        description="Train the Italian transformer language model (dense or MoE)."
    )

    # Model
    g = parser.add_argument_group("model")
    g.add_argument("--arch", type=str, default="dense", choices=["dense", "moe"])
    g.add_argument("--max-sequence-length", type=int, default=DEFAULTS["max_sequence_length"])
    g.add_argument("--vocab-size", type=int, default=DEFAULTS["vocab_size"])
    g.add_argument("--embedding-dim", type=int, default=DEFAULTS["embedding_dim"])
    g.add_argument("--num-heads", type=int, default=DEFAULTS["num_heads"])
    g.add_argument("--num-layers", type=int, default=DEFAULTS["num_layers"])
    g.add_argument("--drop-rate", type=float, default=DEFAULTS["drop_rate"])
    g.add_argument("--qkv-bias", action=argparse.BooleanOptionalAction, default=DEFAULTS["qkv_bias"])

    # MoE (only meaningful when --arch moe)
    g = parser.add_argument_group("mixture-of-experts")
    g.add_argument("--num-experts", type=int, default=DEFAULTS["num_experts"])
    g.add_argument("--num-experts-per-tok", type=int, default=DEFAULTS["num_experts_per_tok"])
    g.add_argument("--moe-aux-loss-coeff", type=float, default=DEFAULTS["moe_aux_loss_coeff"])

    # Training
    g = parser.add_argument_group("training")
    g.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    g.add_argument("--block-size", type=int, default=DEFAULTS["block_size"])
    g.add_argument("--learning-rate", type=float, default=DEFAULTS["learning_rate"])
    g.add_argument("--weight-decay", type=float, default=DEFAULTS["weight_decay"])
    g.add_argument("--num-epochs", type=int, default=DEFAULTS["num_epochs"])
    g.add_argument("--test-ratio", type=float, default=DEFAULTS["test_ratio"])
    g.add_argument("--grad-clip-norm", type=float, default=DEFAULTS["grad_clip_norm"])
    g.add_argument("--num-workers", type=int, default=DEFAULTS["num_workers"])
    g.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=DEFAULTS["pin_memory"])
    g.add_argument("--seed", type=int, default=DEFAULTS["seed"])

    # Inference (used for the end-of-training sample generation)
    g = parser.add_argument_group("inference")
    g.add_argument("--generate-max-tokens", type=int, default=DEFAULTS["generate_max_tokens"])
    g.add_argument("--temperature", type=float, default=DEFAULTS["temperature"])
    g.add_argument("--top-k", type=int, default=DEFAULTS["top_k"])

    # Paths & logging
    g = parser.add_argument_group("paths & logging")
    g.add_argument("--data-dir", type=str, default=DEFAULTS["data_dir"])
    g.add_argument("--checkpoint-dir", type=str, default=DEFAULTS["checkpoint_dir"])
    g.add_argument("--cache-dir", type=str, default=DEFAULTS["cache_dir"])
    g.add_argument("--save-every-epochs", type=int, default=DEFAULTS["save_every_epochs"])

    # W&B
    g = parser.add_argument_group("weights & biases")
    g.add_argument("--wandb-project", type=str, default=DEFAULTS["wandb_project"])
    g.add_argument("--wandb-name", type=str, default=DEFAULTS["wandb_name"])
    g.add_argument("--wandb-entity", type=str, default=DEFAULTS["wandb_entity"])

    return parser.parse_args()


def main() -> None:
    """Full training entrypoint with dataset preparation, training, validation, and inference."""
    args = parse_args()
    model_config, training_config, generation_config = build_configs(args)

    set_seed(training_config.seed)
    device = get_device()
    print(f"Using device: {device}")
    print(f"Architecture: {model_config.arch}")

    amp_dtype, scaler = get_amp_config(device)
    print(
        f"Mixed precision: {'enabled (' + str(amp_dtype) + ')' if amp_dtype != torch.float32 else 'off'}"
    )

    reader = DatasetReader(args.data_dir)

    tokenizer = Tokenizer()
    print(f"Loaded tokenizer with vocab size {tokenizer.get_vocab_size()}")
    model_config.vocab_size = tokenizer.vocab_size

    # Key the token cache on both the data fingerprint and the tokenizer vocab size,
    # so changing special tokens invalidates the cache automatically.
    cache_path = os.path.join(
        args.cache_dir, f"tokens_{reader.fingerprint()}_v{tokenizer.vocab_size}.pt"
    )

    # Skip the slow read + clean + tokenize steps entirely when the cache is warm.
    if os.path.exists(cache_path):
        corpus = None
        print(f"Token cache hit: {cache_path}")
    else:
        corpus = reader.read()

        if not corpus:
            print("No text data was found. Using a fallback sample corpus.")
            corpus = "This is a sample sentence for dataset loader testing. " * 40

    # Print an example of how raw text is tokenized by the current tokenizer.
    sample_text = "The quick brown fox jumps over the lazy dog."
    sample_encoding = tokenizer.encode(sample_text)
    print("\n--- Tokenizer example ---")
    print(f"Sample text: {sample_text}")
    print(f"Token ids: {sample_encoding.ids}")
    print(f"Decoded text: {tokenizer.decode(sample_encoding.ids)}")
    print("-------------------------\n")

    # Create a dataset of fixed-length token blocks for training and testing.
    train_dataset, test_dataset = load_text_datasets(
        corpus,
        tokenizer,
        block_size=training_config.block_size,
        test_ratio=training_config.test_ratio,
        seed=training_config.seed,
        cache_path=cache_path,
    )

    # Wrap the datasets in DataLoader objects for batching and shuffling.
    train_loader, test_loader = create_dataloaders(
        train_dataset,
        test_dataset,
        batch_size=training_config.batch_size,
        num_workers=training_config.num_workers,
        pin_memory=training_config.pin_memory and device.type == "cuda",
    )

    print(f"Train examples: {len(train_dataset)}")
    print(f"Test examples: {len(test_dataset)}")

    # Free the raw corpus string (only needed to build the token cache).
    del corpus

    model = build_model(model_config).to(device)

    # Persist the model architecture to config.json so inference (app.py) can
    # rebuild the exact same model without relying on hard-coded constants.
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    save_model_config(model_config, args.checkpoint_dir)

    # AdamW is the standard optimizer for transformer training.
    optimizer = AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    # Warmup + cosine decay: 5% linear warmup, then cosine down to 5% of peak LR.
    total_steps = len(train_loader) * training_config.num_epochs
    warmup_steps = max(1, int(0.05 * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, make_lr_lambda(warmup_steps, total_steps)
    )
    # CrossEntropyLoss combines LogSoftmax + NLLLoss in one function.
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")

    # Weights & Biases: log all hyperparameters + train/val loss + LR.
    wandb_run = None
    try:
        wandb_run = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=args.wandb_name,
            config=build_wandb_config(
                model_config, training_config, generation_config, args.save_every_epochs
            ),
        )
        wandb_run.config["num_train_examples"] = len(train_dataset)
        wandb_run.config["num_test_examples"] = len(test_dataset)
        wandb_run.config["num_parameters"] = sum(p.numel() for p in model.parameters())
        wandb.watch(model, log="all", log_freq=100)
        print("W&B logging enabled")
    except Exception as exc:  # keep training running even without W&B auth/network
        print(f"W&B init failed, continuing without logging: {exc}")
        wandb_run = None

    step_counter = [0]

    for epoch in range(1, training_config.num_epochs + 1):
        start_time = time.time()
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            epoch,
            tokenizer,
            amp_dtype,
            scaler,
            training_config,
            scheduler,
            moe_aux_loss_coeff=model_config.moe_aux_loss_coeff,
            wandb_run=wandb_run,
            step_counter=step_counter,
        )
        val_loss = evaluate_model(
            model,
            test_loader,
            criterion,
            device,
            amp_dtype,
            moe_aux_loss_coeff=model_config.moe_aux_loss_coeff,
        )
        epoch_time = time.time() - start_time
        lr_now = (
            scheduler.get_last_lr()[0]
            if scheduler is not None
            else training_config.learning_rate
        )

        print(
            f"Epoch {epoch}/{training_config.num_epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"time={epoch_time:.1f}s"
        )

        if wandb_run is not None:
            wandb_run.log(
                {
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "train/lr": lr_now,
                    "epoch": epoch,
                    "epoch_time_s": epoch_time,
                }
            )

        # Save periodic checkpoints to track progress across training.
        if epoch % args.save_every_epochs == 0:
            checkpoint_path = os.path.join(
                args.checkpoint_dir, f"checkpoint_epoch_{epoch}.safetensors"
            )
            save_file(safe_state_dict(model), checkpoint_path)
            print(f"Saved periodic checkpoint: {checkpoint_path}")

        # Save the best model by validation loss.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(args.checkpoint_dir, "best_model.safetensors")
            save_file(safe_state_dict(model), best_path)
            print(f"Saved best validation checkpoint: {best_path}")
            if wandb_run is not None:
                wandb_run.log({"val/best_loss": best_val_loss, "epoch": epoch})

    # Save final checkpoint after all epochs are complete.
    final_path = os.path.join(args.checkpoint_dir, "final_model.safetensors")
    save_file(safe_state_dict(model), final_path)
    print(f"Saved final model checkpoint: {final_path}")

    prompt = "Ciao come stai"
    generated_text = generate_text(
        model,
        tokenizer,
        prompt,
        device,
        config=generation_config,
    )
    print(f"Prompt: {prompt}")
    print(generated_text)

    if wandb_run is not None:
        wandb_run.log({"val/final_loss": val_loss, "generated_text": generated_text})
        wandb_run.finish()


if __name__ == "__main__":
    main()
