"""Training, evaluation, and generation helpers shared by train/inference.

Generic utilities (device selection, seeding, AMP config, LR schedule, safe
state-dict serialization) live in :mod:`lib.utils`; this module re-exports them
so ``from lib.training import ...`` keeps working, but the implementations are
defined exactly once.
"""

import torch
import torch.nn as nn

from .config import GenerationConfig, TrainingConfig
from .tokenizer import Tokenizer
from .utils import (
    get_amp_config,
    get_device,
    make_lr_lambda,
    safe_state_dict,
    set_seed,
)

__all__ = [
    "GenerationConfig",
    "TrainingConfig",
    "Tokenizer",
    "compute_loss",
    "evaluate_model",
    "generate_text",
    "get_amp_config",
    "get_device",
    "make_lr_lambda",
    "safe_state_dict",
    "set_seed",
    "train_epoch",
]


def compute_loss(
    model: nn.Module,
    criterion: nn.Module,
    logits: torch.Tensor,
    targets: torch.Tensor,
    moe_aux_loss_coeff: float,
) -> torch.Tensor:
    """Next-token loss, plus the MoE load-balancing loss when present.

    The MoE aux loss is stored on the model by its forward pass; dense models
    never set it, so this reduces to plain cross-entropy.
    """
    loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
    aux_loss = getattr(model, "moe_aux_loss", None)
    if aux_loss is not None and moe_aux_loss_coeff > 0:
        loss = loss + moe_aux_loss_coeff * aux_loss
    return loss


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
    config: TrainingConfig,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None = None,
    moe_aux_loss_coeff: float = 0.0,
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
        with torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=use_amp
        ):
            logits = model(x)
            # Targets are pre-shifted in the dataset, so each logit row directly
            # predicts the next token without any extra copies in the loop.
            loss = compute_loss(
                model, criterion, logits, y, moe_aux_loss_coeff
            )

        # Backpropagate gradients and update model weights.
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.grad_clip_norm
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.grad_clip_norm
            )
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
            print(
                f"Epoch {epoch} | Batch {batch_idx + 1}/{len(dataloader)} "
                f"| Loss {loss.item():.4f} | GradNorm {grad_norm:.4f}"
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/batch_loss": loss.item(),
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "train/grad_norm": grad_norm,
                        "step": (
                            step_counter[0]
                            if step_counter is not None
                            else batch_idx
                        ),
                    }
                )

    return total_loss / max(1, num_batches)


def evaluate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_dtype: torch.dtype,
    moe_aux_loss_coeff: float = 0.0,
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

            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=use_amp
            ):
                logits = model(x)
                loss = compute_loss(
                    model, criterion, logits, y, moe_aux_loss_coeff
                )

            total_loss += loss.item()
            num_batches += 1

    return total_loss / max(1, num_batches)


def generate_text(
    model: nn.Module,
    tokenizer: Tokenizer,
    prompt: str,
    device: torch.device,
    config: GenerationConfig = GenerationConfig(),
) -> str:
    """Generate text from a prompt using top-k, top-p, and repetition penalty."""
    model.eval()
    encoded = tokenizer.encode(prompt)
    input_ids = torch.tensor(encoded.ids, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        for _ in range(config.generate_max_tokens):
            logits = model(input_ids)
            next_token_logits = logits[:, -1, :] / max(config.temperature, 1e-6)

            if config.repetition_penalty != 1.0:
                for token_id in set(input_ids.squeeze(0).tolist()):
                    if next_token_logits[0, token_id] > 0:
                        next_token_logits[0, token_id] /= config.repetition_penalty
                    else:
                        next_token_logits[0, token_id] *= config.repetition_penalty

            probs = torch.softmax(next_token_logits, dim=-1)

            if config.top_k is not None and config.top_k > 0:
                probs, top_indices = torch.topk(
                    probs, min(config.top_k, probs.size(-1)), dim=-1
                )
            else:
                top_indices = torch.arange(
                    probs.size(-1), device=device
                ).unsqueeze(0)

            if config.top_p is not None and 0.0 < config.top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(
                    probs, descending=True, dim=-1
                )
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > config.top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = False
                probs = probs.masked_fill(sorted_indices_to_remove, 0.0)

            probs = probs / probs.sum(dim=-1, keepdim=True)
            next_token_idx = torch.multinomial(probs.squeeze(0), num_samples=1)
            next_token = top_indices.squeeze(0)[next_token_idx]

            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

    generated_ids = input_ids.squeeze(0).tolist()
    return tokenizer.decode(generated_ids)
