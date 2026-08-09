"""Generic, dependency-light helpers shared across the whole project.

Anything that is pure PyTorch/OS plumbing (device selection, seeding, mixed
precision, LR schedules, safe checkpoint serialization) lives here so it is
defined exactly once and imported everywhere it is needed.
"""

import math

import torch
import torch.nn as nn


def set_seed(seed: int = 42):
    """Set random seed for reproducible training and evaluation."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    """Return the best available device (CUDA, then MPS, then CPU)."""
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.mps.is_available()
        else "cpu"
    )


def get_amp_config( # need this for my lovely macbook
    device: torch.device,
) -> tuple[torch.dtype, torch.amp.GradScaler | None]:
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
