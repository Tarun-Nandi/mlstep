"""Shared precision settings for neural training and offline evaluation."""

import torch

PRECISIONS = ("auto", "fp32", "bf16", "fp16")


def resolve_device(value: torch.device | str) -> torch.device:
    """Resolve an explicit CPU/CUDA device, including the current CUDA index."""
    device = torch.device(value)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("device must be CPU or CUDA")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        device = torch.device("cuda", torch.cuda.current_device() if device.index is None else device.index)
    return device


def autocast_dtype(device: torch.device, precision: str) -> torch.dtype | None:
    """Choose the autocast dtype without converting stored model parameters.

    None means FP32. CUDA auto prefers BF16 when supported, otherwise FP16;
    the trainer supplies gradient scaling for FP16. Explicit CPU mixed-precision
    requests are rejected so the same CLI setting has a clear device contract.
    """
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}")
    if precision == "fp32":
        return None
    if device.type == "cpu":
        if precision != "auto":
            raise ValueError("mixed precision is supported on CUDA; use auto or fp32 on CPU")
        return None
    with torch.cuda.device(device):
        supports_bf16 = torch.cuda.is_bf16_supported()
    if precision == "bf16" and not supports_bf16:
        raise ValueError("this GPU does not support BF16; choose fp16 or fp32")
    return torch.bfloat16 if precision == "bf16" or (precision == "auto" and supports_bf16) else torch.float16
