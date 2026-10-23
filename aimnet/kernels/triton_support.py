"""Hardware gate for the optional Triton inference kernels.

The Triton AEV/ConvSV/geometry kernels are validated on NVIDIA GPUs with compute capability 8.0 or newer only.
Older GPUs and ROCm builds keep the equivalent PyTorch expressions, exactly as when Triton is not installed.
"""

from __future__ import annotations

import functools
import importlib.util

import torch

MIN_COMPUTE_CAPABILITY = (8, 0)


@functools.cache
def _unsupported_reason(index: int) -> str | None:
    if getattr(torch.version, "hip", None):
        return "ROCm builds are not supported"
    if not torch.cuda.is_available():
        return "CUDA is not available"
    capability = tuple(torch.cuda.get_device_capability(index))
    if capability < MIN_COMPUTE_CAPABILITY:
        major, minor = MIN_COMPUTE_CAPABILITY
        return f"compute capability {capability[0]}.{capability[1]} is below the required {major}.{minor}"
    if importlib.util.find_spec("triton") is None:
        return "triton is not installed"
    return None


def triton_unsupported_reason(device: torch.device | str | int | None = None) -> str | None:
    """Why the Triton inference kernels cannot run on ``device`` (a CUDA device), or None when they can."""
    if device is None or isinstance(device, int):
        index = device
    else:
        device = torch.device(device)
        if device.type != "cuda":
            return f"device {device} is not a CUDA device"
        index = device.index
    if index is None:
        if not torch.cuda.is_available():
            return "CUDA is not available"
        index = torch.cuda.current_device()
    return _unsupported_reason(int(index))


def triton_kernels_supported(device: torch.device) -> bool:
    """Cheap per-call check used at the Triton dispatch sites (cached per device index)."""
    return triton_unsupported_reason(device) is None
