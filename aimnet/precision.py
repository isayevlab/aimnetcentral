"""Precision policy helpers for AIMNet inference."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from functools import cache
from typing import Literal

import torch
from torch import nn

PrecisionMode = Literal[
    "strict",
    "tf32_learned",
    "bf16_learned",
    "fp16_learned",
    "fp8_learned_experimental",
    "emulated_reductions",
]

Fp8Backend = Literal["none", "torchao", "transformer_engine", "custom"]
Fp8ScalePolicy = Literal["calibrated_static", "delayed", "hysteretic"]


@dataclass(frozen=True)
class PrecisionPolicy:
    mode: PrecisionMode = "strict"
    learned_amp_dtype: torch.dtype | None = None
    allow_tf32_matmul: bool = False
    allow_tf32_cudnn: bool = False
    geometry_dtype: torch.dtype = torch.float32
    reduction_dtype: torch.dtype = torch.float64
    strict_long_range: bool = True
    strict_derivatives: bool = True
    fp8_backend: Fp8Backend = "none"
    fp8_scale_policy: Fp8ScalePolicy = "calibrated_static"
    fp8_freeze_scales_for_md: bool = True
    fallback_on_accuracy_failure: bool = True
    debug_dtype_assertions: bool = False


_PRESETS: dict[str, PrecisionPolicy] = {
    "strict": PrecisionPolicy(),
    "tf32_learned": PrecisionPolicy(
        mode="tf32_learned",
        allow_tf32_matmul=True,
        allow_tf32_cudnn=False,
    ),
    "bf16_learned": PrecisionPolicy(
        mode="bf16_learned",
        learned_amp_dtype=torch.bfloat16,
    ),
    "fp16_learned": PrecisionPolicy(
        mode="fp16_learned",
        learned_amp_dtype=torch.float16,
    ),
    "fp8_learned_experimental": PrecisionPolicy(
        mode="fp8_learned_experimental",
        fp8_backend="none",
    ),
    "emulated_reductions": PrecisionPolicy(mode="emulated_reductions"),
}


def available_precision_modes() -> tuple[str, ...]:
    return tuple(_PRESETS)


def resolve_precision_policy(policy: str | PrecisionPolicy | None) -> PrecisionPolicy:
    """Resolve a public precision argument to a concrete policy."""
    if policy is None:
        return _PRESETS["strict"]
    if isinstance(policy, PrecisionPolicy):
        _validate_policy(policy)
        return policy
    if isinstance(policy, str):
        try:
            return _PRESETS[policy]
        except KeyError as exc:
            choices = ", ".join(available_precision_modes())
            raise ValueError(f"Unknown precision policy '{policy}'. Valid choices: {choices}") from exc
    raise TypeError("precision must be None, a preset string, or PrecisionPolicy")


def _validate_policy(policy: PrecisionPolicy) -> None:
    if policy.mode not in _PRESETS:
        choices = ", ".join(available_precision_modes())
        raise ValueError(f"Unknown precision mode '{policy.mode}'. Valid choices: {choices}")
    if policy.mode == "fp8_learned_experimental" and policy.fp8_backend == "none":
        return
    if policy.learned_amp_dtype not in (None, torch.bfloat16, torch.float16):
        raise ValueError("learned_amp_dtype must be None, torch.bfloat16, or torch.float16")


def supports_tf32(device: torch.device | str | None = None) -> bool:
    if not torch.cuda.is_available():
        return False
    dev = _cuda_device(device)
    major, _minor = torch.cuda.get_device_capability(dev)
    return major >= 8


def supports_bf16(device: torch.device | str | None = None) -> bool:
    if not torch.cuda.is_available():
        return False
    dev = _cuda_device(device)
    try:
        return bool(torch.cuda.is_bf16_supported(including_emulation=False))
    except TypeError:
        return bool(torch.cuda.is_bf16_supported())
    except AttributeError:
        major, _minor = torch.cuda.get_device_capability(dev)
        return major >= 8


def supports_fp8(device: torch.device | str | None = None, backend: Fp8Backend = "none") -> bool:
    if backend == "none" or not torch.cuda.is_available():
        return False
    dev = _cuda_device(device)
    if backend == "custom":
        return _scaled_mm_fp8_probe(dev.index if dev.index is not None else torch.cuda.current_device())
    if backend == "torchao":
        return _module_available("torchao") and _scaled_mm_fp8_probe(
            dev.index if dev.index is not None else torch.cuda.current_device()
        )
    if backend == "transformer_engine":
        major, _minor = torch.cuda.get_device_capability(dev)
        if major < 9:
            return False
        return _module_available("transformer_engine")
    return False


def apply_model_precision_policy(model: nn.Module, policy: PrecisionPolicy, device: str | torch.device) -> nn.Module:
    """Apply opt-in model transformations that cannot be represented as contexts."""
    if policy.mode != "fp8_learned_experimental" or policy.fp8_backend == "none":
        return model
    if isinstance(model, torch.jit.ScriptModule):
        raise TypeError("FP8 experimental mode is not supported for TorchScript models.")
    if policy.fp8_backend != "torchao":
        raise RuntimeError(f"FP8 backend '{policy.fp8_backend}' is not implemented in AIMNet.")
    if not supports_fp8(device, "torchao"):
        raise RuntimeError("torchao FP8 is not supported on this device/backend combination.")

    quantization = __import__("torchao.quantization", fromlist=["Float8DynamicActivationFloat8WeightConfig", "quantize_"])
    config = quantization.Float8DynamicActivationFloat8WeightConfig()
    quantization.quantize_(model, config, device=str(torch.device(device)))
    return model


def _cuda_device(device: torch.device | str | None) -> torch.device:
    if device is None:
        return torch.device("cuda", torch.cuda.current_device())
    dev = torch.device(device)
    if dev.type != "cuda":
        return torch.device("cuda", torch.cuda.current_device())
    return dev


def _module_available(name: str) -> bool:
    try:
        __import__(name)
    except ImportError:
        return False
    return True


@cache
def _scaled_mm_fp8_probe(device_index: int) -> bool:
    if not hasattr(torch, "_scaled_mm"):
        return False
    try:
        device = torch.device("cuda", device_index)
        a = torch.randn((16, 16), device=device).to(torch.float8_e4m3fn)
        b = torch.randn((16, 16), device=device).to(torch.float8_e4m3fn)
        scale = torch.tensor(1.0, device=device)
        torch._scaled_mm(a, b.t(), scale, scale, out_dtype=torch.float32)  # type: ignore[attr-defined]
        torch.cuda.synchronize(device)
    except Exception:
        return False
    return True


@contextmanager
def torch_precision_context(policy: PrecisionPolicy) -> Iterator[None]:
    """Temporarily apply process-global PyTorch precision controls."""
    state = _snapshot_backend_precision()
    try:
        if policy.allow_tf32_matmul:
            torch.set_float32_matmul_precision("high")
            _set_attr_if_present(torch.backends.cuda.matmul, "allow_tf32", True)
            _set_attr_if_present(torch.backends.cuda.matmul, "fp32_precision", "tf32")
        else:
            torch.set_float32_matmul_precision("highest")
            _set_attr_if_present(torch.backends.cuda.matmul, "allow_tf32", False)
            _set_attr_if_present(torch.backends.cuda.matmul, "fp32_precision", "ieee")

        if policy.allow_tf32_cudnn:
            _set_attr_if_present(torch.backends.cudnn, "allow_tf32", True)
            _set_attr_if_present(torch.backends.cudnn, "fp32_precision", "tf32")
        else:
            _set_attr_if_present(torch.backends.cudnn, "allow_tf32", False)
            _set_attr_if_present(torch.backends.cudnn, "fp32_precision", "ieee")
        yield
    finally:
        _restore_backend_precision(state)


@contextmanager
def learned_autocast_context(policy: PrecisionPolicy | None, device_type: str) -> Iterator[None]:
    """Autocast context for whitelisted learned modules only."""
    if policy is not None and device_type == "cuda" and policy.learned_amp_dtype is not None:
        with torch.amp.autocast("cuda", dtype=policy.learned_amp_dtype):
            yield
    else:
        with nullcontext():
            yield


def _snapshot_backend_precision() -> dict[str, object]:
    state: dict[str, object] = {"float32_matmul_precision": torch.get_float32_matmul_precision()}
    for key, obj, attr in _backend_attrs():
        if hasattr(obj, attr):
            state[key] = getattr(obj, attr)
    return state


def _restore_backend_precision(state: dict[str, object]) -> None:
    torch.set_float32_matmul_precision(str(state["float32_matmul_precision"]))
    for key, obj, attr in _backend_attrs():
        if key in state:
            setattr(obj, attr, state[key])


def _backend_attrs() -> tuple[tuple[str, object, str], ...]:
    return (
        ("cuda_matmul_allow_tf32", torch.backends.cuda.matmul, "allow_tf32"),
        ("cuda_matmul_fp32_precision", torch.backends.cuda.matmul, "fp32_precision"),
        ("cudnn_allow_tf32", torch.backends.cudnn, "allow_tf32"),
        ("cudnn_fp32_precision", torch.backends.cudnn, "fp32_precision"),
    )


def _set_attr_if_present(obj: object, attr: str, value: object) -> None:
    if hasattr(obj, attr):
        setattr(obj, attr, value)
