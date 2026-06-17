"""Lightweight profiling helpers for AIMNet execution paths."""

from __future__ import annotations

import os
from types import TracebackType
from typing import Literal

import torch


class _NullRange:
    def __enter__(self) -> None:
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        return False


class _NvtxRange:
    def __init__(self, name: str):
        self.name = name

    def __enter__(self) -> None:
        torch.cuda.nvtx.range_push(self.name)
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        torch.cuda.nvtx.range_pop()
        return False


_NULL_RANGE = _NullRange()


def nvtx_enabled() -> bool:
    """Return True when AIMNet NVTX annotations should be emitted."""
    value = os.environ.get("AIMNET_NVTX", "")
    return value.lower() in {"1", "true", "yes", "on"} and torch.cuda.is_available()


def nvtx_range(name: str) -> _NullRange | _NvtxRange:
    """Create an NVTX range context, or a no-op context when disabled."""
    if not nvtx_enabled():
        return _NULL_RANGE
    return _NvtxRange(name)
