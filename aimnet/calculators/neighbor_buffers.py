"""Reusable public nvalchemi neighbor buffers for fixed-layout input adapters.

This module owns topology storage, not model arithmetic. Both eager callers
and CUDA-graph adapters may use it. Buffer capacities and the layout are fixed;
coordinate-dependent neighbors are rebuilt for every call.
"""

from __future__ import annotations

import math

import torch
from nvalchemiops.torch.neighbors import batch_naive_neighbor_list
from torch import Tensor


class ReusableNeighborMatrix:
    """A nonperiodic, batched neighbor matrix using nvalchemi's public API.

    ``batch_idx`` and ``batch_ptr`` must describe a validated contiguous layout.
    Result tensors are borrowed and overwritten by :meth:`fill`. Counts may
    exceed ``capacity``; callers must check that condition before consuming a
    result externally and rebuild with larger buffers if necessary. Captured
    callers perform that check after replay, outside the captured region.
    """

    def __init__(
        self,
        cutoff: float,
        batch_idx: Tensor,
        batch_ptr: Tensor,
        capacity: int,
        fill_value: int,
    ):
        if not math.isfinite(cutoff) or cutoff <= 0:
            raise ValueError("cutoff must be finite and positive")
        if not isinstance(capacity, int) or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        if batch_idx.ndim != 1 or batch_ptr.ndim != 1 or batch_ptr.numel() < 2:
            raise ValueError("batch_idx and batch_ptr must describe a flat batched layout")
        if batch_idx.device != batch_ptr.device:
            raise ValueError("batch metadata must share a device")
        self.cutoff = float(cutoff)
        self.capacity = capacity
        self.fill_value = int(fill_value)
        # Own immutable copies: a caller reusing/mutating its buffers must not
        # silently change the ownership of a captured neighbor operation.
        self.batch_idx = batch_idx.to(torch.int32).clone()
        self.batch_ptr = batch_ptr.to(torch.int32).clone()
        self.matrix = torch.empty((batch_idx.numel(), capacity), dtype=torch.int32, device=batch_idx.device)
        self.counts = torch.empty((batch_idx.numel(),), dtype=torch.int32, device=batch_idx.device)

    def fill(self, coord: Tensor) -> tuple[Tensor, Tensor]:
        """Rebuild into the owned buffers on PyTorch's current stream.

        nvalchemi >= 0.5 performs buffer reset and Torch/Warp stream ordering
        internally. No private launcher, extra stream context, host scalar
        read or neighbor-width trimming is needed here.
        """
        if coord.ndim != 2 or coord.shape != (self.batch_idx.numel(), 3):
            raise ValueError("coordinates must match the fixed (total_atoms, 3) layout")
        if coord.device != self.matrix.device:
            raise ValueError("coordinates must be on the neighbor-buffer device")
        if coord.dtype not in (torch.float32, torch.float64):
            raise TypeError("neighbor coordinates must be float32 or float64")
        return batch_naive_neighbor_list(
            positions=coord,
            cutoff=self.cutoff,
            batch_idx=self.batch_idx,
            batch_ptr=self.batch_ptr,
            max_neighbors=self.capacity,
            half_fill=False,
            fill_value=self.fill_value,
            neighbor_matrix=self.matrix,
            num_neighbors=self.counts,
            # Fixed kernel choice. Rows are filled with atomic adds, so the neighbor order (and the floating-point
            # summation order downstream) is not guaranteed to match the regular path's lists.
            strategy="scalar",
        )
