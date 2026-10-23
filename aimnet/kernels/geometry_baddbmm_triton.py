"""Optional out-of-place small-K geometry-gradient accumulation.

Compute ``accum + left @ right`` with one read of ``accum`` and a fresh output.
Unlike an out-of-place BLAS beta operation, the kernel does not first copy the
accumulator into its output. No incoming gradient or saved tensor is mutated.

This module is imported lazily by its caller: Triton remains optional. Calls
that require an autograd graph, or do not fit the CUDA/float32/small-K contract,
use ordinary PyTorch. The kernel deliberately uses IEEE float32 dot products,
not TF32 or reduced-precision inputs.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _geometry_baddbmm_kernel(
    accum_ptr,
    left_ptr,
    right_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    accum_batch: tl.constexpr,
    accum_row: tl.constexpr,
    accum_col: tl.constexpr,
    left_batch: tl.constexpr,
    left_row: tl.constexpr,
    left_col: tl.constexpr,
    right_batch: tl.constexpr,
    right_row: tl.constexpr,
    right_col: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    col = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    batch = tl.program_id(2)
    k = tl.arange(0, BLOCK_K)

    left = tl.load(
        left_ptr + batch * left_batch + row[:, None] * left_row + k[None, :] * left_col,
        mask=(row[:, None] < M) & (k[None, :] < K),
        other=0.0,
    )
    right = tl.load(
        right_ptr + batch * right_batch + k[:, None] * right_row + col[None, :] * right_col,
        mask=(k[:, None] < K) & (col[None, :] < N),
        other=0.0,
    )
    product = tl.dot(left, right, input_precision="ieee")
    mask = (row[:, None] < M) & (col[None, :] < N)
    accum = tl.load(
        accum_ptr + batch * accum_batch + row[:, None] * accum_row + col[None, :] * accum_col,
        mask=mask,
        other=0.0,
    )
    tl.store(out_ptr + (batch * M + row[:, None]) * N + col[None, :], product + accum, mask=mask)


def geometry_baddbmm(accum: Tensor, left: Tensor, right: Tensor) -> Tensor:
    """Return a fresh ``accum + torch.bmm(left, right)`` tensor.

    Optimized operands have shapes ``[batch, M, N]``, ``[batch, M, K]`` and
    ``[batch, K, N]``, respectively, with ``1 <= K <= 64``. Arbitrary strided
    operands, including a transposed left operand and zero-stride expansions,
    are read directly. Other valid ``torch.baddbmm`` inputs use that reference
    operation; invalid inputs therefore keep its normal validation behavior.

    Autograd-enabled calls always take the reference path. This is important
    when called inside a custom backward with ``create_graph=True``.
    """
    if (
        torch.is_grad_enabled()
        or torch.is_autocast_enabled(accum.device.type)
        or accum.ndim != 3
        or left.ndim != 3
        or right.ndim != 3
        or accum.device.type != "cuda"
        or left.device != accum.device
        or right.device != accum.device
        or accum.dtype != torch.float32
        or left.dtype != torch.float32
        or right.dtype != torch.float32
        or left.shape[0] != right.shape[0]
        or left.shape[2] != right.shape[1]
        or accum.shape != (left.shape[0], left.shape[1], right.shape[2])
        or not 1 <= left.shape[2] <= 64
        or accum.numel() == 0
    ):
        return torch.baddbmm(accum, left, right)

    batch, rows, columns = accum.shape
    inner = left.shape[2]
    block_m, block_n = 32, 64
    output = torch.empty((batch, rows, columns), dtype=accum.dtype, device=accum.device)
    _geometry_baddbmm_kernel[(triton.cdiv(rows, block_m), triton.cdiv(columns, block_n), batch)](
        accum,
        left,
        right,
        output,
        rows,
        columns,
        inner,
        *accum.stride(),
        *left.stride(),
        *right.stride(),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=max(16, triton.next_power_of_2(inner)),
        num_warps=4,
    )
    return output
