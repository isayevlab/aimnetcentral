"""Optional dense ConvSV projection, output packing and masking.

The optimized path reads the contraction's strided layout directly and uses
IEEE float32 dots. It is restricted to frozen projection weights; ordinary
PyTorch handles trainable weights, other devices/dtypes, autocast and large
bases. Higher-order backward also uses differentiable PyTorch expressions.
Import this module lazily: Triton is an optional dependency.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _conv_finish_forward_kernel(
    x_ptr,
    weight_ptr,
    mask_ptr,
    projection_ptr,
    out_ptr,
    ATOMS: tl.constexpr,
    N: tl.constexpr,
    F: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
    xb: tl.constexpr,
    xi: tl.constexpr,
    xf: tl.constexpr,
    xs: tl.constexpr,
    xd: tl.constexpr,
    wf: tl.constexpr,
    ws: tl.constexpr,
    wh: tl.constexpr,
    mb: tl.constexpr,
    mi: tl.constexpr,
    HAS_MASK: tl.constexpr,
    BLOCK_A: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    atom = tl.program_id(0) * BLOCK_A + tl.arange(0, BLOCK_A)
    feature = tl.program_id(1)
    shift = tl.arange(0, BLOCK_S)
    combination = tl.arange(0, BLOCK_H)
    batch, center = atom // N, atom % N
    x_offset = batch[:, None] * xb + center[:, None] * xi + feature * xf + shift[None, :] * xs
    x_mask = (atom[:, None] < ATOMS) & (shift[None, :] < S)
    weights = tl.load(
        weight_ptr + feature * wf + shift[:, None] * ws + combination[None, :] * wh,
        mask=(shift[:, None] < S) & (combination[None, :] < H),
        other=0.0,
    )
    vx = tl.dot(tl.load(x_ptr + x_offset + xd, mask=x_mask, other=0.0), weights, input_precision="ieee")
    vy = tl.dot(tl.load(x_ptr + x_offset + 2 * xd, mask=x_mask, other=0.0), weights, input_precision="ieee")
    vz = tl.dot(tl.load(x_ptr + x_offset + 3 * xd, mask=x_mask, other=0.0), weights, input_precision="ieee")
    vector_mask = (atom[:, None] < ATOMS) & (combination[None, :] < H)
    projection_offset = ((atom[:, None] * F + feature) * H + combination[None, :]) * 3
    tl.store(projection_ptr + projection_offset, vx, mask=vector_mask)
    tl.store(projection_ptr + projection_offset + 1, vy, mask=vector_mask)
    tl.store(projection_ptr + projection_offset + 2, vz, mask=vector_mask)
    padded = tl.full((BLOCK_A,), False, tl.int1)
    if HAS_MASK:
        padded = tl.load(mask_ptr + batch * mb + center * mi, mask=atom < ATOMS, other=True)
    scalar = tl.load(x_ptr + x_offset, mask=x_mask, other=0.0)
    scalar = tl.where(padded[:, None], 0.0, scalar)
    vector = tl.where(padded[:, None], 0.0, vx * vx + vy * vy + vz * vz)
    out_base = atom[:, None] * F * (S + H)
    tl.store(out_ptr + out_base + feature * S + shift[None, :], scalar, mask=x_mask)
    tl.store(out_ptr + out_base + F * S + feature * H + combination[None, :], vector, mask=vector_mask)


@triton.jit
def _conv_finish_backward_kernel(
    grad_ptr,
    weight_ptr,
    mask_ptr,
    projection_ptr,
    out_ptr,
    ATOMS: tl.constexpr,
    N: tl.constexpr,
    F: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
    gb: tl.constexpr,
    gi: tl.constexpr,
    gc: tl.constexpr,
    wf: tl.constexpr,
    ws: tl.constexpr,
    wh: tl.constexpr,
    mb: tl.constexpr,
    mi: tl.constexpr,
    HAS_MASK: tl.constexpr,
    BLOCK_A: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    atom = tl.program_id(0) * BLOCK_A + tl.arange(0, BLOCK_A)
    feature = tl.program_id(1)
    shift = tl.arange(0, BLOCK_S)
    combination = tl.arange(0, BLOCK_H)
    batch, center = atom // N, atom % N
    padded = tl.full((BLOCK_A,), False, tl.int1)
    if HAS_MASK:
        padded = tl.load(mask_ptr + batch * mb + center * mi, mask=atom < ATOMS, other=True)
    grad_base = batch[:, None] * gb + center[:, None] * gi
    vector_mask = (atom[:, None] < ATOMS) & (combination[None, :] < H)
    grad_vector = tl.load(
        grad_ptr + grad_base + (F * S + feature * H + combination[None, :]) * gc,
        mask=vector_mask,
        other=0.0,
    )
    grad_vector = tl.where(padded[:, None], 0.0, grad_vector)
    projection_offset = ((atom[:, None] * F + feature) * H + combination[None, :]) * 3
    vx = tl.load(projection_ptr + projection_offset, mask=vector_mask, other=0.0)
    vy = tl.load(projection_ptr + projection_offset + 1, mask=vector_mask, other=0.0)
    vz = tl.load(projection_ptr + projection_offset + 2, mask=vector_mask, other=0.0)
    weights = tl.load(
        weight_ptr + feature * wf + combination[:, None] * wh + shift[None, :] * ws,
        mask=(combination[:, None] < H) & (shift[None, :] < S),
        other=0.0,
    )
    gx = tl.dot((2.0 * vx) * grad_vector, weights, input_precision="ieee")
    gy = tl.dot((2.0 * vy) * grad_vector, weights, input_precision="ieee")
    gz = tl.dot((2.0 * vz) * grad_vector, weights, input_precision="ieee")
    scalar_mask = (atom[:, None] < ATOMS) & (shift[None, :] < S)
    gs = tl.load(
        grad_ptr + grad_base + (feature * S + shift[None, :]) * gc,
        mask=scalar_mask,
        other=0.0,
    )
    gs = tl.where(padded[:, None], 0.0, gs)
    # Write the contraction's native B,S,F,N,4 storage, not a canonical-layout
    # temporary that the preceding convolution would have to transpose/copy.
    offset = (((batch[:, None] * S + shift[None, :]) * F + feature) * N + center[:, None]) * 4
    tl.store(out_ptr + offset, gs, mask=scalar_mask)
    tl.store(out_ptr + offset + 1, gx, mask=scalar_mask)
    tl.store(out_ptr + offset + 2, gy, mask=scalar_mask)
    tl.store(out_ptr + offset + 3, gz, mask=scalar_mask)


def _reference(avf_sv: Tensor, agh: Tensor, mask: Tensor | None) -> Tensor:
    scalar, vector = avf_sv.split((1, 3), dim=-1)
    vector = torch.einsum("fsh,bnfsd->bnfhd", agh, vector).square().sum(-1)
    output = torch.cat((scalar.squeeze(-1).flatten(-2, -1), vector.flatten(-2, -1)), dim=-1)
    return output if mask is None else output.masked_fill(mask.unsqueeze(-1), 0.0)


def _torch_input_grad(avf_sv: Tensor, agh: Tensor, mask: Tensor | None, grad: Tensor) -> Tensor:
    """Differentiable VJP built from original inputs for higher derivatives."""
    features, shifts, combinations = avf_sv.shape[2], avf_sv.shape[3], agh.shape[2]
    if mask is not None:
        grad = grad.masked_fill(mask.unsqueeze(-1), 0.0)
    scalar, vector = grad.split((features * shifts, features * combinations), dim=-1)
    scalar = scalar.unflatten(-1, (features, shifts)).unsqueeze(-1)
    vector = vector.unflatten(-1, (features, combinations)).unsqueeze(-1)
    projection = torch.einsum("fsh,bnfsd->bnfhd", agh, avf_sv[..., 1:])
    vector = torch.einsum("fsh,bnfhd->bnfsd", agh, (2.0 * projection) * vector)
    return torch.cat((scalar, vector), dim=-1)


def _launch_forward(avf_sv: Tensor, agh: Tensor, mask: Tensor | None) -> tuple[Tensor, Tensor]:
    batch, centers, features, shifts, _ = avf_sv.shape
    combinations = agh.shape[2]
    output = torch.empty((batch, centers, features * (shifts + combinations)), device=avf_sv.device, dtype=avf_sv.dtype)
    projection = torch.empty((batch, centers, features, combinations, 3), device=avf_sv.device, dtype=avf_sv.dtype)
    mask_pointer, mask_strides = (avf_sv, (0, 0)) if mask is None else (mask, mask.stride())
    _conv_finish_forward_kernel[(triton.cdiv(batch * centers, 16), features)](
        avf_sv,
        agh,
        mask_pointer,
        projection,
        output,
        batch * centers,
        centers,
        features,
        shifts,
        combinations,
        *avf_sv.stride(),
        *agh.stride(),
        *mask_strides,
        HAS_MASK=mask is not None,
        BLOCK_A=16,
        BLOCK_S=max(16, triton.next_power_of_2(shifts)),
        BLOCK_H=max(16, triton.next_power_of_2(combinations)),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output, projection


def _launch_backward(grad: Tensor, agh: Tensor, mask: Tensor | None, projection: Tensor) -> Tensor:
    batch, centers, features, combinations, _ = projection.shape
    shifts = agh.shape[1]
    native = torch.empty((batch, shifts, features, centers, 4), device=grad.device, dtype=grad.dtype)
    mask_pointer, mask_strides = (grad, (0, 0)) if mask is None else (mask, mask.stride())
    _conv_finish_backward_kernel[(triton.cdiv(batch * centers, 16), features)](
        grad,
        agh,
        mask_pointer,
        projection,
        native,
        batch * centers,
        centers,
        features,
        shifts,
        combinations,
        *grad.stride(),
        *agh.stride(),
        *mask_strides,
        HAS_MASK=mask is not None,
        BLOCK_A=16,
        BLOCK_S=max(16, triton.next_power_of_2(shifts)),
        BLOCK_H=max(16, triton.next_power_of_2(combinations)),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return native.permute(0, 3, 2, 1, 4)


class _ConvFinish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, avf_sv: Tensor, agh: Tensor, mask: Tensor | None) -> Tensor:
        output, projection = _launch_forward(avf_sv, agh, mask)
        ctx.save_for_backward(avf_sv, agh, mask, projection)
        ctx.set_materialize_grads(False)
        return output

    @staticmethod
    def backward(ctx, grad: Tensor | None):
        if grad is None or not ctx.needs_input_grad[0]:
            return None, None, None
        avf_sv, agh, mask, projection = ctx.saved_tensors
        if torch.is_grad_enabled():
            grad_input = _torch_input_grad(avf_sv, agh, mask, grad)
        else:
            grad_input = _launch_backward(grad, agh, mask, projection)
        return grad_input, None, None


def conv_finish(avf_sv: Tensor, agh: Tensor, mask: Tensor | None) -> Tensor:
    """Project ``[B,N,F,S,4]`` into fresh ``[B,N,F*S+F*H]`` masked output.

    ``agh`` has shape ``[F,S,H]`` and an optional boolean mask has shape
    ``[B,N]``. The fast path accepts arbitrary strides, frozen weights, float32
    CUDA inputs, and ``1 <= S,H <= 32``, ``1 <= F <= 64``. Other inputs use the
    ordinary expression; no input tensor or incoming gradient is mutated.
    """
    if (
        avf_sv.ndim != 5
        or agh.ndim != 3
        or avf_sv.device.type != "cuda"
        or agh.device != avf_sv.device
        or avf_sv.dtype != torch.float32
        or agh.dtype != torch.float32
        or agh.requires_grad
        or torch.is_autocast_enabled(avf_sv.device.type)
        or avf_sv.shape[0] < 2
        or avf_sv.shape[-1] != 4
        or agh.shape[:2] != avf_sv.shape[2:4]
        or not 1 <= avf_sv.shape[2] <= 64
        or not 1 <= avf_sv.shape[3] <= 32
        or not 1 <= agh.shape[2] <= 32
        or avf_sv.numel() == 0
        or (
            mask is not None
            and (mask.shape != avf_sv.shape[:2] or mask.device != avf_sv.device or mask.dtype != torch.bool)
        )
    ):
        return _reference(avf_sv, agh, mask)
    return _ConvFinish.apply(avf_sv, agh, mask)
