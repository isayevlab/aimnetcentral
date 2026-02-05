# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# type: ignore

import torch
import warp as wp
from torch import Tensor

wp.init()


def _get_stream(device: torch.device):
    """Get the Warp stream for the given device."""
    if device.type == "cuda":
        return wp.stream_from_torch(torch.cuda.current_stream(device))
    return None


# =============================================================================
# Warp Kernels
# =============================================================================


@wp.kernel(enable_backward=False)
def _conv_sv_2d_sp_kernel(
    a: wp.array3d(dtype=wp.float32),  # (B, A, G)
    idx: wp.array2d(dtype=wp.int32),  # (B, M)
    g: wp.array3d(dtype=wp.vec4f),  # (B, M, G, D)
    output: wp.array3d(dtype=wp.vec4f),  # (B, A, G, D)
):
    """Forward: output[b,a,g] = sum_m a[idx[b,m],a,g] * g[b,m,g]"""
    B, M = idx.shape[0], idx.shape[1]
    padding_value = B - 1  # last row is padding

    _b, _a, _g = wp.tid()

    acc = wp.vec4f()
    for _m in range(M):
        _idx = idx[_b, _m]
        if _idx >= padding_value:
            break
        a_val = a[_idx, _a, _g]
        g_val = g[_b, _m, _g]
        acc += a_val * g_val
    output[_b, _a, _g] = acc


@wp.kernel(enable_backward=False)
def _conv_sv_2d_sp_backward_a_kernel(
    grad_output: wp.array3d(dtype=wp.vec4f),  # (B, A, G, D)
    idx: wp.array2d(dtype=wp.int32),  # (B, M)
    g: wp.array3d(dtype=wp.vec4f),  # (B, M, G, D)
    grad_a: wp.array3d(dtype=wp.float32),  # (B, A, G)
):
    """Backward w.r.t. a: grad_a[idx[b,m],a,g] += dot(grad_output[b,a,g], g[b,m,g])"""
    B, M = idx.shape[0], idx.shape[1]
    padding_value = B - 1  # last row is padding

    _b, _a, _g = wp.tid()

    grad_out = grad_output[_b, _a, _g]
    for _m in range(M):
        _idx = idx[_b, _m]
        if _idx >= padding_value:
            break
        g_val = g[_b, _m, _g]
        val = wp.dot(grad_out, g_val)
        wp.atomic_add(grad_a, _idx, _a, _g, val)


@wp.kernel(enable_backward=False)
def _conv_sv_2d_sp_backward_g_kernel(
    grad_output: wp.array3d(dtype=wp.vec4f),  # (B, A, G, D)
    a: wp.array3d(dtype=wp.float32),  # (B, A, G)
    idx: wp.array2d(dtype=wp.int32),  # (B, M)
    grad_g: wp.array3d(dtype=wp.vec4f),  # (B, M, G, D)
):
    """Backward w.r.t. g: grad_g[b,m,g] = sum_a a[idx[b,m],a,g] * grad_output[b,a,g]"""
    B = idx.shape[0]
    A = a.shape[1]
    padding_value = B - 1  # last row is padding

    _b, _m, _g = wp.tid()

    _idx = idx[_b, _m]
    if _idx >= padding_value:
        return

    acc = wp.vec4f()

    for _a in range(A):
        grad_out = grad_output[_b, _a, _g]
        a_val = a[_idx, _a, _g]
        acc += a_val * grad_out

    grad_g[_b, _m, _g] = acc


@wp.kernel(enable_backward=False)
def _conv_sv_2d_sp_double_backward_a_g_kernel(
    grad_grad_a: wp.array3d(dtype=wp.float32),  # (B, A, G)
    idx: wp.array2d(dtype=wp.int32),  # (B, M)
    grad_output: wp.array3d(dtype=wp.vec4f),  # (B, A, G, D)
    grad_g: wp.array3d(dtype=wp.vec4f),  # (B, M, G, D)
):
    """Double backward: d(grad_a)/dg -> grad_g"""
    B = idx.shape[0]
    A = grad_grad_a.shape[1]
    padding_value = B - 1  # last row is padding

    _b, _m, _g = wp.tid()

    _idx = idx[_b, _m]
    if _idx >= padding_value:
        return

    acc = wp.vec4f()

    for _a in range(A):
        grad_grad_a_val = grad_grad_a[_idx, _a, _g]
        grad_out = grad_output[_b, _a, _g]
        acc += grad_grad_a_val * grad_out

    grad_g[_b, _m, _g] = acc


@wp.kernel(enable_backward=False)
def _conv_sv_2d_sp_double_backward_g_contrib_kernel(
    grad2_g: wp.array3d(dtype=wp.vec4f),  # (B, M, G, D)
    a: wp.array3d(dtype=wp.float32),  # (B, A, G)
    idx: wp.array2d(dtype=wp.int32),  # (B, M)
    grad_output_double: wp.array3d(dtype=wp.vec4f),  # (B, A, G, D) - OUTPUT
):
    """Double backward from grad2_g: einsum('bmgd,bmag->bagd', grad2_g, a_selected)"""
    B, M = idx.shape[0], idx.shape[1]
    padding_value = B - 1  # last row is padding

    _b, _a, _g = wp.tid()

    acc = wp.vec4f()
    for _m in range(M):
        _idx = idx[_b, _m]
        if _idx >= padding_value:
            break
        a_val = a[_idx, _a, _g]
        grad2_g_val = grad2_g[_b, _m, _g]
        acc += a_val * grad2_g_val

    grad_output_double[_b, _a, _g] = acc


@wp.kernel(enable_backward=False)
def _conv_sv_2d_sp_double_backward_a_contrib_kernel(
    grad2_a: wp.array3d(dtype=wp.float32),  # (B, A, G)
    idx: wp.array2d(dtype=wp.int32),  # (B, M)
    g: wp.array3d(dtype=wp.vec4f),  # (B, M, G, D)
    grad_output_double: wp.array3d(dtype=wp.vec4f),  # (B, A, G, D) - OUTPUT
):
    """Double backward from grad2_a: einsum('bmag,bmgd->bagd', grad2_a_selected, g)"""
    B, M = idx.shape[0], idx.shape[1]
    padding_value = B - 1  # last row is padding

    _b, _a, _g = wp.tid()

    acc = wp.vec4f()
    for _m in range(M):
        _idx = idx[_b, _m]
        if _idx >= padding_value:
            break
        grad2_a_val = grad2_a[_idx, _a, _g]
        g_val = g[_b, _m, _g]
        acc += grad2_a_val * g_val

    grad_output_double[_b, _a, _g] = acc


# =============================================================================
# PyTorch Custom Op Primitives
# =============================================================================


@torch.library.custom_op(
    "aimnet::conv_sv_2d_sp_fwd",
    mutates_args=(),
    device_types=["cuda"],
)
def _(a: Tensor, idx: Tensor, g: Tensor) -> Tensor:
    """Forward primitive for conv_sv_2d_sp."""
    stream = _get_stream(a.device)
    device = wp.device_from_torch(a.device)
    B, A, G = a.shape
    output = torch.zeros(B, A, G, 4, dtype=a.dtype, device=a.device)

    wp.launch(
        _conv_sv_2d_sp_kernel,
        dim=(B - 1, A, G),  # B-1: exclude padding row
        stream=stream,
        device=device,
        inputs=(
            wp.from_torch(a.detach(), return_ctype=True),
            wp.from_torch(idx.to(torch.int32), return_ctype=True),
            wp.from_torch(g.detach(), return_ctype=True, dtype=wp.vec4f),
            wp.from_torch(output, return_ctype=True, dtype=wp.vec4f),
        ),
    )
    return output


@torch.library.register_fake("aimnet::conv_sv_2d_sp_fwd")
def _(a: Tensor, idx: Tensor, g: Tensor) -> Tensor:
    B, A, G = a.shape
    return torch.empty(B, A, G, 4, dtype=a.dtype, device=a.device)


@torch.library.custom_op(
    "aimnet::conv_sv_2d_sp_bwd",
    mutates_args=(),
    device_types=["cuda"],
)
def _(grad_output: Tensor, a: Tensor, idx: Tensor, g: Tensor) -> list[Tensor]:
    """Backward primitive for conv_sv_2d_sp."""
    stream = _get_stream(a.device)
    device = wp.device_from_torch(a.device)
    B, A, G = a.shape
    B_out, M = idx.shape

    grad_a = torch.zeros_like(a)
    grad_g = torch.zeros(B_out, M, G, 4, dtype=g.dtype, device=g.device)

    grad_output_contig = grad_output.detach().contiguous()

    # Launch backward w.r.t. a
    wp.launch(
        _conv_sv_2d_sp_backward_a_kernel,
        dim=(B - 1, A, G),  # B-1: exclude padding row
        stream=stream,
        device=device,
        inputs=(
            wp.from_torch(grad_output_contig, return_ctype=True, dtype=wp.vec4f),
            wp.from_torch(idx.to(torch.int32), return_ctype=True),
            wp.from_torch(g.detach(), return_ctype=True, dtype=wp.vec4f),
            wp.from_torch(grad_a, return_ctype=True),
        ),
    )

    # Launch backward w.r.t. g
    wp.launch(
        _conv_sv_2d_sp_backward_g_kernel,
        dim=(B_out - 1, M, G),  # B_out-1: exclude padding row
        stream=stream,
        device=device,
        inputs=(
            wp.from_torch(grad_output_contig, return_ctype=True, dtype=wp.vec4f),
            wp.from_torch(a.detach(), return_ctype=True),
            wp.from_torch(idx.to(torch.int32), return_ctype=True),
            wp.from_torch(grad_g, return_ctype=True, dtype=wp.vec4f),
        ),
    )

    return [grad_a, grad_g]


@torch.library.register_fake("aimnet::conv_sv_2d_sp_bwd")
def _(grad_output: Tensor, a: Tensor, idx: Tensor, g: Tensor) -> list[Tensor]:
    B_out, M = idx.shape
    G = a.shape[2]
    return [
        torch.empty_like(a),
        torch.empty(B_out, M, G, 4, dtype=g.dtype, device=g.device),
    ]


@torch.library.custom_op(
    "aimnet::conv_sv_2d_sp_bwd_bwd",
    mutates_args=(),
    device_types=["cuda"],
)
def _(
    grad_output: Tensor,
    grad2_a: Tensor,
    grad2_g: Tensor,
    a: Tensor,
    idx: Tensor,
    g: Tensor,
) -> list[Tensor]:
    """Double backward primitive for conv_sv_2d_sp."""
    stream = _get_stream(a.device)
    device = wp.device_from_torch(a.device)
    B, A, G = a.shape
    B_out, M = idx.shape

    grad_grad_output = torch.zeros(B, A, G, 4, dtype=a.dtype, device=a.device)
    grad_a_double = torch.zeros_like(a)
    grad_g_double = torch.zeros(B_out, M, G, 4, dtype=a.dtype, device=a.device)

    grad_output_contig = grad_output.detach().contiguous()
    grad2_a_contig = grad2_a.detach().contiguous()
    grad2_g_contig = grad2_g.detach().contiguous()

    # Contribution from grad2_g to grad_grad_output
    wp.launch(
        _conv_sv_2d_sp_double_backward_g_contrib_kernel,
        dim=(B - 1, A, G),  # B-1: exclude padding row
        stream=stream,
        device=device,
        inputs=(
            wp.from_torch(grad2_g_contig, return_ctype=True, dtype=wp.vec4f),
            wp.from_torch(a.detach(), return_ctype=True),
            wp.from_torch(idx.to(torch.int32), return_ctype=True),
            wp.from_torch(grad_grad_output, return_ctype=True, dtype=wp.vec4f),
        ),
    )

    # Contribution from grad2_a to grad_grad_output
    grad_output_2_a = torch.zeros(B, A, G, 4, dtype=a.dtype, device=a.device)
    wp.launch(
        _conv_sv_2d_sp_double_backward_a_contrib_kernel,
        dim=(B - 1, A, G),  # B-1: exclude padding row
        stream=stream,
        device=device,
        inputs=(
            wp.from_torch(grad2_a_contig, return_ctype=True),
            wp.from_torch(idx.to(torch.int32), return_ctype=True),
            wp.from_torch(g.detach(), return_ctype=True, dtype=wp.vec4f),
            wp.from_torch(grad_output_2_a, return_ctype=True, dtype=wp.vec4f),
        ),
    )
    grad_grad_output = grad_grad_output + grad_output_2_a

    # Mixed partial: d(grad_a)/dg -> grad_g_double
    wp.launch(
        _conv_sv_2d_sp_double_backward_a_g_kernel,
        dim=(B_out - 1, M, G),  # B_out-1: exclude padding row
        stream=stream,
        device=device,
        inputs=(
            wp.from_torch(grad2_a_contig, return_ctype=True),
            wp.from_torch(idx.to(torch.int32), return_ctype=True),
            wp.from_torch(grad_output_contig, return_ctype=True, dtype=wp.vec4f),
            wp.from_torch(grad_g_double, return_ctype=True, dtype=wp.vec4f),
        ),
    )

    # Mixed partial: d(grad_g)/da -> grad_a_double
    wp.launch(
        _conv_sv_2d_sp_backward_a_kernel,
        dim=(B - 1, A, G),  # B-1: exclude padding row
        stream=stream,
        device=device,
        inputs=(
            wp.from_torch(grad_output_contig, return_ctype=True, dtype=wp.vec4f),
            wp.from_torch(idx.to(torch.int32), return_ctype=True),
            wp.from_torch(grad2_g_contig, return_ctype=True, dtype=wp.vec4f),
            wp.from_torch(grad_a_double, return_ctype=True),
        ),
    )

    return [grad_grad_output, grad_a_double, grad_g_double]


@torch.library.register_fake("aimnet::conv_sv_2d_sp_bwd_bwd")
def _(
    grad_output: Tensor,
    grad2_a: Tensor,
    grad2_g: Tensor,
    a: Tensor,
    idx: Tensor,
    g: Tensor,
) -> list[Tensor]:
    B, A, G = a.shape
    B_out, M = idx.shape
    return [
        torch.empty(B, A, G, 4, dtype=a.dtype, device=a.device),
        torch.empty_like(a),
        torch.empty(B_out, M, G, 4, dtype=a.dtype, device=a.device),
    ]


# =============================================================================
# Autograd Registration
# =============================================================================


def _conv_sv_2d_sp_setup_fwd_context(ctx, inputs, output):
    """Setup context for forward pass."""
    a, idx, g = inputs
    ctx.save_for_backward(a, idx, g)


def _conv_sv_2d_sp_setup_bwd_context(ctx, inputs, output):
    """Setup context for backward pass."""
    grad_output, a, idx, g = inputs
    ctx.save_for_backward(grad_output, a, idx, g)


@torch.compiler.allow_in_graph
def _conv_sv_2d_sp_bwd(ctx, grad_output):
    """Backward pass for conv_sv_2d_sp."""
    a, idx, g = ctx.saved_tensors
    grad_a, grad_g = torch.ops.aimnet.conv_sv_2d_sp_bwd(grad_output.contiguous(), a, idx, g)
    return grad_a, None, grad_g


@torch.compiler.allow_in_graph
def _conv_sv_2d_sp_bwd_bwd(ctx, *grad_outputs):
    """Double backward pass for conv_sv_2d_sp."""
    grad2_a = grad_outputs[0][0]
    grad2_g = grad_outputs[0][1]

    grad_output_saved, a, idx, g = ctx.saved_tensors

    if grad2_a is None:
        grad2_a = torch.zeros_like(a)
    if grad2_g is None:
        B_out, M = idx.shape
        G = a.shape[2]
        grad2_g = torch.zeros(B_out, M, G, 4, dtype=g.dtype, device=g.device)

    outputs = torch.ops.aimnet.conv_sv_2d_sp_bwd_bwd(grad_output_saved, grad2_a, grad2_g, a, idx, g)

    return outputs[0], outputs[1], None, outputs[2]


torch.library.register_autograd(
    "aimnet::conv_sv_2d_sp_fwd",
    _conv_sv_2d_sp_bwd,
    setup_context=_conv_sv_2d_sp_setup_fwd_context,
)

torch.library.register_autograd(
    "aimnet::conv_sv_2d_sp_bwd",
    _conv_sv_2d_sp_bwd_bwd,
    setup_context=_conv_sv_2d_sp_setup_bwd_context,
)


# =============================================================================
# Public API
# =============================================================================


def conv_sv_2d_sp(a: Tensor, idx: Tensor, g: Tensor) -> Tensor:
    """Compute conv_sv_2d_sp with support for 1st and 2nd order derivatives.

    Parameters
    ----------
    a : Tensor
        Input tensor of shape (B, A, G).
    idx : Tensor
        Index tensor of shape (B, M).
    g : Tensor
        Gate tensor of shape (B, M, G, 4).

    Returns
    -------
    Tensor
        Output tensor of shape (B, A, G, 4).
    """
    return torch.ops.aimnet.conv_sv_2d_sp_fwd(a, idx, g)
