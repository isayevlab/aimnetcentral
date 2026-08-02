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
    padding_value: int,
):
    """Forward: output[b,a,g] = sum_m a[idx[b,m],a,g] * g[b,m,g]"""
    M = idx.shape[1]
    _b, _a, _g = wp.tid()

    acc = wp.vec4f()
    for _m in range(M):
        _idx = idx[_b, _m]
        if _idx >= padding_value:
            # packed-padding contract: sentinels are contiguous at the row end (see conv_sv_2d_sp docstring)
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
    padding_value: int,
):
    """Backward w.r.t. a: grad_a[idx[b,m],a,g] += dot(grad_output[b,a,g], g[b,m,g])"""
    M = idx.shape[1]
    _b, _a, _g = wp.tid()

    grad_out = grad_output[_b, _a, _g]
    for _m in range(M):
        _idx = idx[_b, _m]
        if _idx >= padding_value:
            # packed-padding contract: sentinels are contiguous at the row end (see conv_sv_2d_sp docstring)
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
    padding_value: int,
):
    """Backward w.r.t. g: grad_g[b,m,g] = sum_a a[idx[b,m],a,g] * grad_output[b,a,g]"""
    A = a.shape[1]
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
    padding_value: int,
):
    """Double backward: d(grad_a)/dg -> grad_g"""
    A = grad_grad_a.shape[1]
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
    padding_value: int,
):
    """Double backward from grad2_g: einsum('bmgd,bmag->bagd', grad2_g, a_selected)"""
    M = idx.shape[1]
    _b, _a, _g = wp.tid()

    acc = wp.vec4f()
    for _m in range(M):
        _idx = idx[_b, _m]
        if _idx >= padding_value:
            # packed-padding contract: sentinels are contiguous at the row end (see conv_sv_2d_sp docstring)
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
    padding_value: int,
):
    """Double backward from grad2_a: einsum('bmag,bmgd->bagd', grad2_a_selected, g)"""
    M = idx.shape[1]
    _b, _a, _g = wp.tid()

    acc = wp.vec4f()
    for _m in range(M):
        _idx = idx[_b, _m]
        if _idx >= padding_value:
            # packed-padding contract: sentinels are contiguous at the row end (see conv_sv_2d_sp docstring)
            break
        grad2_a_val = grad2_a[_idx, _a, _g]
        g_val = g[_b, _m, _g]
        acc += grad2_a_val * g_val

    grad_output_double[_b, _a, _g] = acc


# =============================================================================
# PyTorch Custom Op Primitives
# =============================================================================


def _validate_conv_sv_sizes(a: Tensor, idx: Tensor, padding_value: int, num_centers: int) -> None:
    """Validate the flattened atom and center capacities used by Warp."""
    if a.ndim != 3 or idx.ndim != 2:
        raise ValueError("a must be 3D and idx must be 2D.")
    if padding_value < 0 or padding_value > a.shape[0]:
        raise ValueError("padding_value must be in [0, a.shape[0]].")
    if num_centers < 0 or num_centers > idx.shape[0]:
        raise ValueError("num_centers must be in [0, idx.shape[0]].")


@torch.library.custom_op(
    "aimnet::conv_sv_2d_sp_fwd",
    mutates_args=(),
    device_types=["cuda"],
)
def _(a: Tensor, idx: Tensor, g: Tensor, padding_value: int, num_centers: int) -> Tensor:
    """Forward primitive for conv_sv_2d_sp."""
    _validate_conv_sv_sizes(a, idx, padding_value, num_centers)
    stream = _get_stream(a.device)
    device = wp.device_from_torch(a.device)
    _B, A, G = a.shape
    B_out = idx.shape[0]
    output = torch.zeros(B_out, A, G, 4, dtype=a.dtype, device=a.device)

    wp.launch(
        _conv_sv_2d_sp_kernel,
        dim=(num_centers, A, G),
        stream=stream,
        device=device,
        inputs=(
            wp.from_torch(a.detach(), return_ctype=True),
            wp.from_torch(idx.to(torch.int32), return_ctype=True),
            wp.from_torch(g.detach(), return_ctype=True, dtype=wp.vec4f),
            wp.from_torch(output, return_ctype=True, dtype=wp.vec4f),
            padding_value,
        ),
    )
    return output


@torch.library.register_fake("aimnet::conv_sv_2d_sp_fwd")
def _(a: Tensor, idx: Tensor, g: Tensor, padding_value: int, num_centers: int) -> Tensor:
    _validate_conv_sv_sizes(a, idx, padding_value, num_centers)
    _B, A, G = a.shape
    return torch.empty(idx.shape[0], A, G, 4, dtype=a.dtype, device=a.device)


@torch.library.custom_op(
    "aimnet::conv_sv_2d_sp_bwd",
    mutates_args=(),
    device_types=["cuda"],
)
def _(grad_output: Tensor, a: Tensor, idx: Tensor, g: Tensor, padding_value: int, num_centers: int) -> list[Tensor]:
    """Backward primitive for conv_sv_2d_sp."""
    _validate_conv_sv_sizes(a, idx, padding_value, num_centers)
    stream = _get_stream(a.device)
    device = wp.device_from_torch(a.device)
    _B, A, G = a.shape
    B_out, M = idx.shape

    grad_a = torch.zeros_like(a)
    grad_g = torch.zeros(B_out, M, G, 4, dtype=g.dtype, device=g.device)

    grad_output_contig = grad_output.detach().contiguous()

    # Launch backward w.r.t. a
    wp.launch(
        _conv_sv_2d_sp_backward_a_kernel,
        dim=(num_centers, A, G),
        stream=stream,
        device=device,
        inputs=(
            wp.from_torch(grad_output_contig, return_ctype=True, dtype=wp.vec4f),
            wp.from_torch(idx.to(torch.int32), return_ctype=True),
            wp.from_torch(g.detach(), return_ctype=True, dtype=wp.vec4f),
            wp.from_torch(grad_a, return_ctype=True),
            padding_value,
        ),
    )

    # Launch backward w.r.t. g
    wp.launch(
        _conv_sv_2d_sp_backward_g_kernel,
        dim=(num_centers, M, G),
        stream=stream,
        device=device,
        inputs=(
            wp.from_torch(grad_output_contig, return_ctype=True, dtype=wp.vec4f),
            wp.from_torch(a.detach(), return_ctype=True),
            wp.from_torch(idx.to(torch.int32), return_ctype=True),
            wp.from_torch(grad_g, return_ctype=True, dtype=wp.vec4f),
            padding_value,
        ),
    )

    return [grad_a, grad_g]


@torch.library.register_fake("aimnet::conv_sv_2d_sp_bwd")
def _(grad_output: Tensor, a: Tensor, idx: Tensor, g: Tensor, padding_value: int, num_centers: int) -> list[Tensor]:
    B_out, M = idx.shape
    G = a.shape[2]
    _validate_conv_sv_sizes(a, idx, padding_value, num_centers)
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
    padding_value: int,
    num_centers: int,
) -> list[Tensor]:
    """Double backward primitive for conv_sv_2d_sp."""
    _validate_conv_sv_sizes(a, idx, padding_value, num_centers)
    stream = _get_stream(a.device)
    device = wp.device_from_torch(a.device)
    _B, A, G = a.shape
    B_out, M = idx.shape

    grad_grad_output = torch.zeros(B_out, A, G, 4, dtype=a.dtype, device=a.device)
    grad_a_double = torch.zeros_like(a)
    grad_g_double = torch.zeros(B_out, M, G, 4, dtype=a.dtype, device=a.device)

    grad_output_contig = grad_output.detach().contiguous()
    grad2_a_contig = grad2_a.detach().contiguous()
    grad2_g_contig = grad2_g.detach().contiguous()

    # Contribution from grad2_g to grad_grad_output
    wp.launch(
        _conv_sv_2d_sp_double_backward_g_contrib_kernel,
        dim=(num_centers, A, G),
        stream=stream,
        device=device,
        inputs=(
            wp.from_torch(grad2_g_contig, return_ctype=True, dtype=wp.vec4f),
            wp.from_torch(a.detach(), return_ctype=True),
            wp.from_torch(idx.to(torch.int32), return_ctype=True),
            wp.from_torch(grad_grad_output, return_ctype=True, dtype=wp.vec4f),
            padding_value,
        ),
    )

    # Contribution from grad2_a to grad_grad_output
    grad_output_2_a = torch.zeros(B_out, A, G, 4, dtype=a.dtype, device=a.device)
    wp.launch(
        _conv_sv_2d_sp_double_backward_a_contrib_kernel,
        dim=(num_centers, A, G),
        stream=stream,
        device=device,
        inputs=(
            wp.from_torch(grad2_a_contig, return_ctype=True),
            wp.from_torch(idx.to(torch.int32), return_ctype=True),
            wp.from_torch(g.detach(), return_ctype=True, dtype=wp.vec4f),
            wp.from_torch(grad_output_2_a, return_ctype=True, dtype=wp.vec4f),
            padding_value,
        ),
    )
    grad_grad_output = grad_grad_output + grad_output_2_a

    # Mixed partial: d(grad_a)/dg -> grad_g_double
    wp.launch(
        _conv_sv_2d_sp_double_backward_a_g_kernel,
        dim=(num_centers, M, G),
        stream=stream,
        device=device,
        inputs=(
            wp.from_torch(grad2_a_contig, return_ctype=True),
            wp.from_torch(idx.to(torch.int32), return_ctype=True),
            wp.from_torch(grad_output_contig, return_ctype=True, dtype=wp.vec4f),
            wp.from_torch(grad_g_double, return_ctype=True, dtype=wp.vec4f),
            padding_value,
        ),
    )

    # Mixed partial: d(grad_g)/da -> grad_a_double
    wp.launch(
        _conv_sv_2d_sp_backward_a_kernel,
        dim=(num_centers, A, G),
        stream=stream,
        device=device,
        inputs=(
            wp.from_torch(grad_output_contig, return_ctype=True, dtype=wp.vec4f),
            wp.from_torch(idx.to(torch.int32), return_ctype=True),
            wp.from_torch(grad2_g_contig, return_ctype=True, dtype=wp.vec4f),
            wp.from_torch(grad_a_double, return_ctype=True),
            padding_value,
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
    padding_value: int,
    num_centers: int,
) -> list[Tensor]:
    _B, A, G = a.shape
    B_out, M = idx.shape
    _validate_conv_sv_sizes(a, idx, padding_value, num_centers)
    return [
        torch.empty(B_out, A, G, 4, dtype=a.dtype, device=a.device),
        torch.empty_like(a),
        torch.empty(B_out, M, G, 4, dtype=a.dtype, device=a.device),
    ]


# =============================================================================
# Autograd Registration
# =============================================================================


def _conv_sv_2d_sp_setup_fwd_context(ctx, inputs, output):
    """Setup context for forward pass."""
    a, idx, g, padding_value, num_centers = inputs
    ctx.save_for_backward(a, idx, g)
    ctx.padding_value = padding_value
    ctx.num_centers = num_centers


def _conv_sv_2d_sp_setup_bwd_context(ctx, inputs, output):
    """Setup context for backward pass."""
    grad_output, a, idx, g, padding_value, num_centers = inputs
    ctx.save_for_backward(grad_output, a, idx, g)
    ctx.padding_value = padding_value
    ctx.num_centers = num_centers


@torch.compiler.allow_in_graph
def _conv_sv_2d_sp_bwd(ctx, grad_output):
    """Backward pass for conv_sv_2d_sp."""
    a, idx, g = ctx.saved_tensors
    grad_a, grad_g = torch.ops.aimnet.conv_sv_2d_sp_bwd(
        grad_output.contiguous(), a, idx, g, ctx.padding_value, ctx.num_centers
    )
    return grad_a, None, grad_g, None, None


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

    outputs = torch.ops.aimnet.conv_sv_2d_sp_bwd_bwd(
        grad_output_saved, grad2_a, grad2_g, a, idx, g, ctx.padding_value, ctx.num_centers
    )

    return outputs[0], outputs[1], None, outputs[2], None, None


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
# vmap Registration
# =============================================================================


@torch.library.register_vmap("aimnet::conv_sv_2d_sp_fwd")
def _vmap_conv_sv_2d_sp_fwd(info, in_dims, a, idx, g, padding_value, num_centers):
    raise RuntimeError("aimnet::conv_sv_2d_sp_fwd does not support direct vmap.")


def _vmap_slice(t: Tensor, d: int | None, k: int) -> Tensor:
    """Pick the k-th slice along vmap batch dim d, or pass through if not batched.

    Returns a contiguous tensor when slicing — the underlying Warp kernels read
    via wp.from_torch which assumes C-contiguous layout, so a non-contiguous view
    from movedim(d, 0) would silently misread strided memory.
    """
    if d is None:
        return t
    return t.movedim(d, 0)[k].contiguous()


@torch.library.register_vmap("aimnet::conv_sv_2d_sp_bwd")
def _vmap_conv_sv_2d_sp_bwd(info, in_dims, grad_output, a, idx, g, padding_value, num_centers):
    """vmap rule for the first-backward primitive.

    Hit when torch.func.vmap traverses a vjp closure that reaches the first-order
    backward (e.g. vmap over autograd.grad with create_graph=True).  Only
    in_dims = (0, None, None, None) — vmap on the upstream cotangent only — is
    supported.  Batching idx along its leading B dim is unsafe (it scatters the
    kernel's padding-row sentinel at index B-1) and is not detected at runtime.

    Note: register_vmap is consulted ONLY by the functorch dispatch (torch.func.vmap,
    aka torch.vmap).  The legacy batching dispatch used by is_grads_batched=True and
    autograd.functional.hessian(vectorize=True) does not consult this rule.

    Strategy: K-loop, same reasoning as the bwd_bwd rule.
    """
    if any(dim is not None for dim in in_dims[1:]):
        raise RuntimeError(f"aimnet::conv_sv_2d_sp_bwd vmap supports batching only grad_output; got in_dims={in_dims}")
    K = info.batch_size

    out0: list[Tensor] = []
    out1: list[Tensor] = []
    for k in range(K):
        outs = torch.ops.aimnet.conv_sv_2d_sp_bwd(
            _vmap_slice(grad_output, in_dims[0], k),
            _vmap_slice(a, in_dims[1], k),
            _vmap_slice(idx, in_dims[2], k),
            _vmap_slice(g, in_dims[3], k),
            padding_value,
            num_centers,
        )
        out0.append(outs[0])
        out1.append(outs[1])

    return (
        [torch.stack(out0, dim=0), torch.stack(out1, dim=0)],
        [0, 0],
    )


@torch.library.register_vmap("aimnet::conv_sv_2d_sp_bwd_bwd")
def _vmap_conv_sv_2d_sp_bwd_bwd(info, in_dims, grad_output, grad2_a, grad2_g, a, idx, g, padding_value, num_centers):
    """vmap rule for the double-backward primitive.

    Hit when torch.func.vmap traverses a vjp closure that reaches the second-order
    backward (the Hessian-via-vmap path).  Only
    vmap on grad_output and the two upstream cotangents is supported. Batching idx,
    a, or g along their leading B dim is unsafe
    (it scatters the kernel's padding-row sentinel at index B-1) and is not detected
    at runtime.

    Note: register_vmap is consulted ONLY by the functorch dispatch (torch.func.vmap,
    aka torch.vmap).  The legacy batching dispatch used by is_grads_batched=True and
    autograd.functional.hessian(vectorize=True) does not consult this rule.

    Strategy: K-loop. Folding K into the kernel's leading B dim is unsafe because
    the kernels rely on a single padding-row sentinel at index B-1; stacking K
    copies would scatter padding rows. The K calls queue async on the CUDA
    stream, so the loop's per-call cost is dominated by GPU work, not Python.
    """
    if any(dim is not None for dim in in_dims[3:]):
        raise RuntimeError(
            "aimnet::conv_sv_2d_sp_bwd_bwd vmap supports batching only grad_output, "
            f"grad2_a, and grad2_g; got in_dims={in_dims}"
        )
    K = info.batch_size

    out0: list[Tensor] = []
    out1: list[Tensor] = []
    out2: list[Tensor] = []
    for k in range(K):
        outs = torch.ops.aimnet.conv_sv_2d_sp_bwd_bwd(
            _vmap_slice(grad_output, in_dims[0], k),
            _vmap_slice(grad2_a, in_dims[1], k),
            _vmap_slice(grad2_g, in_dims[2], k),
            _vmap_slice(a, in_dims[3], k),
            _vmap_slice(idx, in_dims[4], k),
            _vmap_slice(g, in_dims[5], k),
            padding_value,
            num_centers,
        )
        out0.append(outs[0])
        out1.append(outs[1])
        out2.append(outs[2])

    return (
        [torch.stack(out0, dim=0), torch.stack(out1, dim=0), torch.stack(out2, dim=0)],
        [0, 0, 0],
    )


# =============================================================================
# Public API
# =============================================================================


def conv_sv_2d_sp(
    a: Tensor,
    idx: Tensor,
    g: Tensor,
    padding_value: int | None = None,
    num_centers: int | None = None,
) -> Tensor:
    """Compute conv_sv_2d_sp with support for 1st and 2nd order derivatives.

    Parameters
    ----------
    a : Tensor
        Input tensor of shape (K, A, G), where K is the atom capacity.
    idx : Tensor
        Index tensor of shape (C, M), where C is the center capacity.
    g : Tensor
        Gate tensor of shape (C, M, G, 4).
    padding_value : int, optional
        Sentinel threshold in the atom dimension. Defaults to ``K - 1``.
    num_centers : int, optional
        Number of centers launched by Warp. Defaults to ``C - 1``.

    Notes
    -----
    ``idx`` rows must follow the packed-padding contract: real neighbor indices
    come first and padding sentinels (values >= ``padding_value``) are
    contiguous at the end of each row. The Warp kernels stop scanning a row at
    the first sentinel, so interleaved padding would silently drop real
    neighbors. ``nvalchemiops.torch.neighbors.neighbor_list`` produces this
    layout. The contract is not checked at runtime: validating ``idx`` values
    would require a device-to-host sync on the hot path.

    Only the first- and second-order backward primitives are vmap-registered.
    `torch.func.vmap` directly over the forward (e.g. vmap over a non-vjp
    closure that calls this function) will raise `Batching rule not implemented
    for aimnet::conv_sv_2d_sp_fwd`. The Hessian-via-vmap path uses a vjp closure
    that does not vmap forward, so it is unaffected.

    Returns
    -------
    Tensor
        Output tensor of shape (C, A, G, 4).
    """
    if (padding_value is None) != (num_centers is None):
        raise ValueError("padding_value and num_centers must be supplied together.")
    if a.device.type != "cuda" or idx.device.type != "cuda" or g.device.type != "cuda":
        raise RuntimeError("conv_sv_2d_sp is a CUDA-only Warp kernel")
    if a.dtype != torch.float32 or g.dtype != torch.float32:
        raise TypeError("conv_sv_2d_sp supports float32 tensors only")
    if idx.dtype != torch.int32:
        idx = idx.to(torch.int32)
    if a.ndim != 3 or idx.ndim != 2 or g.ndim != 4 or g.shape[-1] != 4:
        raise ValueError("Expected shapes a=(B,A,G), idx=(B,M), g=(B,M,G,4)")
    if idx.shape[0] != g.shape[0] or a.shape[2] != g.shape[2]:
        raise ValueError("Incompatible conv_sv_2d_sp leading or basis dimensions")
    if padding_value is None:
        padding_value = a.shape[0] - 1
        num_centers = idx.shape[0] - 1
    assert num_centers is not None
    _validate_conv_sv_sizes(a, idx, padding_value, num_centers)
    if not a.is_contiguous():
        a = a.contiguous()
    if not idx.is_contiguous():
        idx = idx.contiguous()
    if not g.is_contiguous():
        g = g.contiguous()
    return torch.ops.aimnet.conv_sv_2d_sp_fwd(a, idx, g, padding_value, num_centers)
