"""Optional dense AEV/ConvSV layout optimizations shared by eager and graph execution.

Only tensor layout and kernel execution change. The model's charge iterations,
output modules and force construction remain in their canonical implementations.
Pure-Torch layout operations retain autograd support; the optional Triton basis
kernel is restricted to first-order inference. Training and higher-derivative
calculator requests do not set the enabling ``_inference_kernels`` flag.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor

from aimnet.kernels.triton_support import triton_kernels_supported

if TYPE_CHECKING:
    from .aev import AEVSV


class _BmmGeometryCarry(torch.autograd.Function):
    """A matrix product and a differentiable continuation of its right operand.

    The continuation makes accumulation between consumers an explicit autograd
    edge. Its VJP is ``incoming + left.T @ grad_output``, so baddbmm can combine
    the matrix product and addition without a separate large add kernel. No
    mutable gradient sink, traversal-order assumption, or pass count is used.
    Backward uses differentiable Torch operations and never mutates its inputs.
    """

    @staticmethod
    def forward(ctx, left: Tensor, right: Tensor) -> tuple[Tensor, Tensor]:
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(left, right)
        return torch.bmm(left, right), right

    @staticmethod
    def backward(ctx, grad_output: Tensor | None, grad_carry: Tensor | None):
        left, right = ctx.saved_tensors
        grad_left = None
        if grad_output is not None and ctx.needs_input_grad[0]:
            grad_left = torch.bmm(grad_output, right.mH)
        grad_right = None
        if ctx.needs_input_grad[1]:
            if grad_output is None:
                grad_right = grad_carry
            elif grad_carry is None:
                grad_right = torch.bmm(left.mH, grad_output)
            else:
                grad_right = _geometry_baddbmm(grad_carry, left.mH, grad_output)
        return grad_left, grad_right


def _geometry_baddbmm(accum: Tensor, left: Tensor, right: Tensor) -> Tensor:
    # Only the first-order CUDA backward needs the optional fused epilogue.
    # Keep ordinary differentiable operations for CPU and higher derivatives.
    if (
        not torch.is_grad_enabled()
        and accum.is_cuda
        and accum.dtype == torch.float32
        and triton_kernels_supported(accum.device)
    ):
        try:
            from aimnet.kernels.geometry_baddbmm_triton import geometry_baddbmm
        except ImportError:
            pass
        else:
            return geometry_baddbmm(accum, left, right)
    return torch.baddbmm(accum, left, right)


def geometry_carry_basis(data: dict[str, Tensor]) -> Tensor:
    """Use the current canonical basis even if custom code replaces it.

    Public tensors keep their identity and derivative meaning. Replacing the
    canonical value starts a new continuation rather than retaining stale data.
    """
    if data["_geometry_carry_root"] is not data["g_sv"]:
        data["_geometry_carry_root"] = data["g_sv"]
        data["_geometry_carry"] = data["g_sv"].permute(0, 3, 2, 1, 4)
    return data["_geometry_carry"]


class _FirstOrderOnly(torch.autograd.Function):
    """Fail explicitly rather than silently drop a requested higher derivative."""

    @staticmethod
    def forward(ctx, value: Tensor) -> Tensor:
        return value

    @staticmethod
    def backward(ctx, grad: Tensor) -> Tensor:
        if torch.is_grad_enabled():
            raise RuntimeError(
                "Optimized Triton AEV supports first-order inference only; "
                "disable inference kernels for create_graph=True, Hessians or force training."
            )
        return grad


def _triton_kernel():
    """Import only on the selected CUDA path; absence preserves Torch behavior."""
    try:
        from aimnet.kernels.aev_sv_dense_triton import aev_sv_dense
    except ImportError:
        return None
    return aev_sv_dense


def dense_aev_layout(aev: AEVSV, data: dict[str, Tensor], d_ij: Tensor, r_ij: Tensor) -> Tensor:
    """Build ``g[b,s,m,i,:]`` directly instead of materializing a permutation.

    Host constants are prepared before execution with ``AEVSV.prepare_inference``.
    The output's view as ``[b,i,m,s,:]`` is the canonical ``g_sv`` tensor.
    """
    constants = data.get("_aev_inference_constants")
    if constants is None or constants.device.type != "cpu" or constants.shape != (3,):
        raise ValueError("Inference AEV requires CPU constants from AEVSV.prepare_inference().")
    eta, rc, scale = constants.unbind()
    kernel = None
    if (
        d_ij.device.type == "cuda"
        and d_ij.dtype == torch.float32
        and d_ij.shape[-1] <= 512
        and triton_kernels_supported(d_ij.device)
    ):
        kernel = _triton_kernel()
    if kernel is not None:
        result = kernel(d_ij, r_ij, aev.shifts_s, float(eta), float(rc), float(scale), pad=data["mask_i"])
        return _FirstOrderOnly.apply(result)

    # Same scalar operations and ordering as AEVSV._calc_aev. CPU zero-dimensional
    # constants preserve the cutoff's FP32 arithmetic without a device sync.
    fc = 0.5 * (torch.cos(d_ij.clamp(min=1e-6, max=rc) * scale) + 1.0)  # type: ignore[call-overload]
    fc = fc.masked_fill(data["mask_ij"], 0.0)
    distance = d_ij.transpose(1, 2).unsqueeze(1)
    eta_s, shifts_s = cast(Tensor, aev.eta_s), cast(Tensor, aev.shifts_s)
    gs = torch.exp(-eta_s * (distance - shifts_s.view(1, -1, 1, 1)).square())
    gs = gs * fc.transpose(1, 2).unsqueeze(1)
    unit = (r_ij / d_ij.unsqueeze(-1)).transpose(1, 2).unsqueeze(1)
    return torch.cat((gs.unsqueeze(-1), gs.unsqueeze(-1) * unit), dim=-1)


def dense_convolution(a: Tensor, geometry: Tensor, d2features: bool) -> Tensor:
    """The mode-0 ConvSV contraction using the basis' native bmm layout.

    Each use has an ordinary autograd edge. There is no mutable gradient sink,
    hard-coded number of passes, detached atom features, or assumed backward
    traversal order, so this primitive can be used by other model architectures.
    """
    batch, shifts, neighbors, centers, components = geometry.shape
    channels = a.shape[2]
    right = geometry.reshape(batch * shifts, neighbors, centers * components)
    if d2features:
        left = a.permute(0, 3, 2, 1).reshape(batch * shifts, channels, neighbors)
    else:
        left = a.permute(0, 2, 1).unsqueeze(1).expand(batch, shifts, channels, neighbors)
        left = left.reshape(batch * shifts, channels, neighbors)
    result = torch.bmm(left, right)
    return result.view(batch, shifts, channels, centers, components).permute(0, 3, 2, 1, 4)


def dense_convolutions(
    features: tuple[Tensor, ...], geometry: Tensor, d2features: tuple[bool, ...]
) -> tuple[Tensor, ...]:
    """Contract independent feature sets against one dense geometry tensor.

    Concatenating feature channels gives one geometry-gradient contraction for
    these consumers. Both feature gradients remain ordinary PyTorch autograd
    edges; this does not collect gradients across sequential model passes.
    """
    if not features or len(features) != len(d2features):
        raise ValueError("Provide one d2features flag for each nonempty feature set.")
    if len(features) == 1:
        return (dense_convolution(features[0], geometry, d2features[0]),)

    batch, shifts, neighbors, centers, components = geometry.shape
    lefts = []
    channels = []
    for a, is_2d in zip(features, d2features, strict=True):
        nchannel = a.shape[2]
        channels.append(nchannel)
        if is_2d:
            left = a.permute(0, 3, 2, 1).reshape(batch * shifts, nchannel, neighbors)
        else:
            left = a.permute(0, 2, 1).unsqueeze(1).expand(batch, shifts, nchannel, neighbors)
            left = left.reshape(batch * shifts, nchannel, neighbors)
        lefts.append(left)
    right = geometry.reshape(batch * shifts, neighbors, centers * components)
    result = torch.bmm(torch.cat(lefts, dim=1), right)
    return tuple(
        part.reshape(batch, shifts, nchannel, centers, components).permute(0, 3, 2, 1, 4)
        for part, nchannel in zip(result.split(channels, dim=1), channels, strict=True)
    )


def dense_convolutions_carry(
    features: tuple[Tensor, ...], geometry: Tensor, d2features: tuple[bool, ...]
) -> tuple[tuple[Tensor, ...], Tensor]:
    """Group this pass and return a private geometry continuation for the next.

    Keep the original geometry tensor as the public model output. The returned
    continuation is an internal dependency, not a replacement for that root.
    Layout preparation happens outside the custom Function so mixed and higher
    derivatives retain the feature/geometry dependencies.
    """
    if not features or len(features) != len(d2features):
        raise ValueError("Provide one d2features flag for each nonempty feature set.")
    if torch.is_autocast_enabled(geometry.device.type):
        # Ordinary BMM owns the casting/saved-input policy in autocast contexts.
        # Do not introduce a custom backward with mismatched operand dtypes.
        return dense_convolutions(features, geometry, d2features), geometry
    batch, shifts, neighbors, centers, components = geometry.shape
    lefts = []
    channels = []
    for a, is_2d in zip(features, d2features, strict=True):
        nchannel = a.shape[2]
        channels.append(nchannel)
        if is_2d:
            left = a.permute(0, 3, 2, 1).reshape(batch * shifts, nchannel, neighbors)
        else:
            left = a.permute(0, 2, 1).unsqueeze(1).expand(batch, shifts, nchannel, neighbors)
            left = left.reshape(batch * shifts, nchannel, neighbors)
        lefts.append(left)
    left = lefts[0] if len(lefts) == 1 else torch.cat(lefts, dim=1)
    right = geometry.reshape(batch * shifts, neighbors, centers * components)
    result, continuation = _BmmGeometryCarry.apply(left, right)
    outputs = tuple(
        part.reshape(batch, shifts, nchannel, centers, components).permute(0, 3, 2, 1, 4)
        for part, nchannel in zip(result.split(channels, dim=1), channels, strict=True)
    )
    return outputs, continuation.reshape_as(geometry)


def dense_convolution_carry(a: Tensor, geometry: Tensor, d2features: bool) -> tuple[Tensor, Tensor]:
    """Single-consumer form of :func:`dense_convolutions_carry`."""
    outputs, continuation = dense_convolutions_carry((a,), geometry, (d2features,))
    return outputs[0], continuation
