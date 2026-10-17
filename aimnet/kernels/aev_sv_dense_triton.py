"""Radial-vector AEV basis of the dense CUDA-graph engine in one kernel each way (Triton).

Reference math (``AEVSV`` with ``ops.calc_distances`` and ``ops.exp_expand``; dense mode, unpadded), per pair
``(b, i, m)`` and shift ``s``::

    fc   = 0.5 * (cos(clamp(d, 1e-6, rc) * (pi / rc)) + 1),   fc = 0 on the diagonal (mask_ij)
    gs_s = exp(-eta * (d - shift_s) ** 2) * fc
    u    = r / d
    g[b, s, m, i, :] = (gs_s, gs_s * u)

The output is ``g_sv[b, i, m, s, :]`` stored in the ``[B, S, M, I, 4]`` layout that ``ConvSV``'s einsum uses for its
``bmm`` (see ``aimnet.calculators.cuda_graph.dense``). The forward writes ``g`` once with the same float32 operations
in the same order as the PyTorch expression (libdevice ``exp``/``cos``, IEEE division), so its values are bitwise
equal. The backward is analytic and first order: from ``G = dL/dg``::

    dL/dgs_s = G0 + Gv . u,   dL/du = sum_s gs_s Gv
    dL/dd    = sum_s dL/dgs_s * (fc * e_s * (-2 eta (d - shift_s)) + e_s * dfc/dd) - (dL/du . r) / d^2
    dL/dr    = dL/du / d

with ``dfc/dd = -0.5 sin(x) pi / rc`` inside the clamp range (inclusive, as ``torch.clamp``'s backward) and 0 on the
diagonal (``masked_fill``'s backward). The inputs ``d [B, N, N]`` and ``r [B, N, N, 3]`` are ``calc_distances``'s
outputs (diagonal ``r`` padded to 1), so autograd continues from them unchanged. Not differentiable twice.
"""

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.autograd.function import once_differentiable
from triton.language.extra import libdevice

MAX_ATOMS = 512  # one program holds a row of next_power_of_2(N) pairs


# =============================================================================
# Triton Kernels
# =============================================================================


@triton.jit
def _aev_sv_dense_fwd_kernel(
    d_ptr,  # (B, N, N) float32
    r_ptr,  # (B, N, N, 3) float32
    sh_ptr,  # (S,) float32
    g_ptr,  # (B, S, N, N, 4) float32, output
    N,
    eta,
    rc,
    scale,  # pi / rc
    S: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    b = tl.program_id(0)
    m = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
    i = tl.arange(0, BLOCK_I)
    msk = (m < N)[:, None] & (i < N)[None, :]
    pair = (b * N + i[None, :]) * N + m[:, None]  # element (b, i, m) of d and r
    d = tl.load(d_ptr + pair, mask=msk, other=1.0)
    rx = tl.load(r_ptr + pair * 3 + 0, mask=msk, other=0.0)
    ry = tl.load(r_ptr + pair * 3 + 1, mask=msk, other=0.0)
    rz = tl.load(r_ptr + pair * 3 + 2, mask=msk, other=0.0)
    dc = tl.minimum(tl.maximum(d, 1e-6), rc)
    fc = 0.5 * (libdevice.cos(dc * scale) + 1.0)
    fc = tl.where(m[:, None] == i[None, :], 0.0, fc)
    ux = libdevice.div_rn(rx, d)
    uy = libdevice.div_rn(ry, d)
    uz = libdevice.div_rn(rz, d)
    for s in tl.static_range(S):
        t = d - tl.load(sh_ptr + s)
        gs = libdevice.exp(-eta * (t * t)) * fc
        base = (((b * S + s) * N + m[:, None]) * N + i[None, :]) * 4
        tl.store(g_ptr + base + 0, gs, mask=msk)
        tl.store(g_ptr + base + 1, gs * ux, mask=msk)
        tl.store(g_ptr + base + 2, gs * uy, mask=msk)
        tl.store(g_ptr + base + 3, gs * uz, mask=msk)


@triton.jit
def _aev_sv_dense_bwd_kernel(
    d_ptr,  # (B, N, N) float32
    r_ptr,  # (B, N, N, 3) float32
    sh_ptr,  # (S,) float32
    gg_ptr,  # (B, S, N, N, 4) float32, dL/dg
    gd_ptr,  # (B, N, N) float32, output dL/dd
    gr_ptr,  # (B, N, N, 3) float32, output dL/dr
    N,
    eta,
    rc,
    scale,  # pi / rc
    S: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    b = tl.program_id(0)
    m = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
    i = tl.arange(0, BLOCK_I)
    msk = (m < N)[:, None] & (i < N)[None, :]
    pair = (b * N + i[None, :]) * N + m[:, None]
    d = tl.load(d_ptr + pair, mask=msk, other=1.0)
    rx = tl.load(r_ptr + pair * 3 + 0, mask=msk, other=0.0)
    ry = tl.load(r_ptr + pair * 3 + 1, mask=msk, other=0.0)
    rz = tl.load(r_ptr + pair * 3 + 2, mask=msk, other=0.0)
    diag = m[:, None] == i[None, :]
    dc = tl.minimum(tl.maximum(d, 1e-6), rc)
    x = dc * scale
    fc = tl.where(diag, 0.0, 0.5 * (libdevice.cos(x) + 1.0))
    inrange = (d >= 1e-6) & (d <= rc)
    dfc = tl.where(inrange & (~diag), -0.5 * libdevice.sin(x) * scale, 0.0)
    ux = libdevice.div_rn(rx, d)
    uy = libdevice.div_rn(ry, d)
    uz = libdevice.div_rn(rz, d)
    gdd = tl.zeros((BLOCK_M, BLOCK_I), dtype=tl.float32)
    gux = tl.zeros((BLOCK_M, BLOCK_I), dtype=tl.float32)
    guy = tl.zeros((BLOCK_M, BLOCK_I), dtype=tl.float32)
    guz = tl.zeros((BLOCK_M, BLOCK_I), dtype=tl.float32)
    for s in tl.static_range(S):
        t = d - tl.load(sh_ptr + s)
        e = libdevice.exp(-eta * (t * t))
        gs = e * fc
        base = (((b * S + s) * N + m[:, None]) * N + i[None, :]) * 4
        G0 = tl.load(gg_ptr + base + 0, mask=msk, other=0.0)
        Gx = tl.load(gg_ptr + base + 1, mask=msk, other=0.0)
        Gy = tl.load(gg_ptr + base + 2, mask=msk, other=0.0)
        Gz = tl.load(gg_ptr + base + 3, mask=msk, other=0.0)
        dgs = G0 + Gx * ux + Gy * uy + Gz * uz
        gdd += dgs * (fc * e * (-2.0 * eta * t) + e * dfc)
        gux += gs * Gx
        guy += gs * Gy
        guz += gs * Gz
    gdd += -libdevice.div_rn(gux * rx + guy * ry + guz * rz, d * d)  # u = r / d: the d part
    tl.store(gd_ptr + pair, gdd, mask=msk)
    tl.store(gr_ptr + pair * 3 + 0, libdevice.div_rn(gux, d), mask=msk)  # u = r / d: the r part
    tl.store(gr_ptr + pair * 3 + 1, libdevice.div_rn(guy, d), mask=msk)
    tl.store(gr_ptr + pair * 3 + 2, libdevice.div_rn(guz, d), mask=msk)


def _tiles(n: int) -> tuple[int, int]:
    block_i = triton.next_power_of_2(n)
    return max(1, 512 // block_i), block_i


# =============================================================================
# Autograd Function
# =============================================================================


class AEVSVDense(torch.autograd.Function):
    """``g`` [B, S, N, N, 4] from ``d`` [B, N, N] and ``r`` [B, N, N, 3] (float32); see the module docstring."""

    @staticmethod
    def forward(ctx, d: Tensor, r: Tensor, shifts: Tensor, eta: float, rc: float, scale: float) -> Tensor:
        if d.dtype != torch.float32 or r.dtype != torch.float32:
            raise TypeError("aev_sv_dense requires float32 distances.")
        B, N = d.shape[0], d.shape[1]
        if N > MAX_ATOMS:
            raise ValueError(f"aev_sv_dense supports at most {MAX_ATOMS} atoms per molecule, got {N}.")
        S = shifts.numel()
        d, r, shifts = d.detach().contiguous(), r.detach().contiguous(), shifts.detach().contiguous()
        g = torch.empty((B, S, N, N, 4), dtype=torch.float32, device=d.device)
        block_m, block_i = _tiles(N)
        _aev_sv_dense_fwd_kernel[(B, triton.cdiv(N, block_m))](
            d, r, shifts, g, N, eta, rc, scale, S=S, BLOCK_M=block_m, BLOCK_I=block_i
        )
        ctx.save_for_backward(d, r, shifts)
        ctx.consts = (eta, rc, scale)
        return g

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_g: Tensor):
        d, r, shifts = ctx.saved_tensors
        eta, rc, scale = ctx.consts
        B, N = d.shape[0], d.shape[1]
        S = shifts.numel()
        gd = torch.empty_like(d)
        gr = torch.empty_like(r)
        block_m, block_i = _tiles(N)
        _aev_sv_dense_bwd_kernel[(B, triton.cdiv(N, block_m))](
            d, r, shifts, grad_g.contiguous(), gd, gr, N, eta, rc, scale, S=S, BLOCK_M=block_m, BLOCK_I=block_i
        )
        return gd, gr, None, None, None, None


def aev_sv_dense(d: Tensor, r: Tensor, shifts: Tensor, eta: float, rc: float, scale: float) -> Tensor:
    """AEV basis ``g[b, s, m, i, :] = g_sv[b, i, m, s, :]``; ``eta``, ``rc`` and ``scale = pi / rc`` as float32 values."""
    return AEVSVDense.apply(d, r, shifts, eta, rc, scale)
