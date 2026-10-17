"""Full pairwise Coulomb energy rows without an N x N pair tensor (Warp).

Reference: ``LRCoulomb.coul_simple`` (``aimnet.modules.lr``) on the all-pairs neighbor matrix with
``subtract_sr=False``::

    e_ij = (q_i * q_j) / d_ij           float32, i != j, same molecule
    e_i  = sum_j e_ij                   accumulated in float64
    E    = factor * mol_sum(e_i)        (applied by the caller)

The reference materializes the ``[T, T-1]`` pair tensors and their gradients; these kernels stream over the pairs.
Layout: ``B`` molecules of ``n_mol`` atoms each, flattened (``n_real = B * n_mol``), followed by padding rows
(zero output, zero gradient).

Backward (analytic, first order), with ``g_i = float32(dL/de_i)``::

    dL/dq_i = sum_j (g_i + g_j) q_j / d_ij
    dL/dR_i = sum_j (g_i + g_j) q_i q_j (R_j - R_i) / d_ij^3

The forward sums every pair term in float64, as the reference does. The backward sums float32 chunks of ``CHUNK``
terms per lane and float64 across chunks and lanes (the reference accumulates these gradients with float32 atomics).
Not differentiable twice.
"""

import torch
import warp as wp
from torch import Tensor
from torch.autograd.function import once_differentiable

LANES = 64  # threads per atom row (block size); a row is reduced with a block-level tile sum
CHUNK = 16  # float32 terms per lane before a float64 flush (backward)


def _get_stream(device: torch.device):
    """Get the Warp stream for the given device."""
    if device.type == "cuda":
        return wp.stream_from_torch(torch.cuda.current_stream(device))
    return None


# =============================================================================
# Warp Kernels
# =============================================================================


@wp.kernel(enable_backward=False)
def _coulomb_rows_fwd_kernel(
    x: wp.array(dtype=wp.vec3f),  # (T,)
    q: wp.array(dtype=wp.float32),  # (T,)
    n_mol: wp.int32,
    n_real: wp.int32,
    lanes: wp.int32,
    e: wp.array(dtype=wp.float64),  # (T,)
):
    """e[i] = sum_{j != i, same molecule} q_i q_j / |R_j - R_i|"""
    i, lane = wp.tid()
    if i >= n_real:
        return
    start = (i // n_mol) * n_mol
    end = start + n_mol
    xi = x[i]
    qi = q[i]
    acc = wp.float64(0.0)
    j = start + lane
    while j < end:
        if j != i:
            d = wp.length(x[j] - xi)
            acc += wp.float64((qi * q[j]) / d)
        j += lanes
    s = wp.tile_sum(wp.tile(acc))[0]
    if lane == 0:
        e[i] = s


@wp.kernel(enable_backward=False)
def _coulomb_rows_bwd_kernel(
    x: wp.array(dtype=wp.vec3f),  # (T,)
    q: wp.array(dtype=wp.float32),  # (T,)
    g: wp.array(dtype=wp.float64),  # (T,) dL/de
    n_mol: wp.int32,
    n_real: wp.int32,
    lanes: wp.int32,
    chunk: wp.int32,
    gx: wp.array(dtype=wp.vec3f),  # (T,) dL/dR
    gq: wp.array(dtype=wp.float32),  # (T,) dL/dq
):
    """Analytic first-order backward of ``_coulomb_rows_fwd_kernel`` (see the module docstring)."""
    i, lane = wp.tid()
    if i >= n_real:
        return
    start = (i // n_mol) * n_mol
    end = start + n_mol
    xi = x[i]
    qi = q[i]
    gi = wp.float32(g[i])
    sq = wp.float64(0.0)
    sx = wp.float64(0.0)
    sy = wp.float64(0.0)
    sz = wp.float64(0.0)
    cq = wp.float32(0.0)
    cx = wp.float32(0.0)
    cy = wp.float32(0.0)
    cz = wp.float32(0.0)
    k = wp.int32(0)  # a dynamic variable (a plain 0 literal would be a Warp constant)
    j = start + lane
    while j < end:
        if j != i:
            r = x[j] - xi
            d = wp.length(r)
            inv = 1.0 / d
            w = gi + wp.float32(g[j])
            qj = q[j]
            cq += w * qj * inv
            b = w * qi * qj * inv * inv * inv
            cx += b * r[0]
            cy += b * r[1]
            cz += b * r[2]
            k += 1
            if k == chunk:
                sq += wp.float64(cq)
                sx += wp.float64(cx)
                sy += wp.float64(cy)
                sz += wp.float64(cz)
                cq = wp.float32(0.0)
                cx = wp.float32(0.0)
                cy = wp.float32(0.0)
                cz = wp.float32(0.0)
                k = wp.int32(0)
        j += lanes
    sq += wp.float64(cq)
    sx += wp.float64(cx)
    sy += wp.float64(cy)
    sz += wp.float64(cz)
    tq = wp.tile_sum(wp.tile(sq))[0]
    tx = wp.tile_sum(wp.tile(sx))[0]
    ty = wp.tile_sum(wp.tile(sy))[0]
    tz = wp.tile_sum(wp.tile(sz))[0]
    if lane == 0:
        gq[i] = wp.float32(tq)
        gx[i] = wp.vec3f(wp.float32(tx), wp.float32(ty), wp.float32(tz))


# =============================================================================
# Autograd Function
# =============================================================================


class FullCoulombRows(torch.autograd.Function):
    """``e`` [T] float64 (padding rows 0) from ``coord`` [T, 3] and ``charges`` [T], both float32."""

    @staticmethod
    def forward(ctx, coord: Tensor, charges: Tensor, n_mol: int, n_real: int) -> Tensor:
        if coord.device.type != "cuda" or charges.device != coord.device:
            raise ValueError("full_coulomb_rows requires coordinates and charges on the same CUDA device.")
        if coord.dtype != torch.float32 or charges.dtype != torch.float32:
            raise TypeError("full_coulomb_rows requires float32 coordinates and charges.")
        if coord.ndim != 2 or coord.shape[1] != 3 or charges.ndim != 1 or charges.shape[0] != coord.shape[0]:
            raise ValueError("full_coulomb_rows expects coord[T, 3] and charges[T].")
        if (
            not isinstance(n_mol, int)
            or not isinstance(n_real, int)
            or n_mol < 1
            or n_real < 1
            or n_real > coord.shape[0]
            or n_real % n_mol
        ):
            raise ValueError("n_mol must be positive atoms/system and divide n_real <= T.")
        x = coord.detach().contiguous()
        q = charges.detach().contiguous()
        e = torch.zeros(x.shape[0], dtype=torch.float64, device=x.device)
        wp.launch(
            _coulomb_rows_fwd_kernel,
            dim=(n_real, LANES),
            block_dim=LANES,
            stream=_get_stream(x.device),
            device=wp.device_from_torch(x.device),
            inputs=(
                wp.from_torch(x, return_ctype=True, dtype=wp.vec3f),
                wp.from_torch(q, return_ctype=True),
                n_mol,
                n_real,
                LANES,
                wp.from_torch(e, return_ctype=True),
            ),
        )
        ctx.save_for_backward(x, q)
        ctx.n_mol, ctx.n_real = n_mol, n_real
        return e

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_e: Tensor):
        x, q = ctx.saved_tensors
        gx = torch.zeros_like(x)
        gq = torch.zeros_like(q)
        g = grad_e.detach().to(torch.float64).contiguous()
        wp.launch(
            _coulomb_rows_bwd_kernel,
            dim=(ctx.n_real, LANES),
            block_dim=LANES,
            stream=_get_stream(x.device),
            device=wp.device_from_torch(x.device),
            inputs=(
                wp.from_torch(x, return_ctype=True, dtype=wp.vec3f),
                wp.from_torch(q, return_ctype=True),
                wp.from_torch(g, return_ctype=True),
                ctx.n_mol,
                ctx.n_real,
                LANES,
                CHUNK,
                wp.from_torch(gx, return_ctype=True, dtype=wp.vec3f),
                wp.from_torch(gq, return_ctype=True),
            ),
        )
        return gx, gq, None, None


def full_coulomb_rows(coord: Tensor, charges: Tensor, n_mol: int, n_real: int) -> Tensor:
    """Per-atom full Coulomb rows ``e_i = sum_j q_i q_j / d_ij`` (float64; multiply by the unit factor and sum)."""
    return FullCoulombRows.apply(coord, charges, n_mol, n_real)
