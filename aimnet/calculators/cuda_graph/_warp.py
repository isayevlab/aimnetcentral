"""Warp interop for the CUDA-graph engines.

nvalchemiops' torch entry points (neighbor matrix, DFT-D3) launch Warp kernels on Warp's current stream, not on
torch's. ``torch_bound_warp_stream`` makes torch's current stream Warp's current stream around those calls:

* eagerly, so the launches are ordered with the surrounding torch kernels (unbound, the neighbor-matrix launch raced
  the torch ``fill_`` and gathers around it and produced an intermittent illegal memory access);
* during CUDA-graph capture, so the launches are recorded into the graph instead of running once outside it
  (unbound, graph replay silently reused stale DFT-D3 results).

Kernels launched by aimnet itself pass ``stream=`` explicitly (``aimnet.kernels``).
"""

from __future__ import annotations

import torch
import warp as wp


def torch_bound_warp_stream(device: torch.device):
    """Context manager: Warp's current stream := torch's current stream on ``device``.

    No entry synchronization while capturing (it would make the capture depend on uncaptured work)."""
    stream = torch.cuda.current_stream(device)
    return wp.ScopedStream(wp.stream_from_torch(stream), sync_enter=not torch.cuda.is_current_stream_capturing())


def dftd3_energy(coord, d3, numbers, batch_idx, neighbor_matrix, num_systems: int, fill_value: int):
    """Per-system DFT-D3(BJ) energy (eV, float64) computed on torch's current stream.

    This is the path ``DFTD3`` takes for ``coord`` that requires grad: the explicit-terms kernel call with its forces
    injected into autograd (``aimnet.modules.lr._DFTD3EnergyFunction``). ``coord`` [M, 3] float32; int32 ``numbers``,
    ``batch_idx`` and ``neighbor_matrix`` (padded with ``fill_value``)."""
    from aimnet.modules.lr import _DFTD3EnergyFunction

    with torch_bound_warp_stream(coord.device):
        return _DFTD3EnergyFunction.apply(
            coord,
            None,
            None,
            numbers,
            batch_idx,
            neighbor_matrix,
            None,
            int(fill_value),
            int(num_systems),
            d3.rcov,
            d3.r4r2,
            d3.c6ab,
            d3.cn_ref,
            float(d3.a1),
            float(d3.a2),
            float(d3.s8),
            float(d3.s6),
            float(d3.smoothing_on),
            float(d3.smoothing_off),
        )
