"""Static dense input preparation and CUDA replay of the shared calculator computation."""

from __future__ import annotations

import weakref

import torch
from torch import nn

from ._capture import warm_up


def check_dense_model(model):
    if not hasattr(model, "forward_prepared"):
        raise NotImplementedError("model has no prepared-input computation entry point")


class StaticDenseAIMNet2(nn.Module):
    """Own input buffers only; neural network, physics and forces are shared."""

    def __init__(self, calc, numbers, charge, padded=False, d3_cutoff=None):
        """``d3_cutoff``: cutoff of the DFT-D3 neighbor list rebuilt on every call; ``None`` is DFT-D3's dense-mode
        list (``smoothing_off``) that the regular path uses for ``[B, N, 3]`` input. Packed ``[N, 3]`` input passes
        the regular packed path's long-range cutoff (all pairs with full Coulomb), so coordination numbers match."""
        super().__init__()
        check_dense_model(calc.model)
        self._calc = calc if isinstance(calc, weakref.ProxyTypes) else weakref.proxy(calc)
        self.model, self.coul, self.d3 = calc.model, calc.external_coulomb, calc.external_dftd3
        self.B, self.N = numbers.shape
        self.padded = padded
        self.static = {
            "numbers": numbers.to(torch.int64).clone(),
            "charge": charge.to(torch.float32).clone(),
            "_nb_mode": torch.tensor(0),
            "_input_padded": torch.tensor(padded),
            "_inference_atoms_per_system": torch.tensor(self.N),
        }
        self._eye = torch.eye(self.N, device=numbers.device, dtype=torch.bool).unsqueeze(0)
        if not padded:
            self.static.update(
                mask_i=numbers == 0,
                mask_ij=self._eye,
                mask_ij_lr=self._eye,
                _natom=torch.tensor(self.N, device=numbers.device),
                mol_sizes=torch.tensor(self.N, device=numbers.device),
            )
        self._d3_neighbors = None
        if self.d3 is not None:
            from aimnet.calculators.neighbor_buffers import ReusableNeighborMatrix

            b, n = self.B, self.N
            batch = torch.arange(b, device=numbers.device, dtype=torch.int32).repeat_interleave(n)
            ptr = torch.arange(b + 1, device=numbers.device, dtype=torch.int32) * n
            cutoff = float(self.d3.smoothing_off) if d3_cutoff is None else float(d3_cutoff)
            self._d3_neighbors = ReusableNeighborMatrix(cutoff, batch, ptr, max(1, n - 1), b * n)

    def set_inputs(self, numbers, charge):
        if not self.padded:
            raise RuntimeError("set_inputs is only defined for padded layouts")
        self.static["numbers"].copy_(numbers)
        self.static["charge"].copy_(charge)

    def prepare(self, coord):
        data = {**self.static, "coord": coord.detach()}
        if self.padded:
            mask = data["numbers"] == 0
            data.update(
                mask_i=mask,
                mask_ij=self._eye | mask.unsqueeze(-1) | mask.unsqueeze(-2),
                mol_sizes=(~mask).sum(-1),
                _natom=(~mask).sum(-1),
            )
            data["mask_ij_lr"] = data["mask_ij"]
        if self._d3_neighbors is not None:
            matrix, _ = self._d3_neighbors.fill(coord.flatten(0, 1))
            data["_dftd3_neighbor_matrix"] = matrix
            data["_dftd3_batch_idx"] = self._d3_neighbors.batch_idx
        return data

    def _evaluate(self, coord, forces):
        with torch.set_grad_enabled(forces):
            out = self._calc._evaluate_prepared(self.prepare(coord), forces=forces, model_prepared=True)
        return out["energy"].detach(), out.get("forces"), out["charges"].detach()

    def forward(self, coord):
        return self._evaluate(coord, True)

    def energy_only(self, coord):
        return self._evaluate(coord, False)


class GraphedEF:
    """CUDA-graph replay of a static engine for a fixed layout (``forces=False``: energy and charges only).

    Coordinate contents may change between calls (copied into a static buffer); everything else is fixed at
    capture. Returned tensors alias graph-owned memory until the next call (clone to keep)."""

    def __init__(self, engine, coord_example, warmup: int = 1, forces: bool = True, pool=None):
        self.engine = engine
        fn = engine if forces else engine.energy_only
        self.x = coord_example.detach().clone()
        # Warm up and capture on the shared capture stream (``_capture.py``): a throwaway side stream per capture left
        # memory allocated after the calculator was deleted (measured: 210 MiB allocated / 470 MiB reserved left over
        # vs 35 / 80 MiB), and a capture on a stream that was not warmed up pins part of the graph pool.
        stream = warm_up(fn, self.x, warmup)
        self.g = torch.cuda.CUDAGraph()
        # ``pool``: a memory pool shared with other graphs that are never replayed concurrently and whose outputs are
        # copied right after their replay (the cache's contract), so their intermediates can reuse the same memory.
        with torch.cuda.graph(self.g, pool=pool, stream=stream):
            self.e, self.f, self.q = fn(self.x)

    def launch(self, coord, numbers=None, charge=None):
        """Enqueue a replay for ``coord`` (and, for a padded layout, ``numbers``/``charge``) without waiting."""
        self.x.copy_(coord)
        if numbers is not None:
            self.engine.set_inputs(numbers, charge)
        self.g.replay()

    def result(self):
        """Outputs of the last launch (graph-owned; stream-ordered after it)."""
        return self.e, self.f, self.q

    def __call__(self, coord, numbers=None, charge=None):
        self.launch(coord, numbers, charge)
        return self.result()
