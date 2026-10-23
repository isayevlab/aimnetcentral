"""Packed input preparation and CUDA replay of the canonical AIMNet computation.

Neighbor construction uses public reusable nvalchemi buffers. Only actual consumers
own long-range lists; streaming Coulomb itself needs no all-pairs matrix. D3 retains
exactly the reference neighbor policy, including its coordination-number support.
"""

from __future__ import annotations

import math
import weakref

import torch
from torch import nn

from aimnet.calculators.neighbor_buffers import ReusableNeighborMatrix

from ._capture import warm_up


def _round16(n):
    return ((int(n) + 15) // 16) * 16


def check_sparse_model(model, coul):
    if not hasattr(model, "forward_prepared"):
        raise NotImplementedError("model has no prepared-input computation entry point")
    if coul is not None and (coul.method != "simple" or coul.subtract_sr):
        raise NotImplementedError("only full simple Coulomb is captured")


class StaticSparseAIMNet2(nn.Module):
    """Static packed metadata, variable coordinates, and reusable neighbor buffers."""

    def __init__(self, calc, numbers, charge, coord_example, capacity=None, margin=1.15, warp_coulomb=None):
        super().__init__()
        check_sparse_model(calc.model, calc.external_coulomb)
        self._calc = calc if isinstance(calc, weakref.ProxyTypes) else weakref.proxy(calc)
        self.model, self.coul, self.d3 = calc.model, calc.external_coulomb, calc.external_dftd3
        b, n = numbers.shape
        t, dev = b * n, numbers.device
        self.B, self.N, self.T = b, n, t
        batch = torch.arange(b, dtype=torch.int32, device=dev).repeat_interleave(n)
        ptr = torch.arange(b + 1, dtype=torch.int32, device=dev) * n
        self.mol_idx = torch.cat((batch, batch[-1:])).long()
        self.numbers_pad = torch.cat((numbers.flatten(), numbers.new_zeros(1))).long()
        self.charge = charge.float().clone()
        self.cutoff = float(calc.cutoff)
        x0 = coord_example.reshape(t, 3)
        if capacity is None:
            trial = ReusableNeighborMatrix(self.cutoff, batch, ptr, min(_round16(n), 512), t)
            _, counts = trial.fill(x0)
            capacity = _round16(max(16, math.ceil(int(counts.max()) * margin)))
        self.capacity = int(capacity)
        self.neighbors = ReusableNeighborMatrix(self.cutoff, batch, ptr, self.capacity, t)
        self._nb_buf, self._nn_buf = self.neighbors.matrix, self.neighbors.counts
        self._pad_row = torch.full((1, self.capacity), t, dtype=torch.int32, device=dev)
        self.mask_i = torch.arange(t + 1, device=dev) == t
        self.static = {
            "numbers": self.numbers_pad,
            "charge": self.charge,
            "mol_idx": self.mol_idx,
            "_nb_mode": torch.tensor(1),
            "_num_mol": torch.tensor(b),
            "_input_padded": torch.tensor(True),
            "mol_sizes": torch.full((b,), n, device=dev, dtype=torch.long),
            "mask_i": self.mask_i,
            "_inference_atoms_per_system": torch.tensor(n),
        }
        # The regular simple-Coulomb path shares all pairs with D3. Keep those
        # pairs for D3 CN even when Coulomb switches to its streaming kernel.
        use_stream = calc.inference_kernels and not calc._deterministic
        need_pairs = self.coul is not None and (self.d3 is not None or not use_stream)
        self.nbmat_lr = None
        self._d3_neighbors = None
        if need_pairs:
            indices = torch.arange(n, device=dev, dtype=torch.int32)
            template = indices.expand(n, n)[~torch.eye(n, device=dev, dtype=torch.bool)].reshape(n, n - 1)
            pairs = (template.unsqueeze(0) + torch.arange(b, device=dev, dtype=torch.int32).view(b, 1, 1) * n).reshape(
                t, n - 1
            )
            self.nbmat_lr = torch.cat((pairs, torch.full((1, n - 1), t, dtype=torch.int32, device=dev)))
            mask = self.nbmat_lr == t
            self.static.update(nbmat_lr=self.nbmat_lr, mask_ij_lr=mask)
            if self.coul is not None and not use_stream:
                self.static.update(nbmat_coulomb=self.nbmat_lr, mask_ij_coulomb=mask)
            if self.d3 is not None:
                self.static.update(nbmat_dftd3=self.nbmat_lr, mask_ij_dftd3=mask)
        elif self.d3 is not None:
            self._d3_neighbors = ReusableNeighborMatrix(float(calc._dftd3_cutoff), batch, ptr, max(1, n - 1), t)

    def prepare(self, coord):
        x = torch.cat((coord.reshape(self.T, 3), coord.new_zeros((1, 3)))).detach()
        matrix, counts = self.neighbors.fill(x[: self.T])
        nbmat = torch.cat((matrix, self._pad_row))
        data = {**self.static, "coord": x, "nbmat": nbmat, "mask_ij": nbmat == self.T}
        if self._d3_neighbors is not None:
            pairs, _ = self._d3_neighbors.fill(x[: self.T])
            pairs = torch.cat((pairs, torch.full_like(pairs[:1], self.T)))
            data.update(nbmat_dftd3=pairs, mask_ij_dftd3=pairs == self.T)
        return data, counts.max() > self.capacity

    def _evaluate(self, coord, forces):
        with torch.set_grad_enabled(forces):
            data, overflow = self.prepare(coord)
            out = self._calc._evaluate_prepared(data, forces=forces, model_prepared=True)
        force = out["forces"][: self.T].reshape(self.B, self.N, 3) if forces else None
        return out["energy"].detach(), force, out["charges"][: self.T].detach().reshape(self.B, self.N), overflow

    def forward(self, coord):
        return self._evaluate(coord, True)

    def energy_only(self, coord):
        return self._evaluate(coord, False)


class NeighborCapacityError(RuntimeError):
    pass


class GraphedSparseEF:
    """CUDA-graph replay of ``StaticSparseAIMNet2`` (fixed layout and neighbor capacity).

    After every replay the overflow flag is read (one sync; the caller reads the results anyway); on overflow
    ``NeighborCapacityError`` is raised."""

    def __init__(self, engine, coord_example, warmup: int = 1, forces: bool = True, pool=None):
        self.engine = engine
        fn = engine if forces else engine.energy_only
        self.x = coord_example.detach().clone()
        stream = warm_up(fn, self.x, warmup)  # as GraphedEF (dense.py)
        self.g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.g, pool=pool, stream=stream):
            self.e, self.f, self.q, self.overflow = fn(self.x)

    def launch(self, coord):
        """Enqueue a replay for ``coord`` without waiting for it."""
        self.x.copy_(coord)
        self.g.replay()

    def result(self, check: bool = True):
        """Outputs of the last launch; reading the overflow flag waits for the replay."""
        if check and bool(self.overflow):
            raise NeighborCapacityError(f"5 A neighbor capacity {self.engine.capacity} exceeded")
        return self.e, self.f, self.q

    def __call__(self, coord, check: bool = True):
        self.launch(coord)
        return self.result(check)


class AutoGraphedSparseEF:
    """``GraphedSparseEF`` that rebuilds itself when a call overflows the neighbor capacity.

    The overflowing call's result is discarded, engine + graph are rebuilt with capacity from the true count (the
    nvalchemiops kernel counts past capacity), and the same coordinates are evaluated again.

    ``pool``: owner of the graph memory pool (anything with an ``id``, e.g. ``torch.cuda.MemPool``), read at every
    capture, so a rebuild fails instead of capturing into a pool its owner has released."""

    def __init__(
        self, calc, numbers, charge, coord_example, margin: float = 1.15, forces: bool = True, pool=None, **engine_kw
    ):
        # Keep rebuilds independent of mutations to the caller's input buffers.
        self._args = (calc, numbers.detach().clone(), charge.detach().clone())
        self._kw, self.margin, self.forces, self._pool = engine_kw, margin, forces, pool
        self.rebuilds = 0
        self._launched = None
        self._build(coord_example, None)

    def _build(self, coord, capacity):
        calc, numbers, charge = self._args
        engine = StaticSparseAIMNet2(calc, numbers, charge, coord, capacity=capacity, margin=self.margin, **self._kw)
        pool = None if self._pool is None else self._pool.id
        self.graph = GraphedSparseEF(engine, coord, forces=self.forces, pool=pool)

    @property
    def capacity(self) -> int:
        return self.graph.engine.capacity

    def launch(self, coord):
        """Enqueue a replay for ``coord`` without waiting for it (``result`` handles an overflow)."""
        self._launched = coord
        self.graph.launch(coord)

    def result(self):
        coord, self._launched = self._launched, None
        try:
            return self.graph.result()
        except NeighborCapacityError:
            need = int(self.graph.engine._nn_buf.max())
        # Rebuild after the handler: its traceback references the old graph, whose pool blocks the new capture reuses
        # once it is gone. The caller accounts for the rebuilt graph's memory.
        self.graph = None
        self._build(coord, _round16(math.ceil(need * self.margin)))
        self.rebuilds += 1
        return self.graph(coord)

    def __call__(self, coord):
        self.launch(coord)
        return self.result()
