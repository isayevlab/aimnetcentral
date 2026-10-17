"""Static-layout, host-sync-free E+F for AIMNet2 in aimnetcentral's neighbor-list mode (N > nb_threshold).

Reproduces ``AIMNet2Calculator.eval`` for inputs ``[B, N, 3]`` with ``N > nb_threshold`` (aimnetcentral nb_mode 1):
``B`` equal-size systems flattened, a 5 A neighbor matrix, one padding atom, and one all-pairs matrix shared by the
full Coulomb and DFT-D3 terms; forces ``-dE/dR`` by autograd.

Differences from the regular code path (the arithmetic of every term is unchanged):

* Layout-only quantities (molecule index, batch pointers, sizes, padding masks, the all-pairs matrix whose content
  depends only on the layout) are computed once instead of on every call.
* The 5 A neighbor matrix is filled every call by the same nvalchemiops kernel into a fixed-capacity buffer
  (the regular path trims it to the per-call maximum, a host sync). The kernel keeps counting past the capacity, so an
  overflow is detected on the device and reported; ``AutoGraphedSparseEF`` then rebuilds with a larger capacity.
* The full Coulomb rows come from a Warp kernel that streams over pairs (``aimnet.kernels.coulomb_wp``) instead
  of materializing the ``[T, T-1]`` pair tensors and their gradients. The regular-path math is kept behind
  ``warp_coulomb=False`` for validation.
* DFT-D3 is the explicit-terms kernel call with its forces injected into autograd, i.e. the path ``DFTD3`` takes for
  ``coord`` that requires grad (``aimnet.modules.lr._DFTD3EnergyFunction``).
"""

from __future__ import annotations

import math

import torch
from torch import nn

from ._warp import dftd3_energy, torch_bound_warp_stream


def _round16(n: int) -> int:
    return ((int(n) + 15) // 16) * 16


def check_sparse_model(model, coul) -> None:
    """Raise NotImplementedError unless the model / Coulomb setup is the one this engine reproduces."""
    if getattr(model, "num_charge_channels", 1) != 1:
        raise NotImplementedError("closed-shell (one charge channel) models only")
    names = [name for name, _ in model.outputs.named_children()]
    if not set(names) <= {"energy_mlp", "atomic_shift", "atomic_sum", "srcoulomb"}:
        raise NotImplementedError(f"output modules {names} not supported")
    if "srcoulomb" in names and model.outputs.srcoulomb.envelope != "exp":
        raise NotImplementedError("exp short-range Coulomb envelope only")
    if coul is not None and (coul.method != "simple" or coul.subtract_sr):
        raise NotImplementedError("full 'simple' Coulomb with embedded short range only")


class StaticSparseAIMNet2(nn.Module):
    """E+F for a fixed layout: ``numbers`` [B, N] (all atoms real), ``charge`` [B]; coordinates vary per call."""

    def __init__(
        self,
        calc,
        numbers,
        charge,
        coord_example,
        capacity: int | None = None,
        margin: float = 1.15,
        warp_coulomb: bool = True,
    ):
        super().__init__()
        from nvalchemiops.neighbors import NeighborOverflowError
        from nvalchemiops.torch.neighbors import neighbor_list
        from nvalchemiops.torch.neighbors.batch_naive import _batch_naive_neighbor_matrix_no_pbc

        from aimnet import constants, nbops, ops

        if numbers.ndim != 2 or bool((numbers <= 0).any()):
            raise ValueError("static layout requires unpadded numbers[B, N]")
        model = calc.model
        check_sparse_model(model, calc.external_coulomb)
        self._nbops, self._ops = nbops, ops
        self._nb_fill = _batch_naive_neighbor_matrix_no_pbc
        self.model, self.coul, self.d3 = model, calc.external_coulomb, calc.external_dftd3
        self.warp_coulomb = warp_coulomb
        self.cutoff = float(calc.cutoff)
        dev = numbers.device
        B, N = numbers.shape
        T = B * N  # real atoms; index T is the padding atom
        self.B, self.N, self.T = B, N, T
        mol_idx = torch.repeat_interleave(torch.arange(B, device=dev), N)
        self.mol_idx = torch.cat((mol_idx, mol_idx[-1:]))
        self.mol_idx_i32 = self.mol_idx.to(torch.int32)
        self.batch_idx = mol_idx.to(torch.int32)
        self.batch_ptr = torch.arange(0, T + 1, N, device=dev, dtype=torch.int32)
        self.mol_sizes = torch.full((B,), N, device=dev, dtype=torch.long)
        self.numbers_pad = torch.cat((numbers.flatten(), numbers.new_zeros(1))).to(torch.int64)
        self.numbers_pad_i32 = self.numbers_pad.to(torch.int32)
        self.charge = charge.to(torch.float32)
        self._rc_s = model.aev.rc_s.detach().to("cpu")
        self._factor = constants.half_Hartree * constants.Bohr
        x0 = coord_example.reshape(T, 3).to(torch.float32)
        # all-pairs matrix exactly as the regular path builds it (cutoff 1e6, max_neighbors = T), trimmed to
        # its true width, plus the padding row
        with torch_bound_warp_stream(dev):
            nbl, nnl = neighbor_list(
                positions=x0,
                cutoff=1e6,
                batch_idx=self.batch_idx,
                max_neighbors=T,
                half_fill=False,
                fill_value=T,
                method="batch_naive",
            )
        w = max(1, int(nnl.max().item()))
        nbl = torch.cat((nbl[:, :w], torch.full((1, w), T, dtype=nbl.dtype, device=dev)))
        self.nbmat_lr = nbl.to(torch.int64)
        self.nbmat_lr_i32 = nbl.to(torch.int32)
        self.mask_ij_lr = self.nbmat_lr == T
        if capacity is None:  # 5 A capacity from the example geometry
            trial = min(_round16(N), 512)
            while True:
                try:
                    with torch_bound_warp_stream(dev):
                        _, nns = neighbor_list(
                            positions=x0,
                            cutoff=self.cutoff,
                            batch_idx=self.batch_idx,
                            max_neighbors=trial,
                            half_fill=False,
                            fill_value=T,
                            method="batch_naive",
                        )
                    break
                except NeighborOverflowError:
                    trial = _round16(trial * 2)
            capacity = _round16(max(16, math.ceil(int(nns.max().item()) * margin)))
        self.capacity = int(capacity)
        self._nb_buf = torch.empty((T, self.capacity), dtype=torch.int32, device=dev)
        self._nn_buf = torch.empty((T,), dtype=torch.int32, device=dev)
        self._pad_row = torch.full((1, self.capacity), T, dtype=torch.int32, device=dev)

    # --- static-layout helpers (same arithmetic as aimnet nbops/ops in nb_mode 1) -----------------------------
    def _mol_sum(self, x):
        idx = self.mol_idx
        if x.ndim == 1:
            res = torch.zeros(self.B, device=x.device, dtype=x.dtype)
        else:
            idx = idx.unsqueeze(-1).expand(-1, x.shape[1])
            res = torch.zeros(self.B, x.shape[1], device=x.device, dtype=x.dtype)
        return res.scatter_add_(0, idx, x)

    def _nse(self, Q, q_u, f_u, epsilon=1.0e-6):  # aimnet.ops.nse, nb_mode 1
        F_u = self._mol_sum(f_u) + epsilon
        dQ = Q - self._mol_sum(q_u)
        F_u = torch.index_select(F_u, 0, self.mol_idx)
        dQ = torch.index_select(dQ, 0, self.mol_idx)
        return q_u + (f_u / F_u) * dQ

    def _update_q(self, data, x, delta_q):  # AIMNet2._update_q with the static helpers
        c = self.model.num_charge_channels
        _q, _f, delta_a = x.split([c, c, x.shape[-1] - 2 * c], dim=-1)
        q = data["charges"] + _q if delta_q else _q
        data["charges"] = self._nse(data["charge"], q, _f.pow(2))
        data["a"] = data["a"] + delta_a.view_as(data["a"])
        return data

    @staticmethod
    def _zero_pad_row(x):
        x[-1].zero_()
        return x

    def _neighbors(self, coord_real):
        self._nb_buf.fill_(self.T)
        self._nn_buf.zero_()
        with torch_bound_warp_stream(coord_real.device):  # ordered with the torch fills / gathers around it
            self._nb_fill(coord_real, self.cutoff, self.batch_idx, self.batch_ptr, self._nb_buf, self._nn_buf, False)
        nbmat = torch.cat((self._nb_buf, self._pad_row)).to(torch.int64)
        overflow = self._nn_buf.max() > self.capacity  # the kernel counts past capacity
        return nbmat, overflow

    def energy(self, coord_pad):
        """coord_pad [T+1, 3] (last row: padding atom at the origin) -> (energy [B] float64, data, overflow)."""
        m, ops = self.model, self._ops
        T = self.T
        nbmat, overflow = self._neighbors(coord_pad[:T].detach())
        data = {
            "coord": coord_pad,
            "numbers": self.numbers_pad,
            "charge": self.charge,
            "mol_idx": self.mol_idx,
            "nbmat": nbmat,
            "nbmat_lr": self.nbmat_lr,
            "nbmat_coulomb": self.nbmat_lr,
            "nbmat_dftd3": self.nbmat_lr,
        }
        data["_nb_mode"] = torch.tensor(1)
        data["_num_mol"] = torch.tensor(self.B)
        mask_i = torch.zeros(T + 1, device=coord_pad.device, dtype=torch.bool)
        mask_i[-1:] = True
        data["mask_i"] = mask_i
        data["mask_ij"] = nbmat == T
        data["mask_ij_lr"] = data["mask_ij_coulomb"] = data["mask_ij_dftd3"] = self.mask_ij_lr
        data["_input_padded"] = torch.tensor(True)
        data["mol_sizes"] = self.mol_sizes
        a = m.afv(data["numbers"])
        if m.d2features:
            a = a.unflatten(-1, (m.nfeature, m.nshifts_s))
        data["a"] = a
        data["charge"] = data["charge"].unsqueeze(-1)
        d_ij, r_ij = ops.calc_distances(data)
        data["d_ij"] = d_ij
        rc = self._rc_s
        fc_ij = 0.5 * (torch.cos(d_ij.clamp(min=1e-6, max=rc) * (torch.pi / rc)) + 1.0)
        fc_ij = fc_ij.masked_fill(data["mask_ij"], 0.0)
        aev = m.aev
        gs = ops.exp_expand(d_ij, aev.shifts_s, aev.eta_s) * fc_ij.unsqueeze(-1)
        u_ij = r_ij / d_ij.unsqueeze(-1)
        data["g_sv"] = torch.cat([gs.unsqueeze(-1), gs.unsqueeze(-1) * u_ij.unsqueeze(-2)], dim=-1)
        npass = len(m.mlps)
        for ipass, mlp in enumerate(m.mlps):
            _in = (
                m._prepare_in_a(data)
                if ipass == 0
                else torch.cat([m._prepare_in_a(data), m._prepare_in_q(data)], dim=-1)
            )
            _out = self._zero_pad_row(mlp(_in))
            if ipass == 0:
                data = self._update_q(data, _out, delta_q=False)
            elif ipass < npass - 1:
                data = self._update_q(data, _out, delta_q=True)
            else:
                data["aim"] = _out
        data["charges"] = data["charges"].squeeze(-1)
        data["charge"] = data["charge"].squeeze(-1)
        for name, mod in m.outputs.named_children():
            if name == "energy_mlp":
                data[mod.key_out] = self._zero_pad_row(mod.mlp(data[mod.key_in]).squeeze(-1))
            elif name == "atomic_sum":
                data[mod.key_out] = self._mol_sum(data[mod.key_in])
            elif name == "srcoulomb":
                data = self._sr_coulomb(mod, data)
            else:  # atomic_shift
                data = mod(data)
        if self.coul is not None:
            data = self._lr_coulomb(data)
        if self.d3 is not None:
            e_d3 = dftd3_energy(
                coord_pad, self.d3, self.numbers_pad_i32, self.mol_idx_i32, self.nbmat_lr_i32, self.B, T + 1
            )
            data["energy"] = data["energy"] + e_d3.to(data["energy"].dtype)
        return data["energy"], data, overflow

    def _sr_coulomb(self, mod, data):  # aimnet.modules.lr._calc_coulomb_sr, exp envelope
        nbops, ops = self._nbops, self._ops
        q_i, q_j = nbops.get_ij(data[mod.key_in], data)
        e_ij = ops.exp_cutoff(data["d_ij"], mod.rc) * (q_i * q_j) / data["d_ij"]
        e_ij = nbops.mask_ij_(e_ij, data, 0.0)
        e_sr = self._factor * self._mol_sum(e_ij.sum(-1, dtype=torch.float64))
        data[mod.key_out] = data[mod.key_out].double() - e_sr
        return data

    def _lr_coulomb(self, data):  # LRCoulomb.coul_simple, subtract_sr=False
        c = self.coul
        if self.warp_coulomb:
            from aimnet.kernels.coulomb_wp import full_coulomb_rows

            e_i = full_coulomb_rows(data["coord"], data[c.key_in], self.N, self.T)
        else:
            nbops, ops = self._nbops, self._ops
            data = ops.lazy_calc_dij(data, "_coulomb")
            q_i, q_j = nbops.get_ij(data[c.key_in], data, suffix="_coulomb")
            e_ij = nbops.mask_ij_(q_i * q_j / data["d_ij_coulomb"], data, 0.0, suffix="_coulomb")
            e_i = e_ij.sum(-1, dtype=torch.float64)
        data[c.key_out] = data[c.key_out].double() + self._factor * self._mol_sum(e_i)
        return data

    def forward(self, coord):
        """coord [B, N, 3] -> energy [B] (float64), forces [B, N, 3], charges [B, N], overflow flag (device bool)."""
        B, N, T = self.B, self.N, self.T
        with torch.enable_grad():
            x = torch.cat((coord.reshape(T, 3).to(torch.float32), coord.new_zeros((1, 3))))
            x = x.detach().requires_grad_(True)
            e, data, overflow = self.energy(x)
            g = torch.autograd.grad(e.sum(), x)[0]
        return e.detach(), -g[:T].reshape(B, N, 3), data["charges"][:T].detach().reshape(B, N), overflow

    def energy_only(self, coord):
        """Energy [B] (float64), None, charges [B, N], overflow flag: ``forward`` without the backward."""
        B, N, T = self.B, self.N, self.T
        with torch.no_grad():
            x = torch.cat((coord.reshape(T, 3).to(torch.float32), coord.new_zeros((1, 3))))
            e, data, overflow = self.energy(x)
        return e, None, data["charges"][:T].reshape(B, N), overflow


class NeighborCapacityError(RuntimeError):
    pass


class GraphedSparseEF:
    """CUDA-graph replay of ``StaticSparseAIMNet2`` (fixed layout and neighbor capacity).

    After every replay the overflow flag is read (one sync; the caller reads the results anyway); on overflow
    ``NeighborCapacityError`` is raised."""

    def __init__(self, engine, coord_example, warmup: int = 3, forces: bool = True):
        self.engine = engine
        fn = engine if forces else engine.energy_only
        self.x = coord_example.detach().clone()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(warmup):
                fn(self.x)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        self.g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.g), torch_bound_warp_stream(self.x.device):
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
    nvalchemiops kernel counts past capacity), and the same coordinates are evaluated again."""

    def __init__(self, calc, numbers, charge, coord_example, margin: float = 1.15, forces: bool = True, **engine_kw):
        # Keep rebuilds independent of mutations to the caller's input buffers.
        self._args = (calc, numbers.detach().clone(), charge.detach().clone())
        self._kw, self.margin, self.forces = engine_kw, margin, forces
        self.rebuilds = 0
        self.unaccounted_memory_growth = 0
        self._launched = None
        self._build(coord_example, None)

    def _build(self, coord, capacity):
        calc, numbers, charge = self._args
        engine = StaticSparseAIMNet2(calc, numbers, charge, coord, capacity=capacity, margin=self.margin, **self._kw)
        self.graph = GraphedSparseEF(engine, coord, forces=self.forces)

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
            before = torch.cuda.memory_reserved(coord.device)
            self.graph = None
            self._build(coord, _round16(math.ceil(need * self.margin)))
            self.rebuilds += 1
            self.unaccounted_memory_growth += max(0, torch.cuda.memory_reserved(coord.device) - before)
            return self.graph(coord)

    def __call__(self, coord):
        self.launch(coord)
        return self.result()
