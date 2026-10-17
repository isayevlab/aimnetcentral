"""Static-layout, host-sync-free E+F for AIMNet2 in aimnetcentral's dense mode (N <= nb_threshold), pure PyTorch.

Reproduces ``AIMNet2Calculator.eval`` for unpadded inputs ``[B, N, 3]`` (aimnetcentral nb_mode 0):
``E = E_NN + atomic shifts - SRCoulomb(exp) + full Coulomb + DFT-D3(BJ)``, forces ``-dE/dR`` by autograd.

Differences from the regular code path (the arithmetic of every term is unchanged):

* Layout-only fields (masks, sizes, the D3 all-pairs template) are built once for a fixed ``(numbers, charge)``
  instead of on every call; no ``.item()`` / pageable host-to-device copies remain, so the whole E+F can be
  captured in one CUDA graph (``GraphedEF``).
* The radial-angular basis ``g_sv[b, i, m, s, :]`` is built directly in the layout ``torch.einsum`` permutes it to
  inside ``ConvSV`` (``[B, S, M, I, 4]``), and both convolutions call that ``bmm`` without the permute copy. The
  ``conv_a`` contraction is the identical ``bmm`` on identical operands (bitwise equal to the einsum of the regular path);
  ``conv_q`` batches the same contraction over ``(b, s)`` (float32 rounding level).
* The five ``g_sv`` uses (conv_a in every pass, conv_q from pass 1) return one gradient computed as a single
  ``bmm`` (``_SharedGBmm``) instead of five 4-D gradients summed by autograd.
* With Triton available, ``g_sv`` and its gradient are computed by one kernel each way
  (``aimnet.kernels.aev_sv_dense_triton``; forward bitwise equal to the PyTorch expression) instead of a chain of
  elementwise passes over the ``[B, S, M, I, 4]`` tensor; without Triton the PyTorch expression is used.
* DFT-D3 is the explicit-terms kernel call with its forces injected into autograd, i.e. the path ``DFTD3`` takes for
  ``coord`` that requires grad (``aimnet.modules.lr._DFTD3EnergyFunction``), on the all-pairs template.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.autograd.function import once_differentiable

from ._warp import dftd3_energy

_AEV_KERNEL_MAX_ATOMS = 512  # aimnet.kernels.aev_sv_dense_triton.MAX_ATOMS (not imported: Triton is optional)


class _SharedGBmm(torch.autograd.Function):
    """``out = bmm(xl, g)`` for every ConvSV use of one ``g``; the gradient of ``g`` is assembled once.

    Non-collector uses stash ``(xl, dL/dout)``; the collector (the first use in forward, pass-0 conv_a, whose
    backward runs last because every other use is downstream of its output) returns
    ``dL/dg = cat(xl)^T @ cat(dL/dout)`` as one bmm."""

    @staticmethod
    def forward(ctx, xl, g, sink, collector: bool, expected: int):
        ctx.save_for_backward(xl, g)
        ctx.sink, ctx.collector, ctx.expected = sink, collector, expected
        return torch.bmm(xl, g)

    @staticmethod
    @once_differentiable
    def backward(ctx, gout):
        xl, g = ctx.saved_tensors
        gxl = torch.bmm(gout, g.transpose(1, 2)) if ctx.needs_input_grad[0] else None
        if not ctx.collector:
            ctx.sink.append((xl, gout))
            return gxl, None, None, None, None
        parts = [*ctx.sink, (xl, gout)]
        ctx.sink.clear()
        if len(parts) != ctx.expected:
            raise RuntimeError(f"shared-g backward order violated: {len(parts)} of {ctx.expected} uses collected")
        gg = torch.bmm(torch.cat([p[0] for p in parts], 1).transpose(1, 2), torch.cat([p[1] for p in parts], 1))
        return gxl, gg, None, None, None


def check_dense_model(model) -> None:
    """Raise NotImplementedError unless ``model`` is the AIMNet2 architecture this engine reproduces."""
    if getattr(model, "num_charge_channels", 1) != 1:
        raise NotImplementedError("closed-shell (one charge channel) models only")
    if not (getattr(model, "d2features", False) and model.conv_a.d2features and not model.conv_q.d2features):
        raise NotImplementedError("expects d2features conv_a and plain conv_q (AIMNet2 default)")


class StaticDenseAIMNet2(nn.Module):
    """E+F for a fixed layout: ``numbers`` [B, N] (all atoms real), ``charge`` [B]; coordinates vary per call."""

    def __init__(
        self, calc, numbers, charge, layout_conv: bool = True, fused_grad: bool = True, fused_aev: bool = True
    ):
        super().__init__()
        from aimnet import ops

        self._ops = ops
        if numbers.ndim != 2 or bool((numbers <= 0).any()):
            raise ValueError("dense static layout requires unpadded numbers[B, N]")
        model = calc.model
        check_dense_model(model)
        self.model, self.coul, self.d3 = model, calc.external_coulomb, calc.external_dftd3
        self.layout_conv = layout_conv
        self.fused_grad = fused_grad and layout_conv
        dev = numbers.device
        B, N = numbers.shape
        self.B, self.N = B, N
        st = {"numbers": numbers.to(torch.int64), "charge": charge.to(torch.float32)}
        st["_nb_mode"] = torch.tensor(0)
        st["mask_i"] = st["numbers"] == 0
        st["mask_ij"] = torch.eye(N, device=dev, dtype=torch.bool).unsqueeze(0)
        st["_input_padded"] = torch.tensor(False)
        st["_natom"] = torch.tensor(N, device=dev)
        st["mol_sizes"] = torch.tensor(N, device=dev)
        st["mask_ij_lr"] = st["mask_ij"]
        self.static = st
        aev = model.aev
        self._rc_s = aev.rc_s.detach().to("cpu")  # float32 0-dim on CPU: same fp32 math, no device scalar sync
        self._aev_kernel = None
        if layout_conv and fused_aev and dev.type == "cuda" and N <= _AEV_KERNEL_MAX_ATOMS:
            try:
                from aimnet.kernels.aev_sv_dense_triton import aev_sv_dense
            except ImportError:  # no Triton (e.g. Windows): the PyTorch expression in _aev_bmm_layout
                aev_sv_dense = None
            self._aev_kernel = aev_sv_dense
            # float32 constants exactly as the PyTorch expression computes them (pi / rc in CPU float32)
            self._aev_consts = (float(aev.eta_s), float(self._rc_s), float(torch.pi / self._rc_s))
        if self.d3 is not None:
            arange_n = torch.arange(N, device=dev, dtype=torch.int32)
            all_indices = arange_n.unsqueeze(0).expand(N, -1)
            template = all_indices[all_indices != arange_n.unsqueeze(1)].view(N, N - 1)
            offs = torch.arange(B, device=dev, dtype=torch.int32).view(B, 1, 1) * N
            self._d3_nbmat = (template.unsqueeze(0) + offs).view(B * N, N - 1)
            self._d3_batch = torch.arange(B, device=dev, dtype=torch.int32).repeat_interleave(N)
            self._d3_numbers = st["numbers"].to(torch.int32).flatten()

    # --- radial-angular basis -------------------------------------------------------------------------------
    def _aev(self, data):
        """Regular-path layout g_sv[b, i, m, s, :] (used when layout_conv=False)."""
        ops, aev = self._ops, self.model.aev
        d_ij, r_ij = ops.calc_distances(data)
        data["d_ij"] = d_ij
        rc = self._rc_s
        fc_ij = 0.5 * (torch.cos(d_ij.clamp(min=1e-6, max=rc) * (torch.pi / rc)) + 1.0)
        fc_ij = fc_ij.masked_fill(data["mask_ij"], 0.0)
        gs = ops.exp_expand(d_ij, aev.shifts_s, aev.eta_s) * fc_ij.unsqueeze(-1)
        u_ij = r_ij / d_ij.unsqueeze(-1)
        gv = gs.unsqueeze(-1) * u_ij.unsqueeze(-2)
        data["g_sv"] = torch.cat([gs.unsqueeze(-1), gv], dim=-1)
        return data

    def _aev_bmm_layout(self, data):
        """Same per-element arithmetic as the regular-path AEV, stored as g[b, s, m, i, :] = g_sv[b, i, m, s, :]."""
        ops, aev = self._ops, self.model.aev
        d_ij, r_ij = ops.calc_distances(data)  # diagonal r padded to 1 by calc_distances
        data["d_ij"] = d_ij
        if self._aev_kernel is not None:
            data["g_L"] = self._aev_kernel(d_ij, r_ij, aev.shifts_s, *self._aev_consts)
            return data
        rc = self._rc_s
        fc_ij = 0.5 * (torch.cos(d_ij.clamp(min=1e-6, max=rc) * (torch.pi / rc)) + 1.0)
        fc_ij = fc_ij.masked_fill(data["mask_ij"], 0.0)
        eta = torch.as_tensor(aev.eta_s, dtype=d_ij.dtype, device=d_ij.device)
        dT = d_ij.transpose(1, 2).unsqueeze(1)  # [B, 1, M, I]
        gs = torch.exp(-eta * (dT - aev.shifts_s.view(1, -1, 1, 1)) ** 2) * fc_ij.transpose(1, 2).unsqueeze(1)
        u = (r_ij / d_ij.unsqueeze(-1)).transpose(1, 2).unsqueeze(1)  # [B, 1, M, I, 3]
        data["g_L"] = torch.cat([gs.unsqueeze(-1), gs.unsqueeze(-1) * u], dim=-1)  # [B, S, M, I, 4]
        return data

    # --- convolutions on the bmm layout ---------------------------------------------------------------------
    def _conv(self, conv, data, x, d2: bool):
        """ConvSV.forward (mode 0, unpadded) on g_L. x: [B, M, F, S] (conv_a) or [B, M, C] (conv_q)."""
        B, N = self.B, self.N
        gL = data["g_L"]
        S = gL.shape[1]
        g2 = gL.view(B * S, N, N * 4)
        F = x.shape[2]
        if d2:
            xl = x.permute(0, 3, 2, 1).reshape(B * S, F, N)  # the einsum's left operand [b, s, f, m]
        else:
            xl = x.permute(0, 2, 1).unsqueeze(1).expand(B, S, F, N).reshape(B * S, F, N)
        if self.fused_grad:
            collector = data["_g_uses"] == 0
            data["_g_uses"] += 1
            y = _SharedGBmm.apply(xl, g2, data["_g_sink"], collector, data["_g_expected"])
        else:
            y = torch.bmm(xl, g2)
        avf_sv = y.view(B, S, F, N, 4).permute(0, 3, 2, 1, 4)  # [B, I, F, S, 4] like the einsum output
        avf_s, avf_v = avf_sv.split([1, 3], dim=-1)
        avf_v = torch.einsum("agh,...agd->...ahd", conv.agh, avf_v).pow(2).sum(-1)
        return torch.cat([avf_s.squeeze(-1).flatten(-2, -1), avf_v.flatten(-2, -1)], dim=-1)

    def _prep_in_a(self, data):  # AIMNet2._prepare_in_a, mode 0
        m = self.model
        a = data["a"]
        avf_a = self._conv(m.conv_a, data, a, True)
        return torch.cat([a.unsqueeze(2).flatten(-2, -1).squeeze(-2), avf_a], dim=-1)

    def _prep_in_q(self, data):  # AIMNet2._prepare_in_q, mode 0
        q = data["charges"]
        avf_q = self._conv(self.model.conv_q, data, q, False)
        return torch.cat([q.unsqueeze(2).squeeze(-2), avf_q], dim=-1)

    # --- energy ---------------------------------------------------------------------------------------------
    def energy(self, coord):
        """coord [B, N, 3] (requires_grad for forces) -> (energy [B] float64, data)."""
        m = self.model
        data = dict(self.static)
        data["coord"] = coord
        a = m.afv(data["numbers"])
        if m.d2features:
            a = a.unflatten(-1, (m.nfeature, m.nshifts_s))
        data["a"] = a
        data["charge"] = data["charge"].unsqueeze(-1)
        if self.layout_conv:
            data = self._aev_bmm_layout(data)
            prep_a, prep_q = self._prep_in_a, self._prep_in_q
            if self.fused_grad:
                data["a"] = data["a"].detach()  # species embedding: no coordinate dependence
                data["_g_sink"], data["_g_uses"] = [], 0
                data["_g_expected"] = len(m.mlps) + (len(m.mlps) - 1)
        else:
            data = self._aev(data)
            prep_a, prep_q = m._prepare_in_a, m._prepare_in_q
        npass = len(m.mlps)
        for ipass, mlp in enumerate(m.mlps):
            _in = prep_a(data) if ipass == 0 else torch.cat([prep_a(data), prep_q(data)], dim=-1)
            _out = mlp(_in)
            if ipass == 0:
                data = m._update_q(data, _out, delta_q=False)
            elif ipass < npass - 1:
                data = m._update_q(data, _out, delta_q=True)
            else:
                data["aim"] = _out
        data["charges"] = data["charges"].squeeze(-1)
        data["charge"] = data["charge"].squeeze(-1)
        for mod in m.outputs.children():
            data = mod(data)
        if self.coul is not None:
            data = self.coul(data)
        if self.d3 is not None:
            e_d3 = dftd3_energy(
                coord.flatten(0, 1), self.d3, self._d3_numbers, self._d3_batch, self._d3_nbmat, self.B, self.B * self.N
            )
            data["energy"] = data["energy"] + e_d3.to(data["energy"].dtype)
        return data["energy"], data

    def forward(self, coord):
        """Energy [B] (float64), forces [B, N, 3] = -dE/dR, charges [B, N]."""
        with torch.enable_grad():
            x = coord.detach().requires_grad_(True)
            e, data = self.energy(x)
            f = -torch.autograd.grad(e.sum(), x)[0]
        return e.detach(), f, data["charges"].detach()

    def energy_only(self, coord):
        """Energy [B] (float64), None, charges [B, N]: the forward pass of ``forward`` without the backward."""
        with torch.no_grad():
            e, data = self.energy(coord)
        return e, None, data["charges"]


class GraphedEF:
    """CUDA-graph replay of a static engine for a fixed layout (``forces=False``: energy and charges only).

    Coordinate contents may change between calls (copied into a static buffer); everything else is fixed at
    capture. Returned tensors alias graph-owned memory until the next call (clone to keep)."""

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
        with torch.cuda.graph(self.g):
            self.e, self.f, self.q = fn(self.x)

    def launch(self, coord):
        """Enqueue a replay for ``coord`` without waiting for it."""
        self.x.copy_(coord)
        self.g.replay()

    def result(self):
        """Outputs of the last launch (graph-owned; stream-ordered after it)."""
        return self.e, self.f, self.q

    def __call__(self, coord):
        self.launch(coord)
        return self.result()
