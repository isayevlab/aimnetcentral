"""TorchScript wrapper exposing AIMNet2 via the GROMACS NNPot interface.

STATUS: PARKED / WORK IN PROGRESS. Do not import this module for production
use. ``build_gromacs_nnpot_model`` will raise for every shipped AIMNet2
model because the upstream pipeline is not currently ``torch.jit.script``-
able. Specifically:

    * the core ``AIMNet2`` module uses ``tensor.data_ptr()`` in
      ``aimnet/nbops.py`` for neighbor-cache identity, which TorchScript
      rejects;
    * the ``DFTD3`` external module uses an ``aten::grad`` call signature
      that TorchScript cannot match;
    * shipped v2 ``.pt`` assets are plain ``torch.save`` state dicts
      (loaded into Python ``nn.Module``s), not TorchScript archives.

Tracking issue / plan:
    docs/superpowers/plans/2026-04-26-torchscript-export.md
Status doc:
    docs/external/gromacs.md

The code below is preserved as a starting point for the eventual GROMACS
NNPot wrapper once the scriptability blockers are resolved. The forward
contract, unit conversions, and all-pairs neighbor-list construction were
sanity-checked against a dummy inner ScriptModule and round-trip via
``torch.jit.save`` / ``torch.jit.load``.

Intended limitations (v1, when unblocked):
    * non-PBC only -- the QM region in QM/MM is the realistic use case
    * single QM region with a fixed total charge (set at construction)
    * returns energy only; GROMACS autograds forces from ``positions``
    * NSE / open-shell models are not supported here
"""

from typing import Optional

import torch
from torch import Tensor, nn

# Exact-SI conversion: e * NA / 1000 (post-2019 SI fixed values).
_EV_TO_KJ_MOL: float = 96.48533212331
_NM_TO_ANG: float = 10.0


class GromacsNNPotWrapper(nn.Module):
    """TorchScript-able adapter around an AIMNet2 ScriptModule.

    Parameters
    ----------
    model : torch.jit.ScriptModule
        Inner AIMNet2 scripted model loaded from a v2 ``.pt`` asset.
    charge : float, optional
        Total charge of the QM region. Default ``0.0``.
    """

    length_conversion: float
    energy_conversion: float

    def __init__(self, model: torch.jit.ScriptModule, charge: float = 0.0) -> None:
        super().__init__()
        self.model = model
        self.length_conversion = _NM_TO_ANG
        self.energy_conversion = _EV_TO_KJ_MOL
        # Buffer survives jit.script + jit.save; serialized with the model.
        self.register_buffer(
            "_charge",
            torch.tensor([float(charge)], dtype=torch.float32),
        )

    def forward(
        self,
        positions: Tensor,
        atomic_numbers: Tensor,
        box: Optional[Tensor] = None,
        pbc: Optional[Tensor] = None,
    ) -> Tensor:
        # box / pbc accepted for GROMACS interface compatibility but ignored:
        # PBC is out of scope for the v1 wrapper.
        n = positions.shape[0]
        device = positions.device

        coord = (positions * self.length_conversion).to(torch.float32)
        numbers = atomic_numbers.to(torch.int32)

        # All-pairs neighbor list, shape (N, N-1). Inner model masks
        # contributions from index N (padding) and uses internal cutoff
        # functions to zero distant interactions, so passing every other
        # atom is correct for any cutoff.
        idx = torch.arange(n, device=device, dtype=torch.int32)
        full = idx.unsqueeze(0).expand(n, n)
        mask = idx.unsqueeze(0) != idx.unsqueeze(1)
        nbmat = full[mask].view(n, n - 1)
        # Add padding row at index N filled with N (the calculator's convention).
        pad_row = torch.full((1, n - 1), n, dtype=torch.int32, device=device)
        nbmat = torch.cat([nbmat, pad_row], dim=0)

        coord_p = torch.cat(
            [coord, torch.zeros(1, 3, dtype=coord.dtype, device=device)],
            dim=0,
        )
        numbers_p = torch.cat(
            [numbers, torch.zeros(1, dtype=torch.int32, device=device)],
            dim=0,
        )
        mol_idx = torch.zeros(n + 1, dtype=torch.int32, device=device)

        data: dict[str, Tensor] = {
            "coord": coord_p,
            "numbers": numbers_p,
            "charge": self._charge.to(coord.dtype),
            "mol_idx": mol_idx,
            "nbmat": nbmat,
            "nbmat_lr": nbmat,
        }
        out = self.model(data)
        return out["energy"][0] * self.energy_conversion


def build_gromacs_nnpot_model(
    model_name: str = "aimnet2",
    *,
    charge: float = 0.0,
    output_path: Optional[str] = None,
) -> torch.jit.ScriptModule:
    """Build, script, and optionally save a GROMACS-ready ``.pt`` file.

    Parameters
    ----------
    model_name : str, optional
        Registry alias (e.g. ``"aimnet2"``), HF repo id, or path to a v2
        ``.pt`` file. Default ``"aimnet2"``.
    charge : float, optional
        Total charge of the QM region. Default ``0.0``.
    output_path : str | None, optional
        If given, the scripted wrapper is written to this path via
        ``torch.jit.save``.

    Returns
    -------
    torch.jit.ScriptModule
        The scripted ``GromacsNNPotWrapper``.
    """
    from aimnet.calculators import AIMNet2Calculator

    base = AIMNet2Calculator(model_name, device="cpu")
    inner = base.model
    if not isinstance(inner, torch.jit.ScriptModule):
        raise TypeError(
            f"Model '{model_name}' did not load as a TorchScript ScriptModule "
            f"(got {type(inner).__name__}). The GROMACS wrapper requires a "
            f"v2 .pt asset."
        )
    if base.has_external_coulomb or base.has_external_dftd3:
        raise RuntimeError(
            f"Model '{model_name}' uses external long-range modules "
            f"(coulomb={base.has_external_coulomb}, dftd3={base.has_external_dftd3}). "
            f"Only models with all long-range terms baked into the scripted "
            f"asset are supported by the GROMACS wrapper. The v2 .pt assets "
            f"shipped with this package satisfy this constraint."
        )

    wrapper = GromacsNNPotWrapper(inner.cpu(), charge=charge).cpu().eval()
    scripted = torch.jit.script(wrapper)
    if output_path is not None:
        scripted.save(output_path)
    return scripted
