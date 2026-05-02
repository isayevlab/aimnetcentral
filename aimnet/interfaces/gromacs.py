"""TorchScript wrapper exposing AIMNet2 via the GROMACS NNPot interface.

Status: parked. ``build_gromacs_nnpot_model`` raises ``NotImplementedError``
because the upstream pipeline is not yet ``torch.jit.script``-able. The class
definition below is preserved as a starting point for the eventual wrapper.
See ``docs/external/gromacs.md`` for blocker details.
"""

import torch
from torch import Tensor, nn

# Exact-SI conversion: e * NA / 1000 (post-2019 SI fixed values).
_EV_TO_KJ_MOL: float = 96.48533212331
_NM_TO_ANG: float = 10.0


class AIMNet2Gromacs(nn.Module):
    """TorchScript-able adapter around an AIMNet2 ScriptModule for GROMACS NNPot.

    Currently unreachable in production -- ``build_gromacs_nnpot_model`` raises
    ``NotImplementedError``. The class is preserved as a starting point for
    the eventual export pathway. The forward() body has been verified to
    ``torch.jit.script`` cleanly with a dummy inner ScriptModule and to
    round-trip via ``torch.jit.save`` / ``torch.jit.load``.

    Intended limitations (when unblocked):
        * non-PBC only
        * single QM region with a fixed total charge
        * returns energy only; GROMACS autograds forces from positions
        * float32 precision (positions are downcast internally)
        * NSE / open-shell models not supported
    """

    length_conversion: float
    energy_conversion: float

    def __init__(self, model: torch.jit.ScriptModule, charge: float = 0.0) -> None:
        super().__init__()
        self.model = model
        self.length_conversion = _NM_TO_ANG
        self.energy_conversion = _EV_TO_KJ_MOL
        # Buffer survives jit.script + jit.save; serialized with the saved .pt.
        self.register_buffer(
            "total_charge",
            torch.tensor([float(charge)], dtype=torch.float32),
        )

    def forward(
        self,
        positions: Tensor,
        atomic_numbers: Tensor,
        box: Tensor | None = None,
        pbc: Tensor | None = None,
    ) -> Tensor:
        n = positions.shape[0]
        # n==1 produces a (1, 0) nbmat which is degenerate downstream.
        assert n >= 2, "AIMNet2Gromacs requires at least 2 atoms"
        # PBC is out of scope for v1. Refuse non-trivial box/pbc rather than
        # silently producing open-boundary energies for a periodic GROMACS run.
        if box is not None:
            assert torch.all(box == 0), "AIMNet2Gromacs does not support periodic systems; pass box=None or zeros"
        if pbc is not None:
            assert not bool(pbc.any()), "AIMNet2Gromacs does not support periodic boundary conditions"
        device = positions.device

        # GROMACS may feed float64 positions in double-precision builds; the
        # inner AIMNet2 model expects float32. Downcast costs ~7 decimal digits
        # in the autograd-derived forces path -- documented v1 limitation.
        coord = (positions * self.length_conversion).to(torch.float32)
        numbers = atomic_numbers.to(torch.int32)

        # All-pairs neighbor list, shape (N, N-1). Inner model masks
        # contributions from the padding row (index N) and applies internal
        # cutoff functions to zero distant interactions, so passing every
        # other atom is correct for any cutoff (short-range or long-range).
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

        # nbmat_lr aliases nbmat -- safe only because both are all-pairs lists
        # that already cover any LR cutoff for a small QM region.
        data: dict[str, Tensor] = {
            "coord": coord_p,
            "numbers": numbers_p,
            "charge": self.total_charge,
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
    output_path: str | None = None,
) -> torch.jit.ScriptModule:
    """Stub. Always raises ``NotImplementedError``.

    The wrapper is parked because the upstream AIMNet2 pipeline is not
    currently ``torch.jit.script``-able end-to-end. See
    ``docs/external/gromacs.md`` for blocker details.
    """
    del model_name, charge, output_path
    raise NotImplementedError(
        "GROMACS NNPot wrapper is parked: AIMNet2 is not currently "
        "torch.jit.script-able end-to-end. See docs/external/gromacs.md."
    )
