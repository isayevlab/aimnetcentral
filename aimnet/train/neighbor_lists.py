"""Packed periodic neighbor lists for stress training with the built-in loader.

``SizeGroupedDataset`` batches are dense ``(B, N)`` arrays, while stress needs
explicit neighbor topology with periodic shifts. :class:`PeriodicNeighborCollate`
converts each batch to the packed mode-1 layout the calculator uses: real
atoms only, one trailing dummy atom, ``mol_idx``, and ``nbmat``/``shifts``
(plus long-range lists) whose fill value is the dummy's index.
"""

from collections.abc import Mapping
from typing import cast

import torch
from nvalchemiops.torch.interactions.electrostatics import estimate_ewald_parameters, estimate_pme_parameters
from torch import Tensor, nn

from aimnet import nbops
from aimnet.calculators.neighbors import AdaptiveNeighborList, _add_padding_row, move_coord_to_cell, normalize_pbc
from aimnet.data.sgdataset import _squeeze_loader_batch
from aimnet.modules import AEVSV, D3TS, DFTD3, LRCoulomb, SRCoulomb

# D3TS has no cutoff of its own and sums every listed pair, so its list radius
# is part of the model. This is the calculator's default dispersion cutoff.
D3TS_CUTOFF = 15.0

# Per-system tensors whose shape can coincide with a per-atom (B, N) prefix
# (for example a (B, 3, 3) cell in a group of 3-atom cells).
PER_SYSTEM_KEYS = frozenset({
    "cell",
    "pbc",
    "charge",
    "mult",
    "energy",
    "stress",
    "virial",
    "dipole",
    "quadrupole",
})

# A neighbor-list radius in Angstrom, or ("ewald" | "pme", accuracy) for a
# Coulomb real-space radius estimated per batch, as the calculator does.
CutoffSpec = float | tuple[str, float]


def _set_cutoff(cutoffs: dict[str, CutoffSpec], suffix: str, spec: CutoffSpec) -> None:
    if suffix in cutoffs and cutoffs[suffix] != spec:
        raise ValueError(f"Conflicting neighbor-list cutoffs for nbmat{suffix}: {cutoffs[suffix]} and {spec}.")
    cutoffs[suffix] = spec


def neighbor_cutoffs(model: nn.Module) -> dict[str, CutoffSpec]:
    """Return the neighbor lists a model needs for periodic mode-1 input.

    Keys are neighbor-matrix suffixes: ``""`` for the short-range list,
    ``"_coulomb"`` and ``"_dftd3"`` for embedded long-range modules. The
    short-range list must reach both the AEV cutoff and the radius of any
    short-range Coulomb term, which is evaluated on that same list.
    """
    radii = [
        max(float(cast(Tensor, module.rc_s)), float(cast(Tensor, module.rc_v)))
        for module in model.modules()
        if isinstance(module, AEVSV)
    ]
    if not radii:
        raise ValueError("Cannot infer the short-range cutoff: the model has no AEVSV module.")
    for module in model.modules():
        if isinstance(module, SRCoulomb) or (isinstance(module, LRCoulomb) and module.subtract_sr):
            radii.append(float(module.rc))
    cutoffs: dict[str, CutoffSpec] = {"": max(radii)}
    for module in model.modules():
        if isinstance(module, LRCoulomb):
            if module.method == "simple":
                raise ValueError(
                    "Stress training needs a periodic Coulomb method: the model's LRCoulomb uses method='simple' "
                    "(all-pairs 1/r), which has no periodic form. Set method: ewald (or pme) in the model YAML."
                )
            if module.method == "dsf":
                # Called from the model's own forward, DSF runs the kernel on
                # detached coordinates and cell, so forces and stress would
                # miss its fixed-charge geometry derivative.
                raise NotImplementedError(
                    "Stress training does not support an embedded LRCoulomb with method='dsf': inside the model "
                    "forward it contributes no geometry derivative, so forces and stress would be wrong. Use "
                    "method: ewald or pme."
                )
            _set_cutoff(cutoffs, "_coulomb", (module.method, float(module.ewald_accuracy)))
        elif isinstance(module, DFTD3):
            raise NotImplementedError(
                "Stress training is not available for a model carrying a tabulated DFT-D3 module: it contributes "
                "no cell gradient, so the stress would silently omit most of the dispersion virial. Use D3TS, or "
                "train without the module and add dispersion in the calculator."
            )
        elif isinstance(module, D3TS):
            _set_cutoff(cutoffs, "_dftd3", D3TS_CUTOFF)
    return cutoffs


class PeriodicNeighborCollate:
    """Collate dense loader batches into packed mode-1 periodic batches.

    Padded atoms (``numbers == 0``) are dropped and one dummy atom is
    appended, so per-atom tensors in ``x`` and ``y`` become ``(L + 1, ...)``
    with ``mol_idx`` sorted by system. Coordinates are wrapped into the cell
    before the lists are built, as in the calculator; energies, forces, and
    stress are invariant to that wrap.

    Lists are built on CPU in the loader process (or worker), one per distinct
    cutoff, each with integer shifts and the dummy index as the fill value.
    The adaptive buffers are kept on the instance, so their size carries over
    from batch to batch.
    """

    def __init__(self, cutoffs: Mapping[str, CutoffSpec]):
        if "" not in cutoffs:
            raise ValueError("A short-range cutoff (key '') is required.")
        self.cutoffs = dict(cutoffs)
        self._nblists: dict[str, AdaptiveNeighborList] = {}

    def __call__(self, batch) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        x, y = _squeeze_loader_batch(batch)
        return self.pack(x, y)

    def _neighbors(
        self, suffix: str, cutoff: float, coord: Tensor, cell: Tensor, pbc: Tensor, batch_idx: Tensor
    ) -> tuple[Tensor, Tensor]:
        # One adaptive buffer per suffix: an Ewald/PME radius changes from
        # batch to batch, so keying by radius would never reuse a buffer.
        nblist = self._nblists.get(suffix)
        if nblist is None:
            nblist = self._nblists[suffix] = AdaptiveNeighborList(cutoff=cutoff)
        nblist.cutoff = cutoff
        n_real = coord.shape[0]
        nbmat, _, shifts = nblist(positions=coord, cell=cell, pbc=pbc, batch_idx=batch_idx, fill_value=n_real)
        assert shifts is not None
        nbmat, shifts = _add_padding_row(nbmat, shifts, n_real)
        assert shifts is not None
        return nbmat, shifts

    def _radius(self, spec: CutoffSpec, coord: Tensor, cell: Tensor, batch_idx: Tensor) -> float:
        if not isinstance(spec, tuple):
            return float(spec)
        method, accuracy = spec
        estimate = estimate_pme_parameters if method == "pme" else estimate_ewald_parameters
        params = estimate(positions=coord, cell=cell, batch_idx=batch_idx, accuracy=accuracy)
        return float(params.real_space_cutoff.max().item())

    def pack(self, x: dict[str, Tensor], y: dict[str, Tensor]) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        neighbor_keys = sorted(k for k in x if k.startswith(("nbmat", "shifts")))
        if neighbor_keys:
            raise ValueError(
                f"The batch already carries {neighbor_keys}; the loader builds neighbor lists itself. "
                "Drop them from data.x, or supply a dataset whose loader yields complete mode-1/2 batches."
            )
        if "cell" not in x:
            raise ValueError("Stress training with the built-in loader needs 'cell' in data.x.")
        numbers, coord = x["numbers"], x["coord"]
        if numbers.ndim != 2 or coord.shape[:2] != numbers.shape:
            raise ValueError("Expected dense (B, N) numbers and (B, N, 3) coord from the loader.")
        B, N = numbers.shape
        cell = x["cell"]
        if cell.shape == (3, 3):
            cell = cell.expand(B, 3, 3)
        if cell.shape != (B, 3, 3):
            raise ValueError(f"cell must have shape (3, 3) or ({B}, 3, 3), got {tuple(cell.shape)}.")
        real = (numbers != 0).flatten()
        mol_idx = torch.arange(B).repeat_interleave(N)[real]
        if mol_idx.numel() == 0 or bool((torch.bincount(mol_idx, minlength=B) == 0).any()):
            raise ValueError("Every system in a stress batch needs at least one atom.")

        def pack_atoms(value: Tensor) -> Tensor:
            flat = value.flatten(0, 1)[real]
            return torch.cat((flat, flat.new_zeros((1, *flat.shape[1:]))))

        def pack_dict(data: dict[str, Tensor]) -> dict[str, Tensor]:
            return {
                k: pack_atoms(v) if k not in PER_SYSTEM_KEYS and v.shape[:2] == (B, N) else v for k, v in data.items()
            }

        packed_x, packed_y = pack_dict(x), pack_dict(y)

        # Lists are built on the real atoms; the dummy (index L) is the fill value.
        coord_real = packed_x["coord"][:-1]
        lattice = cell.to(coord_real.dtype)
        pbc = normalize_pbc(x.get("pbc"), lattice, coord_real.device)
        coord_real = move_coord_to_cell(coord_real, lattice, mol_idx, pbc)
        batch_idx = mol_idx.to(torch.int32)
        radii = {suffix: self._radius(spec, coord_real, lattice, batch_idx) for suffix, spec in self.cutoffs.items()}
        lists: dict[float, tuple[Tensor, Tensor]] = {}
        for suffix, radius in radii.items():
            if radius not in lists:
                lists[radius] = self._neighbors(suffix, radius, coord_real, lattice, pbc, batch_idx)
        packed_x["coord"] = torch.cat((coord_real, packed_x["coord"][-1:]))
        # The dummy belongs to the last system, the calculator's padding convention.
        packed_x["mol_idx"] = torch.cat((mol_idx, mol_idx[-1:]))
        packed_x["cell"] = cell.contiguous()
        for suffix, radius in radii.items():
            packed_x[f"nbmat{suffix}"], packed_x[f"shifts{suffix}"] = lists[radius]
        long_range = [radius for suffix, radius in radii.items() if suffix]
        if long_range:
            # Modules without their own suffix fall back to the shared list.
            packed_x["nbmat_lr"], packed_x["shifts_lr"] = lists[max(long_range)]
        assert nbops.infer_nb_mode(packed_x) == 1
        return packed_x, packed_y
