"""Neighbor-list and static-input-cache utilities for the AIMNet2 calculator.

Holds :class:`AdaptiveNeighborList` (auto-sizing wrapper over the
nvalchemiops neighbor list), :class:`StaticInputCache` (bounded storage for
static-geometry MD-loop reuse), and the geometry/padding helpers used when
building neighbor matrices. Cache-key *construction* stays on the
calculator — the keys encode calculator policy — while this module owns the
storage and identity-validation mechanics.
"""

import math
import weakref
from typing import Any

import torch
from nvalchemiops.neighbors import NeighborOverflowError
from nvalchemiops.torch.neighbors import neighbor_list
from torch import Tensor


class AdaptiveNeighborList:
    """Adaptive neighbor list with automatic buffer sizing.

    Wraps nvalchemiops.torch.neighbors.neighbor_list with automatic max_neighbors adjustment.
    Maintains ~75% utilization to balance memory and recomputation.

    Parameters
    ----------
    cutoff : float
        Cutoff distance for neighbor detection in Angstroms.
    density : float, optional
        Initial atomic density estimate for allocation sizing.
        Used to compute initial max_neighbors as density * (4/3 * pi * cutoff^3).
        Default is 0.2.
    target_utilization : float, optional
        Target ratio of actual neighbors to allocated max_neighbors.
        Default is 0.75 (75% utilization).

    Attributes
    ----------
    cutoff : float
        Cutoff distance for neighbor detection.
    target_utilization : float
        Target ratio of actual to allocated neighbors.
    max_neighbors : int
        Current maximum neighbor allocation (rounded to 16).
    """

    def __init__(
        self,
        cutoff: float,
        density: float = 0.2,
        target_utilization: float = 0.75,
    ) -> None:
        self.cutoff = cutoff
        self.target_utilization = target_utilization
        sphere_volume = 4 / 3 * math.pi * cutoff**3
        self.max_neighbors = self._round_to_16(int(density * sphere_volume))

    @staticmethod
    def _round_to_16(n: int) -> int:
        """Round up to the next multiple of 16 for memory alignment."""
        return ((n + 15) // 16) * 16

    def __call__(
        self,
        positions: Tensor,
        cell: Tensor | None = None,
        pbc: Tensor | None = None,
        batch_idx: Tensor | None = None,
        fill_value: int | None = None,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """Compute neighbor list with automatic buffer adjustment.

        Parameters
        ----------
        positions : Tensor
            Atomic coordinates, shape (N, 3).
        cell : Tensor | None
            Unit cell vectors, shape (num_systems, 3, 3). None for non-periodic.
        pbc : Tensor | None
            Periodic boundary conditions, shape (num_systems, 3). None for non-periodic.
        batch_idx : Tensor | None
            Batch index for each atom, shape (N,). None for single system.
        fill_value : int | None
            Fill value for padding. Default is N (number of atoms).

        Returns
        -------
        nbmat : Tensor
            Neighbor indices, shape (N, actual_max_neighbors).
        num_neighbors : Tensor
            Number of neighbors per atom, shape (N,).
        shifts : Tensor | None
            Integer unit cell shifts for PBC, shape (N, actual_max_neighbors, 3).
            None for non-periodic systems.
        """
        N = positions.shape[0]
        if fill_value is None:
            fill_value = N
        _pbc = cell is not None

        while True:
            try:
                if _pbc:
                    nbmat, num_neighbors, shifts = neighbor_list(
                        positions=positions,
                        cutoff=self.cutoff,
                        cell=cell,
                        pbc=pbc,
                        batch_idx=batch_idx,
                        max_neighbors=self.max_neighbors,
                        half_fill=False,
                        fill_value=fill_value,
                    )
                else:
                    nbmat, num_neighbors = neighbor_list(
                        positions=positions,
                        cutoff=self.cutoff,
                        batch_idx=batch_idx,
                        max_neighbors=self.max_neighbors,
                        half_fill=False,
                        fill_value=fill_value,
                        method="batch_naive",
                    )
                    shifts = None
            except NeighborOverflowError:
                # Increase buffer by 1.5x and retry
                self.max_neighbors = self._round_to_16(int(self.max_neighbors * 1.5))
                continue

            # Get actual max neighbors from result
            actual_max = int(num_neighbors.max().item())

            # Adjust buffer if under-utilized (shrink at 2/3 of target for hysteresis)
            # Use 2/3 threshold to prevent thrashing from small fluctuations
            if actual_max < (2 / 3) * self.target_utilization * self.max_neighbors:
                new_max = self._round_to_16(int(actual_max / self.target_utilization))
                self.max_neighbors = max(new_max, 16)  # Ensure minimum of 16

            # Trim to actual max neighbors
            actual_nnb = max(1, actual_max)
            nbmat = nbmat[:, :actual_nnb]
            if shifts is not None:
                shifts = shifts[:, :actual_nnb]

            return nbmat, num_neighbors, shifts


class StaticInputCache:
    """Bounded storage for static-geometry reuse across MD-loop evaluations.

    Entries are keyed by calculator-built policy keys and validated against
    weak references to the caller's original coord/numbers tensors, so a
    caller that rebuilds its input tensors never receives stale data.
    """

    _NBMAT_KEYS = (
        "nbmat",
        "shifts",
        "nbmat_lr",
        "shifts_lr",
        "nbmat_coulomb",
        "shifts_coulomb",
        "nbmat_dftd3",
        "shifts_dftd3",
    )

    def __init__(self, max_entries: int = 8) -> None:
        self.max_entries = max_entries
        self._nbmat: dict[tuple[Any, ...], tuple[Any, Any, dict[str, Tensor]]] = {}
        self._dftd3: dict[tuple[Any, ...], tuple[Any, Any, dict[str, Any]]] = {}

    @staticmethod
    def tensor_identity_key(value: Any) -> tuple[Any, ...] | None:
        if not isinstance(value, Tensor) or value.device.type != "cuda" or value.is_sparse:
            return None
        try:
            version = int(value._version)
        except RuntimeError:
            return None
        return (
            id(value),
            str(value.device),
            str(value.dtype),
            tuple(int(dim) for dim in value.shape),
            tuple(int(stride) for stride in value.stride()),
            int(value.data_ptr()),
            int(value.storage_offset()),
            version,
        )

    @staticmethod
    def tensor_refs(raw_data: dict[str, Any]) -> tuple[Any, Any] | None:
        raw_coord = raw_data.get("coord")
        raw_numbers = raw_data.get("numbers")
        if not isinstance(raw_coord, Tensor) or not isinstance(raw_numbers, Tensor):
            return None
        return weakref.ref(raw_coord), weakref.ref(raw_numbers)

    def get_nbmat(self, cache_key: tuple[Any, ...], raw_data: dict[str, Any]) -> dict[str, Tensor] | None:
        entry = self._nbmat.get(cache_key)
        if entry is None:
            return None
        coord_ref, numbers_ref, cached = entry
        if coord_ref() is not raw_data.get("coord") or numbers_ref() is not raw_data.get("numbers"):
            self._nbmat.pop(cache_key, None)
            return None
        return cached

    def remember_nbmat(
        self,
        cache_key: tuple[Any, ...],
        raw_data: dict[str, Any],
        data: dict[str, Tensor],
    ) -> None:
        raw_coord = raw_data.get("coord")
        raw_numbers = raw_data.get("numbers")
        if not isinstance(raw_coord, Tensor) or not isinstance(raw_numbers, Tensor):
            return
        cached = {key: data[key].detach() for key in self._NBMAT_KEYS if isinstance(data.get(key), Tensor)}
        if not cached:
            return
        if len(self._nbmat) >= self.max_entries:
            self._nbmat.pop(next(iter(self._nbmat)))
        self._nbmat[cache_key] = (weakref.ref(raw_coord), weakref.ref(raw_numbers), cached)

    def get_dftd3(self, cache_key: tuple[Any, ...], current_refs: tuple[Any, Any] | None) -> dict[str, Any] | None:
        entry = self._dftd3.get(cache_key)
        if entry is None or current_refs is None:
            return None
        coord_ref, numbers_ref, cached = entry
        if coord_ref() is not current_refs[0]() or numbers_ref() is not current_refs[1]():
            self._dftd3.pop(cache_key, None)
            return None
        return cached

    def store_dftd3(
        self,
        cache_key: tuple[Any, ...],
        current_refs: tuple[Any, Any],
        cached: dict[str, Any],
    ) -> None:
        if len(self._dftd3) >= self.max_entries:
            self._dftd3.pop(next(iter(self._dftd3)))
        self._dftd3[cache_key] = (current_refs[0], current_refs[1], cached)

    def clear(self) -> None:
        self._nbmat.clear()
        self._dftd3.clear()


def _add_padding_row(
    nbmat: Tensor,
    shifts: Tensor | None,
    N: int,
) -> tuple[Tensor, Tensor | None]:
    """Add padding row to neighbor matrix and shifts.

    Parameters
    ----------
    nbmat : Tensor
        Neighbor matrix, shape (N, max_neighbors).
    shifts : Tensor | None
        Shift vectors for PBC or None, shape (N, max_neighbors, 3).
    N : int
        Number of atoms (used as fill value for padding row).

    Returns
    -------
    tuple[Tensor, Tensor | None]
        Tuple of (nbmat, shifts) with padding row added.
    """
    device = nbmat.device
    dtype = nbmat.dtype
    nnb_max = nbmat.shape[1]
    padding_row = torch.full((1, nnb_max), N, dtype=dtype, device=device)
    nbmat = torch.cat([nbmat, padding_row], dim=0)

    if shifts is not None:
        shifts_padding = torch.zeros((1, nnb_max, 3), dtype=shifts.dtype, device=device)
        shifts = torch.cat([shifts, shifts_padding], dim=0)

    return nbmat, shifts


def maybe_pad_dim0(a: Tensor, N: int, value=0.0) -> Tensor:
    _shape_diff = N - a.shape[0]
    assert _shape_diff == 0 or _shape_diff == 1, "Invalid shape"
    if _shape_diff == 1:
        a = pad_dim0(a, value=value)
    return a


def pad_dim0(a: Tensor, value=0.0) -> Tensor:
    shapes = [0] * ((a.ndim - 1) * 2) + [0, 1]
    a = torch.nn.functional.pad(a, shapes, mode="constant", value=value)
    return a


def maybe_unpad_dim0(a: Tensor, N: int) -> Tensor:
    _shape_diff = a.shape[0] - N
    assert _shape_diff == 0 or _shape_diff == 1, "Invalid shape"
    if _shape_diff == 1:
        a = a[:-1]
    return a


def normalize_pbc(pbc: Tensor | None, cell: Tensor, device: str | torch.device) -> Tensor:
    """Return PBC flags as ``(B, 3)`` bool tensor matching ``cell``."""
    num_systems = 1 if cell.ndim == 2 else cell.shape[0]
    if pbc is None:
        return torch.ones((num_systems, 3), dtype=torch.bool, device=device)
    pbc = torch.as_tensor(pbc, dtype=torch.bool, device=device)
    if pbc.ndim == 1:
        if pbc.shape[0] != 3:
            raise ValueError("pbc must have shape (3,) or (B, 3)")
        return pbc.unsqueeze(0).expand(num_systems, -1)
    if pbc.ndim == 2 and pbc.shape == (num_systems, 3):
        return pbc
    raise ValueError(f"pbc must have shape (3,) or ({num_systems}, 3), got {tuple(pbc.shape)}")


def _wrap_fractional(coord_f: Tensor, pbc: Tensor) -> Tensor:
    pbc = pbc.to(device=coord_f.device, dtype=torch.bool)
    while pbc.ndim < coord_f.ndim:
        pbc = pbc.unsqueeze(-2)
    return torch.where(pbc, coord_f % 1, coord_f)


def move_coord_to_cell(
    coord: Tensor,
    cell: Tensor,
    mol_idx: Tensor | None = None,
    pbc: Tensor | None = None,
) -> Tensor:
    """Move coordinates into the periodic cell.

    Parameters
    ----------
    coord : Tensor
        Coordinates tensor, shape (N, 3) or (B, N, 3).
    cell : Tensor
        Cell tensor, shape (3, 3) or (B, 3, 3).
    mol_idx : Tensor | None
        Molecule index for each atom, shape (N,).
        Required for batched cells with flat coordinates.
    pbc : Tensor | None
        Periodic axes, shape (3,) or (B, 3). Defaults to all periodic axes.

    Returns
    -------
    Tensor
        Coordinates wrapped into the cell.
    """
    pbc = normalize_pbc(pbc, cell, coord.device)
    if cell.ndim == 2:
        # Single cell (3, 3)
        cell_inv = torch.linalg.inv(cell)
        coord_f = coord @ cell_inv
        coord_f = _wrap_fractional(coord_f, pbc[0])
        return coord_f @ cell
    else:
        # Batched cells (B, 3, 3)
        if coord.ndim == 3:
            # Batched coords (B, N, 3) with batched cells (B, 3, 3)
            cell_inv = torch.linalg.inv(cell)  # (B, 3, 3)
            coord_f = torch.bmm(coord, cell_inv)  # (B, N, 3)
            coord_f = _wrap_fractional(coord_f, pbc)
            return torch.bmm(coord_f, cell)
        else:
            # Flat coords (N_total, 3) with batched cells (B, 3, 3) - need mol_idx
            assert mol_idx is not None, "mol_idx required for batched cells with flat coordinates"
            cell_inv = torch.linalg.inv(cell)  # (B, 3, 3)
            # Get cell and cell_inv for each atom
            atom_cell = cell[mol_idx]  # (N_total, 3, 3)
            atom_cell_inv = cell_inv[mol_idx]  # (N_total, 3, 3)
            atom_pbc = pbc[mol_idx]
            coord_f = torch.bmm(coord.unsqueeze(1), atom_cell_inv).squeeze(1)  # (N_total, 3)
            coord_f = _wrap_fractional(coord_f, atom_pbc)
            return torch.bmm(coord_f.unsqueeze(1), atom_cell).squeeze(1)
