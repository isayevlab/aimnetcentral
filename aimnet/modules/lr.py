import math
import os
from dataclasses import dataclass
from typing import Any, NamedTuple

import torch
from nvalchemiops.neighbors import NeighborOverflowError
from nvalchemiops.torch.interactions.dispersion import dftd3
from nvalchemiops.torch.interactions.electrostatics import (
    dsf_coulomb,
    ewald_summation,
    particle_mesh_ewald,
)
from nvalchemiops.torch.neighbors import neighbor_list
from torch import Tensor, nn
from torch.autograd import Function

from aimnet import constants, nbops, ops


def _calc_coulomb_sr(
    data: dict[str, Tensor],
    rc: Tensor,
    envelope: str,
    key_in: str,
    factor: float,
) -> Tensor:
    """Shared short-range Coulomb energy calculation.

    Computes pairwise Coulomb energy with envelope-weighted cutoff.

    Parameters
    ----------
    data : dict
        Data dictionary containing d_ij distances and charges.
    rc : Tensor
        Cutoff radius tensor.
    envelope : str
        Envelope function: "exp" or "cosine".
    key_in : str
        Key for charges in data dict.
    factor : float
        Unit conversion factor (half_Hartree * Bohr).

    Returns
    -------
    Tensor
        Short-range Coulomb energy per molecule.
    """
    d_ij = data["d_ij"]
    q = data[key_in]
    q_i, q_j = nbops.get_ij(q, data)
    q_ij = q_i * q_j
    if envelope == "exp":
        fc = ops.exp_cutoff(d_ij, rc)
    else:  # cosine
        fc = ops.cosine_cutoff(d_ij, rc.item())
    e_ij = fc * q_ij / d_ij
    e_ij = nbops.mask_ij_(e_ij, data, 0.0)
    # Accumulate in float64 for precision
    e_i = e_ij.sum(-1, dtype=torch.float64)
    return factor * nbops.mol_sum(e_i, data)


@dataclass(slots=True)
class ExternalDerivativeTerms:
    """Explicit derivative terms returned by external nvalchemiops backends."""

    forces: Tensor | None = None
    virial: Tensor | None = None


def _periodic_coulomb_hybrid(
    *,
    coord: Tensor,
    cell: Tensor,
    charges: Tensor,
    batch_idx: Tensor,
    neighbor_matrix: Tensor,
    shifts: Tensor,
    mask_value: int,
    num_systems: int,
    accuracy: float,
    compute_virial: bool,
    is_pme: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Call nvalchemiops Ewald/PME in hybrid mode and convert outputs to eV."""
    fn = particle_mesh_ewald if is_pme else ewald_summation
    result = fn(
        positions=coord,
        charges=charges,
        cell=cell,
        batch_idx=batch_idx,
        neighbor_matrix=neighbor_matrix,
        neighbor_matrix_shifts=shifts,
        mask_value=mask_value,
        accuracy=accuracy,
        compute_forces=True,
        compute_charge_gradients=True,
        compute_virial=compute_virial,
        hybrid_forces=True,
    )

    ke = constants.Hartree * constants.Bohr
    energies_per_atom = result[0] * ke
    forces = result[1] * ke
    charge_grad = result[2] * ke
    virial = result[3] * ke if compute_virial else coord.new_empty(0)

    batch_idx_long = batch_idx.to(torch.int64)
    energies = torch.zeros(num_systems, dtype=torch.float64, device=coord.device)
    energies = energies.scatter_add(0, batch_idx_long, energies_per_atom.double())
    return energies, forces, charge_grad, virial


class _PeriodicCoulombFunction(Function):
    """Local training wrapper for Ewald/PME force and strain losses."""

    @staticmethod
    def forward(
        ctx: Any,
        coord: Tensor,
        cell: Tensor,
        scaling: Tensor | None,
        charges: Tensor,
        batch_idx: Tensor,
        neighbor_matrix: Tensor,
        shifts: Tensor,
        mask_value: int,
        num_systems: int,
        accuracy: float,
        is_pme: bool,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        batch_idx_long = batch_idx.to(torch.int64)
        if scaling is None:
            coord_eval = coord
            cell_eval = cell
        else:
            if scaling.ndim == 2:
                coord_eval = coord @ scaling
                cell_eval = cell @ scaling
            elif scaling.ndim == 3:
                atom_scaling = scaling.index_select(0, batch_idx_long)
                coord_eval = (coord.unsqueeze(1) @ atom_scaling).squeeze(1)
                cell_eval = cell @ scaling
            else:
                raise ValueError("scaling must have shape (3, 3) or (B, 3, 3)")

        energies, forces, charge_grad, virial = _periodic_coulomb_hybrid(
            coord=coord_eval,
            cell=cell_eval,
            charges=charges,
            batch_idx=batch_idx,
            neighbor_matrix=neighbor_matrix,
            shifts=shifts,
            mask_value=mask_value,
            num_systems=num_systems,
            accuracy=accuracy,
            compute_virial=scaling is not None,
            is_pme=is_pme,
        )
        if scaling is None:
            ctx.save_for_backward(forces, charge_grad, batch_idx_long)
        else:
            ctx.save_for_backward(forces, charge_grad, virial, batch_idx_long, scaling)
        return energies, forces, charge_grad, virial

    @staticmethod
    def backward(
        ctx: Any,
        grad_energies: Tensor,
        _grad_forces: Tensor,
        _grad_charge_grad: Tensor,
        _grad_virial: Tensor,
    ) -> tuple[Tensor | None, ...]:
        if len(ctx.saved_tensors) == 5:
            forces, charge_grad, virial, batch_idx, scaling = ctx.saved_tensors
        else:
            forces, charge_grad, batch_idx = ctx.saved_tensors
            scaling = None
        g = grad_energies.to(forces.dtype).index_select(0, batch_idx)
        grad_coord_eval = -forces * g.unsqueeze(-1)
        grad_charges = charge_grad * g

        if scaling is None:
            grad_coord = grad_coord_eval
            grad_scaling = None
        elif scaling.ndim == 2:
            grad_coord = grad_coord_eval @ scaling.mT
            grad_scaling = (-virial.mT * grad_energies.to(virial.dtype).view(-1, 1, 1)).sum(dim=0)
        elif scaling.ndim == 3:
            atom_scaling = scaling.index_select(0, batch_idx)
            grad_coord = (grad_coord_eval.unsqueeze(1) @ atom_scaling.mT).squeeze(1)
            grad_scaling = -virial.mT * grad_energies.to(virial.dtype).view(-1, 1, 1)

        return (
            grad_coord,
            None,
            grad_scaling,
            grad_charges,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class LRCoulomb(nn.Module):
    """Long-range Coulomb energy module.

    Computes electrostatic energy using one of several methods:
    simple (all pairs), DSF (damped shifted force), Ewald summation, or
    Particle Mesh Ewald (PME). DSF, Ewald, and PME are backed by
    ``nvalchemiops``; Ewald and PME require periodic systems with a ``cell``.

    Parameters
    ----------
    key_in : str
        Key for input charges in data dict. Default is "charges".
    key_out : str
        Key for output energy in data dict. Default is "e_h".
    rc : float
        Short-range cutoff radius. Default is 4.6 Angstrom.
    method : str
        Coulomb method: "simple", "dsf", "ewald", or "pme". Default is "simple".
    dsf_alpha : float
        Alpha parameter for DSF method. Default is 0.2.
    dsf_rc : float
        Cutoff for DSF method. Default is 15.0.
    ewald_accuracy : float
        Target accuracy for Ewald and PME summation. Controls real-space and
        reciprocal-space cutoffs (and PME mesh dimensions). Lower values give
        higher accuracy at higher cost. Default is 1e-6.
    subtract_sr : bool
        Whether to subtract short-range contribution. Default is True.
    envelope : str
        Envelope function for SR cutoff: "exp" or "cosine". Default is "exp".

    Notes
    -----
    Energy accumulation uses float64 for numerical precision, particularly
    important for large systems where many small contributions can suffer
    from floating-point error accumulation.

    Neighbor list keys follow a suffix resolution pattern: methods first look
    for module-specific keys (e.g., nbmat_coulomb, shifts_coulomb), falling
    back to shared _lr suffix (nbmat_lr, shifts_lr) if not found.

    DSF uses ``nvalchemiops.torch.interactions.electrostatics.dsf_coulomb``.
    Its energy is differentiable through charges, but not through positions
    or cell; the calculator consumes explicit DSF forces/virial for inference
    and rejects DSF force/stress training and Hessian requests.

    Ewald/PME call ``nvalchemiops`` directly. Inference uses
    ``hybrid_forces=True`` so energy remains differentiable through charges
    and fixed-charge geometry derivatives are returned as explicit terms.
    Training derivative paths use a small local ``autograd.Function`` wrapper
    because the installed nvalchemiops coordinate backward kernels do not
    currently provide a registered backward-of-backward.
    """

    def __init__(
        self,
        key_in: str = "charges",
        key_out: str = "e_h",
        rc: float = 4.6,
        method: str = "simple",
        dsf_alpha: float = 0.2,
        dsf_rc: float = 15.0,
        ewald_accuracy: float = 1e-6,
        subtract_sr: bool = True,
        envelope: str = "exp",
    ):
        super().__init__()
        self.key_in = key_in
        self.key_out = key_out
        # Pairwise convention factor used by simple/dsf (sums over ordered pairs).
        # Ewald/PME nvalchemiops outputs are converted with k_e = Hartree * Bohr.
        self._factor = constants.half_Hartree * constants.Bohr
        self.register_buffer("rc", torch.tensor(rc))
        self.dsf_alpha = dsf_alpha
        self.dsf_rc = dsf_rc
        self.ewald_accuracy = ewald_accuracy
        self.subtract_sr = subtract_sr
        if envelope not in ("exp", "cosine"):
            raise ValueError(f"Unknown envelope {envelope}, must be 'exp' or 'cosine'")
        self.envelope = envelope
        if method in ("simple", "dsf", "ewald", "pme"):
            self.method = method
        else:
            raise ValueError(f"Unknown method {method}")

    def coul_simple(self, data: dict[str, Tensor]) -> Tensor:
        """Compute pairwise Coulomb energy.

        With subtract_sr=True (default): Returns LR only (FULL - SR)
        With subtract_sr=False: Returns FULL pairwise Coulomb
        """
        suffix = nbops.resolve_suffix(data, ["_coulomb", "_lr"])
        data = ops.lazy_calc_dij(data, suffix)
        d_ij = data[f"d_ij{suffix}"]
        q = data[self.key_in]
        q_i, q_j = nbops.get_ij(q, data, suffix=suffix)
        q_ij = q_i * q_j
        # Compute FULL pairwise Coulomb (no exp_cutoff weighting)
        e_ij = q_ij / d_ij
        e_ij = nbops.mask_ij_(e_ij, data, 0.0, suffix=suffix)
        e_i = e_ij.sum(-1, dtype=torch.float64)
        e = self._factor * nbops.mol_sum(e_i, data)
        # Same pattern as dsf/ewald - subtract SR to get LR
        if self.subtract_sr:
            e = e - self.coul_simple_sr(data)
        return e

    def coul_simple_sr(self, data: dict[str, Tensor]) -> Tensor:
        return _calc_coulomb_sr(data, self.rc, self.envelope, self.key_in, self._factor)

    def _dsf_inputs_mode0(
        self,
        data: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor | None, Tensor | None, int, int]:
        """Flatten dense, unlisted molecular batches and build a cutoff-bounded
        neighbor list for nvalchemiops DSF.

        Uses ``nvalchemiops.torch.neighbors.neighbor_list`` with
        ``cutoff=self.dsf_rc`` so the kernel only sees pairs within range,
        not the full O(B·N²) all-pairs set. Padded atoms in the batch are
        shifted far away before NL construction so they cannot become
        neighbors of real atoms; their charges are zeroed regardless.
        """
        if data.get("cell") is not None:
            raise ValueError("Direct DSF with periodic cells requires flat nb_mode=1 neighbor matrices.")

        coord = data["coord"]
        charges = data[self.key_in].to(coord.dtype)
        mask_i = data["mask_i"]
        charges = charges.masked_fill(mask_i, 0.0)

        B, N = coord.shape[:2]
        total_atoms = B * N
        fill_value = total_atoms

        # Flatten atoms and place padded entries at deterministic sentinel
        # coordinates outside the real coordinate extent. This keeps padding out
        # of the cutoff-bounded neighbor list even for unwrapped large systems.
        coord_real = coord.reshape(total_atoms, 3)
        mask_i_flat = mask_i.reshape(total_atoms)
        if mask_i_flat.any():
            real_coord = coord_real[~mask_i_flat]
            extent = real_coord.abs().amax() if real_coord.numel() else coord_real.new_tensor(0.0)
            margin = coord_real.new_tensor(float(self.dsf_rc) + 1.0)
            pad_rank = torch.cumsum(mask_i_flat.to(coord_real.dtype), dim=0)[mask_i_flat]
            pad_coord = coord_real.new_zeros((pad_rank.shape[0], 3))
            pad_coord[:, 0] = extent + margin * pad_rank
            coord_real = coord_real.clone()
            coord_real[mask_i_flat] = pad_coord

        charges_real = charges.reshape(total_atoms)
        batch_idx_real = torch.arange(B, device=coord.device, dtype=torch.int32).repeat_interleave(N)

        # Build NL with overflow retry. Initial guess is a density-based
        # estimate (matching AdaptiveNeighborList's heuristic) clamped to the
        # absolute upper bound of N-1.
        sphere_volume = 4.0 / 3.0 * math.pi * float(self.dsf_rc) ** 3
        max_neighbors = max(16, min(N - 1, ((int(0.2 * sphere_volume) + 15) // 16) * 16))
        if max_neighbors < 1:
            max_neighbors = 1
        while True:
            try:
                nbmat_real, _ = neighbor_list(
                    positions=coord_real,
                    cutoff=float(self.dsf_rc),
                    batch_idx=batch_idx_real,
                    max_neighbors=max_neighbors,
                    half_fill=False,
                    fill_value=fill_value,
                    method="batch_naive",
                )
                break
            except NeighborOverflowError:
                if max_neighbors >= max(1, N - 1):
                    raise
                max_neighbors = min(max(1, N - 1), ((int(max_neighbors * 1.5) + 15) // 16) * 16)

        # Append the explicit pad atom (charges=0, coord=0) and matching pad row.
        positions = torch.cat([coord_real, coord_real.new_zeros(1, 3)], dim=0)
        charges_flat = torch.cat([charges_real, charges_real.new_zeros(1)], dim=0)
        batch_idx = torch.cat(
            [batch_idx_real, torch.zeros(1, device=coord.device, dtype=torch.int32)],
            dim=0,
        )
        pad_row = torch.full((1, nbmat_real.shape[1]), fill_value, dtype=torch.int32, device=coord.device)
        nbmat = torch.cat([nbmat_real.to(torch.int32), pad_row], dim=0)

        return positions, charges_flat, batch_idx, nbmat, None, None, fill_value, B

    def _dsf_inputs_mode1(
        self,
        data: dict[str, Tensor],
        suffix: str,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor | None, Tensor | None, int, int]:
        """Use padded flat nb_mode=1 tensors directly for nvalchemiops DSF."""
        coord = data["coord"]
        charges = nbops.mask_i_(data[self.key_in].to(coord.dtype), data, 0.0, inplace=False)
        batch_idx = data["mol_idx"].to(torch.int32)
        fill_value = coord.shape[0] - 1
        num_systems = int(batch_idx.max().item()) + 1

        nbmat = data[f"nbmat{suffix}"].to(torch.int32)
        cell = data.get("cell")
        shifts = None
        if cell is not None:
            cell = cell.to(coord.dtype)
            if cell.ndim == 2:
                cell = cell.unsqueeze(0)
            shifts = data[f"shifts{suffix}"].to(torch.int32)

        return coord, charges, batch_idx, nbmat, cell, shifts, fill_value, num_systems

    def _dsf_inputs_mode2(
        self,
        data: dict[str, Tensor],
        suffix: str,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor | None, Tensor | None, int, int]:
        """Flatten batched neighbor-matrix inputs for nvalchemiops DSF."""
        coord = data["coord"]
        charges = data[self.key_in].to(coord.dtype).masked_fill(data["mask_i"], 0.0)
        B, N = coord.shape[:2]
        fill_value = B * N

        positions = torch.cat([coord.reshape(B * N, 3), coord.new_zeros(1, 3)], dim=0)
        charges_flat = torch.cat([charges.reshape(B * N), charges.new_zeros(1)], dim=0)
        batch_idx = torch.cat(
            [
                torch.repeat_interleave(torch.arange(B, device=coord.device, dtype=torch.int32), N),
                torch.zeros(1, device=coord.device, dtype=torch.int32),
            ],
            dim=0,
        )

        # nbmat values are atom indices LOCAL to each batch; the flattened
        # `positions` tensor uses GLOBAL indices (b * N + i), so each batch's
        # neighbor entries must be offset by b * N. Without this, valid
        # neighbors in batch b > 0 silently point into batch 0 and DSF energy
        # is wrong. Mirrors the offset that DFTD3 mode-2 already applies.
        nbmat_local = data[f"nbmat{suffix}"].to(torch.int32)
        offsets = (torch.arange(B, device=coord.device, dtype=torch.int32) * N).view(B, 1, 1)
        nbmat = (nbmat_local + offsets).flatten(0, 1)
        mask_ij = data[f"mask_ij{suffix}"].flatten(0, 1)
        nbmat = torch.where(mask_ij, torch.full_like(nbmat, fill_value), nbmat)
        nbmat = torch.cat(
            [nbmat, torch.full((1, nbmat.shape[1]), fill_value, dtype=torch.int32, device=coord.device)],
            dim=0,
        )

        cell = data.get("cell")
        shifts = None
        if cell is not None:
            cell = cell.to(coord.dtype)
            if cell.ndim == 2:
                cell = cell.unsqueeze(0).expand(B, -1, -1)
            shifts = data[f"shifts{suffix}"].flatten(0, 1).to(torch.int32)
            shifts = torch.cat([shifts, torch.zeros((1, shifts.shape[1], 3), dtype=torch.int32, device=coord.device)])

        return positions, charges_flat, batch_idx, nbmat, cell, shifts, fill_value, B

    def _dsf_inputs(
        self,
        data: dict[str, Tensor],
        suffix: str,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor | None, Tensor | None, int, int]:
        nb_mode = nbops.get_nb_mode(data)
        if nb_mode == 0:
            return self._dsf_inputs_mode0(data)
        if nb_mode == 1:
            return self._dsf_inputs_mode1(data, suffix)
        if nb_mode == 2:
            return self._dsf_inputs_mode2(data, suffix)
        raise ValueError(f"Invalid neighbor mode: {nb_mode}")

    @staticmethod
    def _restore_dsf_forces_shape(data: dict[str, Tensor], forces: Tensor) -> Tensor:
        """Map flattened nvalchemiops DSF forces back to the input coordinate shape."""
        nb_mode = nbops.get_nb_mode(data)
        if nb_mode == 1:
            return forces
        if nb_mode in (0, 2):
            return forces[:-1].reshape_as(data["coord"])
        raise ValueError(f"Invalid neighbor mode: {nb_mode}")

    def _coul_dsf_nvalchemi(
        self,
        data: dict[str, Tensor],
        *,
        compute_forces: bool = False,
        compute_virial: bool = False,
    ) -> tuple[Tensor, ExternalDerivativeTerms | None]:
        """Compute DSF through nvalchemiops.

        DSF is a split-derivative external backend: geometry derivatives come
        from explicit fixed-charge forces/virial, while ``charges`` stays
        graph-attached so autograd can supply the charge-response term.
        """
        suffix = nbops.resolve_suffix(data, ["_coulomb", "_lr"])
        coord, charges, batch_idx, nbmat, cell, shifts, fill_value, num_systems = self._dsf_inputs(data, suffix)
        coord_for_kernel = coord.detach()
        cell_for_kernel = cell.detach() if cell is not None else None
        result = dsf_coulomb(
            positions=coord_for_kernel,
            charges=charges,
            cutoff=float(self.dsf_rc),
            alpha=float(self.dsf_alpha),
            cell=cell_for_kernel,
            batch_idx=batch_idx,
            neighbor_matrix=nbmat,
            neighbor_matrix_shifts=shifts,
            fill_value=int(fill_value),
            compute_forces=compute_forces or compute_virial,
            compute_virial=compute_virial,
            num_systems=int(num_systems),
            device=str(coord.device),
        )

        ke = constants.Hartree * constants.Bohr
        energy = result[0] * ke
        terms = None
        if compute_forces or compute_virial:
            forces = self._restore_dsf_forces_shape(data, result[1].detach() * ke) if compute_forces else None
            virial = result[2].detach() * ke if compute_virial else None
            terms = ExternalDerivativeTerms(forces=forces, virial=virial)

        if self.subtract_sr:
            data = ops.lazy_calc_dij(data, "")
            energy = energy - self.coul_simple_sr(data)
        return energy, terms

    def coul_dsf(self, data: dict[str, Tensor]) -> Tensor:
        energy, _terms = self._coul_dsf_nvalchemi(data)
        return energy

    def _coul_nvalchemi(
        self,
        data: dict[str, Tensor],
        backend: str,
        *,
        compute_forces: bool = False,
        compute_virial: bool = False,
        training_derivatives: bool = False,
        scaling: Tensor | None = None,
        coord_unstrained: Tensor | None = None,
        cell_unstrained: Tensor | None = None,
    ) -> tuple[Tensor, ExternalDerivativeTerms | None]:
        """Compute periodic Coulomb energy via nvalchemiops Ewald/PME.

        ``training_derivatives=True`` uses a local autograd wrapper only when
        coordinate, charge, or strain inputs require gradients. This keeps
        force/stress training on the explicit nvalchemiops forces/virial while
        avoiding an autograd wrapper for plain energy calls. ``False`` returns
        detached fixed-charge geometry derivatives as explicit terms while
        energy stays differentiable through charges.

        Requires ``cell`` in ``data`` and a PBC neighbor list under
        ``nbmat_coulomb``/``shifts_coulomb`` (preferred) or the shared
        ``nbmat_lr``/``shifts_lr``. Drops the trailing padding row before
        invoking the backend and re-adds a zero pad row so downstream
        ``unpad_output`` contracts are preserved.
        """
        suffix = nbops.resolve_suffix(data, ["_coulomb", "_lr"])

        coord = data["coord"]
        cell = data["cell"]
        assert cell is not None

        charges = data[self.key_in]
        mol_idx = data["mol_idx"]
        nbmat = data[f"nbmat{suffix}"]
        shifts = data[f"shifts{suffix}"]

        # Drop the trailing padding atom (flat mode includes one at index N).
        N_padded = coord.shape[0]
        N = N_padded - 1
        coord_real = coord[:-1]
        charges_real = charges[:-1]
        mol_idx_real = mol_idx[:-1].to(torch.int32)
        nbmat_real = nbmat[:-1].to(torch.int32)
        shifts_real = shifts[:-1].to(torch.int32)

        if backend not in ("ewald", "pme"):
            raise ValueError(f"backend must be 'ewald' or 'pme', got {backend!r}")
        num_systems = int(mol_idx_real.max().item()) + 1
        fn = particle_mesh_ewald if backend == "pme" else ewald_summation

        if training_derivatives:
            is_pme = backend == "pme"
            needs_strain_grad = scaling is not None and scaling.requires_grad
            needs_coord_or_charge_grad = coord_real.requires_grad or charges_real.requires_grad
            if needs_strain_grad:
                if coord_unstrained is None or cell_unstrained is None:
                    raise ValueError("scaling-aware Coulomb requires coord_unstrained and cell_unstrained")
                e_periodic, _forces, _charge_grad, _virial = _PeriodicCoulombFunction.apply(
                    coord_unstrained[:-1],
                    cell_unstrained,
                    scaling,
                    charges_real,
                    mol_idx_real,
                    nbmat_real,
                    shifts_real,
                    N,
                    num_systems,
                    float(self.ewald_accuracy),
                    is_pme,
                )
            elif needs_coord_or_charge_grad:
                e_periodic, _forces, _charge_grad, _virial = _PeriodicCoulombFunction.apply(
                    coord_real,
                    cell,
                    None,
                    charges_real,
                    mol_idx_real,
                    nbmat_real,
                    shifts_real,
                    N,
                    num_systems,
                    float(self.ewald_accuracy),
                    is_pme,
                )
            else:
                e_periodic = None

            if e_periodic is not None:
                if self.subtract_sr:
                    data = ops.lazy_calc_dij(data, "")
                    e_periodic = e_periodic - self.coul_simple_sr(data)
                return e_periodic, None

        result = fn(
            positions=coord_real.detach(),
            charges=charges_real,
            cell=cell.detach(),
            batch_idx=mol_idx_real,
            neighbor_matrix=nbmat_real,
            neighbor_matrix_shifts=shifts_real,
            mask_value=N,
            accuracy=float(self.ewald_accuracy),
            compute_forces=compute_forces,
            compute_charge_gradients=True,
            compute_virial=compute_virial,
            hybrid_forces=True,
        )

        result_tuple = result if isinstance(result, tuple) else (result,)
        idx = 0
        energies_per_atom = result_tuple[idx]
        idx += 1

        forces_real: Tensor | None = None
        if compute_forces:
            forces_real = result_tuple[idx]
            idx += 1

        # Charge gradients are injected into ``energies_per_atom`` by
        # nvalchemiops when charges require grad; the explicit tensor is not
        # needed by AIMNet.
        idx += 1

        virial: Tensor | None = None
        if compute_virial:
            virial = result_tuple[idx]

        ke = constants.Hartree * constants.Bohr
        energies_per_atom = energies_per_atom * ke
        energies_per_system = torch.zeros(num_systems, dtype=torch.float64, device=coord.device)
        energies_per_system = energies_per_system.scatter_add(
            0,
            mol_idx_real.to(torch.int64),
            energies_per_atom.double(),
        )

        terms = None
        if compute_forces or compute_virial:
            forces = None
            if forces_real is not None:
                forces = torch.cat([forces_real.detach() * ke, forces_real.new_zeros((1, 3))], dim=0)
            virial_ev = virial.detach() * ke if virial is not None else None
            terms = ExternalDerivativeTerms(forces=forces, virial=virial_ev)
        e_periodic = energies_per_system

        if self.subtract_sr:
            data = ops.lazy_calc_dij(data, "")
            e_periodic = e_periodic - self.coul_simple_sr(data)
        return e_periodic, terms

    def coul_ewald(self, data: dict[str, Tensor]) -> Tensor:
        """Per-system Ewald energy in eV. Requires ``cell`` and ``nbmat_lr``/``shifts_lr``."""
        energy, _terms = self._coul_nvalchemi(data, backend="ewald")
        return energy

    def coul_pme(self, data: dict[str, Tensor]) -> Tensor:
        """Per-system PME energy in eV. Requires ``cell`` and ``nbmat_lr``/``shifts_lr``."""
        energy, _terms = self._coul_nvalchemi(data, backend="pme")
        return energy

    def forward(
        self,
        data: dict[str, Tensor],
        *,
        compute_forces: bool = False,
        compute_virial: bool = False,
        return_terms: bool = False,
        training_derivatives: bool = False,
        scaling: Tensor | None = None,
        coord_unstrained: Tensor | None = None,
        cell_unstrained: Tensor | None = None,
    ) -> dict[str, Tensor] | tuple[dict[str, Tensor], ExternalDerivativeTerms | None]:
        if (compute_forces or compute_virial) and not return_terms:
            raise ValueError("compute_forces/compute_virial require return_terms=True")

        if self.method == "simple":
            e = self.coul_simple(data)
            terms = None
        elif self.method == "dsf":
            if training_derivatives:
                raise ValueError("DSF Coulomb does not support training derivatives")
            e, terms = self._coul_dsf_nvalchemi(
                data,
                compute_forces=compute_forces,
                compute_virial=compute_virial,
            )
        elif self.method == "ewald":
            e, terms = self._coul_nvalchemi(
                data,
                backend="ewald",
                compute_forces=compute_forces,
                compute_virial=compute_virial,
                training_derivatives=training_derivatives,
                scaling=scaling,
                coord_unstrained=coord_unstrained,
                cell_unstrained=cell_unstrained,
            )
        elif self.method == "pme":
            e, terms = self._coul_nvalchemi(
                data,
                backend="pme",
                compute_forces=compute_forces,
                compute_virial=compute_virial,
                training_derivatives=training_derivatives,
                scaling=scaling,
                coord_unstrained=coord_unstrained,
                cell_unstrained=cell_unstrained,
            )
        else:
            raise ValueError(f"Unknown method {self.method}")

        if self.key_out in data:
            data[self.key_out] = data[self.key_out].double() + e
        else:
            data[self.key_out] = e
        if return_terms:
            return data, terms
        return data


class SRCoulomb(nn.Module):
    """Subtract short-range Coulomb contribution from energy.

    For models trained with "simple" Coulomb mode, the NN has implicitly learned
    the short-range Coulomb interaction. When using DSF or Ewald summation for
    the full Coulomb energy, we need to subtract this short-range contribution
    to avoid double-counting.

    Parameters
    ----------
    rc : float
        Cutoff radius for short-range Coulomb. Default is 4.6 Angstrom.
    key_in : str
        Key for input charges in data dict. Default is "charges".
    key_out : str
        Key for output energy in data dict. Default is "energy".
    envelope : str
        Envelope function for cutoff: "exp" (mollifier) or "cosine". Default is "exp".
    """

    def __init__(
        self,
        rc: float = 4.6,
        key_in: str = "charges",
        key_out: str = "energy",
        envelope: str = "exp",
    ):
        super().__init__()
        self.key_in = key_in
        self.key_out = key_out
        self._factor = constants.half_Hartree * constants.Bohr
        self.register_buffer("rc", torch.tensor(rc))
        if envelope not in ("exp", "cosine"):
            raise ValueError(f"Unknown envelope {envelope}, must be 'exp' or 'cosine'")
        self.envelope = envelope

    def forward(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        """Subtract short-range Coulomb from energy."""
        e_sr = _calc_coulomb_sr(data, self.rc, self.envelope, self.key_in, self._factor)

        # Subtract short-range Coulomb from energy (in float64)
        if self.key_out in data:
            data[self.key_out] = data[self.key_out].double() - e_sr
        else:
            data[self.key_out] = -e_sr
        return data


class DispParam(nn.Module):
    def __init__(
        self,
        ref_c6: dict[int, Tensor] | Tensor | None = None,
        ref_alpha: dict[int, Tensor] | Tensor | None = None,
        ptfile: str | None = None,
        key_in: str = "disp_param",
        key_out: str = "disp_param",
    ):
        super().__init__()
        # Validate: cannot mix ptfile with ref_c6/ref_alpha
        if ptfile is not None and (ref_c6 is not None or ref_alpha is not None):
            raise ValueError("Cannot specify both ptfile and ref_c6/ref_alpha.")

        # Load reference data
        if ptfile is not None:
            ref = torch.load(ptfile, weights_only=True)
        elif ref_c6 is not None or ref_alpha is not None:
            ref = torch.zeros(87, 2)
            for i, p in enumerate([ref_c6, ref_alpha]):
                if p is not None:
                    if isinstance(p, Tensor):
                        ref[: p.shape[0], i] = p
                    else:
                        for k, v in p.items():
                            ref[k, i] = v
        else:
            # Placeholder - will be populated by load_state_dict
            ref = torch.zeros(87, 2)

        # Element 0 represents dummy atoms with c6=0 and alpha=1
        ref[0, 0] = 0.0
        ref[0, 1] = 1.0
        self.register_buffer("disp_param0", ref)
        self.key_in = key_in
        self.key_out = key_out

    def _load_from_state_dict(
        self,
        state_dict: dict,
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: list,
        unexpected_keys: list,
        error_msgs: list,
    ) -> None:
        # Resize placeholder buffer to match checkpoint size before loading
        key = prefix + "disp_param0"
        if key in state_dict:
            buf = state_dict[key]
            if buf.shape != self.disp_param0.shape:
                # Resize placeholder to match checkpoint
                self.disp_param0 = torch.zeros_like(buf)

            # Validate buffer has non-zero values (safety check)
            nonzero = (buf != 0).sum() / buf.numel()
            if nonzero < 0.1:
                import warnings

                warnings.warn(
                    f"DispParam buffer appears to have mostly zero values (nonzero: {nonzero:.1%}). "
                    "This may indicate a loading issue.",
                    stacklevel=2,
                )

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        disp_param_mult = data[self.key_in].clamp(min=-4, max=4).exp()
        disp_param = self.disp_param0[data["numbers"]]
        vals = disp_param * disp_param_mult
        data[self.key_out] = vals
        return data


class D3TS(nn.Module):
    """DFT-D3-like pairwise dispersion with TS combination rule"""

    def __init__(self, a1: float, a2: float, s8: float, s6: float = 1.0, key_in="disp_param", key_out="energy"):
        super().__init__()
        self.register_buffer("r4r2", constants.get_r4r2())
        self.a1 = a1
        self.a2 = a2
        self.s6 = s6
        self.s8 = s8
        self.key_in = key_in
        self.key_out = key_out

    def forward(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        suffix = nbops.resolve_suffix(data, ["_dftd3", "_lr"])

        disp_param = data[self.key_in]
        disp_param_i, disp_param_j = nbops.get_ij(disp_param, data, suffix=suffix)
        c6_i, alpha_i = disp_param_i.unbind(dim=-1)
        c6_j, alpha_j = disp_param_j.unbind(dim=-1)

        # TS combination rule
        c6ij = 2 * c6_i * c6_j / (c6_i * alpha_j / alpha_i + c6_j * alpha_i / alpha_j).clamp(min=1e-4)
        c6ij = nbops.mask_ij_(c6ij, data, 0.0, suffix=suffix)

        rr = self.r4r2[data["numbers"]]
        rr_i, rr_j = nbops.get_ij(rr, data, suffix=suffix)
        rrij = 3 * rr_i * rr_j
        rrij = nbops.mask_ij_(rrij, data, 1.0, suffix=suffix)
        r0ij = self.a1 * rrij.sqrt() + self.a2

        ops.lazy_calc_dij(data, suffix)
        d_ij = data[f"d_ij{suffix}"] * constants.Bohr_inv
        e_ij = c6ij * (self.s6 / (d_ij.pow(6) + r0ij.pow(6)) + self.s8 * rrij / (d_ij.pow(8) + r0ij.pow(8)))

        e = -constants.half_Hartree * nbops.mol_sum(e_ij.sum(-1), data)

        if self.key_out in data:
            data[self.key_out] = data[self.key_out] + e
        else:
            data[self.key_out] = e

        return data


class _DFTD3KernelInputs(NamedTuple):
    """Flat-atom inputs for the nvalchemi DFTD3 kernel call."""

    nb_mode: int
    coord: Tensor
    coord_flat: Tensor
    numbers_flat: Tensor
    batch_idx: Tensor
    neighbor_matrix: Tensor
    neighbor_matrix_shifts: Tensor | None
    num_systems: int
    fill_value: int
    cell_for_kernel: Tensor | None


def _call_dftd3_kernel(
    *,
    coord: Tensor,
    numbers: Tensor,
    batch_idx: Tensor,
    neighbor_matrix: Tensor,
    neighbor_matrix_shifts: Tensor | None,
    fill_value: int,
    num_systems: int,
    cell: Tensor | None,
    rcov: Tensor,
    r4r2: Tensor,
    c6_reference: Tensor,
    coord_num_ref: Tensor,
    a1: float,
    a2: float,
    s8: float,
    s6: float,
    smoothing_on: float,
    smoothing_off: float,
    compute_virial: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    """Call nvalchemiops DFT-D3 and convert energy, forces, and virial to eV units."""
    cell_bohr = None
    if cell is not None:
        cell_bohr = cell * constants.Bohr_inv
        if cell_bohr.ndim == 2:
            cell_bohr = cell_bohr.unsqueeze(0)

    dftd3_kwargs: dict[str, Any] = {
        "positions": coord * constants.Bohr_inv,
        "numbers": numbers,
        "a1": float(a1),
        "a2": float(a2),
        "s8": float(s8),
        "s6": float(s6),
        "covalent_radii": rcov,
        "r4r2": r4r2,
        "c6_reference": c6_reference,
        "coord_num_ref": coord_num_ref,
        "batch_idx": batch_idx,
        "cell": cell_bohr,
        "neighbor_matrix": neighbor_matrix,
        "neighbor_matrix_shifts": neighbor_matrix_shifts,
        "fill_value": int(fill_value),
        "num_systems": int(num_systems),
        "compute_virial": compute_virial,
        "device": str(coord.device),
    }
    if smoothing_on < smoothing_off:
        dftd3_kwargs["s5_smoothing_on"] = float(smoothing_on) * constants.Bohr_inv
        dftd3_kwargs["s5_smoothing_off"] = float(smoothing_off) * constants.Bohr_inv

    result = dftd3(**dftd3_kwargs)
    if compute_virial:
        energy_h, forces_h_bohr, _coord_num, virial_h = result
    else:
        energy_h, forces_h_bohr, _coord_num = result
        virial_h = coord.new_empty(0)

    energy_ev = energy_h * constants.Hartree
    forces_ev = forces_h_bohr * constants.Hartree * constants.Bohr_inv
    virial_ev = virial_h * constants.Hartree if compute_virial else virial_h
    return energy_ev.double(), forces_ev, virial_ev


class _DFTD3EnergyFunction(Function):
    """Embedded DFT-D3 wrapper that injects explicit forces/strain into autograd."""

    @staticmethod
    def forward(
        ctx: Any,
        coord: Tensor,
        cell: Tensor | None,
        scaling: Tensor | None,
        numbers: Tensor,
        batch_idx: Tensor,
        neighbor_matrix: Tensor,
        neighbor_matrix_shifts: Tensor | None,
        fill_value: int,
        num_systems: int,
        rcov: Tensor,
        r4r2: Tensor,
        c6_reference: Tensor,
        coord_num_ref: Tensor,
        a1: float,
        a2: float,
        s8: float,
        s6: float,
        smoothing_on: float,
        smoothing_off: float,
    ) -> Tensor:
        batch_idx_long = batch_idx.to(torch.int64)
        if scaling is None:
            coord_eval = coord
            cell_eval = cell
        else:
            if cell is None:
                raise ValueError("strain-aware DFTD3 requires cell")
            if scaling.ndim == 2:
                coord_eval = coord @ scaling
                cell_eval = cell @ scaling
            elif scaling.ndim == 3:
                atom_scaling = scaling.index_select(0, batch_idx_long)
                coord_eval = (coord.unsqueeze(1) @ atom_scaling).squeeze(1)
                cell_eval = cell @ scaling
            else:
                raise ValueError("scaling must have shape (3, 3) or (B, 3, 3)")

        energy, forces, virial = _call_dftd3_kernel(
            coord=coord_eval,
            numbers=numbers,
            batch_idx=batch_idx,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            fill_value=fill_value,
            num_systems=num_systems,
            cell=cell_eval,
            rcov=rcov,
            r4r2=r4r2,
            c6_reference=c6_reference,
            coord_num_ref=coord_num_ref,
            a1=a1,
            a2=a2,
            s8=s8,
            s6=s6,
            smoothing_on=smoothing_on,
            smoothing_off=smoothing_off,
            compute_virial=scaling is not None,
        )
        if scaling is None:
            ctx.save_for_backward(forces, batch_idx_long)
        else:
            ctx.save_for_backward(forces, virial, batch_idx_long, scaling)
        return energy

    @staticmethod
    def backward(ctx: Any, grad_energies: Tensor) -> tuple[Tensor | None, ...]:
        if len(ctx.saved_tensors) == 4:
            forces, virial, batch_idx, scaling = ctx.saved_tensors
        else:
            forces, batch_idx = ctx.saved_tensors
            scaling = None
        g = grad_energies.to(forces.dtype).index_select(0, batch_idx)
        grad_coord_eval = -forces * g.unsqueeze(-1)

        if scaling is None:
            grad_coord = grad_coord_eval
            grad_scaling = None
        elif scaling.ndim == 2:
            grad_coord = grad_coord_eval @ scaling.mT
            grad_scaling = (-virial.mT * grad_energies.to(virial.dtype).view(-1, 1, 1)).sum(dim=0)
        elif scaling.ndim == 3:
            atom_scaling = scaling.index_select(0, batch_idx)
            grad_coord = (grad_coord_eval.unsqueeze(1) @ atom_scaling.mT).squeeze(1)
            grad_scaling = -virial.mT * grad_energies.to(virial.dtype).view(-1, 1, 1)

        return (grad_coord, None, grad_scaling) + (None,) * 16


class DFTD3(nn.Module):
    """DFT-D3 implementation using nvalchemiops GPU-accelerated kernels.

    BJ damping, C6 and C8 terms, without 3-body term.

    This implementation uses nvalchemiops.torch.interactions.dispersion.dftd3 for
    GPU-accelerated computation of dispersion energies, forces, and virial. The
    embedded model path injects explicit forces/virial into autograd only when
    coordinate or strain gradients are requested; the external calculator path
    returns detached derivative terms.

    Parameters
    ----------
    s8 : float
        Scaling factor for C8 term.
    a1 : float
        BJ damping parameter 1.
    a2 : float
        BJ damping parameter 2.
    s6 : float, optional
        Scaling factor for C6 term. Default is 1.0.
    cutoff : float, optional
        Cutoff distance in Angstroms for smoothing. Default is 15.0.
    smoothing_fraction : float, optional
        Fraction of cutoff distance used for smoothing window width.
        Smoothing starts at cutoff * (1 - smoothing_fraction) and ends at cutoff.
        Example: With cutoff=15.0 and smoothing_fraction=0.2:
          - Smoothing starts at 12.0 Å (15.0 * 0.8)
          - Smoothing ends at 15.0 Å
        Default is 0.2 (20% of cutoff as smoothing window).
    key_out : str, optional
        Key for output energy in data dict. Default is "energy".
    Attributes
    ----------
    smoothing_on : float
        Distance where smoothing starts (Angstroms).
    smoothing_off : float
        Distance where smoothing ends / cutoff (Angstroms).
    s6, s8, a1, a2 : float
        BJ damping parameters.

    Notes
    -----
    Neighbor list keys follow a suffix resolution pattern: methods first look
    for module-specific keys (e.g., nbmat_dftd3, shifts_dftd3), falling back
    to shared _lr suffix (nbmat_lr, shifts_lr) if not found.
    """

    def __init__(
        self,
        s8: float,
        a1: float,
        a2: float,
        s6: float = 1.0,
        cutoff: float = 15.0,
        smoothing_fraction: float = 0.2,
        key_out: str = "energy",
    ):
        super().__init__()
        self.key_out = key_out
        # BJ damping parameters
        self.s6 = s6
        self.s8 = s8
        self.a1 = a1
        self.a2 = a2

        # Smoothing parameters as module attributes
        self.smoothing_on: float = cutoff * (1 - smoothing_fraction)
        self.smoothing_off: float = cutoff

        # Load D3 reference parameters and convert to nvalchemiops format
        dirname = os.path.dirname(os.path.dirname(__file__))
        filename = os.path.join(dirname, "dftd3_data.pt")
        param = torch.load(filename, map_location="cpu", weights_only=True)

        c6ab_packed = param["c6ab"]
        c6ab = c6ab_packed[..., 0].contiguous()
        cn_ref = c6ab_packed[..., 1].contiguous()

        # Register buffers for D3 parameters
        self.register_buffer("rcov", param["rcov"].float())
        self.register_buffer("r4r2", param["r4r2"].float())
        self.register_buffer("c6ab", c6ab.float())
        self.register_buffer("cn_ref", cn_ref.float())

    def set_smoothing(self, cutoff: float, smoothing_fraction: float = 0.2) -> None:
        """Update smoothing parameters based on new cutoff and fraction.

        Parameters
        ----------
        cutoff : float
            Cutoff distance in Angstroms.
        smoothing_fraction : float
            Fraction of cutoff used as smoothing window width.
            Smoothing occurs from cutoff * (1 - smoothing_fraction) to cutoff.
            Example: smoothing_fraction=0.2 means smoothing over last 20%
            of cutoff distance (from 0.8*cutoff to cutoff). Default is 0.2.
        """
        self.smoothing_on = cutoff * (1 - smoothing_fraction)
        self.smoothing_off = cutoff

    def _load_from_state_dict(
        self,
        state_dict: dict,
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: list,
        unexpected_keys: list,
        error_msgs: list,
    ) -> None:
        """Handle loading from old state dict format with packed c6ab.

        Migrates from legacy format where c6ab had shape [95, 95, 5, 5, 3]
        with last dimension containing (c6ref, cnref_i, cnref_j) to new format
        where c6ab is [95, 95, 5, 5] and cn_ref is separate [95, 95, 5, 5].
        Also removes deprecated cnmax parameter if present.
        """
        c6ab_key = prefix + "c6ab"
        cn_ref_key = prefix + "cn_ref"
        cnmax_key = prefix + "cnmax"

        if c6ab_key in state_dict and state_dict[c6ab_key].ndim == 5:
            c6ab_packed = state_dict[c6ab_key]
            state_dict[c6ab_key] = c6ab_packed[..., 0].contiguous()
            state_dict[cn_ref_key] = c6ab_packed[..., 1].contiguous()

        state_dict.pop(cnmax_key, None)

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def _prepare_dftd3_inputs(self, data: dict[str, Tensor]) -> "_DFTD3KernelInputs":
        """Build the flat-atom kernel arguments for the active ``nb_mode``.

        ``cell_for_kernel`` is ``None`` for mode 0 (no cell available); the
        original ``coord`` tensor is returned alongside the flat one so
        callers can recover the per-batch shape.
        """
        nb_mode = nbops.get_nb_mode(data)
        coord = data["coord"]
        numbers = data["numbers"].to(torch.int32)
        cell = data.get("cell")

        if nb_mode == 0:
            B, N = coord.shape[:2]
            coord_flat = coord.flatten(0, 1)
            numbers_flat = numbers.flatten()
            batch_idx = torch.arange(B, device=coord.device, dtype=torch.int32).repeat_interleave(N)
            num_systems = B
            total_atoms = B * N
            max_neighbors = N - 1

            arange_n = torch.arange(N, device=coord.device, dtype=torch.int32)
            all_indices = arange_n.unsqueeze(0).expand(N, -1)
            mask = all_indices != arange_n.unsqueeze(1)
            template = all_indices[mask].view(N, N - 1)
            batch_offsets = torch.arange(B, device=coord.device, dtype=torch.int32).unsqueeze(1).unsqueeze(2) * N
            neighbor_matrix = (template.unsqueeze(0) + batch_offsets).view(total_atoms, max_neighbors)

            fill_value = total_atoms
            neighbor_matrix_shifts: Tensor | None = None
            cell_for_kernel: Tensor | None = None

        elif nb_mode == 1:
            suffix = nbops.resolve_suffix(data, ["_dftd3", "_lr"])
            N = coord.shape[0]
            coord_flat = coord
            numbers_flat = numbers
            neighbor_matrix = data[f"nbmat{suffix}"].to(torch.int32)

            mol_idx = data.get("mol_idx")
            if mol_idx is not None:
                batch_idx = mol_idx.to(torch.int32)
                num_systems = int(mol_idx.max().item()) + 1
            else:
                batch_idx = torch.zeros(N, dtype=torch.int32, device=coord.device)
                num_systems = 1

            shifts = data.get(f"shifts{suffix}")
            neighbor_matrix_shifts = shifts.to(torch.int32) if shifts is not None else None
            fill_value = N
            cell_for_kernel = cell

        elif nb_mode == 2:
            suffix = nbops.resolve_suffix(data, ["_dftd3", "_lr"])
            B, N = coord.shape[:2]
            coord_flat = coord.flatten(0, 1)
            numbers_flat = numbers.flatten()
            batch_idx = torch.arange(B, device=coord.device, dtype=torch.int32).repeat_interleave(N)
            num_systems = B

            nbmat = data[f"nbmat{suffix}"]
            offsets = torch.arange(B, device=coord.device).unsqueeze(1) * N
            neighbor_matrix = (nbmat + offsets.unsqueeze(-1)).flatten(0, 1).to(torch.int32)

            shifts = data.get(f"shifts{suffix}")
            neighbor_matrix_shifts = shifts.flatten(0, 1).to(torch.int32) if shifts is not None else None
            fill_value = B * N
            cell_for_kernel = cell

        else:
            raise ValueError(f"Unsupported neighbor mode: {nb_mode}")

        return _DFTD3KernelInputs(
            nb_mode=nb_mode,
            coord=coord,
            coord_flat=coord_flat,
            numbers_flat=numbers_flat,
            batch_idx=batch_idx,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            num_systems=num_systems,
            fill_value=fill_value,
            cell_for_kernel=cell_for_kernel,
        )

    @staticmethod
    def _restore_dftd3_forces_shape(forces_flat: Tensor, nb_mode: int, coord_shape: torch.Size) -> Tensor:
        """Reshape flat per-atom forces back to the input layout for the active nb_mode."""
        if nb_mode == 0 or nb_mode == 2:
            B, N = coord_shape[:2]
            return forces_flat.view(B, N, 3)
        return forces_flat

    def _s5_switch_torch(self, d_ij_bohr: Tensor) -> Tensor:
        """nvalchemiops S5 cutoff switch for the differentiable Hessian path."""
        r_on = float(self.smoothing_on) * constants.Bohr_inv
        r_off = float(self.smoothing_off) * constants.Bohr_inv
        if r_off <= r_on:
            return torch.ones_like(d_ij_bohr)

        t = ((d_ij_bohr - r_on) / (r_off - r_on)).clamp(0.0, 1.0)
        t2 = t * t
        t3 = t2 * t
        t4 = t3 * t
        t5 = t4 * t
        switch = 1.0 - (10.0 * t3 - 15.0 * t4 + 6.0 * t5)
        return torch.where(d_ij_bohr <= r_on, torch.ones_like(switch), switch)

    def _calc_torch_coord_num(self, data: dict[str, Tensor], suffix: str, d_ij_bohr: Tensor) -> Tensor:
        """Coordination numbers matching nvalchemiops DFT-D3."""
        numbers = data["numbers"].to(torch.long)
        numbers_i, numbers_j = nbops.get_ij(numbers, data, suffix=suffix)
        rcov_i = self.rcov[numbers_i]
        rcov_j = self.rcov[numbers_j]
        cn_ij = torch.sigmoid(16.0 * ((rcov_i + rcov_j) / d_ij_bohr.clamp_min(1.0e-12) - 1.0))
        cn_ij = nbops.mask_ij_(cn_ij, data, 0.0, inplace=False, suffix=suffix)
        return cn_ij.sum(-1)

    def _calc_torch_c6ij(self, data: dict[str, Tensor], suffix: str, cn: Tensor) -> Tensor:
        """C6 interpolation matching nvalchemiops' single ``cn_ref`` tensor contract."""
        numbers = data["numbers"].to(torch.long)
        numbers_i, numbers_j = nbops.get_ij(numbers, data, suffix=suffix)
        cn_i, cn_j = nbops.get_ij(cn.unsqueeze(-1).unsqueeze(-1), data, suffix=suffix)

        c6ref = self.c6ab[numbers_i, numbers_j]
        cnref_i = self.cn_ref[numbers_i, numbers_j]
        cnref_j = self.cn_ref[numbers_j, numbers_i].transpose(-1, -2)

        valid = c6ref != 0
        exp_arg = -4.0 * ((cn_i - cnref_i).pow(2) + (cn_j - cnref_j).pow(2))
        max_exp = exp_arg.masked_fill(~valid, -torch.inf).amax(dim=(-1, -2), keepdim=True)
        finite_max = torch.isfinite(max_exp)
        shifted = torch.where(finite_max, exp_arg - max_exp, torch.zeros_like(exp_arg))
        weights = torch.where(valid & finite_max & (shifted >= -12.0), shifted.exp(), torch.zeros_like(shifted))

        weight_sum = weights.sum(dim=(-1, -2))
        c6_sum = (c6ref * weights).sum(dim=(-1, -2))
        return torch.where(weight_sum > 1.0e-12, c6_sum / weight_sum.clamp_min(1.0e-12), torch.zeros_like(weight_sum))

    def _compute_energy_torch(self, data: dict[str, Tensor]) -> Tensor:
        """Differentiable DFT-D3 energy used only when true Hessians are requested."""
        suffix = nbops.resolve_suffix(data, ["_dftd3", "_lr"])
        distance_data = data
        shifts_key = f"shifts{suffix}"
        if shifts_key in data and not data[shifts_key].is_floating_point():
            distance_data = {**data, shifts_key: data[shifts_key].to(dtype=data["coord"].dtype)}
        d_ij_bohr = ops.calc_distances(distance_data, suffix=suffix)[0].clamp_min(1.0e-12) * constants.Bohr_inv
        cn = self._calc_torch_coord_num(distance_data, suffix, d_ij_bohr)
        c6ij = self._calc_torch_c6ij(distance_data, suffix, cn)

        numbers = distance_data["numbers"].to(torch.long)
        numbers_i, numbers_j = nbops.get_ij(numbers, distance_data, suffix=suffix)
        r4r2_i = self.r4r2[numbers_i]
        r4r2_j = self.r4r2[numbers_j]
        r4r2_ij = 3.0 * r4r2_i * r4r2_j
        r0ij = self.a1 * r4r2_ij.sqrt() + self.a2

        d2 = d_ij_bohr.pow(2)
        d4 = d2.pow(2)
        d6 = d4 * d2
        d8 = d4.pow(2)
        r0_2 = r0ij.pow(2)
        r0_4 = r0_2.pow(2)
        r0_6 = r0_4 * r0_2
        r0_8 = r0_4.pow(2)

        damping = self.s6 / (d6 + r0_6) + self.s8 * r4r2_ij / (d8 + r0_8)
        switch = self._s5_switch_torch(d_ij_bohr)
        e_ij = -c6ij * damping * switch
        e_ij = nbops.mask_ij_(e_ij, distance_data, 0.0, inplace=False, suffix=suffix)
        return constants.half_Hartree * nbops.mol_sum(e_ij.sum(-1), distance_data)

    def forward(
        self,
        data: dict[str, Tensor],
        *,
        compute_forces: bool = False,
        compute_virial: bool = False,
        return_terms: bool = False,
        hessian: bool = False,
    ) -> dict[str, Tensor] | tuple[dict[str, Tensor], ExternalDerivativeTerms | None]:
        """Compute DFT-D3 energy and optional explicit derivative terms.

        The embedded path returns an autograd-capable energy only when the
        coordinate or calculator strain inputs require it. The external
        ``return_terms=True`` path always returns detached energy/terms.

        The returned virial follows the calculator-side external-derivative
        convention: ``get_derivatives`` subtracts ``terms.virial.mT`` from the
        strain gradient (the same path DSF uses). FD-validated against
        ``dE/dscaling`` in :class:`tests.test_dftd3.TestDFTD3ForwardTerms`.
        """
        if (compute_forces or compute_virial) and not return_terms:
            raise ValueError("compute_forces/compute_virial require return_terms=True")
        if hessian:
            if return_terms or compute_forces or compute_virial:
                raise ValueError("hessian=True uses differentiable DFTD3 energy; do not request explicit terms")
            energy_ev = self._compute_energy_torch(data).double()
            if self.key_out in data:
                data[self.key_out] = data[self.key_out].double() + energy_ev
            else:
                data[self.key_out] = energy_ev
            return data

        scaling = data.get("_dftd3_scaling")
        coord_unstrained = data.get("_dftd3_coord_unstrained")
        cell_unstrained = data.get("_dftd3_cell_unstrained")
        use_strain_wrapper = False
        if not return_terms and isinstance(scaling, Tensor) and scaling.requires_grad:
            if not isinstance(coord_unstrained, Tensor) or not isinstance(cell_unstrained, Tensor):
                raise ValueError("strain-aware DFTD3 requires coord_unstrained and cell_unstrained")
            use_strain_wrapper = True

        kernel_data = data
        if use_strain_wrapper:
            kernel_data = {**data, "coord": coord_unstrained, "cell": cell_unstrained}
        kernel_inputs = self._prepare_dftd3_inputs(kernel_data)

        common_args = (
            kernel_inputs.numbers_flat,
            kernel_inputs.batch_idx,
            kernel_inputs.neighbor_matrix,
            kernel_inputs.neighbor_matrix_shifts,
            int(kernel_inputs.fill_value),
            int(kernel_inputs.num_systems),
            self.rcov,
            self.r4r2,
            self.c6ab,
            self.cn_ref,
            float(self.a1),
            float(self.a2),
            float(self.s8),
            float(self.s6),
            float(self.smoothing_on),
            float(self.smoothing_off),
        )

        if return_terms:
            with torch.no_grad():
                energy_ev, forces_ev_flat, virial_kernel = _call_dftd3_kernel(
                    coord=kernel_inputs.coord_flat.detach(),
                    numbers=kernel_inputs.numbers_flat,
                    batch_idx=kernel_inputs.batch_idx,
                    neighbor_matrix=kernel_inputs.neighbor_matrix,
                    neighbor_matrix_shifts=kernel_inputs.neighbor_matrix_shifts,
                    fill_value=int(kernel_inputs.fill_value),
                    num_systems=int(kernel_inputs.num_systems),
                    cell=kernel_inputs.cell_for_kernel.detach() if kernel_inputs.cell_for_kernel is not None else None,
                    rcov=self.rcov,
                    r4r2=self.r4r2,
                    c6_reference=self.c6ab,
                    coord_num_ref=self.cn_ref,
                    a1=float(self.a1),
                    a2=float(self.a2),
                    s8=float(self.s8),
                    s6=float(self.s6),
                    smoothing_on=float(self.smoothing_on),
                    smoothing_off=float(self.smoothing_off),
                    compute_virial=compute_virial,
                )
            energy_ev = energy_ev.detach().double()
        elif use_strain_wrapper:
            energy_ev = _DFTD3EnergyFunction.apply(
                kernel_inputs.coord_flat,
                kernel_inputs.cell_for_kernel,
                scaling,
                *common_args,
            )
            forces_ev_flat = kernel_inputs.coord_flat.new_empty(0)
            virial_kernel = kernel_inputs.coord_flat.new_empty(0)
        elif kernel_inputs.coord_flat.requires_grad:
            energy_ev = _DFTD3EnergyFunction.apply(
                kernel_inputs.coord_flat,
                kernel_inputs.cell_for_kernel,
                None,
                *common_args,
            )
            forces_ev_flat = kernel_inputs.coord_flat.new_empty(0)
            virial_kernel = kernel_inputs.coord_flat.new_empty(0)
        else:
            with torch.no_grad():
                energy_ev, forces_ev_flat, virial_kernel = _call_dftd3_kernel(
                    coord=kernel_inputs.coord_flat.detach(),
                    numbers=kernel_inputs.numbers_flat,
                    batch_idx=kernel_inputs.batch_idx,
                    neighbor_matrix=kernel_inputs.neighbor_matrix,
                    neighbor_matrix_shifts=kernel_inputs.neighbor_matrix_shifts,
                    fill_value=int(kernel_inputs.fill_value),
                    num_systems=int(kernel_inputs.num_systems),
                    cell=kernel_inputs.cell_for_kernel.detach() if kernel_inputs.cell_for_kernel is not None else None,
                    rcov=self.rcov,
                    r4r2=self.r4r2,
                    c6_reference=self.c6ab,
                    coord_num_ref=self.cn_ref,
                    a1=float(self.a1),
                    a2=float(self.a2),
                    s8=float(self.s8),
                    s6=float(self.s6),
                    smoothing_on=float(self.smoothing_on),
                    smoothing_off=float(self.smoothing_off),
                    compute_virial=False,
                )
            energy_ev = energy_ev.detach().double()

        if self.key_out in data:
            data[self.key_out] = data[self.key_out].double() + energy_ev
        else:
            data[self.key_out] = energy_ev

        forces_ev: Tensor | None = None
        if compute_forces:
            forces_ev = self._restore_dftd3_forces_shape(
                forces_ev_flat.detach(),
                kernel_inputs.nb_mode,
                kernel_inputs.coord.shape,
            )

        # The nvalchemi DFTD3 kernel already returns the strain virial in the
        # external-derivative-term convention used by the calculator
        # (``dedc -= terms.virial.mT``). Verified by FD against
        # ``dE/dscaling`` under row-vector strain - see
        # ``tests/test_dftd3.py::TestDFTD3ForwardTerms``.
        external_virial: Tensor | None = None
        if compute_virial and virial_kernel.numel() > 0:
            external_virial = virial_kernel.detach().contiguous()

        terms = None
        if compute_forces or compute_virial:
            terms = ExternalDerivativeTerms(forces=forces_ev, virial=external_virial)
        if return_terms:
            return data, terms
        return data
