# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
"""
Custom PyTorch ops for AIMNet long-range interactions.

This module provides custom ops for DFT-D3 dispersion and periodic Coulomb
(Ewald/PME) energy computation using nvalchemiops GPU-accelerated kernels.

Both custom-op families intentionally use a dual autograd-registration pattern:
the custom-op body calls a ``torch.autograd.Function.apply`` implementation for
eager Python execution, and the ``torch.library`` op also registers a separate
``register_autograd(..., setup_context=...)`` formula for traced/custom-op
execution such as ``torch.compile``. These two paths must remain behaviorally
aligned. Removing either one can leave eager mode working while compiled mode
breaks, or the reverse.

Note: Creating new TorchScript modules via torch.jit.script() is no longer
supported. Loading legacy .jpt files remains functional.

Stress Sign Convention
----------------------
This module follows the Cauchy (physical) stress convention:

    - Positive stress = tensile (material being stretched)
    - Negative stress = compressive (material being compressed)

The relationship between virial and Cauchy stress is:

    stress = -virial / volume

where virial = -0.5 * sum_ij(F_ij outer r_ij).

The cell gradient for autograd is computed as:

    dE/dcell = -virial @ inv(cell).T

This ensures the calculator returns physical Cauchy stress compatible with
ASE and standard MD conventions.
"""

from typing import Any

import torch
from nvalchemiops.torch.interactions.dispersion import dftd3
from nvalchemiops.torch.interactions.electrostatics import (
    ewald_summation,
    particle_mesh_ewald,
)
from torch import Tensor
from torch.autograd import Function

from aimnet import constants

# =============================================================================
# Autograd Function Classes
# =============================================================================


class _DFTD3Function(Function):
    """Autograd Function for DFT-D3 dispersion energy computation.

    This class wraps the nvalchemiops dftd3 implementation with proper
    autograd support, enabling gradient computation and custom op
    serialization.

    Notes
    -----
    Input coordinates are in Angstroms, internally converted to Bohr.
    Output energies are in eV, forces in eV/Angstrom.
    """

    @staticmethod
    def forward(
        ctx: Any,
        coord: Tensor,
        cell: Tensor,
        numbers: Tensor,
        batch_idx: Tensor,
        neighbor_matrix: Tensor,
        shifts: Tensor,
        rcov: Tensor,
        r4r2: Tensor,
        c6ab: Tensor,
        cn_ref: Tensor,
        a1: float,
        a2: float,
        s6: float,
        s8: float,
        num_systems: int,
        fill_value: int,
        smoothing_on: float,
        smoothing_off: float,
        compute_virial: bool,
        has_cell: bool,
        has_shifts: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Forward pass: compute DFT-D3 dispersion energy."""
        # Convert coordinates to Bohr
        coord_bohr = coord * constants.Bohr_inv

        # Convert cell to Bohr if present
        cell_bohr = None
        if has_cell:
            cell_bohr = cell * constants.Bohr_inv
            if cell_bohr.ndim == 2:
                cell_bohr = cell_bohr.unsqueeze(0)

        # Handle shifts
        shifts_arg = None
        if has_shifts:
            shifts_arg = shifts

        # Build kwargs for nvalchemiops dftd3 call
        dftd3_kwargs: dict[str, Any] = {
            "positions": coord_bohr,
            "numbers": numbers,
            "a1": a1,
            "a2": a2,
            "s8": s8,
            "s6": s6,
            "covalent_radii": rcov,
            "r4r2": r4r2,
            "c6_reference": c6ab,
            "coord_num_ref": cn_ref,
            "batch_idx": batch_idx,
            "cell": cell_bohr,
            "neighbor_matrix": neighbor_matrix,
            "neighbor_matrix_shifts": shifts_arg,
            "fill_value": fill_value,
            "num_systems": num_systems,
            "compute_virial": compute_virial,
            "device": str(coord.device),
        }

        # Only pass smoothing parameters if smoothing is enabled
        # When smoothing_on >= smoothing_off, omit to use nvalchemiops defaults (1e10)
        if smoothing_on < smoothing_off:
            dftd3_kwargs["s5_smoothing_on"] = smoothing_on * constants.Bohr_inv
            dftd3_kwargs["s5_smoothing_off"] = smoothing_off * constants.Bohr_inv

        # Call nvalchemiops dftd3
        result = dftd3(**dftd3_kwargs)

        if compute_virial:
            energy, forces, _coord_num, virial = result
        else:
            energy, forces, _coord_num = result
            virial = torch.empty(0, device=coord.device)

        # Convert to eV/Angstrom units
        energy_ev = energy * constants.Hartree
        forces_ev_ang = forces * constants.Hartree * constants.Bohr_inv

        # Save tensors for backward - convert cell_bohr for gradient computation
        cell_bohr_saved = torch.empty(0, device=coord.device)
        if has_cell:
            cell_bohr_saved = cell * constants.Bohr_inv
            if cell_bohr_saved.ndim == 2:
                cell_bohr_saved = cell_bohr_saved.unsqueeze(0)

        ctx.save_for_backward(forces_ev_ang, virial, batch_idx, cell_bohr_saved)
        ctx.has_cell = has_cell
        ctx.compute_virial = compute_virial

        return energy_ev, forces_ev_ang, virial

    @staticmethod
    def backward(
        ctx: Any,
        grad_energy: Tensor,
        grad_forces: Tensor,
        grad_virial: Tensor,
    ) -> tuple[Tensor | None, ...]:
        """Backward pass: compute gradients w.r.t. coord and cell."""
        forces, virial, batch_idx, cell_bohr = ctx.saved_tensors
        has_cell = ctx.has_cell
        compute_virial = ctx.compute_virial

        # Coord gradient: forces = -dE/dR, so dE/dR = -forces
        grad_coord = -forces * grad_energy[batch_idx].unsqueeze(-1)

        # Cell gradient for physical Cauchy stress
        # Cauchy stress = -virial / volume, so dE/dcell = -virial @ inv(cell).T
        grad_cell = None
        if has_cell and compute_virial and virial.numel() > 0:
            cell_inv_t = torch.linalg.inv(cell_bohr).mT
            dE_dcell_bohr = -virial @ cell_inv_t  # Negative sign for Cauchy stress
            dE_dcell_ang = dE_dcell_bohr * constants.Hartree * constants.Bohr_inv
            grad_cell = dE_dcell_ang * grad_energy.view(-1, 1, 1)

        # Return gradients for all 21 inputs (only coord and cell have gradients)
        return (
            grad_coord,
            grad_cell,
            None,  # numbers
            None,  # batch_idx
            None,  # neighbor_matrix
            None,  # shifts
            None,  # rcov
            None,  # r4r2
            None,  # c6ab
            None,  # cn_ref
            None,  # a1
            None,  # a2
            None,  # s6
            None,  # s8
            None,  # num_systems
            None,  # fill_value
            None,  # smoothing_on
            None,  # smoothing_off
            None,  # compute_virial
            None,  # has_cell
            None,  # has_shifts
        )


# =============================================================================
# PyTorch Custom Op Registration
# =============================================================================


@torch.library.custom_op("aimnet::dftd3_fwd", mutates_args=())
def dftd3_fwd(
    coord: Tensor,
    cell: Tensor,
    numbers: Tensor,
    batch_idx: Tensor,
    neighbor_matrix: Tensor,
    shifts: Tensor,
    rcov: Tensor,
    r4r2: Tensor,
    c6ab: Tensor,
    cn_ref: Tensor,
    a1: float,
    a2: float,
    s6: float,
    s8: float,
    num_systems: int,
    fill_value: int,
    smoothing_on: float,
    smoothing_off: float,
    compute_virial: bool,
    has_cell: bool,
    has_shifts: bool,
) -> list[Tensor]:
    """
    Forward primitive for DFT-D3 energy computation.

    Returns [energy, forces, virial] tensors.
    """
    result = _DFTD3Function.apply(
        coord,
        cell,
        numbers,
        batch_idx,
        neighbor_matrix,
        shifts,
        rcov,
        r4r2,
        c6ab,
        cn_ref,
        a1,
        a2,
        s6,
        s8,
        num_systems,
        fill_value,
        smoothing_on,
        smoothing_off,
        compute_virial,
        has_cell,
        has_shifts,
    )
    return list(result)


@torch.library.register_fake("aimnet::dftd3_fwd")
def _(
    coord: Tensor,
    cell: Tensor,
    numbers: Tensor,
    batch_idx: Tensor,
    neighbor_matrix: Tensor,
    shifts: Tensor,
    rcov: Tensor,
    r4r2: Tensor,
    c6ab: Tensor,
    cn_ref: Tensor,
    a1: float,
    a2: float,
    s6: float,
    s8: float,
    num_systems: int,
    fill_value: int,
    smoothing_on: float,
    smoothing_off: float,
    compute_virial: bool,
    has_cell: bool,
    has_shifts: bool,
) -> list[Tensor]:
    """Fake implementation for torch.compile tracing."""
    n_atoms = coord.shape[0]
    energy = coord.new_empty(num_systems)
    forces = coord.new_empty(n_atoms, 3)
    if compute_virial:
        virial = coord.new_empty(num_systems, 3, 3)
    else:
        virial = coord.new_empty(0)
    return [energy, forces, virial]


# =============================================================================
# Autograd Registration (Method-based)
# =============================================================================


def _dftd3_setup_context(ctx: Any, inputs: tuple[Any, ...], output: list[Tensor]) -> None:
    """Setup context for backward pass."""
    (
        coord,
        cell,
        _numbers,
        batch_idx,
        _neighbor_matrix,
        _shifts,
        _rcov,
        _r4r2,
        _c6ab,
        _cn_ref,
        _a1,
        _a2,
        _s6,
        _s8,
        _num_systems,
        _fill_value,
        _smoothing_on,
        _smoothing_off,
        compute_virial,
        has_cell,
        _has_shifts,
    ) = inputs
    _energy, forces, virial = output

    # Convert cell to Bohr for backward
    cell_bohr = torch.empty(0, device=coord.device)
    if has_cell:
        cell_bohr = cell * constants.Bohr_inv
        if cell_bohr.ndim == 2:
            cell_bohr = cell_bohr.unsqueeze(0)

    ctx.save_for_backward(forces, virial, batch_idx, cell_bohr)
    ctx.has_cell = has_cell
    ctx.compute_virial = compute_virial


def _dftd3_backward(
    ctx: Any,
    grad_outputs: list[Tensor],
) -> tuple[Tensor | None, ...]:
    """Backward pass for dftd3 energy."""
    grad_energy = grad_outputs[0]
    # grad_outputs[1] and grad_outputs[2] are grad_forces and grad_virial (unused)
    forces, virial, batch_idx, cell_bohr = ctx.saved_tensors
    has_cell = ctx.has_cell
    compute_virial = ctx.compute_virial

    # Coord gradient: forces = -dE/dR, so dE/dR = -forces
    grad_coord = -forces * grad_energy[batch_idx].unsqueeze(-1)

    # Cell gradient for physical Cauchy stress
    # Cauchy stress = -virial / volume, so dE/dcell = -virial @ inv(cell).T
    grad_cell = None
    if has_cell and compute_virial and virial.numel() > 0:
        cell_inv_t = torch.linalg.inv(cell_bohr).mT
        dE_dcell_bohr = -virial @ cell_inv_t  # Negative sign for Cauchy stress
        dE_dcell_ang = dE_dcell_bohr * constants.Hartree * constants.Bohr_inv
        grad_cell = dE_dcell_ang * grad_energy.view(-1, 1, 1)

    # Return gradients for all 21 inputs (only coord and cell have gradients)
    return (
        grad_coord,
        grad_cell,
        None,  # numbers
        None,  # batch_idx
        None,  # neighbor_matrix
        None,  # shifts
        None,  # rcov
        None,  # r4r2
        None,  # c6ab
        None,  # cn_ref
        None,  # a1
        None,  # a2
        None,  # s6
        None,  # s8
        None,  # num_systems
        None,  # fill_value
        None,  # smoothing_on
        None,  # smoothing_off
        None,  # compute_virial
        None,  # has_cell
        None,  # has_shifts
    )


# Use method-based autograd registration for serializability
dftd3_fwd.register_autograd(
    _dftd3_backward,
    setup_context=_dftd3_setup_context,
)


# =============================================================================
# Public API
# =============================================================================


def dftd3_energy(
    coord: Tensor,
    cell: Tensor | None,
    numbers: Tensor,
    batch_idx: Tensor,
    neighbor_matrix: Tensor,
    shifts: Tensor | None,
    rcov: Tensor,
    r4r2: Tensor,
    c6ab: Tensor,
    cn_ref: Tensor,
    a1: float,
    a2: float,
    s6: float,
    s8: float,
    num_systems: int,
    fill_value: int,
    smoothing_on: float,
    smoothing_off: float,
    compute_virial: bool = False,
) -> Tensor:
    """
    Compute DFT-D3 dispersion energy with automatic differentiation support.

    This function wraps the nvalchemiops DFT-D3 implementation as a PyTorch
    custom op with proper autograd support for computing gradients.

    Parameters
    ----------
    coord : Tensor
        Atomic coordinates in Angstrom, shape (N, 3)
    cell : Tensor or None
        Unit cell vectors in Angstrom, shape (3, 3) or (B, 3, 3)
    numbers : Tensor
        Atomic numbers, shape (N,)
    batch_idx : Tensor
        Batch index for each atom, shape (N,)
    neighbor_matrix : Tensor
        Neighbor indices, shape (N, M)
    shifts : Tensor or None
        Periodic shift vectors as integers, shape (N, M, 3)
    rcov : Tensor
        Covalent radii, shape (95,)
    r4r2 : Tensor
        R4/R2 expectation values, shape (95,)
    c6ab : Tensor
        C6 reference values, shape (95, 95, 5, 5)
    cn_ref : Tensor
        Coordination number references, shape (95, 95, 5, 5)
    a1 : float
        BJ damping parameter a1
    a2 : float
        BJ damping parameter a2
    s6 : float
        Scaling factor for C6 term
    s8 : float
        Scaling factor for C8 term
    num_systems : int
        Number of systems in batch
    fill_value : int
        Fill value for invalid neighbor indices
    smoothing_on : float
        Distance at which smoothing starts, in Angstrom.
    smoothing_off : float
        Distance at which smoothing ends (cutoff), in Angstrom.
    compute_virial : bool
        Whether to compute virial tensor for cell gradients

    Returns
    -------
    Tensor
        Dispersion energy per system in eV, shape (num_systems,)

    Notes
    -----
    Input coordinates are in Angstroms, internally converted to Bohr.
    Output energies are in eV, forces in eV/Angstrom.
    """
    # Prepare tensors - custom op requires non-None tensors
    cell_tensor = cell if cell is not None else torch.empty(0, device=coord.device)
    shifts_tensor = shifts if shifts is not None else torch.empty(0, device=coord.device, dtype=torch.int32)

    has_cell = cell is not None
    has_shifts = shifts is not None

    result = torch.ops.aimnet.dftd3_fwd(
        coord,
        cell_tensor,
        numbers,
        batch_idx,
        neighbor_matrix,
        shifts_tensor,
        rcov,
        r4r2,
        c6ab,
        cn_ref,
        a1,
        a2,
        s6,
        s8,
        num_systems,
        fill_value,
        smoothing_on,
        smoothing_off,
        compute_virial,
        has_cell,
        has_shifts,
    )

    return result[0]  # Return only energy


# =============================================================================
# LRCoulomb (Ewald / PME) Custom Op
# =============================================================================
#
# Same pattern as DFT-D3: wrap the nvalchemiops call in an autograd.Function
# so the Warp kernels (which have no registered 2nd-order autograd) can be
# used safely under `create_graph=True`. The Function's backward publishes
# analytic 1st-order gradients from saved detached tensors and produces
# zero 2nd-order contribution for the periodic piece — the standard
# ML-potential approximation for kernels that lack 2nd-order support.
#
# `charges` is the exception: nvalchemiops is invoked with `hybrid_forces=True`
# which internally detaches positions/cell but keeps charges attached via a
# straight-through trick. Our Function's backward surfaces the explicit
# `charge_gradients` as the grad w.r.t. `charges`; PyTorch autograd then
# continues propagating through the NN's charge-producing subgraph on its
# own, so `d(E)/d(params)` through the charge chain is preserved.
#
# Energies returned by nvalchemiops are in e²/Å (formula already includes the
# 1/2 prefactor); they are converted to eV using the Coulomb constant
# k_e = Hartree * Bohr at the Function boundary.


def _cell_grad_from_virial(
    coord: Tensor,
    grad_coord: Tensor,
    virial: Tensor,
    batch_idx: Tensor,
    cell: Tensor,
    grad_energies: Tensor,
    batched_cell: bool,
) -> Tensor:
    """Return cell gradients whose row-vector strain derivative is ``-virial.T``.

    AIMNet applies strain as ``coord @ scaling`` and ``cell @ scaling``. The
    nvalchemiops virial convention is ``W = -dE/dstrain`` for the same
    row-vector deformation, so the total derivative with respect to
    ``scaling`` must be ``-W.T``. PyTorch will already account for the
    coordinate contribution through ``grad_coord``; the cell gradient is the
    residual needed to make ``coord.T @ grad_coord + cell.T @ grad_cell`` equal
    that target.
    """
    g_sys = grad_energies.to(virial.dtype)
    target = -virial.mT * g_sys.view(-1, 1, 1)

    atom_outer = coord.to(grad_coord.dtype).unsqueeze(2) * grad_coord.unsqueeze(1)
    coord_term = torch.zeros_like(target)
    coord_term.index_add_(0, batch_idx, atom_outer.to(target.dtype))

    residual = target - coord_term
    if batched_cell:
        return torch.linalg.solve(cell.mT, residual)
    return torch.linalg.solve(cell.mT, residual.sum(dim=0))


class _LRCoulombFunction(Function):
    """Autograd Function for periodic Coulomb (Ewald or PME) via nvalchemiops.

    Forward calls ``ewald_summation`` / ``particle_mesh_ewald`` with
    ``hybrid_forces=True``, always requesting explicit forces, charge
    gradients and optionally the virial. Per-atom energies are summed into
    per-system totals inside the Function so the backward math matches the
    DFTD3 pattern (per-system ``grad_energy`` scatter-indexed per atom by
    ``batch_idx``).

    Notes
    -----
    The 2nd-order backward returns zero for the periodic force-on-position
    piece (``explicit_forces`` is a detached constant). This is the standard
    approximation for ML potentials backed by non-differentiable MD kernels.
    The charge chain through the NN is exact because ``grad_charges`` is a
    pure PyTorch product of detached ``charge_grad`` with the graph-attached
    ``grad_energy``.
    """

    @staticmethod
    def forward(
        ctx: Any,
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
        """Forward pass: compute periodic Coulomb energy, forces, charge-grads, virial."""
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
        energies_per_atom = result[0]
        forces = result[1]
        charge_grad = result[2]
        virial = result[3] if compute_virial else coord.new_empty(0)

        # Convert units: nvalchemiops returns e²/Å; multiply by k_e = Hartree * Bohr
        ke = constants.Hartree * constants.Bohr
        energies_per_atom = energies_per_atom * ke
        forces = forces * ke
        charge_grad = charge_grad * ke
        if compute_virial:
            virial = virial * ke

        # Scatter-reduce per-atom energies to per-system totals (matches
        # nbops.mol_sum). Using float64 for numerical precision.
        batch_idx_long = batch_idx.to(torch.int64)
        energies = torch.zeros(num_systems, dtype=torch.float64, device=coord.device)
        energies = energies.scatter_add(0, batch_idx_long, energies_per_atom.double())

        ctx.save_for_backward(coord.detach(), forces, charge_grad, virial, batch_idx_long, cell.detach())
        ctx.compute_virial = compute_virial
        ctx.batched_cell = cell.ndim == 3

        return energies, forces, charge_grad, virial

    @staticmethod
    def backward(
        ctx: Any,
        grad_energies: Tensor,
        _grad_forces: Tensor,
        _grad_charge_grad: Tensor,
        _grad_virial: Tensor,
    ) -> tuple[Tensor | None, ...]:
        """Backward: analytic grads for coord, cell, charges; None for the rest."""
        coord, forces, charge_grad, virial, batch_idx, cell = ctx.saved_tensors

        # Broadcast per-system grad_energy to per-atom via batch_idx.
        g = grad_energies.to(forces.dtype).index_select(0, batch_idx)

        # Coord: forces = -dE/dR, so dE/dR = -forces.
        grad_coord = -forces * g.unsqueeze(-1)

        # Charges: straight-through of dE/dq through the Function boundary.
        grad_charges = charge_grad * g

        grad_cell: Tensor | None = None
        if ctx.compute_virial and virial.numel() > 0:
            grad_cell = _cell_grad_from_virial(
                coord=coord,
                grad_coord=grad_coord,
                virial=virial,
                batch_idx=batch_idx,
                cell=cell,
                grad_energies=grad_energies,
                batched_cell=ctx.batched_cell,
            )

        return (
            grad_coord,
            grad_cell,
            grad_charges,
            None,  # batch_idx
            None,  # neighbor_matrix
            None,  # shifts
            None,  # mask_value
            None,  # num_systems
            None,  # accuracy
            None,  # compute_virial
            None,  # is_pme
        )


@torch.library.custom_op("aimnet::lr_coulomb_fwd", mutates_args=())
def lr_coulomb_fwd(
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
) -> list[Tensor]:
    """Forward primitive for periodic Coulomb (Ewald or PME).

    Returns ``[energies_per_system, forces_per_atom, charge_grad_per_atom, virial]``.
    ``virial`` is an empty tensor when ``compute_virial=False``.
    """
    result = _LRCoulombFunction.apply(
        coord,
        cell,
        charges,
        batch_idx,
        neighbor_matrix,
        shifts,
        mask_value,
        num_systems,
        accuracy,
        compute_virial,
        is_pme,
    )
    return list(result)


@torch.library.register_fake("aimnet::lr_coulomb_fwd")
def _(
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
) -> list[Tensor]:
    """Fake implementation for torch.compile tracing."""
    n_atoms = coord.shape[0]
    energies = coord.new_empty(num_systems, dtype=torch.float64)
    forces = coord.new_empty(n_atoms, 3)
    charge_grad = coord.new_empty(n_atoms)
    if compute_virial:
        n_systems = num_systems if cell.ndim == 3 else 1
        virial = coord.new_empty(n_systems, 3, 3)
    else:
        virial = coord.new_empty(0)
    return [energies, forces, charge_grad, virial]


def _lr_coulomb_setup_context(ctx: Any, inputs: tuple[Any, ...], output: list[Tensor]) -> None:
    """Setup context for method-based autograd registration."""
    (
        coord,
        cell,
        _charges,
        batch_idx,
        _neighbor_matrix,
        _shifts,
        _mask_value,
        _num_systems,
        _accuracy,
        compute_virial,
        _is_pme,
    ) = inputs
    _energies, forces, charge_grad, virial = output
    batch_idx_long = batch_idx.to(torch.int64)
    ctx.save_for_backward(coord.detach(), forces, charge_grad, virial, batch_idx_long, cell.detach())
    ctx.compute_virial = compute_virial
    ctx.batched_cell = cell.ndim == 3


def _lr_coulomb_backward(
    ctx: Any,
    grad_outputs: list[Tensor],
) -> tuple[Tensor | None, ...]:
    """Backward for lr_coulomb_fwd."""
    grad_energies = grad_outputs[0]
    coord, forces, charge_grad, virial, batch_idx, cell = ctx.saved_tensors

    g = grad_energies.to(forces.dtype).index_select(0, batch_idx)
    grad_coord = -forces * g.unsqueeze(-1)
    grad_charges = charge_grad * g

    grad_cell: Tensor | None = None
    if ctx.compute_virial and virial.numel() > 0:
        grad_cell = _cell_grad_from_virial(
            coord=coord,
            grad_coord=grad_coord,
            virial=virial,
            batch_idx=batch_idx,
            cell=cell,
            grad_energies=grad_energies,
            batched_cell=ctx.batched_cell,
        )

    return (
        grad_coord,
        grad_cell,
        grad_charges,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


lr_coulomb_fwd.register_autograd(
    _lr_coulomb_backward,
    setup_context=_lr_coulomb_setup_context,
)


def lr_coulomb_energy(
    coord: Tensor,
    cell: Tensor,
    charges: Tensor,
    batch_idx: Tensor,
    neighbor_matrix: Tensor,
    shifts: Tensor,
    mask_value: int,
    num_systems: int,
    accuracy: float,
    backend: str,
    compute_virial: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute periodic Coulomb energy with autograd support.

    Wraps the ``nvalchemiops`` Ewald/PME call in a custom op whose backward
    uses the explicit forces, charge gradients, and virial returned by the
    same call. This makes the Function safe under ``create_graph=True``
    (e.g., force-loss training via ``aimnet.modules.core.Forces``) while the
    underlying Warp kernels have no registered 2nd-order autograd.

    Parameters
    ----------
    coord : Tensor, shape (N, 3)
        Atomic positions in Angstrom.
    cell : Tensor, shape (3, 3) or (B, 3, 3)
        Unit cell matrix.
    charges : Tensor, shape (N,)
        Atomic partial charges. Must be graph-attached for the charge chain
        to backprop to model parameters.
    batch_idx : Tensor, shape (N,)
        Per-atom system index (int32 or int64).
    neighbor_matrix : Tensor, shape (N, M)
        Dense neighbor list. Invalid entries should equal ``mask_value``.
    shifts : Tensor, shape (N, M, 3)
        Periodic image shifts for ``neighbor_matrix``.
    mask_value : int
        Value indicating invalid neighbor entries (conventionally ``N``).
    num_systems : int
        Number of independent systems (B).
    accuracy : float
        Target accuracy for parameter estimation inside nvalchemiops.
    backend : str
        ``"ewald"`` or ``"pme"``.
    compute_virial : bool, default False
        If True, compute the virial tensor and enable cell gradients.

    Returns
    -------
    energies : Tensor, shape (num_systems,)
        Per-system total Coulomb energy in eV.
    forces : Tensor, shape (N, 3)
        Explicit forces in eV/Å (constant w.r.t. autograd).
    charge_grad : Tensor, shape (N,)
        Explicit ``dE/dq`` in eV/e (constant w.r.t. autograd).
    virial : Tensor
        Virial tensor in eV when ``compute_virial=True``, else empty.
    """
    if backend not in ("ewald", "pme"):
        raise ValueError(f"backend must be 'ewald' or 'pme', got {backend!r}")
    is_pme = backend == "pme"
    result = torch.ops.aimnet.lr_coulomb_fwd(
        coord,
        cell,
        charges,
        batch_idx,
        neighbor_matrix,
        shifts,
        int(mask_value),
        int(num_systems),
        float(accuracy),
        bool(compute_virial),
        bool(is_pme),
    )
    return result[0], result[1], result[2], result[3]
