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
DFT-D3 Custom Op for PyTorch.

This module provides a TorchScript-compatible custom op for DFT-D3 dispersion
energy computation using nvalchemiops GPU-accelerated kernels.

TorchScript Compatibility
-------------------------
- torch.jit.script(): SUPPORTED - Models using this op can be scripted
- torch.jit.save(): SUPPORTED - Uses torch.autograd.Function pattern for serialization

The implementation wraps nvalchemiops calls in torch.autograd.Function classes,
which enables proper serialization with TorchScript.
"""

from typing import Any

import torch
from nvalchemiops.interactions.dispersion.dftd3 import dftd3
from torch import Tensor
from torch.autograd import Function

from aimnet import constants

# =============================================================================
# Autograd Function Classes
# =============================================================================


class _DFTD3Function(Function):
    """Autograd Function for DFT-D3 dispersion energy computation.

    This class wraps the nvalchemiops dftd3 implementation with proper
    autograd support, enabling both gradient computation and TorchScript
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

        # Cell gradient (if periodic and virial computed)
        grad_cell = None
        if has_cell and compute_virial and virial.numel() > 0:
            cell_inv_t = torch.linalg.inv(cell_bohr).transpose(-1, -2)
            dE_dcell_bohr = virial @ cell_inv_t
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

    # Cell gradient (if periodic and virial computed)
    grad_cell = None
    if has_cell and compute_virial and virial.numel() > 0:
        cell_inv_t = torch.linalg.inv(cell_bohr).transpose(-1, -2)
        dE_dcell_bohr = virial @ cell_inv_t
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
