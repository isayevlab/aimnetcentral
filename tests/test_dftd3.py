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
Comprehensive tests for DFT-D3 dispersion implementation.

Tests cover:
- Forward computation
- Explicit force and virial terms
- Physical correctness
"""

import pytest
import torch
from conftest import add_dftd3_keys

from aimnet import nbops
from aimnet.modules.lr import DFTD3

# =============================================================================
# Test Data Fixtures
# =============================================================================


def setup_dftd3_data_mode_0(device, n_atoms=5):
    """Create test data in nb_mode=0 format for DFTD3."""
    torch.manual_seed(42)
    coord = torch.rand((1, n_atoms, 3), device=device) * 5  # 5 Angstrom box
    # Use common organic elements (H, C, N, O)
    numbers = torch.tensor([[6, 1, 1, 7, 8]], device=device)[:, :n_atoms]

    # Create neighbor matrix for DFTD3
    max_nb = n_atoms - 1
    nbmat_dftd3 = torch.zeros((1, n_atoms, max_nb), dtype=torch.long, device=device)
    for i in range(n_atoms):
        neighbors = [j for j in range(n_atoms) if j != i]
        for k, nb in enumerate(neighbors[:max_nb]):
            nbmat_dftd3[0, i, k] = nb

    data = {
        "coord": coord,
        "numbers": numbers,
        "nbmat_dftd3": nbmat_dftd3,
        "cutoff_dftd3": torch.tensor(15.0),
    }
    data = nbops.set_nb_mode(data)
    data = nbops.calc_masks(data)

    return data


def setup_dftd3_data_mode_1(device, n_atoms=5):
    """Create test data in nb_mode=1 format (flat with mol_idx) for DFTD3."""
    torch.manual_seed(42)
    coord = torch.rand((n_atoms + 1, 3), device=device) * 5  # +1 for padding
    # Use common organic elements, padding atom has number 0
    numbers = torch.tensor([6, 1, 1, 7, 8, 0], device=device)[: n_atoms + 1]
    mol_idx = torch.cat([torch.zeros(n_atoms, dtype=torch.long, device=device), torch.tensor([0], device=device)])

    # Create neighbor matrix (all atoms see each other)
    max_nb = n_atoms
    nbmat = torch.zeros((n_atoms + 1, max_nb), dtype=torch.long, device=device)
    nbmat_lr = torch.zeros((n_atoms + 1, max_nb), dtype=torch.long, device=device)
    nbmat_dftd3 = torch.zeros((n_atoms + 1, max_nb), dtype=torch.long, device=device)
    for i in range(n_atoms):
        neighbors = [j for j in range(n_atoms + 1) if j != i][:max_nb]
        for k, nb in enumerate(neighbors):
            nbmat[i, k] = nb
            nbmat_lr[i, k] = nb
            nbmat_dftd3[i, k] = nb
    # Padding row points to padding atom
    nbmat[-1] = n_atoms
    nbmat_lr[-1] = n_atoms
    nbmat_dftd3[-1] = n_atoms

    data = {
        "coord": coord,
        "numbers": numbers,
        "mol_idx": mol_idx,
        "nbmat": nbmat,
        "nbmat_lr": nbmat_lr,
        "nbmat_dftd3": nbmat_dftd3,
        "cutoff_dftd3": torch.tensor(15.0),
    }
    data = nbops.set_nb_mode(data)
    data = nbops.calc_masks(data)

    return data


def _central_fd_hessian(energy_fn, coord_flat: torch.Tensor, step: float = 5e-3) -> torch.Tensor:
    """Central finite-difference Hessian for small coordinate-only test systems."""
    coord0 = coord_flat.detach()
    ndim = coord0.numel()
    hessian = torch.empty(ndim, ndim, dtype=coord0.dtype, device=coord0.device)
    eye = torch.eye(ndim, dtype=coord0.dtype, device=coord0.device)
    energy0 = energy_fn(coord0)

    for i in range(ndim):
        e_i = eye[i]
        e_plus = energy_fn(coord0 + step * e_i)
        e_minus = energy_fn(coord0 - step * e_i)
        hessian[i, i] = (e_plus - 2.0 * energy0 + e_minus) / (step * step)
        for j in range(i + 1, ndim):
            e_j = eye[j]
            e_pp = energy_fn(coord0 + step * e_i + step * e_j)
            e_pm = energy_fn(coord0 + step * e_i - step * e_j)
            e_mp = energy_fn(coord0 - step * e_i + step * e_j)
            e_mm = energy_fn(coord0 - step * e_i - step * e_j)
            value = (e_pp - e_pm - e_mp + e_mm) / (4.0 * step * step)
            hessian[i, j] = value
            hessian[j, i] = value
    return hessian


# =============================================================================
# Initialization Tests
# =============================================================================


@pytest.mark.gpu
class TestDFTD3Init:
    """Tests for DFTD3 initialization."""

    def test_default_init(self, device):
        """Test default initialization with required parameters."""
        module = DFTD3(s8=0.3908, a1=0.5660, a2=3.1280).to(device)
        assert module.s8 == 0.3908
        assert module.a1 == 0.5660
        assert module.a2 == 3.1280
        assert module.s6 == 1.0
        assert module.key_out == "energy"

    def test_custom_s6(self, device):
        """Test initialization with custom s6."""
        module = DFTD3(s8=0.5, a1=0.4, a2=4.0, s6=0.9).to(device)
        assert module.s6 == 0.9

    def test_custom_key_out(self, device):
        """Test initialization with custom key_out."""
        module = DFTD3(s8=0.5, a1=0.4, a2=4.0, key_out="dispersion").to(device)
        assert module.key_out == "dispersion"

    def test_buffers_loaded(self, device):
        """Test that D3 parameter buffers are loaded correctly."""
        module = DFTD3(s8=0.5, a1=0.4, a2=4.0).to(device)
        assert module.rcov.shape == (95,)
        assert module.r4r2.shape == (95,)
        assert module.c6ab.shape == (95, 95, 5, 5)
        assert module.cn_ref.shape == (95, 95, 5, 5)


# =============================================================================
# Forward Pass Tests
# =============================================================================


@pytest.mark.gpu
class TestDFTD3Forward:
    """Tests for DFTD3 forward pass."""

    def test_output_shape_mode_0(self, device):
        """Test that DFTD3 produces correct output shape in mode 0."""
        module = DFTD3(s8=0.3908, a1=0.5660, a2=3.1280).to(device)
        data = setup_dftd3_data_mode_0(device, n_atoms=5)
        data["coord"] = data["coord"].requires_grad_(True)

        result, terms = module(data, compute_forces=True)

        assert "energy" in result
        assert result["energy"].shape == (1,)  # Per-molecule energy
        assert terms is not None and terms.forces is not None
        assert terms.forces.shape == (1, 5, 3)

    def test_output_shape_mode_1(self, device):
        """Test that DFTD3 produces correct output shape in mode 1."""
        module = DFTD3(s8=0.3908, a1=0.5660, a2=3.1280).to(device)
        data = setup_dftd3_data_mode_1(device, n_atoms=5)
        data["coord"] = data["coord"].requires_grad_(True)

        result, terms = module(data, compute_forces=True)

        assert "energy" in result
        assert result["energy"].shape == (1,)
        assert terms is not None and terms.forces is not None
        assert terms.forces.shape == (6, 3)  # n_atoms + 1 for padding

    def test_mode_1_uses_trailing_padding_atom_as_fill_value(self, device):
        """Flat padded DFTD3 inputs use the final atom index as the kernel sentinel."""
        module = DFTD3(s8=0.3908, a1=0.5660, a2=3.1280).to(device)
        data = setup_dftd3_data_mode_1(device, n_atoms=5)

        kernel_inputs = module._prepare_dftd3_inputs(data)

        assert kernel_inputs.fill_value == data["coord"].shape[0] - 1
        assert torch.equal(
            kernel_inputs.neighbor_matrix[data["mask_ij_dftd3"]],
            torch.full_like(kernel_inputs.neighbor_matrix[data["mask_ij_dftd3"]], kernel_inputs.fill_value),
        )

    def test_mode_2_masks_padded_neighbors_after_global_offset(self, device):
        """Batched sparse DFTD3 inputs convert local padding indices to the global fill value."""
        module = DFTD3(s8=0.3908, a1=0.5660, a2=3.1280).to(device)
        coord = torch.tensor(
            [
                [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [1.10, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            dtype=torch.float32,
            device=device,
        )
        numbers = torch.tensor([[8, 1, 0], [6, 1, 0]], device=device)
        nbmat = torch.tensor(
            [
                [[1, 2], [0, 2], [2, 2]],
                [[1, 2], [0, 2], [2, 2]],
            ],
            device=device,
        )
        data = {
            "coord": coord,
            "numbers": numbers,
            "nbmat": nbmat,
            "nbmat_dftd3": nbmat,
            "cutoff_dftd3": torch.tensor(15.0, device=device),
        }
        data = nbops.set_nb_mode(data)
        data = nbops.calc_masks(data)

        kernel_inputs = module._prepare_dftd3_inputs(data)

        assert kernel_inputs.fill_value == coord.numel() // 3
        mask_flat = data["mask_ij_dftd3"].flatten(0, 1)
        assert torch.equal(
            kernel_inputs.neighbor_matrix[mask_flat],
            torch.full_like(kernel_inputs.neighbor_matrix[mask_flat], kernel_inputs.fill_value),
        )
        assert kernel_inputs.neighbor_matrix[3, 0].item() == 4

    def test_energy_is_negative(self, device):
        """Test that dispersion energy is negative (attractive)."""
        module = DFTD3(s8=0.3908, a1=0.5660, a2=3.1280).to(device)

        # Water molecule
        coord = torch.tensor([[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]], device=device)
        numbers = torch.tensor([[8, 1, 1]], device=device)

        data = {"coord": coord, "numbers": numbers}
        data = add_dftd3_keys(data, device)
        data = nbops.set_nb_mode(data)
        data = nbops.calc_masks(data)

        result = module(data)

        # Dispersion energy should be negative (attractive)
        assert result["energy"].item() < 0

    def test_energy_finite(self, device):
        """Test that energy is finite for typical molecules."""
        module = DFTD3(s8=0.3908, a1=0.5660, a2=3.1280).to(device)
        data = setup_dftd3_data_mode_0(device, n_atoms=5)
        data["coord"] = data["coord"].requires_grad_(True)

        result, terms = module(data, compute_forces=True)

        assert torch.isfinite(result["energy"]).all()
        assert terms is not None and terms.forces is not None
        assert torch.isfinite(terms.forces).all()

    def test_deterministic(self, device):
        """Test that forward pass is deterministic."""
        module = DFTD3(s8=0.3908, a1=0.5660, a2=3.1280).to(device)

        coord = torch.tensor([[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0]]], device=device)
        numbers = torch.tensor([[6, 1, 1]], device=device)

        data1 = {"coord": coord.clone(), "numbers": numbers.clone()}
        data1 = add_dftd3_keys(data1, device)
        data1 = nbops.set_nb_mode(data1)
        data1 = nbops.calc_masks(data1)
        result1 = module(data1)

        data2 = {"coord": coord.clone(), "numbers": numbers.clone()}
        data2 = add_dftd3_keys(data2, device)
        data2 = nbops.set_nb_mode(data2)
        data2 = nbops.calc_masks(data2)
        result2 = module(data2)

        torch.testing.assert_close(result1["energy"], result2["energy"])


# =============================================================================
# Additive Energy Tests
# =============================================================================


@pytest.mark.gpu
class TestDFTD3Additive:
    """Tests for DFTD3 additive energy and forces."""

    def test_energy_addition(self, device):
        """Test that energy is added to existing key if present."""
        module = DFTD3(s8=0.3908, a1=0.5660, a2=3.1280).to(device)

        coord = torch.tensor([[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0]]], device=device)
        numbers = torch.tensor([[6, 1, 1]], device=device)

        data = {"coord": coord, "numbers": numbers}
        data = add_dftd3_keys(data, device)
        data = nbops.set_nb_mode(data)
        data = nbops.calc_masks(data)

        # Set initial energy
        data["energy"] = torch.tensor([10.0], device=device)

        result = module(data)

        # Energy should be added to existing value (dispersion is negative)
        assert result["energy"].item() < 10.0

    def test_explicit_forces_do_not_modify_data_forces(self, device):
        """DFT-D3 returns explicit forces through terms, not ``data["forces"]``."""
        module = DFTD3(s8=0.3908, a1=0.5660, a2=3.1280).to(device)

        coord = torch.tensor([[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0]]], device=device, requires_grad=True)
        numbers = torch.tensor([[6, 1, 1]], device=device)

        data = {"coord": coord, "numbers": numbers}
        data = add_dftd3_keys(data, device)
        data = nbops.set_nb_mode(data)
        data = nbops.calc_masks(data)

        # Set initial forces
        initial_forces = torch.ones((1, 3, 3), device=device) * 0.5
        data["forces"] = initial_forces.clone()

        result, terms = module(data, compute_forces=True)

        torch.testing.assert_close(result["forces"], initial_forces)
        assert terms is not None and terms.forces is not None
        assert not torch.allclose(terms.forces, initial_forces)


# =============================================================================
# Physical Correctness Tests
# =============================================================================


@pytest.mark.gpu
class TestDFTD3PhysicalCorrectness:
    """Tests for physical correctness of DFTD3."""

    def test_distance_dependence(self, device):
        """Test that dispersion energy decreases with distance."""
        module = DFTD3(s8=0.3908, a1=0.5660, a2=3.1280).to(device)

        # Close atoms
        coord_close = torch.tensor([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]], device=device)
        numbers = torch.tensor([[6, 6]], device=device)

        data_close = {"coord": coord_close, "numbers": numbers}
        data_close = add_dftd3_keys(data_close, device)
        data_close = nbops.set_nb_mode(data_close)
        data_close = nbops.calc_masks(data_close)
        result_close = module(data_close)

        # Far atoms
        coord_far = torch.tensor([[[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]], device=device)
        data_far = {"coord": coord_far, "numbers": numbers.clone()}
        data_far = add_dftd3_keys(data_far, device)
        data_far = nbops.set_nb_mode(data_far)
        data_far = nbops.calc_masks(data_far)

        module2 = DFTD3(s8=0.3908, a1=0.5660, a2=3.1280).to(device)
        result_far = module2(data_far)

        # Close atoms should have stronger (more negative) dispersion
        assert result_close["energy"].item() < result_far["energy"].item()

    def test_heavier_atoms_stronger_dispersion(self, device):
        """Test that heavier atoms have stronger dispersion."""
        module = DFTD3(s8=0.3908, a1=0.5660, a2=3.1280).to(device)

        coord = torch.tensor([[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]], device=device)

        # H-H pair
        numbers_hh = torch.tensor([[1, 1]], device=device)
        data_hh = {"coord": coord.clone(), "numbers": numbers_hh}
        data_hh = add_dftd3_keys(data_hh, device)
        data_hh = nbops.set_nb_mode(data_hh)
        data_hh = nbops.calc_masks(data_hh)
        result_hh = module(data_hh)

        # C-C pair
        module2 = DFTD3(s8=0.3908, a1=0.5660, a2=3.1280).to(device)
        numbers_cc = torch.tensor([[6, 6]], device=device)
        data_cc = {"coord": coord.clone(), "numbers": numbers_cc}
        data_cc = add_dftd3_keys(data_cc, device)
        data_cc = nbops.set_nb_mode(data_cc)
        data_cc = nbops.calc_masks(data_cc)
        result_cc = module2(data_cc)

        # C-C should have stronger dispersion than H-H
        assert result_cc["energy"].item() < result_hh["energy"].item()


# =============================================================================
# Batching Tests
# =============================================================================


@pytest.mark.gpu
class TestDFTD3Batching:
    """Tests for DFTD3 with batched input."""

    def test_batched_input_mode_0(self, device):
        """Test DFTD3 with batched input in mode 0."""
        module = DFTD3(s8=0.3908, a1=0.5660, a2=3.1280).to(device)

        # Batch of 2 molecules, 3 atoms each
        coord = torch.tensor(
            [
                [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0]],
                [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
            ],
            device=device,
            requires_grad=True,
        )
        numbers = torch.tensor([[6, 1, 1], [6, 1, 1]], device=device)

        data = {"coord": coord, "numbers": numbers}
        data = add_dftd3_keys(data, device)
        data = nbops.set_nb_mode(data)
        data = nbops.calc_masks(data)

        result, terms = module(data, compute_forces=True)

        assert result["energy"].shape == (2,)
        assert torch.isfinite(result["energy"]).all()
        assert terms is not None and terms.forces is not None
        assert terms.forces.shape == (2, 3, 3)
        assert torch.isfinite(terms.forces).all()


# =============================================================================
# Explicit Derivative Tests
# =============================================================================


@pytest.mark.gpu
class TestDFTD3ExplicitDerivatives:
    """Tests for DFTD3's embedded and external derivative contracts."""

    def test_embedded_energy_backward_matches_explicit_forces(self, device):
        """Embedded DFTD3 injects explicit nvalchemiops forces into coord autograd."""
        module = DFTD3(s8=0.3908, a1=0.5660, a2=3.1280).to(device)

        coord = torch.tensor(
            [[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]], device=device, requires_grad=True
        )
        numbers = torch.tensor([[8, 1, 1]], device=device)

        data = {"coord": coord, "numbers": numbers}
        data = add_dftd3_keys(data, device)
        data = nbops.set_nb_mode(data)
        data = nbops.calc_masks(data)

        result = module(data)
        assert result["energy"].requires_grad

        result["energy"].sum().backward()
        assert coord.grad is not None

        terms_data = {**data, "coord": coord.detach()}
        _, terms = module(terms_data, compute_forces=True)
        assert terms is not None and terms.forces is not None
        torch.testing.assert_close(-coord.grad, terms.forces, rtol=1e-4, atol=1e-5)

    def test_energy_only_without_grad_inputs_is_detached(self, device):
        """Plain inference avoids an autograd wrapper when no input needs gradients."""
        module = DFTD3(s8=0.3908, a1=0.5660, a2=3.1280).to(device)

        coord = torch.tensor([[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]], device=device)
        numbers = torch.tensor([[8, 1, 1]], device=device)

        data = {"coord": coord, "numbers": numbers}
        data = add_dftd3_keys(data, device)
        data = nbops.set_nb_mode(data)
        data = nbops.calc_masks(data)

        result = module(data)

        assert not result["energy"].requires_grad
        assert result["energy"].grad_fn is None

    def test_explicit_forces_are_returned_as_terms(self, device):
        """Forces are exposed as detached external derivative terms."""
        module = DFTD3(s8=0.3908, a1=0.5660, a2=3.1280).to(device)

        coord = torch.tensor([[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0]]], device=device, requires_grad=True)
        numbers = torch.tensor([[6, 1, 1]], device=device)

        data = {"coord": coord, "numbers": numbers}
        data = add_dftd3_keys(data, device)
        data = nbops.set_nb_mode(data)
        data = nbops.calc_masks(data)

        result, terms = module(data.copy(), compute_forces=True)

        assert terms is not None and terms.forces is not None
        assert not result["energy"].requires_grad
        assert not terms.forces.requires_grad
        assert torch.isfinite(terms.forces).all()

    def test_hessian_path_matches_explicit_forces_and_double_backward(self, device):
        """Hessian fallback is differentiable and keeps first derivatives aligned with nvalchemiops."""
        module = DFTD3(s8=0.3908, a1=0.5660, a2=3.1280).to(device)

        coord = torch.tensor(
            [[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0]]],
            device=device,
            requires_grad=True,
        )
        numbers = torch.tensor([[6, 1, 1]], device=device)

        data = {"coord": coord, "numbers": numbers}
        data = add_dftd3_keys(data, device)
        data = nbops.set_nb_mode(data)
        data = nbops.calc_masks(data)

        out = module(data, hessian=True)
        grad_coord = torch.autograd.grad(out["energy"].sum(), coord, create_graph=True)[0]

        terms_data = {**data, "coord": coord.detach()}
        _, terms = module(terms_data, compute_forces=True)
        assert terms is not None and terms.forces is not None
        torch.testing.assert_close(-grad_coord, terms.forces, rtol=2e-3, atol=2e-4)

        second = torch.autograd.grad(grad_coord.pow(2).sum(), coord)[0]
        assert torch.isfinite(second).all()
        assert second.abs().sum() > 0

    def test_hessian_path_matches_energy_finite_difference(self, device):
        """Pure-torch DFTD3 Hessian matches central FD on its own energy."""
        module = DFTD3(s8=0.3908, a1=0.5660, a2=3.1280, cutoff=8.0).to(device=device, dtype=torch.float64)

        coord0 = torch.tensor([[0.0, 0.0, 0.0], [1.65, 0.10, 0.0]], dtype=torch.float64, device=device)
        numbers = torch.tensor([[6, 8]], device=device)
        template = {"coord": coord0.unsqueeze(0), "numbers": numbers}
        template = add_dftd3_keys(template, device, cutoff=8.0)
        template = nbops.set_nb_mode(template)
        template = nbops.calc_masks(template)

        def energy_fn(coord_flat: torch.Tensor) -> torch.Tensor:
            data = {**template, "coord": coord_flat.reshape(1, 2, 3)}
            return module._compute_energy_torch(data).sum()

        coord_flat = coord0.reshape(-1)
        h_auto = torch.autograd.functional.hessian(energy_fn, coord_flat)
        h_fd = _central_fd_hessian(energy_fn, coord_flat)

        torch.testing.assert_close(h_auto, h_fd, rtol=5e-3, atol=5e-3)


# =============================================================================
# Forward explicit terms (calculator stress path)
# =============================================================================


def _setup_pbc_dftd3_data(coord_real, cell_value, numbers_real, device, cutoff=8.0):
    """Build flat nb_mode=1 PBC data for DFTD3 with a real periodic neighbor list."""
    from nvalchemiops.torch.neighbors import neighbor_list

    n_real = coord_real.shape[0]
    pad_idx = n_real
    pbc = torch.tensor([True, True, True], device=device)
    batch_idx_real = torch.zeros(n_real, dtype=torch.int32, device=device)

    nbmat_real, _, shifts_real = neighbor_list(
        positions=coord_real,
        cutoff=cutoff,
        cell=cell_value.unsqueeze(0),
        pbc=pbc,
        batch_idx=batch_idx_real,
        max_neighbors=64,
        half_fill=False,
        fill_value=pad_idx,
    )
    actual_max = int(nbmat_real.shape[1])

    coord = torch.cat([coord_real, coord_real.new_zeros((1, 3))], dim=0)
    numbers = torch.cat([numbers_real, numbers_real.new_zeros((1,))], dim=0)
    mol_idx = torch.zeros(coord.shape[0], dtype=torch.long, device=device)

    nbmat = torch.full((n_real + 1, actual_max), pad_idx, dtype=torch.long, device=device)
    nbmat[:n_real] = nbmat_real
    shifts = torch.zeros((n_real + 1, actual_max, 3), dtype=torch.int32, device=device)
    shifts[:n_real] = shifts_real.to(torch.int32)

    data = {
        "coord": coord,
        "numbers": numbers,
        "cell": cell_value,
        "mol_idx": mol_idx,
        "nbmat": nbmat,
        "shifts": shifts,
        "nbmat_dftd3": nbmat,
        "shifts_dftd3": shifts,
        "cutoff_dftd3": torch.tensor(float(cutoff), device=device),
    }
    data = nbops.set_nb_mode(data)
    data = nbops.calc_masks(data)
    return data


def _data_with_strain(template: dict, scaling: torch.Tensor, coord_real: torch.Tensor, cell0: torch.Tensor) -> dict:
    """Reuse the neighbor list from ``template`` but apply ``scaling`` to coord/cell."""
    coord_s = coord_real @ scaling
    cell_s = cell0 @ scaling
    coord_pad = torch.cat([coord_s, coord_s.new_zeros((1, 3))], dim=0)
    return {**template, "coord": coord_pad, "cell": cell_s}


@pytest.mark.gpu
class TestDFTD3ForwardTerms:
    """Tests for the calculator stress path: DFTD3 forward terms."""

    def test_embedded_strain_grad_matches_explicit_virial(self, pbc_crystal_small, device):
        """Embedded strain autograd uses the same direct virial convention as external terms."""
        coord_real = pbc_crystal_small["coord"].to(device)
        cell0 = pbc_crystal_small["cell"].to(device)
        numbers_real = pbc_crystal_small["numbers"].to(device)
        module = DFTD3(s8=0.3908, a1=0.5660, a2=3.1280, cutoff=8.0).to(device)

        template = _setup_pbc_dftd3_data(coord_real, cell0, numbers_real, device)
        coord_unstrained = template["coord"].detach().clone().requires_grad_(True)
        scaling = torch.eye(3, device=device, requires_grad=True)
        data = {
            **template,
            "coord": coord_unstrained @ scaling,
            "cell": cell0 @ scaling,
        }

        out = module(data, scaling=scaling, coord_unstrained=coord_unstrained, cell_unstrained=cell0)
        grad_coord, grad_scaling = torch.autograd.grad(out["energy"].sum(), (coord_unstrained, scaling))

        terms_data = _data_with_strain(template, torch.eye(3, device=device), coord_real, cell0)
        _, terms = module(terms_data, compute_forces=True, compute_virial=True)
        assert terms is not None and terms.forces is not None and terms.virial is not None
        external_virial = terms.virial
        if external_virial.ndim == 3:
            external_virial = external_virial.sum(dim=0)

        torch.testing.assert_close(grad_coord, -terms.forces, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(grad_scaling, -external_virial.mT, rtol=1e-4, atol=1e-5)

    def test_hessian_path_pbc_matches_finite_difference_and_double_backward(self, pbc_crystal_small, device):
        """PBC Hessian fallback has finite-difference first derivatives and double backward."""
        coord_real = pbc_crystal_small["coord"].to(device=device, dtype=torch.float64)
        cell0 = pbc_crystal_small["cell"].to(device=device, dtype=torch.float64)
        numbers_real = pbc_crystal_small["numbers"].to(device)
        module = DFTD3(s8=0.3908, a1=0.5660, a2=3.1280, cutoff=8.0).to(device=device, dtype=torch.float64)

        template = _setup_pbc_dftd3_data(coord_real, cell0, numbers_real, device)
        coord_unstrained = template["coord"].detach().clone().requires_grad_(True)
        scaling = torch.eye(3, device=device, dtype=torch.float64, requires_grad=True)
        data = {
            **template,
            "coord": coord_unstrained @ scaling,
            "cell": cell0 @ scaling,
        }

        out = module(data, hessian=True)
        grad_coord, grad_scaling = torch.autograd.grad(
            out["energy"].sum(), (coord_unstrained, scaling), create_graph=True
        )

        def energy_at(coord_base: torch.Tensor, scaling_value: torch.Tensor) -> torch.Tensor:
            data_fd = {
                **template,
                "coord": coord_base @ scaling_value,
                "cell": cell0 @ scaling_value,
            }
            with torch.no_grad():
                return module(data_fd, hessian=True)["energy"].sum()

        delta = 1e-3
        atom_idx, comp_idx = 3, 1
        coord_p = coord_unstrained.detach().clone()
        coord_m = coord_unstrained.detach().clone()
        coord_p[atom_idx, comp_idx] += delta
        coord_m[atom_idx, comp_idx] -= delta
        fd_coord_grad = (energy_at(coord_p, scaling.detach()) - energy_at(coord_m, scaling.detach())) / (2 * delta)
        fd_coord_grad = fd_coord_grad.to(grad_coord.dtype)
        torch.testing.assert_close(grad_coord[atom_idx, comp_idx], fd_coord_grad, rtol=5e-3, atol=5e-4)

        fd_scaling_grad = torch.zeros_like(grad_scaling)
        coord_detached = coord_unstrained.detach()
        identity = scaling.detach()
        for i in range(3):
            for j in range(3):
                s_p = identity.clone()
                s_m = identity.clone()
                s_p[i, j] += delta
                s_m[i, j] -= delta
                fd_scaling_grad[i, j] = (energy_at(coord_detached, s_p) - energy_at(coord_detached, s_m)) / (2 * delta)
        torch.testing.assert_close(grad_scaling, fd_scaling_grad, rtol=5e-3, atol=5e-3)

        second = torch.autograd.grad(grad_coord[:-1].pow(2).sum(), coord_unstrained)[0]
        assert torch.isfinite(second).all()
        assert second.abs().sum() > 0

    def test_hessian_path_pbc_hessian_matches_energy_finite_difference(self, device):
        """PBC pure-torch DFTD3 Hessian matches central FD with a fixed neighbor list."""
        module = DFTD3(s8=0.3908, a1=0.5660, a2=3.1280, cutoff=8.0).to(device=device, dtype=torch.float64)

        cell = torch.diag(torch.tensor([10.0, 11.0, 12.0], dtype=torch.float64, device=device))
        coord0 = torch.tensor([[2.0, 2.0, 2.0], [3.65, 2.15, 2.05]], dtype=torch.float64, device=device)
        numbers = torch.tensor([6, 8], device=device)
        template = _setup_pbc_dftd3_data(coord0, cell, numbers, device, cutoff=8.0)

        def energy_fn(coord_flat: torch.Tensor) -> torch.Tensor:
            coord_real = coord_flat.reshape(2, 3)
            coord = torch.cat([coord_real, coord_real.new_zeros((1, 3))], dim=0)
            data = {**template, "coord": coord}
            return module._compute_energy_torch(data).sum()

        coord_flat = coord0.reshape(-1)
        h_auto = torch.autograd.functional.hessian(energy_fn, coord_flat)
        h_fd = _central_fd_hessian(energy_fn, coord_flat)

        torch.testing.assert_close(h_auto, h_fd, rtol=5e-3, atol=5e-3)

    def test_explicit_virial_matches_finite_difference(self, pbc_crystal_small, device):
        """Explicit virial from ``forward(...)`` matches row-vector strain FD.

        The calculator consumes ``terms.virial`` via ``dedc -= terms.virial.mT``,
        so the contract is ``-terms.virial.mT == dE/dscaling``. FD is computed
        against the same ``compute_virial=True`` kernel branch the terms call uses,
        with the neighbor list built once at the unstrained cell so the strain
        perturbation only changes coord/cell.
        """
        coord_real = pbc_crystal_small["coord"].to(device)
        cell0 = pbc_crystal_small["cell"].to(device)
        numbers_real = pbc_crystal_small["numbers"].to(device)
        module = DFTD3(s8=0.3908, a1=0.5660, a2=3.1280, cutoff=8.0).to(device)

        template = _setup_pbc_dftd3_data(coord_real, cell0, numbers_real, device)

        def energy_at(scaling: torch.Tensor) -> float:
            data = _data_with_strain(template, scaling, coord_real, cell0)
            with torch.no_grad():
                # Use the same kernel branch (compute_virial=True) as the terms
                # call below to avoid mixing branches across forward and FD.
                out, _ = module(data, compute_virial=True)
            return float(out["energy"].sum().item())

        identity = torch.eye(3, device=device)
        data0 = _data_with_strain(template, identity, coord_real, cell0)
        _, terms = module(data0, compute_virial=True)
        assert terms is not None and terms.virial is not None
        external_virial = terms.virial.detach().cpu()
        if external_virial.ndim == 3:
            external_virial = external_virial.sum(dim=0)

        delta = 1e-3
        dE_dscaling = torch.zeros(3, 3)
        for i in range(3):
            for j in range(3):
                s_p = identity.clone()
                s_p[i, j] += delta
                s_m = identity.clone()
                s_m[i, j] -= delta
                dE_dscaling[i, j] = (energy_at(s_p) - energy_at(s_m)) / (2 * delta)

        torch.testing.assert_close(-external_virial.mT, dE_dscaling, rtol=5e-3, atol=5e-3)

    def test_explicit_forces_match_forward_forces(self, pbc_crystal_small, device):
        """Repeated forward-term calls return the same explicit forces."""
        coord_real = pbc_crystal_small["coord"].to(device)
        cell0 = pbc_crystal_small["cell"].to(device)
        numbers_real = pbc_crystal_small["numbers"].to(device)
        module = DFTD3(s8=0.3908, a1=0.5660, a2=3.1280, cutoff=8.0).to(device)

        # Share a single neighbor list across paths so kernel inputs are bit-identical.
        template = _setup_pbc_dftd3_data(coord_real, cell0, numbers_real, device)

        data_a = dict(template)
        _, terms_a = module(data_a, compute_forces=True)
        assert terms_a is not None and terms_a.forces is not None
        forces_first = terms_a.forces.detach()

        data_b = dict(template)
        _, terms = module(data_b, compute_forces=True)
        assert terms is not None and terms.forces is not None
        forces_explicit = terms.forces.detach()

        torch.testing.assert_close(forces_explicit, forces_first, rtol=1e-4, atol=1e-5)

    def test_no_virial_no_forces_returns_data(self, pbc_crystal_small, device):
        """When neither derivative flag is requested, forward returns only data."""
        coord_real = pbc_crystal_small["coord"].to(device)
        cell0 = pbc_crystal_small["cell"].to(device)
        numbers_real = pbc_crystal_small["numbers"].to(device)
        module = DFTD3(s8=0.3908, a1=0.5660, a2=3.1280, cutoff=8.0).to(device)

        data = _setup_pbc_dftd3_data(coord_real, cell0, numbers_real, device)
        out = module(data)
        assert "energy" in out

    def test_forces_and_virial_in_one_call(self, pbc_crystal_small, device):
        """Both flags in a single call populate both fields and agree with single-flag calls."""
        coord_real = pbc_crystal_small["coord"].to(device)
        cell0 = pbc_crystal_small["cell"].to(device)
        numbers_real = pbc_crystal_small["numbers"].to(device)
        module = DFTD3(s8=0.3908, a1=0.5660, a2=3.1280, cutoff=8.0).to(device)
        template = _setup_pbc_dftd3_data(coord_real, cell0, numbers_real, device)

        # forces+virial in one call
        _, terms_both = module(dict(template), compute_forces=True, compute_virial=True)
        assert terms_both is not None
        assert terms_both.forces is not None
        assert terms_both.virial is not None

        # forces alone - virial omitted, forces populated
        _, terms_f = module(dict(template), compute_forces=True)
        assert terms_f is not None
        assert terms_f.forces is not None
        assert terms_f.virial is None
        torch.testing.assert_close(terms_both.forces, terms_f.forces, rtol=1e-4, atol=1e-5)

        # virial alone - forces omitted, virial populated and agrees with combined.
        _, terms_v = module(dict(template), compute_virial=True)
        assert terms_v is not None
        assert terms_v.virial is not None
        assert terms_v.forces is None
        torch.testing.assert_close(terms_both.virial, terms_v.virial, rtol=1e-4, atol=1e-5)
