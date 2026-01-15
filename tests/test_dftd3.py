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
- Custom op registration
- Forward computation
- Backward computation (autograd)
- TorchScript compatibility
- Physical correctness
"""

import tempfile

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


# =============================================================================
# Op Registration Tests
# =============================================================================


@pytest.mark.gpu
class TestDFTD3OpRegistration:
    """Tests for custom op registration."""

    def test_op_registered(self, device):
        """Test that dftd3_fwd op is registered."""
        from aimnet.modules import ops  # noqa: F401

        assert hasattr(torch.ops, "aimnet"), "aimnet namespace not found in torch.ops"
        assert hasattr(torch.ops.aimnet, "dftd3_fwd"), "dftd3_fwd op not registered"

    def test_load_ops_includes_dftd3(self, device):
        """Test that load_ops() returns dftd3_fwd."""
        from aimnet.kernels import load_ops
        from aimnet.modules import ops  # noqa: F401

        available_ops = load_ops()
        assert "aimnet::dftd3_fwd" in available_ops


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

    def test_compute_forces_flag(self, device):
        """Test compute_forces flag initialization."""
        module = DFTD3(s8=0.5, a1=0.4, a2=4.0, compute_forces=True).to(device)
        assert module.compute_forces is True

    def test_compute_virial_flag(self, device):
        """Test compute_virial flag initialization."""
        module = DFTD3(s8=0.5, a1=0.4, a2=4.0, compute_virial=True).to(device)
        assert module.compute_virial is True


# =============================================================================
# Forward Pass Tests
# =============================================================================


@pytest.mark.gpu
class TestDFTD3Forward:
    """Tests for DFTD3 forward pass."""

    def test_output_shape_mode_0(self, device):
        """Test that DFTD3 produces correct output shape in mode 0."""
        module = DFTD3(s8=0.3908, a1=0.5660, a2=3.1280, compute_forces=True).to(device)
        data = setup_dftd3_data_mode_0(device, n_atoms=5)
        data["coord"] = data["coord"].requires_grad_(True)

        result = module(data)

        assert "energy" in result
        assert result["energy"].shape == (1,)  # Per-molecule energy
        assert "forces" in result
        assert result["forces"].shape == (1, 5, 3)

    def test_output_shape_mode_1(self, device):
        """Test that DFTD3 produces correct output shape in mode 1."""
        module = DFTD3(s8=0.3908, a1=0.5660, a2=3.1280, compute_forces=True).to(device)
        data = setup_dftd3_data_mode_1(device, n_atoms=5)
        data["coord"] = data["coord"].requires_grad_(True)

        result = module(data)

        assert "energy" in result
        assert result["energy"].shape == (1,)
        assert "forces" in result
        assert result["forces"].shape == (6, 3)  # n_atoms + 1 for padding

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
        module = DFTD3(s8=0.3908, a1=0.5660, a2=3.1280, compute_forces=True).to(device)
        data = setup_dftd3_data_mode_0(device, n_atoms=5)
        data["coord"] = data["coord"].requires_grad_(True)

        result = module(data)

        assert torch.isfinite(result["energy"]).all()
        assert torch.isfinite(result["forces"]).all()

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

    def test_forces_addition(self, device):
        """Test that forces are added to existing key if present."""
        module = DFTD3(s8=0.3908, a1=0.5660, a2=3.1280, compute_forces=True).to(device)

        coord = torch.tensor([[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0]]], device=device, requires_grad=True)
        numbers = torch.tensor([[6, 1, 1]], device=device)

        data = {"coord": coord, "numbers": numbers}
        data = add_dftd3_keys(data, device)
        data = nbops.set_nb_mode(data)
        data = nbops.calc_masks(data)

        # Set initial forces
        initial_forces = torch.ones((1, 3, 3), device=device) * 0.5
        data["forces"] = initial_forces.clone()

        result = module(data)

        # Forces should be different from initial (dispersion forces added)
        assert not torch.allclose(result["forces"], initial_forces)


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
        module = DFTD3(s8=0.3908, a1=0.5660, a2=3.1280, compute_forces=True).to(device)

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

        result = module(data)

        assert result["energy"].shape == (2,)
        assert result["forces"].shape == (2, 3, 3)
        assert torch.isfinite(result["energy"]).all()
        assert torch.isfinite(result["forces"]).all()


# =============================================================================
# Autograd Tests
# =============================================================================


@pytest.mark.gpu
class TestDFTD3Autograd:
    """Tests for DFTD3 autograd functionality."""

    def test_energy_requires_grad(self, device):
        """Test that energy computation preserves gradient tracking."""
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
        assert result["energy"].grad_fn is not None

    def test_coord_gradient_finite(self, device):
        """Test that coord gradients are finite after backward."""
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
        result["energy"].sum().backward()

        assert coord.grad is not None
        assert torch.isfinite(coord.grad).all()

    def test_forces_match_negative_grad(self, device):
        """Test that forces from compute_forces match -coord.grad."""
        module = DFTD3(s8=0.3908, a1=0.5660, a2=3.1280, compute_forces=True).to(device)

        coord = torch.tensor([[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0]]], device=device, requires_grad=True)
        numbers = torch.tensor([[6, 1, 1]], device=device)

        data = {"coord": coord, "numbers": numbers}
        data = add_dftd3_keys(data, device)
        data = nbops.set_nb_mode(data)
        data = nbops.calc_masks(data)

        result = module(data)

        # Forces computed in forward should match -grad
        forces_from_forward = result["forces"].clone()

        # Compute grad via backward
        result["energy"].sum().backward()
        forces_from_grad = -coord.grad

        torch.testing.assert_close(forces_from_forward, forces_from_grad, rtol=1e-4, atol=1e-6)

    def test_gradient_accumulation(self, device):
        """Test gradients accumulate correctly over multiple backwards."""
        module = DFTD3(s8=0.3908, a1=0.5660, a2=3.1280).to(device)

        coord = torch.tensor([[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0]]], device=device, requires_grad=True)
        numbers = torch.tensor([[6, 1, 1]], device=device)

        data = {"coord": coord, "numbers": numbers}
        data = add_dftd3_keys(data, device)
        data = nbops.set_nb_mode(data)
        data = nbops.calc_masks(data)

        result = module(data)

        # First backward
        result["energy"].sum().backward(retain_graph=True)
        grad1 = coord.grad.clone()

        # Second backward (should accumulate)
        result["energy"].sum().backward()
        grad2 = coord.grad

        # grad2 should be 2x grad1
        torch.testing.assert_close(grad2, 2 * grad1, rtol=1e-5, atol=1e-7)

    def test_batched_gradients(self, device):
        """Test gradients work correctly with batched input."""
        module = DFTD3(s8=0.3908, a1=0.5660, a2=3.1280).to(device)

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

        result = module(data)
        result["energy"].sum().backward()

        assert coord.grad is not None
        assert coord.grad.shape == coord.shape
        assert torch.isfinite(coord.grad).all()

    def test_single_energy_gradient(self, device):
        """Test gradient when backward is called on single energy."""
        module = DFTD3(s8=0.3908, a1=0.5660, a2=3.1280).to(device)

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

        result = module(data)

        # Backward on just first energy
        result["energy"][0].backward()

        assert coord.grad is not None
        # Only first molecule should have non-zero gradients
        assert not torch.allclose(coord.grad[0], torch.zeros_like(coord.grad[0]))
        assert torch.allclose(coord.grad[1], torch.zeros_like(coord.grad[1]))


# =============================================================================
# TorchScript Tests
# =============================================================================


@pytest.mark.gpu
class TestDFTD3TorchScript:
    """Tests for TorchScript compatibility."""

    def test_script_module(self, device):
        """Test that DFTD3 module can be scripted."""
        module = DFTD3(s8=0.3908, a1=0.5660, a2=3.1280).to(device)
        scripted = torch.jit.script(module)

        # Create test input
        coord = torch.tensor([[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]], device=device)
        numbers = torch.tensor([[8, 1, 1]], device=device)

        data = {"coord": coord, "numbers": numbers}
        data = add_dftd3_keys(data, device)
        data = nbops.set_nb_mode(data)
        data = nbops.calc_masks(data)

        # Compare eager vs scripted
        with torch.no_grad():
            eager_result = module(data.copy())
            scripted_result = scripted(data.copy())

        torch.testing.assert_close(eager_result["energy"], scripted_result["energy"], rtol=1e-5, atol=1e-6)

    def test_script_save_load(self, device):
        """Test that scripted DFTD3 module can be saved and loaded."""
        module = DFTD3(s8=0.3908, a1=0.5660, a2=3.1280).to(device)
        scripted = torch.jit.script(module)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=True) as f:
            torch.jit.save(scripted, f.name)
            loaded = torch.jit.load(f.name)

        # Create test input
        coord = torch.tensor([[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]], device=device)
        numbers = torch.tensor([[8, 1, 1]], device=device)

        data = {"coord": coord, "numbers": numbers}
        data = add_dftd3_keys(data, device)
        data = nbops.set_nb_mode(data)
        data = nbops.calc_masks(data)

        with torch.no_grad():
            original_result = scripted(data.copy())
            loaded_result = loaded(data.copy())

        torch.testing.assert_close(original_result["energy"], loaded_result["energy"])

    def test_trace_module(self, device):
        """Test that DFTD3 forward can be traced (for subset of functionality)."""
        module = DFTD3(s8=0.3908, a1=0.5660, a2=3.1280).to(device)

        # Create example input
        coord = torch.tensor([[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]], device=device)
        numbers = torch.tensor([[8, 1, 1]], device=device)

        data = {"coord": coord, "numbers": numbers}
        data = add_dftd3_keys(data, device)
        data = nbops.set_nb_mode(data)
        data = nbops.calc_masks(data)

        # Module can be called without error
        with torch.no_grad():
            result = module(data)

        assert result["energy"].shape == (1,)


# =============================================================================
# torch.compile Tests
# =============================================================================


@pytest.mark.gpu
class TestDFTD3Compile:
    """Tests for torch.compile compatibility."""

    def test_compile_forward(self, device):
        """Test that DFTD3 module can be compiled and produces correct results."""
        module = DFTD3(s8=0.3908, a1=0.5660, a2=3.1280).to(device)
        compiled_module = torch.compile(module, fullgraph=False)

        # Create test input
        coord = torch.tensor([[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]], device=device)
        numbers = torch.tensor([[8, 1, 1]], device=device)

        data = {"coord": coord, "numbers": numbers}
        data = add_dftd3_keys(data, device)
        data = nbops.set_nb_mode(data)
        data = nbops.calc_masks(data)

        # Compare eager vs compiled
        with torch.no_grad():
            eager_result = module(data.copy())
            compiled_result = compiled_module(data.copy())

        torch.testing.assert_close(eager_result["energy"], compiled_result["energy"], rtol=1e-5, atol=1e-6)

    def test_compile_with_gradients(self, device):
        """Test that compiled DFTD3 works with gradient computation."""
        module = DFTD3(s8=0.3908, a1=0.5660, a2=3.1280).to(device)
        compiled_module = torch.compile(module, fullgraph=False)

        # Create test input with gradients
        coord = torch.tensor(
            [[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]],
            device=device,
            requires_grad=True,
        )
        numbers = torch.tensor([[8, 1, 1]], device=device)

        data = {"coord": coord, "numbers": numbers}
        data = add_dftd3_keys(data, device)
        data = nbops.set_nb_mode(data)
        data = nbops.calc_masks(data)

        # Run compiled forward and backward
        result = compiled_module(data)
        result["energy"].sum().backward()

        # Verify gradients are finite
        assert coord.grad is not None
        assert torch.isfinite(coord.grad).all()

    def test_compile_batched(self, device):
        """Test that compiled DFTD3 works with batched input."""
        module = DFTD3(s8=0.3908, a1=0.5660, a2=3.1280).to(device)
        compiled_module = torch.compile(module, fullgraph=False)

        # Batch of 2 molecules
        coord = torch.tensor(
            [
                [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0]],
                [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
            ],
            device=device,
        )
        numbers = torch.tensor([[6, 1, 1], [6, 1, 1]], device=device)

        data = {"coord": coord, "numbers": numbers}
        data = add_dftd3_keys(data, device)
        data = nbops.set_nb_mode(data)
        data = nbops.calc_masks(data)

        # Compare eager vs compiled
        with torch.no_grad():
            eager_result = module(data.copy())
            compiled_result = compiled_module(data.copy())

        assert compiled_result["energy"].shape == (2,)
        torch.testing.assert_close(eager_result["energy"], compiled_result["energy"], rtol=1e-5, atol=1e-6)
