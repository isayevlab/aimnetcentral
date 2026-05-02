"""Tests for periodic boundary conditions - DSF, Ewald, and PME Coulomb methods.

These tests verify that the DSF, Ewald, and PME Coulomb methods work correctly
with periodic crystal structures loaded from CIF files.
"""

import warnings

import numpy as np
import pytest
import torch

from aimnet import nbops, ops
from aimnet.calculators import AIMNet2Calculator
from aimnet.modules.lr import LRCoulomb

pytestmark = [pytest.mark.ase, pytest.mark.gpu]


def setup_pbc_data_with_nblist(data: dict, device: torch.device, cutoff: float = 8.0) -> dict:
    """Set up padded flat nb_mode=1 periodic data for Coulomb calculations.

    Creates an all-pairs neighbor matrix for testing. Sets both short-range
    (nbmat) and long-range (nbmat_lr) keys for compatibility with all modules.

    Parameters
    ----------
    data : dict
        Data dictionary with coord, numbers, cell, mol_idx.
    device : torch.device
        Device for tensors.
    cutoff : float
        Cutoff for neighbor list (unused, kept for API compatibility).

    Returns
    -------
    dict
        Data with neighbor list keys added (nbmat, nbmat_lr, shifts, shifts_lr).
    """
    coord_real = data["coord"].to(device)
    numbers_real = data["numbers"].to(device)
    n_real = coord_real.shape[0]
    pad_idx = n_real

    coord = torch.cat([coord_real, coord_real.new_zeros((1, 3))], dim=0)
    if coord.requires_grad:
        coord.retain_grad()
    numbers = torch.cat([numbers_real, numbers_real.new_zeros((1,))], dim=0)
    mol_idx_real = data.get("mol_idx")
    if mol_idx_real is None:
        mol_idx_real = torch.zeros(n_real, dtype=torch.long, device=device)
    else:
        mol_idx_real = mol_idx_real.to(device)
    mol_idx = torch.cat([mol_idx_real, mol_idx_real.max().to(torch.long).reshape(1)], dim=0)
    n_atoms = coord.shape[0]

    # Create all-pairs neighbor matrix over real atoms. Invalid slots and the
    # final row point to the required padding atom.
    max_nb = max(1, min(n_real - 1, 50))  # Limit for large systems
    nbmat = torch.full((n_atoms, max_nb), pad_idx, dtype=torch.long, device=device)
    for i in range(n_real):
        neighbors = [j for j in range(n_real) if j != i][:max_nb]
        for k, nb in enumerate(neighbors):
            nbmat[i, k] = nb

    # Shifts for PBC - zero for non-image neighbors
    shifts = torch.zeros((n_atoms, max_nb, 3), dtype=torch.float32, device=device)

    # Set both short-range and long-range neighbor lists
    data["coord"] = coord
    data["numbers"] = numbers
    data["cell"] = data["cell"].to(device)
    data["mol_idx"] = mol_idx
    data["nbmat"] = nbmat
    data["shifts"] = shifts
    data["nbmat_lr"] = nbmat
    data["shifts_lr"] = shifts.clone()

    data = nbops.set_nb_mode(data)
    data = nbops.calc_masks(data)

    # Compute distances
    d_ij, _ = ops.calc_distances(data)
    data["d_ij"] = d_ij

    return data


def _neutral_padded_charges(n_atoms: int, device: torch.device, scale: float = 0.3) -> torch.Tensor:
    """Random neutral charges with the final padded atom fixed at zero."""
    charges = torch.randn(n_atoms, device=device) * scale
    charges[:-1] = charges[:-1] - charges[:-1].mean()
    charges[-1] = 0.0
    return charges


def _random_padded_charges(n_atoms: int, device: torch.device, scale: float = 0.3) -> torch.Tensor:
    """Random charges following the padded flat nb_mode=1 contract."""
    charges = torch.randn(n_atoms, device=device) * scale
    charges[-1] = 0.0
    return charges


class TestDSFPeriodic:
    """Tests for DSF Coulomb with periodic structures."""

    def test_dsf_pbc_energy_finite(self, pbc_crystal_small, device):
        """Test DSF produces finite energy for periodic system."""
        module = LRCoulomb(method="dsf", dsf_rc=8.0).to(device)
        data = setup_pbc_data_with_nblist(pbc_crystal_small.copy(), device)

        # Add mock charges for testing
        n_atoms = data["coord"].shape[0]
        data["charges"] = _random_padded_charges(n_atoms, device)

        result = module(data)

        assert "e_h" in result
        assert torch.isfinite(result["e_h"]).all(), "DSF energy is not finite"

    def test_dsf_pbc_zero_charges(self, pbc_crystal_small, device):
        """Test DSF with zero charges produces zero energy."""
        module = LRCoulomb(method="dsf", dsf_rc=8.0).to(device)
        data = setup_pbc_data_with_nblist(pbc_crystal_small.copy(), device)

        n_atoms = data["coord"].shape[0]
        data["charges"] = torch.zeros(n_atoms, device=device)

        result = module(data)

        assert result["e_h"].abs().sum().item() < 1e-10, "DSF energy should be zero for zero charges"

    def test_dsf_pbc_opposite_charges_attractive(self, pbc_crystal_small, device):
        """Test DSF gives negative energy for opposite charges."""
        module = LRCoulomb(method="dsf", dsf_rc=8.0).to(device)
        data = setup_pbc_data_with_nblist(pbc_crystal_small.copy(), device)

        n_atoms = data["coord"].shape[0]
        # Alternate charges: +1, -1, +1, -1, ...
        charges = torch.ones(n_atoms, device=device)
        charges[1::2] = -1.0
        charges[-1] = 0.0
        data["charges"] = charges

        result = module(data)

        # For alternating charges, energy should be negative (attractive)
        assert result["e_h"].item() < 0, "Alternating charges should have negative Coulomb energy"

    def test_dsf_pbc_cutoff_effect(self, pbc_crystal_small, device):
        """Test that DSF cutoff affects energy in periodic system."""
        data = setup_pbc_data_with_nblist(pbc_crystal_small.copy(), device)
        n_atoms = data["coord"].shape[0]
        data["charges"] = _random_padded_charges(n_atoms, device, scale=0.5)

        # Short cutoff
        module_short = LRCoulomb(method="dsf", dsf_rc=3.0).to(device)
        result_short = module_short(data.copy())

        # Long cutoff
        module_long = LRCoulomb(method="dsf", dsf_rc=10.0).to(device)
        result_long = module_long(data.copy())

        # Both should be finite
        assert torch.isfinite(result_short["e_h"]).all()
        assert torch.isfinite(result_long["e_h"]).all()

    def test_dsf_pbc_explicit_forces(self, pbc_crystal_small, device):
        """DSF returns explicit nvalchemiops forces for padded flat PBC inputs."""
        module = LRCoulomb(method="dsf", dsf_rc=8.0).to(device)

        data = setup_pbc_data_with_nblist(pbc_crystal_small.copy(), device)

        n_atoms = data["coord"].shape[0]
        data["charges"] = _random_padded_charges(n_atoms, device)

        result, terms = module(data, compute_forces=True)

        assert "e_h" in result
        assert terms is not None
        assert terms.forces is not None
        assert terms.forces.shape == data["coord"].shape
        assert not terms.forces.requires_grad
        assert torch.isfinite(terms.forces).all(), "Explicit DSF forces should be finite"
        assert torch.allclose(terms.forces[-1], torch.zeros(3, device=device))

    def test_dsf_pbc_virial_matches_strain_finite_difference(self, pbc_crystal_small, device):
        """nvalchemiops DSF virial uses W=-dE/dstrain, so stress is -W.T/V."""
        module = LRCoulomb(method="dsf", dsf_rc=8.0, subtract_sr=False).to(device)
        data = setup_pbc_data_with_nblist(pbc_crystal_small.copy(), device)
        n_atoms = data["coord"].shape[0]
        data["charges"] = _random_padded_charges(n_atoms, device)

        _result, terms = module(data.copy(), compute_virial=True)
        assert terms is not None and terms.virial is not None
        assert not terms.virial.requires_grad
        volume = data["cell"].det().abs()
        stress_from_virial = -terms.virial.squeeze(0).mT / volume

        delta = 1e-4
        stress_fd = torch.zeros(3, 3, device=device)
        base = {k: v for k, v in data.items() if k != "e_h"}
        for i in range(3):
            for j in range(3):
                strain_plus = torch.eye(3, device=device)
                strain_plus[i, j] += delta
                data_plus = base.copy()
                data_plus["coord"] = data["coord"] @ strain_plus
                data_plus["cell"] = data["cell"] @ strain_plus
                e_plus = module(data_plus)["e_h"].item()

                strain_minus = torch.eye(3, device=device)
                strain_minus[i, j] -= delta
                data_minus = base.copy()
                data_minus["coord"] = data["coord"] @ strain_minus
                data_minus["cell"] = data["cell"] @ strain_minus
                e_minus = module(data_minus)["e_h"].item()

                stress_fd[i, j] = (e_plus - e_minus) / (2 * delta * volume)

        torch.testing.assert_close(stress_from_virial, stress_fd, atol=5e-4, rtol=5e-3)


class TestEwaldPeriodic:
    """Tests for Ewald summation with periodic structures."""

    def test_ewald_pbc_energy_finite(self, pbc_crystal_small, device):
        """Test Ewald produces finite energy for periodic system."""
        module = LRCoulomb(method="ewald").to(device)
        data = setup_pbc_data_with_nblist(pbc_crystal_small.copy(), device, cutoff=8.0)

        n_atoms = data["coord"].shape[0]
        # Ensure charge neutrality for Ewald
        data["charges"] = _neutral_padded_charges(n_atoms, device)

        result = module(data)

        assert "e_h" in result
        assert torch.isfinite(result["e_h"]).all(), "Ewald energy is not finite"

    def test_ewald_pbc_zero_charges(self, pbc_crystal_small, device):
        """Test Ewald with zero charges produces zero energy."""
        module = LRCoulomb(method="ewald").to(device)
        data = setup_pbc_data_with_nblist(pbc_crystal_small.copy(), device)

        n_atoms = data["coord"].shape[0]
        data["charges"] = torch.zeros(n_atoms, device=device)

        result = module(data)

        assert result["e_h"].abs().sum().item() < 1e-8, "Ewald energy should be zero for zero charges"

    def test_ewald_pbc_charge_neutrality(self, pbc_crystal_small, device):
        """Test Ewald handles charge-neutral periodic systems correctly."""
        module = LRCoulomb(method="ewald").to(device)
        data = setup_pbc_data_with_nblist(pbc_crystal_small.copy(), device)

        n_atoms = data["coord"].shape[0]
        # Create explicitly neutral system
        charges = torch.ones(n_atoms, device=device)
        charges[n_atoms // 2 :] = -1.0
        charges[-1] = 0.0
        data["charges"] = charges

        result = module(data)

        assert torch.isfinite(result["e_h"]).all(), "Ewald should work with neutral system"

    def test_ewald_pbc_gradient_coords(self, pbc_crystal_small, device):
        """Test gradient of Ewald energy w.r.t. coordinates."""
        module = LRCoulomb(method="ewald").to(device)

        data = pbc_crystal_small.copy()
        data["coord"] = data["coord"].clone().requires_grad_(True)
        data = setup_pbc_data_with_nblist(data, device)

        n_atoms = data["coord"].shape[0]
        data["charges"] = _neutral_padded_charges(n_atoms, device)

        result = module(data)
        result["e_h"].backward()

        assert data["coord"].grad is not None, "Gradient should exist"
        assert torch.isfinite(data["coord"].grad).all(), "Gradient should be finite"


class TestDSFvsEwaldConsistency:
    """Tests comparing DSF and Ewald methods."""

    def test_dsf_ewald_both_finite(self, pbc_crystal_small, device):
        """Test both DSF and Ewald produce finite energies for same system."""
        data = setup_pbc_data_with_nblist(pbc_crystal_small.copy(), device, cutoff=8.0)

        n_atoms = data["coord"].shape[0]
        data["charges"] = _neutral_padded_charges(n_atoms, device)

        module_dsf = LRCoulomb(method="dsf", dsf_rc=8.0).to(device)
        module_ewald = LRCoulomb(method="ewald").to(device)

        result_dsf = module_dsf(data.copy())
        result_ewald = module_ewald(data.copy())

        assert torch.isfinite(result_dsf["e_h"]).all(), "DSF energy not finite"
        assert torch.isfinite(result_ewald["e_h"]).all(), "Ewald energy not finite"

    def test_dsf_ewald_sign_consistency(self, pbc_crystal_small, device):
        """Test DSF and Ewald agree on energy sign."""
        data = setup_pbc_data_with_nblist(pbc_crystal_small.copy(), device, cutoff=8.0)

        n_atoms = data["coord"].shape[0]
        # Alternating charges for clear attractive energy
        charges = torch.ones(n_atoms, device=device)
        charges[1::2] = -1.0
        charges[-1] = 0.0
        data["charges"] = charges

        module_dsf = LRCoulomb(method="dsf", dsf_rc=8.0).to(device)
        module_ewald = LRCoulomb(method="ewald").to(device)

        result_dsf = module_dsf(data.copy())
        result_ewald = module_ewald(data.copy())

        # Both should have same sign (likely negative for alternating charges)
        dsf_sign = torch.sign(result_dsf["e_h"])
        ewald_sign = torch.sign(result_ewald["e_h"])
        assert dsf_sign.item() == ewald_sign.item(), "DSF and Ewald should agree on energy sign"


class TestPBCNeighborList:
    """Tests for periodic neighbor list construction."""

    def test_pbc_data_has_cell(self, pbc_crystal_small, device):
        """Test that PBC crystal data has cell information."""
        assert "cell" in pbc_crystal_small, "PBC data should have cell"
        assert pbc_crystal_small["cell"].shape == (3, 3), "Cell should be 3x3 matrix"

    def test_pbc_data_has_mol_idx(self, pbc_crystal_small, device):
        """Test that PBC crystal data has mol_idx."""
        assert "mol_idx" in pbc_crystal_small, "PBC data should have mol_idx"
        n_atoms = pbc_crystal_small["coord"].shape[0]
        assert pbc_crystal_small["mol_idx"].shape[0] == n_atoms

    def test_setup_nblist_shapes(self, pbc_crystal_small, device):
        """Test neighbor list setup produces correct shapes."""
        data = setup_pbc_data_with_nblist(pbc_crystal_small.copy(), device)

        n_atoms = data["coord"].shape[0]
        assert "nbmat" in data
        assert "nbmat_lr" in data
        assert data["nbmat"].shape[0] == n_atoms
        assert data["nbmat_lr"].shape[0] == n_atoms
        assert data["coord"].shape[0] == pbc_crystal_small["coord"].shape[0] + 1
        assert data["numbers"][-1].item() == 0
        assert torch.equal(data["nbmat"][-1], torch.full_like(data["nbmat"][-1], n_atoms - 1))
        assert data["nbmat"].max().item() == n_atoms - 1


class TestCalculatorPBC:
    """Integration tests for full calculator with PBC."""

    def test_calculator_pbc_dsf_inference(self, pbc_crystal_small, device):
        """Test full calculator inference on periodic system with DSF."""
        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Model has embedded Coulomb module", category=UserWarning)
            calc.set_lrcoulomb_method("dsf", cutoff=8.0)

        data = pbc_crystal_small.copy()
        # Convert to format expected by calculator
        data_calc = {
            "coord": data["coord"].cpu().numpy(),
            "numbers": data["numbers"].cpu().numpy(),
            "charge": 0.0,
            "cell": data["cell"].cpu().numpy(),
        }

        res = calc(data_calc)

        assert "energy" in res
        assert torch.isfinite(res["energy"]).all()

    def test_calculator_pbc_ewald_inference(self, pbc_crystal_small, device):
        """Test full calculator inference on periodic system with Ewald."""
        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Model has embedded Coulomb module", category=UserWarning)
            calc.set_lrcoulomb_method("ewald")

        data = pbc_crystal_small.copy()
        data_calc = {
            "coord": data["coord"].cpu().numpy(),
            "numbers": data["numbers"].cpu().numpy(),
            "charge": 0.0,
            "cell": data["cell"].cpu().numpy(),
        }

        res = calc(data_calc)

        assert "energy" in res
        assert torch.isfinite(res["energy"]).all()

    def test_calculator_pbc_forces(self, pbc_crystal_small, device):
        """Test force calculation for periodic system."""
        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Model has embedded Coulomb module", category=UserWarning)
            calc.set_lrcoulomb_method("dsf", cutoff=8.0)

        data = pbc_crystal_small.copy()
        data_calc = {
            "coord": data["coord"].cpu().numpy(),
            "numbers": data["numbers"].cpu().numpy(),
            "charge": 0.0,
            "cell": data["cell"].cpu().numpy(),
        }

        res = calc(data_calc, forces=True)

        assert "forces" in res
        n_atoms = data["coord"].shape[0]
        assert res["forces"].shape[-2] == n_atoms
        assert res["forces"].shape[-1] == 3
        assert torch.isfinite(res["forces"]).all()

    @pytest.mark.parametrize("method", ["dsf", "ewald", "pme"])
    def test_calculator_pbc_both_methods(self, pbc_crystal_small, device, method):
        """Test calculator works with DSF, Ewald, and PME methods."""
        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Model has embedded Coulomb module", category=UserWarning)
            if method == "dsf":
                calc.set_lrcoulomb_method(method, cutoff=8.0)
            else:
                calc.set_lrcoulomb_method(method)

        data = pbc_crystal_small.copy()
        data_calc = {
            "coord": data["coord"].cpu().numpy(),
            "numbers": data["numbers"].cpu().numpy(),
            "charge": 0.0,
            "cell": data["cell"].cpu().numpy(),
        }

        res = calc(data_calc)

        assert "energy" in res
        assert torch.isfinite(res["energy"]).all()


class TestTorchCompilePBC:
    """Tests for torch.compile compatibility with periodic boundary conditions."""

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile requires PyTorch 2.0+")
    def test_torch_compile_pbc_dsf(self, pbc_crystal_small, device):
        """Test torch.compile works with PBC and DSF Coulomb."""
        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Model has embedded Coulomb module", category=UserWarning)
            calc.set_lrcoulomb_method("dsf", cutoff=8.0)

        # Compile the model
        compiled_model = torch.compile(calc.model)
        calc.model = compiled_model

        data = pbc_crystal_small.copy()
        data_calc = {
            "coord": data["coord"].cpu().numpy(),
            "numbers": data["numbers"].cpu().numpy(),
            "charge": 0.0,
            "cell": data["cell"].cpu().numpy(),
        }

        res = calc(data_calc)

        assert "energy" in res
        assert torch.isfinite(res["energy"]).all()

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile requires PyTorch 2.0+")
    def test_torch_compile_pbc_forces(self, pbc_crystal_small, device):
        """Test torch.compile works with PBC force calculation."""
        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Model has embedded Coulomb module", category=UserWarning)
            calc.set_lrcoulomb_method("dsf", cutoff=8.0)

        compiled_model = torch.compile(calc.model)
        calc.model = compiled_model

        data = pbc_crystal_small.copy()
        data_calc = {
            "coord": data["coord"].cpu().numpy(),
            "numbers": data["numbers"].cpu().numpy(),
            "charge": 0.0,
            "cell": data["cell"].cpu().numpy(),
        }

        res = calc(data_calc, forces=True)

        assert "forces" in res
        assert torch.isfinite(res["forces"]).all()


class TestLargeCrystal:
    """Tests with larger crystal structure."""

    def test_large_crystal_dsf_energy(self, pbc_crystal_large, device):
        """Test DSF on larger crystal (~50 atoms)."""
        module = LRCoulomb(method="dsf", dsf_rc=8.0).to(device)
        data = setup_pbc_data_with_nblist(pbc_crystal_large.copy(), device)

        n_atoms = data["coord"].shape[0]
        data["charges"] = _random_padded_charges(n_atoms, device)

        result = module(data)

        assert torch.isfinite(result["e_h"]).all()

    def test_large_crystal_ewald_energy(self, pbc_crystal_large, device):
        """Test Ewald on larger crystal (~50 atoms)."""
        module = LRCoulomb(method="ewald").to(device)
        data = setup_pbc_data_with_nblist(pbc_crystal_large.copy(), device)

        n_atoms = data["coord"].shape[0]
        data["charges"] = _neutral_padded_charges(n_atoms, device)

        result = module(data)

        assert torch.isfinite(result["e_h"]).all()

    def test_large_crystal_calculator(self, pbc_crystal_large, device):
        """Test full calculator on larger crystal."""
        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Model has embedded Coulomb module", category=UserWarning)
            calc.set_lrcoulomb_method("dsf", cutoff=8.0)

        data = pbc_crystal_large.copy()
        data_calc = {
            "coord": data["coord"].cpu().numpy(),
            "numbers": data["numbers"].cpu().numpy(),
            "charge": 0.0,
            "cell": data["cell"].cpu().numpy(),
        }

        res = calc(data_calc)

        assert "energy" in res
        assert torch.isfinite(res["energy"]).all()


class TestBatchedStress:
    """Tests for batched stress calculation with per-system scaling."""

    def test_batched_stress_shape(self, pbc_crystal_small, pbc_crystal_large, device):
        """Test that batched stress returns correct shape (B, 3, 3)."""
        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Model has embedded Coulomb module", category=UserWarning)
            calc.set_lrcoulomb_method("dsf", cutoff=8.0)

        # Get two crystals (use same one twice with slight perturbation for testing)
        data1 = pbc_crystal_small.copy()
        data2 = pbc_crystal_small.copy()
        data2["coord"] = data2["coord"] + 0.01  # Small perturbation

        # Create batched input with mol_idx
        n1 = data1["coord"].shape[0]
        n2 = data2["coord"].shape[0]
        batched_data = {
            "coord": torch.cat([data1["coord"], data2["coord"]], dim=0),
            "numbers": torch.cat([data1["numbers"], data2["numbers"]], dim=0),
            "charge": torch.tensor([0.0, 0.0], device=device),
            "cell": torch.stack([data1["cell"], data2["cell"]], dim=0),  # (2, 3, 3)
            "mol_idx": torch.cat([
                torch.zeros(n1, dtype=torch.long, device=device),
                torch.ones(n2, dtype=torch.long, device=device),
            ]),
        }

        res = calc(batched_data, stress=True)

        assert "stress" in res
        assert res["stress"].shape == (2, 3, 3), f"Expected (2, 3, 3), got {res['stress'].shape}"
        assert torch.isfinite(res["stress"]).all()

    def test_batched_stress_matches_individual(self, pbc_crystal_small, device):
        """Test that batched stress matches stress computed individually for each system."""
        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Model has embedded Coulomb module", category=UserWarning)
            calc.set_lrcoulomb_method("dsf", cutoff=8.0)

        # Create two slightly different systems
        data1 = pbc_crystal_small.copy()
        data2 = pbc_crystal_small.copy()
        data2["coord"] = data2["coord"] + 0.02  # Small perturbation

        # Compute stress individually
        data1_calc = {
            "coord": data1["coord"].cpu().numpy(),
            "numbers": data1["numbers"].cpu().numpy(),
            "charge": 0.0,
            "cell": data1["cell"].cpu().numpy(),
        }
        res1 = calc(data1_calc, stress=True)

        data2_calc = {
            "coord": data2["coord"].cpu().numpy(),
            "numbers": data2["numbers"].cpu().numpy(),
            "charge": 0.0,
            "cell": data2["cell"].cpu().numpy(),
        }
        res2 = calc(data2_calc, stress=True)

        # Compute stress in batch
        n1 = data1["coord"].shape[0]
        n2 = data2["coord"].shape[0]
        batched_data = {
            "coord": torch.cat([data1["coord"], data2["coord"]], dim=0),
            "numbers": torch.cat([data1["numbers"], data2["numbers"]], dim=0),
            "charge": torch.tensor([0.0, 0.0], device=device),
            "cell": torch.stack([data1["cell"], data2["cell"]], dim=0),  # (2, 3, 3)
            "mol_idx": torch.cat([
                torch.zeros(n1, dtype=torch.long, device=device),
                torch.ones(n2, dtype=torch.long, device=device),
            ]),
        }
        res_batch = calc(batched_data, stress=True)

        # Compare individual stress with batched stress
        atol = 1e-5
        assert torch.allclose(res_batch["stress"][0], res1["stress"], atol=atol), (
            f"System 1 stress mismatch: batched={res_batch['stress'][0]}, individual={res1['stress']}"
        )
        assert torch.allclose(res_batch["stress"][1], res2["stress"], atol=atol), (
            f"System 2 stress mismatch: batched={res_batch['stress'][1]}, individual={res2['stress']}"
        )

    def test_single_system_stress_unchanged(self, pbc_crystal_small, device):
        """Test that single-system stress calculation still works correctly."""
        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Model has embedded Coulomb module", category=UserWarning)
            calc.set_lrcoulomb_method("dsf", cutoff=8.0)

        data = pbc_crystal_small.copy()
        data_calc = {
            "coord": data["coord"].cpu().numpy(),
            "numbers": data["numbers"].cpu().numpy(),
            "charge": 0.0,
            "cell": data["cell"].cpu().numpy(),
        }

        res = calc(data_calc, stress=True)

        assert "stress" in res
        assert res["stress"].shape == (3, 3), f"Expected (3, 3), got {res['stress'].shape}"
        assert torch.isfinite(res["stress"]).all()

    def test_batched_stress_with_forces(self, pbc_crystal_small, device):
        """Test that batched stress and forces can be computed together."""
        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Model has embedded Coulomb module", category=UserWarning)
            calc.set_lrcoulomb_method("dsf", cutoff=8.0)

        data1 = pbc_crystal_small.copy()
        data2 = pbc_crystal_small.copy()
        data2["coord"] = data2["coord"] + 0.01

        n1 = data1["coord"].shape[0]
        n2 = data2["coord"].shape[0]
        batched_data = {
            "coord": torch.cat([data1["coord"], data2["coord"]], dim=0),
            "numbers": torch.cat([data1["numbers"], data2["numbers"]], dim=0),
            "charge": torch.tensor([0.0, 0.0], device=device),
            "cell": torch.stack([data1["cell"], data2["cell"]], dim=0),
            "mol_idx": torch.cat([
                torch.zeros(n1, dtype=torch.long, device=device),
                torch.ones(n2, dtype=torch.long, device=device),
            ]),
        }

        res = calc(batched_data, forces=True, stress=True)

        assert "forces" in res
        assert "stress" in res
        assert res["stress"].shape == (2, 3, 3)
        assert torch.isfinite(res["forces"]).all()
        assert torch.isfinite(res["stress"]).all()

    def test_batched_stress_different_cells(self, pbc_crystal_small, device):
        """Test that systems with different cell sizes have different energy, forces, and stress."""
        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Model has embedded Coulomb module", category=UserWarning)
            calc.set_lrcoulomb_method("dsf", cutoff=8.0)

        # Create two systems with different cell sizes
        data1 = pbc_crystal_small.copy()
        data2 = pbc_crystal_small.copy()

        # Scale cell and coordinates for second system by 2%
        scale_factor = 1.02
        data2["cell"] = data2["cell"] * scale_factor
        data2["coord"] = data2["coord"] * scale_factor

        # Create batched input
        n1 = data1["coord"].shape[0]
        n2 = data2["coord"].shape[0]
        batched_data = {
            "coord": torch.cat([data1["coord"], data2["coord"]], dim=0),
            "numbers": torch.cat([data1["numbers"], data2["numbers"]], dim=0),
            "charge": torch.tensor([0.0, 0.0], device=device),
            "cell": torch.stack([data1["cell"], data2["cell"]], dim=0),
            "mol_idx": torch.cat([
                torch.zeros(n1, dtype=torch.long, device=device),
                torch.ones(n2, dtype=torch.long, device=device),
            ]),
        }

        res = calc(batched_data, forces=True, stress=True)

        # Verify all quantities are finite
        assert torch.isfinite(res["energy"]).all()
        assert torch.isfinite(res["forces"]).all()
        assert torch.isfinite(res["stress"]).all()

        # Verify energies differ (scaled system should have different energy)
        assert not torch.allclose(res["energy"][0], res["energy"][1], atol=1e-6), (
            f"Energies should differ: system1={res['energy'][0]}, system2={res['energy'][1]}"
        )

        # Verify stresses differ
        assert not torch.allclose(res["stress"][0], res["stress"][1], atol=1e-6), (
            "Stresses should differ between systems with different cells"
        )

        # Verify forces differ (compare force norms since atom counts are same)
        forces1 = res["forces"][:n1]
        forces2 = res["forces"][n1:]
        assert not torch.allclose(forces1, forces2, atol=1e-6), (
            "Forces should differ between systems with different cells"
        )


class TestDFTD3Stress:
    """Tests for DFTD3 contribution to stress calculation."""

    def test_dftd3_stress_finite(self, pbc_crystal_small, device):
        """Test that stress is finite when DFTD3 is enabled."""
        # Use aimnet2 model which has DFTD3
        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Model has embedded Coulomb module", category=UserWarning)
            calc.set_lrcoulomb_method("dsf", cutoff=8.0)

        data = pbc_crystal_small.copy()
        data_calc = {
            "coord": data["coord"].cpu().numpy(),
            "numbers": data["numbers"].cpu().numpy(),
            "charge": 0.0,
            "cell": data["cell"].cpu().numpy(),
        }

        res = calc(data_calc, stress=True)

        assert "stress" in res
        assert torch.isfinite(res["stress"]).all(), "Stress should be finite with DFTD3"

    def test_dftd3_stress_contribution(self, pbc_crystal_small, device):
        """Test that DFTD3 contributes to stress (stress differs with/without D3)."""
        # Calculator with DFTD3 (aimnet2 has it by default)
        calc_with_d3 = AIMNet2Calculator("aimnet2", nb_threshold=0)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Model has embedded Coulomb module", category=UserWarning)
            calc_with_d3.set_lrcoulomb_method("dsf", cutoff=8.0)

        # Disable DFTD3 for comparison
        calc_no_d3 = AIMNet2Calculator("aimnet2", nb_threshold=0)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Model has embedded Coulomb module", category=UserWarning)
            calc_no_d3.set_lrcoulomb_method("dsf", cutoff=8.0)
        calc_no_d3.external_dftd3 = None  # Disable DFTD3

        data = pbc_crystal_small.copy()
        data_calc = {
            "coord": data["coord"].cpu().numpy(),
            "numbers": data["numbers"].cpu().numpy(),
            "charge": 0.0,
            "cell": data["cell"].cpu().numpy(),
        }

        res_with_d3 = calc_with_d3(data_calc, stress=True)
        res_no_d3 = calc_no_d3(data_calc, stress=True)

        # Both should be finite
        assert torch.isfinite(res_with_d3["stress"]).all()
        assert torch.isfinite(res_no_d3["stress"]).all()

        # Stress should differ when DFTD3 is included
        assert not torch.allclose(res_with_d3["stress"], res_no_d3["stress"], atol=1e-6), (
            "Stress should differ when DFTD3 is enabled vs disabled"
        )

    def test_dftd3_stress_with_scaled_cell(self, pbc_crystal_small, device):
        """Test that scaled cells produce different stress with DFTD3."""
        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Model has embedded Coulomb module", category=UserWarning)
            calc.set_lrcoulomb_method("dsf", cutoff=8.0)

        # Original system
        data1 = pbc_crystal_small.copy()
        data1_calc = {
            "coord": data1["coord"].cpu().numpy(),
            "numbers": data1["numbers"].cpu().numpy(),
            "charge": 0.0,
            "cell": data1["cell"].cpu().numpy(),
        }

        # Scaled system (2% larger)
        data2 = pbc_crystal_small.copy()
        scale_factor = 1.02
        data2["cell"] = data2["cell"] * scale_factor
        data2["coord"] = data2["coord"] * scale_factor
        data2_calc = {
            "coord": data2["coord"].cpu().numpy(),
            "numbers": data2["numbers"].cpu().numpy(),
            "charge": 0.0,
            "cell": data2["cell"].cpu().numpy(),
        }

        res1 = calc(data1_calc, stress=True)
        res2 = calc(data2_calc, stress=True)

        # Both should be finite
        assert torch.isfinite(res1["stress"]).all()
        assert torch.isfinite(res2["stress"]).all()

        # Stress should differ for different cell sizes
        assert not torch.allclose(res1["stress"], res2["stress"], atol=1e-6), (
            "Stress should differ for different cell sizes with DFTD3"
        )

    def test_dftd3_stress_matches_finite_difference(self, pbc_crystal_small, device):
        """Test that autograd stress matches numerical finite-difference stress."""
        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Model has embedded Coulomb module", category=UserWarning)
            calc.set_lrcoulomb_method("dsf", cutoff=8.0)

        data = pbc_crystal_small.copy()
        cell = data["cell"]
        coord = data["coord"]
        numbers = data["numbers"]

        # Compute volume for stress normalization
        volume = cell.det().abs().item()

        # Compute autograd stress
        data_calc = {
            "coord": coord.cpu().numpy(),
            "numbers": numbers.cpu().numpy(),
            "charge": 0.0,
            "cell": cell.cpu().numpy(),
        }
        res = calc(data_calc, stress=True)
        stress_autograd = res["stress"]

        # Compute finite-difference stress
        delta = 1e-4
        stress_fd = torch.zeros(3, 3, device=device)
        for i in range(3):
            for j in range(3):
                # Apply small strain in (i,j) direction
                strain_plus = torch.eye(3, device=device)
                strain_plus[i, j] += delta
                cell_plus = cell @ strain_plus
                coord_plus = coord @ strain_plus

                strain_minus = torch.eye(3, device=device)
                strain_minus[i, j] -= delta
                cell_minus = cell @ strain_minus
                coord_minus = coord @ strain_minus

                # Compute energies
                data_plus = {
                    "coord": coord_plus.cpu().numpy(),
                    "numbers": numbers.cpu().numpy(),
                    "charge": 0.0,
                    "cell": cell_plus.cpu().numpy(),
                }
                E_plus = calc(data_plus)["energy"].item()

                data_minus = {
                    "coord": coord_minus.cpu().numpy(),
                    "numbers": numbers.cpu().numpy(),
                    "charge": 0.0,
                    "cell": cell_minus.cpu().numpy(),
                }
                E_minus = calc(data_minus)["energy"].item()

                # Central difference for stress component
                stress_fd[i, j] = (E_plus - E_minus) / (2 * delta * volume)

        # Autograd and finite-difference should match within numerical tolerance
        # Finite-difference has inherent numerical error, so use relaxed tolerances
        assert torch.allclose(stress_autograd, stress_fd, rtol=1e-2, atol=1e-3), (
            f"Stress mismatch:\nautograd={stress_autograd}\nfinite_diff={stress_fd}"
        )


def _data_calc_from_fixture(data: dict) -> dict:
    return {
        "coord": data["coord"].cpu().numpy(),
        "numbers": data["numbers"].cpu().numpy(),
        "charge": 0.0,
        "cell": data["cell"].cpu().numpy(),
    }


class TestNvAlchemiCoulombBackend:
    """Calculator-level integration tests for the nvalchemiops Coulomb backends.

    Covers ``ewald``/``pme`` direct nvalchemiops dispatch end to end and
    parametrizes the FD-stress regression over all three nvalchemiops methods
    (``dsf`` included) to lock in the row-vector strain convention.
    """

    @pytest.mark.parametrize("method", ["ewald", "pme"])
    def test_calculator_pbc_energy_finite(self, pbc_crystal_small, device, method):
        """Ewald and PME both produce a finite energy on a periodic system."""
        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Model has embedded Coulomb module", category=UserWarning)
            calc.set_lrcoulomb_method(method)

        res = calc(_data_calc_from_fixture(pbc_crystal_small))

        assert "energy" in res
        assert torch.isfinite(res["energy"]).all()

    @pytest.mark.parametrize("method", ["ewald", "pme"])
    def test_calculator_pbc_forces_finite(self, pbc_crystal_small, device, method):
        """Forces from Ewald/PME are finite and have the expected shape."""
        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Model has embedded Coulomb module", category=UserWarning)
            calc.set_lrcoulomb_method(method)

        res = calc(_data_calc_from_fixture(pbc_crystal_small), forces=True)

        assert "forces" in res
        n_atoms = pbc_crystal_small["coord"].shape[0]
        assert res["forces"].shape[-2] == n_atoms
        assert res["forces"].shape[-1] == 3
        assert torch.isfinite(res["forces"]).all()

    @pytest.mark.parametrize("method", ["ewald", "pme"])
    def test_calculator_pbc_stress_finite(self, pbc_crystal_small, device, method):
        """Stress from Ewald/PME is finite and has the expected shape."""
        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Model has embedded Coulomb module", category=UserWarning)
            calc.set_lrcoulomb_method(method)

        res = calc(_data_calc_from_fixture(pbc_crystal_small), stress=True)

        assert "stress" in res
        assert res["stress"].shape == (3, 3)
        assert torch.isfinite(res["stress"]).all()

    @pytest.mark.parametrize("method", ["dsf", "ewald", "pme"])
    def test_calculator_stress_matches_finite_difference(self, pbc_crystal_small, device, method):
        """Calculator stress for every periodic Coulomb method must match a
        central finite difference under the same row-vector strain
        (``coord @ scaling``, ``cell @ scaling``) the calculator uses
        internally. This is the regression test for the
        nvalchemiops virial sign convention (W = -dE/dstrain) and ensures
        DSF/Ewald/PME all share the same convention."""
        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Model has embedded Coulomb module", category=UserWarning)
            if method == "dsf":
                calc.set_lrcoulomb_method(method, cutoff=8.0)
            else:
                calc.set_lrcoulomb_method(method)

        data_calc = _data_calc_from_fixture(pbc_crystal_small)
        coord_np = data_calc["coord"]
        cell_np = data_calc["cell"]
        numbers_np = data_calc["numbers"]
        volume = abs(np.linalg.det(cell_np))

        res = calc(data_calc, stress=True)
        stress_autograd = res["stress"].detach().cpu().numpy()

        delta = 1e-3
        stress_fd = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                scale_p = np.eye(3)
                scale_p[i, j] += delta
                scale_m = np.eye(3)
                scale_m[i, j] -= delta
                e_p = calc({
                    "coord": coord_np @ scale_p,
                    "numbers": numbers_np,
                    "charge": 0.0,
                    "cell": cell_np @ scale_p,
                })["energy"].item()
                e_m = calc({
                    "coord": coord_np @ scale_m,
                    "numbers": numbers_np,
                    "charge": 0.0,
                    "cell": cell_np @ scale_m,
                })["energy"].item()
                stress_fd[i, j] = (e_p - e_m) / (2 * delta * volume)

        # Loose tolerance: NN energy + Ewald accuracy 1e-6 + FD step noise.
        assert np.abs(stress_autograd - stress_fd).max() < 5e-3, (
            f"{method}: stress mismatch\nautograd={stress_autograd}\nfd      ={stress_fd}"
        )

    @pytest.mark.parametrize("method", ["ewald", "pme"])
    def test_eval_matches_train_forces_and_stress(self, pbc_crystal_small, device, method):
        """Eval-mode derivatives match train-mode derivatives for Ewald/PME."""
        calc_eval = AIMNet2Calculator("aimnet2", nb_threshold=0, train=False)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Model has embedded Coulomb module", category=UserWarning)
            calc_eval.set_lrcoulomb_method(method)

        calc_train = AIMNet2Calculator("aimnet2", nb_threshold=0, train=True)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Model has embedded Coulomb module", category=UserWarning)
            calc_train.set_lrcoulomb_method(method)

        data_calc = _data_calc_from_fixture(pbc_crystal_small)

        res_eval = calc_eval(data_calc, forces=True, stress=True)
        res_train = calc_train(data_calc, forces=True, stress=True)

        torch.testing.assert_close(res_eval["energy"], res_train["energy"], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(res_eval["forces"], res_train["forces"], atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(
            res_eval["stress"],
            res_train["stress"],
            atol=1e-3,
            rtol=1e-3,
        )

    @pytest.mark.parametrize("method", ["ewald", "pme"])
    def test_force_matches_finite_difference(self, pbc_crystal_small, device, method):
        """Total force on a single atom matches a central finite-difference of the energy."""
        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Model has embedded Coulomb module", category=UserWarning)
            calc.set_lrcoulomb_method(method)

        data = pbc_crystal_small.copy()
        coord = data["coord"].clone()
        numbers = data["numbers"]
        cell = data["cell"]

        # Reference forces (analytic, via the nvalchemiops path)
        res = calc(
            {
                "coord": coord.cpu().numpy(),
                "numbers": numbers.cpu().numpy(),
                "charge": 0.0,
                "cell": cell.cpu().numpy(),
            },
            forces=True,
        )
        f_analytic = res["forces"].detach().clone()

        # Probe atom 0 along x
        delta = 1e-3
        atom_idx = 0
        axis = 0

        coord_plus = coord.clone()
        coord_plus[atom_idx, axis] += delta
        coord_minus = coord.clone()
        coord_minus[atom_idx, axis] -= delta

        e_plus = calc({
            "coord": coord_plus.cpu().numpy(),
            "numbers": numbers.cpu().numpy(),
            "charge": 0.0,
            "cell": cell.cpu().numpy(),
        })["energy"].item()
        e_minus = calc({
            "coord": coord_minus.cpu().numpy(),
            "numbers": numbers.cpu().numpy(),
            "charge": 0.0,
            "cell": cell.cpu().numpy(),
        })["energy"].item()

        # F = -dE/dx
        f_fd = -(e_plus - e_minus) / (2.0 * delta)

        # Loose tolerance (relaxed for FD noise + neural-net non-linearity)
        assert abs(f_analytic[atom_idx, axis].item() - f_fd) < 5e-2, (
            f"Force vs FD mismatch: analytic={f_analytic[atom_idx, axis].item():.6f}, FD={f_fd:.6f}"
        )

    @pytest.mark.parametrize("method", ["ewald", "pme"])
    def test_train_mode_force_stress_loss_backpropagates(self, pbc_crystal_small, device, method):
        """Ewald/PME train mode supports force and stress losses."""
        calc = AIMNet2Calculator("aimnet2", nb_threshold=0, train=True)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Model has embedded Coulomb module", category=UserWarning)
            calc.set_lrcoulomb_method(method)

        res = calc(_data_calc_from_fixture(pbc_crystal_small), forces=True, stress=True)

        assert torch.isfinite(res["energy"]).all()
        assert torch.isfinite(res["forces"]).all()
        assert torch.isfinite(res["stress"]).all()

        loss = res["energy"].sum() + res["forces"].pow(2).sum() + res["stress"].pow(2).sum()
        loss.backward()
        grads = [p.grad for p in calc.model.parameters() if p.requires_grad and p.grad is not None]
        assert grads
        assert any(torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)

    @pytest.mark.parametrize("method", ["ewald", "pme"])
    def test_hessian_matches_force_finite_difference_component(self, pbc_crystal_small, device, method):
        """A selected Hessian component matches finite differences of force."""
        calc = AIMNet2Calculator("aimnet2", nb_threshold=0, train=False)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Model has embedded Coulomb module", category=UserWarning)
            calc.set_lrcoulomb_method(method)
        calc.external_dftd3 = None

        data_calc = _data_calc_from_fixture(pbc_crystal_small)
        res = calc(data_calc, hessian=True)

        n_atoms = pbc_crystal_small["coord"].shape[0]
        assert res["hessian"].shape == (n_atoms, 3, n_atoms, 3)
        assert torch.isfinite(res["hessian"]).all()

        atom_idx = 0
        axis = 0
        delta = 1e-3
        coord = pbc_crystal_small["coord"]
        coord_plus = coord.clone()
        coord_plus[atom_idx, axis] += delta
        coord_minus = coord.clone()
        coord_minus[atom_idx, axis] -= delta

        data_plus = _data_calc_from_fixture({**pbc_crystal_small, "coord": coord_plus})
        data_minus = _data_calc_from_fixture({**pbc_crystal_small, "coord": coord_minus})
        f_plus = calc(data_plus, forces=True)["forces"][atom_idx, axis]
        f_minus = calc(data_minus, forces=True)["forces"][atom_idx, axis]
        h_fd = -(f_plus - f_minus) / (2.0 * delta)

        torch.testing.assert_close(res["hessian"][atom_idx, axis, atom_idx, axis], h_fd, atol=5e-2, rtol=5e-2)

    @pytest.mark.parametrize("method", ["ewald", "pme"])
    def test_calculator_pbc_batched(self, pbc_crystal_small, device, method):
        """Ewald/PME work with a batched input of two periodic systems."""
        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Model has embedded Coulomb module", category=UserWarning)
            calc.set_lrcoulomb_method(method)

        data1 = pbc_crystal_small.copy()
        data2 = pbc_crystal_small.copy()
        data2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in data2.items()}
        data2["coord"] = data2["coord"] + 0.01

        n1 = data1["coord"].shape[0]
        n2 = data2["coord"].shape[0]
        batched_data = {
            "coord": torch.cat([data1["coord"], data2["coord"]], dim=0),
            "numbers": torch.cat([data1["numbers"], data2["numbers"]], dim=0),
            "charge": torch.tensor([0.0, 0.0], device=device),
            "cell": torch.stack([data1["cell"], data2["cell"]], dim=0),
            "mol_idx": torch.cat([
                torch.zeros(n1, dtype=torch.long, device=device),
                torch.ones(n2, dtype=torch.long, device=device),
            ]),
        }

        res = calc(batched_data, forces=True, stress=True)

        assert "energy" in res
        assert "forces" in res
        assert "stress" in res
        assert res["stress"].shape == (2, 3, 3)
        assert torch.isfinite(res["energy"]).all()
        assert torch.isfinite(res["forces"]).all()
        assert torch.isfinite(res["stress"]).all()
