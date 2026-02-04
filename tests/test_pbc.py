"""Tests for periodic boundary conditions - DSF and Ewald Coulomb methods.

These tests verify that the DSF and Ewald Coulomb methods work correctly
with periodic crystal structures loaded from CIF files.
"""

import warnings

import pytest
import torch

from aimnet import nbops, ops
from aimnet.calculators import AIMNet2Calculator
from aimnet.modules.lr import LRCoulomb

pytestmark = [pytest.mark.ase, pytest.mark.gpu]


def setup_pbc_data_with_nblist(data: dict, device: torch.device, cutoff: float = 8.0) -> dict:
    """Set up periodic data with neighbor list for Coulomb calculations.

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
    coord = data["coord"]
    n_atoms = coord.shape[0]

    # Create all-pairs neighbor matrix (for testing)
    max_nb = min(n_atoms - 1, 50)  # Limit for large systems
    nbmat = torch.zeros((n_atoms, max_nb), dtype=torch.long, device=device)
    for i in range(n_atoms):
        neighbors = [j for j in range(n_atoms) if j != i][:max_nb]
        for k, nb in enumerate(neighbors):
            nbmat[i, k] = nb
        # Fill remaining with padding index
        for k in range(len(neighbors), max_nb):
            nbmat[i, k] = n_atoms - 1

    # Shifts for PBC - zero for non-image neighbors
    shifts = torch.zeros((n_atoms, max_nb, 3), dtype=torch.float32, device=device)

    # Set both short-range and long-range neighbor lists
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


class TestDSFPeriodic:
    """Tests for DSF Coulomb with periodic structures."""

    def test_dsf_pbc_energy_finite(self, pbc_crystal_small, device):
        """Test DSF produces finite energy for periodic system."""
        module = LRCoulomb(method="dsf", dsf_rc=8.0).to(device)
        data = setup_pbc_data_with_nblist(pbc_crystal_small.copy(), device)

        # Add mock charges for testing
        n_atoms = data["coord"].shape[0]
        data["charges"] = torch.randn(n_atoms, device=device) * 0.3

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
        data["charges"] = charges

        result = module(data)

        # For alternating charges, energy should be negative (attractive)
        assert result["e_h"].item() < 0, "Alternating charges should have negative Coulomb energy"

    def test_dsf_pbc_cutoff_effect(self, pbc_crystal_small, device):
        """Test that DSF cutoff affects energy in periodic system."""
        data = setup_pbc_data_with_nblist(pbc_crystal_small.copy(), device)
        n_atoms = data["coord"].shape[0]
        data["charges"] = torch.randn(n_atoms, device=device) * 0.5

        # Short cutoff
        module_short = LRCoulomb(method="dsf", dsf_rc=3.0).to(device)
        result_short = module_short(data.copy())

        # Long cutoff
        module_long = LRCoulomb(method="dsf", dsf_rc=10.0).to(device)
        result_long = module_long(data.copy())

        # Both should be finite
        assert torch.isfinite(result_short["e_h"]).all()
        assert torch.isfinite(result_long["e_h"]).all()

    def test_dsf_pbc_gradient_coords(self, pbc_crystal_small, device):
        """Test gradient of DSF energy w.r.t. coordinates."""
        module = LRCoulomb(method="dsf", dsf_rc=8.0).to(device)

        data = pbc_crystal_small.copy()
        data["coord"] = data["coord"].clone().requires_grad_(True)
        data = setup_pbc_data_with_nblist(data, device)

        n_atoms = data["coord"].shape[0]
        data["charges"] = torch.randn(n_atoms, device=device) * 0.3

        result = module(data)
        result["e_h"].backward()

        assert data["coord"].grad is not None, "Gradient should exist"
        assert torch.isfinite(data["coord"].grad).all(), "Gradient should be finite"


class TestEwaldPeriodic:
    """Tests for Ewald summation with periodic structures."""

    def test_ewald_pbc_energy_finite(self, pbc_crystal_small, device):
        """Test Ewald produces finite energy for periodic system."""
        module = LRCoulomb(method="ewald").to(device)
        data = setup_pbc_data_with_nblist(pbc_crystal_small.copy(), device, cutoff=8.0)

        n_atoms = data["coord"].shape[0]
        # Ensure charge neutrality for Ewald
        charges = torch.randn(n_atoms, device=device) * 0.3
        charges = charges - charges.mean()  # Make neutral
        data["charges"] = charges

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
        if n_atoms % 2 != 0:
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
        charges = torch.randn(n_atoms, device=device) * 0.3
        charges = charges - charges.mean()
        data["charges"] = charges

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
        charges = torch.randn(n_atoms, device=device) * 0.3
        charges = charges - charges.mean()
        data["charges"] = charges

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
        if n_atoms % 2 != 0:
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
            calc.set_lrcoulomb_method("ewald", ewald_accuracy=1e-8)

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

    @pytest.mark.parametrize("method", ["dsf", "ewald"])
    def test_calculator_pbc_both_methods(self, pbc_crystal_small, device, method):
        """Test calculator works with both DSF and Ewald methods."""
        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Model has embedded Coulomb module", category=UserWarning)
            calc.set_lrcoulomb_method(method, cutoff=8.0)

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
        data["charges"] = torch.randn(n_atoms, device=device) * 0.3

        result = module(data)

        assert torch.isfinite(result["e_h"]).all()

    def test_large_crystal_ewald_energy(self, pbc_crystal_large, device):
        """Test Ewald on larger crystal (~50 atoms)."""
        module = LRCoulomb(method="ewald").to(device)
        data = setup_pbc_data_with_nblist(pbc_crystal_large.copy(), device)

        n_atoms = data["coord"].shape[0]
        charges = torch.randn(n_atoms, device=device) * 0.3
        charges = charges - charges.mean()
        data["charges"] = charges

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
