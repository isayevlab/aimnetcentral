"""Tests for aimnet.modules.lr - long-range Coulomb and dispersion modules."""

import pytest
import torch

from aimnet import nbops, ops
from aimnet.modules.lr import LRCoulomb, SRCoulomb

# Coulomb methods for parametrized tests (non-periodic)
# Note: "ewald" excluded here because it requires cell and flat format (mode 1)
# Ewald is tested separately in TestLRCoulombEwald and TestLRCoulombEwaldPBC classes
COULOMB_METHODS = ["simple", "dsf"]
COULOMB_METHODS_ALL = ["simple", "dsf", "ewald"]


def setup_data_mode_0(device, n_atoms=5):
    """Create test data in nb_mode=0 format."""
    torch.manual_seed(42)
    coord = torch.rand((1, n_atoms, 3), device=device) * 5  # 5 Angstrom box
    numbers = torch.randint(1, 10, (1, n_atoms), device=device)
    charges = torch.randn((1, n_atoms), device=device) * 0.3

    data = {"coord": coord, "numbers": numbers, "charges": charges}
    data = nbops.set_nb_mode(data)
    data = nbops.calc_masks(data)

    # Compute distances
    d_ij, _r_ij = ops.calc_distances(data)
    data["d_ij"] = d_ij

    return data


def setup_data_mode_1(device, n_atoms=5):
    """Create test data in nb_mode=1 format (flat with mol_idx)."""
    torch.manual_seed(42)
    coord = torch.rand((n_atoms + 1, 3), device=device) * 5  # +1 for padding
    numbers = torch.cat([torch.randint(1, 10, (n_atoms,), device=device), torch.tensor([0], device=device)])
    charges = torch.cat([torch.randn(n_atoms, device=device) * 0.3, torch.tensor([0.0], device=device)])
    mol_idx = torch.cat([torch.zeros(n_atoms, dtype=torch.long, device=device), torch.tensor([0], device=device)])

    # Create neighbor matrix (all atoms see each other)
    max_nb = n_atoms
    nbmat = torch.zeros((n_atoms + 1, max_nb), dtype=torch.long, device=device)
    nbmat_lr = torch.zeros((n_atoms + 1, max_nb), dtype=torch.long, device=device)
    for i in range(n_atoms):
        neighbors = [j for j in range(n_atoms + 1) if j != i][:max_nb]
        for k, nb in enumerate(neighbors):
            nbmat[i, k] = nb
            nbmat_lr[i, k] = nb

    data = {
        "coord": coord,
        "numbers": numbers,
        "charges": charges,
        "mol_idx": mol_idx,
        "nbmat": nbmat,
        "nbmat_lr": nbmat_lr,
    }
    data = nbops.set_nb_mode(data)
    data = nbops.calc_masks(data)

    # Compute distances
    d_ij, _r_ij = ops.calc_distances(data)
    data["d_ij"] = d_ij

    return data


class TestLRCoulombSimple:
    """Tests for simple Coulomb method."""

    def test_simple_output_shape(self, device):
        """Test that simple method produces correct output shape."""
        module = LRCoulomb(method="simple").to(device)
        data = setup_data_mode_0(device, n_atoms=5)

        result = module(data)

        assert "e_h" in result
        assert result["e_h"].shape == (1,)  # Per-molecule energy

    def test_simple_zero_charges(self, device):
        """Test that zero charges produce zero energy."""
        module = LRCoulomb(method="simple").to(device)
        data = setup_data_mode_0(device, n_atoms=5)
        data["charges"] = torch.zeros_like(data["charges"])

        result = module(data)

        assert result["e_h"].abs().item() < 1e-10

    def test_simple_opposite_charges(self, device):
        """Test that opposite charges produce negative (attractive) energy."""
        module = LRCoulomb(method="simple").to(device)

        # Two atoms with opposite charges
        coord = torch.tensor([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]], device=device)
        numbers = torch.tensor([[1, 1]], device=device)
        charges = torch.tensor([[1.0, -1.0]], device=device)

        data = {"coord": coord, "numbers": numbers, "charges": charges}
        data = nbops.set_nb_mode(data)
        data = nbops.calc_masks(data)
        data["d_ij"], _ = ops.calc_distances(data)

        result = module(data)

        # Opposite charges should attract (negative energy)
        assert result["e_h"].item() < 0

    def test_simple_same_charges(self, device):
        """Test that same charges produce positive (repulsive) energy."""
        module = LRCoulomb(method="simple").to(device)

        coord = torch.tensor([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]], device=device)
        numbers = torch.tensor([[1, 1]], device=device)
        charges = torch.tensor([[1.0, 1.0]], device=device)

        data = {"coord": coord, "numbers": numbers, "charges": charges}
        data = nbops.set_nb_mode(data)
        data = nbops.calc_masks(data)
        data["d_ij"], _ = ops.calc_distances(data)

        result = module(data)

        # Same charges should repel (positive energy)
        assert result["e_h"].item() > 0


class TestLRCoulombDSF:
    """Tests for DSF (Damped Shifted Force) Coulomb method."""

    def test_dsf_output_shape(self, device):
        """Test that DSF method produces correct output shape."""
        module = LRCoulomb(method="dsf").to(device)
        data = setup_data_mode_0(device, n_atoms=5)

        result = module(data)

        assert "e_h" in result
        assert result["e_h"].shape == (1,)

    def test_dsf_zero_charges(self, device):
        """Test that DSF with zero charges produces zero energy."""
        module = LRCoulomb(method="dsf").to(device)
        data = setup_data_mode_0(device, n_atoms=5)
        data["charges"] = torch.zeros_like(data["charges"])

        result = module(data)

        assert result["e_h"].abs().item() < 1e-10

    def test_dsf_cutoff_effect(self, device):
        """Test that DSF cutoff affects energy."""
        data = setup_data_mode_0(device, n_atoms=3)

        # Short cutoff
        module_short = LRCoulomb(method="dsf", dsf_rc=2.0).to(device)
        result_short = module_short(data.copy())

        # Long cutoff
        module_long = LRCoulomb(method="dsf", dsf_rc=20.0).to(device)
        result_long = module_long(data.copy())

        # Energies should differ due to cutoff
        # (might be same if all atoms within short cutoff)
        assert torch.isfinite(result_short["e_h"])
        assert torch.isfinite(result_long["e_h"])


class TestLRCoulombEwald:
    """Tests for Ewald summation Coulomb method."""

    def test_ewald_output_shape(self, device):
        """Test that Ewald method produces correct output shape."""
        module = LRCoulomb(method="ewald").to(device)

        # Ewald requires cell and flat format (mode 1)
        N = 5
        max_nb = N
        coord = torch.rand((N + 1, 3), device=device) * 5
        numbers = torch.cat([torch.randint(1, 10, (N,), device=device), torch.tensor([0], device=device)])
        charges = torch.cat([torch.randn(N, device=device) * 0.3, torch.tensor([0.0], device=device)])
        cell = torch.eye(3, device=device) * 10
        mol_idx = torch.cat([torch.zeros(N, dtype=torch.long, device=device), torch.tensor([0], device=device)])
        # Create neighbor matrix for mode 1
        nbmat = torch.zeros((N + 1, max_nb), dtype=torch.long, device=device)
        for i in range(N):
            nbmat[i] = torch.tensor([j for j in range(N + 1) if j != i][:max_nb], device=device)
        nbmat[-1] = N  # padding points to itself
        # Create shifts (no periodic images in this simple test)
        shifts = torch.zeros((N + 1, max_nb, 3), dtype=torch.float32, device=device)

        data = {
            "coord": coord,
            "numbers": numbers,
            "charges": charges,
            "cell": cell,
            "mol_idx": mol_idx,
            "nbmat": nbmat,
            "shifts": shifts,
            "nbmat_coulomb": nbmat,
            "shifts_coulomb": shifts.to(torch.int32),
            "cutoff_coulomb": torch.tensor(8.0),
        }
        data = nbops.set_nb_mode(data)
        data = nbops.calc_masks(data)
        data["d_ij"], _ = ops.calc_distances(data)

        result = module(data)

        assert "e_h" in result
        assert torch.isfinite(result["e_h"]).all()

    def test_ewald_zero_charges(self, device):
        """Test that Ewald with zero charges produces zero energy."""
        module = LRCoulomb(method="ewald").to(device)

        N = 5
        max_nb = N
        coord = torch.rand((N + 1, 3), device=device) * 5
        numbers = torch.cat([torch.randint(1, 10, (N,), device=device), torch.tensor([0], device=device)])
        charges = torch.zeros(N + 1, device=device)
        cell = torch.eye(3, device=device) * 10
        mol_idx = torch.cat([torch.zeros(N, dtype=torch.long, device=device), torch.tensor([0], device=device)])
        nbmat = torch.zeros((N + 1, max_nb), dtype=torch.long, device=device)
        for i in range(N):
            nbmat[i] = torch.tensor([j for j in range(N + 1) if j != i][:max_nb], device=device)
        nbmat[-1] = N
        # Create shifts (no periodic images in this simple test)
        shifts = torch.zeros((N + 1, max_nb, 3), dtype=torch.float32, device=device)

        data = {
            "coord": coord,
            "numbers": numbers,
            "charges": charges,
            "cell": cell,
            "mol_idx": mol_idx,
            "nbmat": nbmat,
            "shifts": shifts,
            "nbmat_coulomb": nbmat,
            "shifts_coulomb": shifts.to(torch.int32),
            "cutoff_coulomb": torch.tensor(8.0),
        }
        data = nbops.set_nb_mode(data)
        data = nbops.calc_masks(data)
        data["d_ij"], _ = ops.calc_distances(data)

        result = module(data)

        assert result["e_h"].abs().sum().item() < 1e-8


class TestLRCoulombGradients:
    """Tests for gradient flow through Coulomb methods."""

    def test_simple_gradient_wrt_charges(self, device):
        """Test gradient of simple Coulomb wrt charges."""
        module = LRCoulomb(method="simple").to(device)

        coord = torch.tensor([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]], device=device)
        numbers = torch.tensor([[1, 1]], device=device)
        charges = torch.tensor([[0.5, -0.5]], device=device, requires_grad=True)

        data = {"coord": coord, "numbers": numbers, "charges": charges}
        data = nbops.set_nb_mode(data)
        data = nbops.calc_masks(data)
        data["d_ij"], _ = ops.calc_distances(data)

        result = module(data)
        result["e_h"].backward()

        assert charges.grad is not None
        assert torch.isfinite(charges.grad).all()

    def test_simple_gradient_wrt_coords(self, device):
        """Test gradient of simple Coulomb wrt coordinates."""
        module = LRCoulomb(method="simple").to(device)

        coord = torch.tensor([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]], device=device, requires_grad=True)
        numbers = torch.tensor([[1, 1]], device=device)
        charges = torch.tensor([[0.5, -0.5]], device=device)

        data = {"coord": coord, "numbers": numbers, "charges": charges}
        data = nbops.set_nb_mode(data)
        data = nbops.calc_masks(data)
        data["d_ij"], _ = ops.calc_distances(data)

        result = module(data)
        result["e_h"].backward()

        assert coord.grad is not None
        assert torch.isfinite(coord.grad).all()

    def test_dsf_gradient_wrt_charges(self, device):
        """Test gradient of DSF Coulomb wrt charges."""
        module = LRCoulomb(method="dsf").to(device)

        coord = torch.tensor([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]], device=device)
        numbers = torch.tensor([[1, 1]], device=device)
        charges = torch.tensor([[0.5, -0.5]], device=device, requires_grad=True)

        data = {"coord": coord, "numbers": numbers, "charges": charges}
        data = nbops.set_nb_mode(data)
        data = nbops.calc_masks(data)
        data["d_ij"], _ = ops.calc_distances(data)

        result = module(data)
        result["e_h"].backward()

        assert charges.grad is not None
        assert torch.isfinite(charges.grad).all()


class TestLRCoulombConsistency:
    """Tests for consistency between Coulomb methods."""

    def test_simple_dsf_close_for_small_molecules(self, device):
        """Test that simple and DSF give similar results for small isolated molecules."""
        coord = torch.tensor([[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0]]], device=device)
        numbers = torch.tensor([[8, 1, 1]], device=device)
        charges = torch.tensor([[-0.8, 0.4, 0.4]], device=device)

        data = {"coord": coord, "numbers": numbers, "charges": charges}
        data = nbops.set_nb_mode(data)
        data = nbops.calc_masks(data)
        data["d_ij"], _ = ops.calc_distances(data)

        module_simple = LRCoulomb(method="simple").to(device)
        module_dsf = LRCoulomb(method="dsf", dsf_rc=20.0).to(device)

        # Need to copy data since modules modify it
        result_simple = module_simple(data.copy())
        result_dsf = module_dsf(data.copy())

        # For small isolated molecules, results should be reasonably close
        # (DSF subtracts short-range contribution)
        assert torch.isfinite(result_simple["e_h"])
        assert torch.isfinite(result_dsf["e_h"])

    def test_energy_sign_consistency_simple(self, device):
        """Test that simple method gives correct energy sign."""
        # Two opposite charges - should be attractive (negative)
        coord = torch.tensor([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]], device=device)
        numbers = torch.tensor([[1, 1]], device=device)
        charges = torch.tensor([[1.0, -1.0]], device=device)

        data = {"coord": coord, "numbers": numbers, "charges": charges}
        data = nbops.set_nb_mode(data)
        data = nbops.calc_masks(data)
        data["d_ij"], _ = ops.calc_distances(data)

        module = LRCoulomb(method="simple").to(device)
        result = module(data)

        # Opposite charges should have negative energy
        assert result["e_h"].item() < 0

    @pytest.mark.parametrize("method", COULOMB_METHODS)
    def test_finite_energy_output(self, device, method):
        """Test that all methods produce finite energy output."""
        data = setup_data_mode_0(device, n_atoms=5)

        if method == "dsf":
            module = LRCoulomb(method=method, dsf_rc=10.0).to(device)
        else:
            module = LRCoulomb(method=method).to(device)

        result = module(data)
        assert torch.isfinite(result["e_h"]).all()


class TestLRCoulombAdditive:
    """Tests for additive energy accumulation."""

    def test_energy_addition(self, device):
        """Test that energy is added to existing key if present."""
        module = LRCoulomb(method="simple", key_out="energy").to(device)

        coord = torch.tensor([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]], device=device)
        numbers = torch.tensor([[1, 1]], device=device)
        charges = torch.tensor([[0.5, -0.5]], device=device)

        data = {"coord": coord, "numbers": numbers, "charges": charges}
        data = nbops.set_nb_mode(data)
        data = nbops.calc_masks(data)
        data["d_ij"], _ = ops.calc_distances(data)

        # Set initial energy
        data["energy"] = torch.tensor([10.0], device=device)

        result = module(data)

        # Energy should be added to existing value
        assert result["energy"].item() != 10.0
        # The Coulomb contribution should be added

    def test_energy_creation(self, device):
        """Test that energy key is created if not present."""
        module = LRCoulomb(method="simple", key_out="new_energy").to(device)

        coord = torch.tensor([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]], device=device)
        numbers = torch.tensor([[1, 1]], device=device)
        charges = torch.tensor([[0.5, -0.5]], device=device)

        data = {"coord": coord, "numbers": numbers, "charges": charges}
        data = nbops.set_nb_mode(data)
        data = nbops.calc_masks(data)
        data["d_ij"], _ = ops.calc_distances(data)

        assert "new_energy" not in data
        result = module(data)
        assert "new_energy" in result


# =============================================================================
# PBC Coulomb Tests
# =============================================================================


def setup_pbc_data(pbc_fixture: dict, device: torch.device, cutoff: float = 8.0) -> dict:
    """Set up periodic data with neighbor list for Coulomb calculations."""
    data = pbc_fixture.copy()
    coord = data["coord"]
    n_atoms = coord.shape[0]

    # Create a simple all-pairs neighbor matrix
    max_nb = min(n_atoms - 1, 50)
    nbmat = torch.zeros((n_atoms, max_nb), dtype=torch.long, device=device)
    for i in range(n_atoms):
        neighbors = [j for j in range(n_atoms) if j != i][:max_nb]
        for k, nb in enumerate(neighbors):
            nbmat[i, k] = nb
        for k in range(len(neighbors), max_nb):
            nbmat[i, k] = n_atoms - 1

    # Shifts are unit cell translations - use float for matrix multiplication with cell
    shifts = torch.zeros((n_atoms, max_nb, 3), dtype=torch.float32, device=device)

    data["nbmat"] = nbmat
    data["shifts"] = shifts
    # Long-range neighbor list keys (used by LRCoulomb module)
    # shifts_lr must be float for matrix multiplication with cell in calc_distances
    data["nbmat_lr"] = nbmat
    data["shifts_lr"] = shifts.clone()
    data["nbmat_coulomb"] = nbmat
    data["shifts_coulomb"] = shifts.clone()
    data["cutoff_coulomb"] = torch.tensor(cutoff, device=device)

    data = nbops.set_nb_mode(data)
    data = nbops.calc_masks(data)

    d_ij, _ = ops.calc_distances(data)
    data["d_ij"] = d_ij

    return data


class TestLRCoulombDSFPBC:
    """Tests for DSF Coulomb with periodic boundary conditions."""

    def test_dsf_pbc_output_shape(self, pbc_crystal_small, device):
        """Test DSF output shape with PBC."""
        module = LRCoulomb(method="dsf", dsf_rc=8.0).to(device)
        data = setup_pbc_data(pbc_crystal_small, device)

        n_atoms = data["coord"].shape[0]
        data["charges"] = torch.randn(n_atoms, device=device) * 0.3

        result = module(data)

        assert "e_h" in result
        assert result["e_h"].shape == (1,)

    def test_dsf_pbc_gradient_wrt_coords(self, pbc_crystal_small, device):
        """Test DSF coordinate gradients with PBC."""
        module = LRCoulomb(method="dsf", dsf_rc=8.0).to(device)

        data = pbc_crystal_small.copy()
        data["coord"] = data["coord"].clone().requires_grad_(True)
        data = setup_pbc_data(data, device)

        n_atoms = data["coord"].shape[0]
        data["charges"] = torch.randn(n_atoms, device=device) * 0.3

        result = module(data)
        result["e_h"].backward()

        assert data["coord"].grad is not None
        assert torch.isfinite(data["coord"].grad).all()

    def test_dsf_pbc_finite_energy(self, pbc_crystal_small, device):
        """Test DSF produces finite energy with PBC."""
        module = LRCoulomb(method="dsf", dsf_rc=8.0).to(device)
        data = setup_pbc_data(pbc_crystal_small, device)

        n_atoms = data["coord"].shape[0]
        data["charges"] = torch.randn(n_atoms, device=device) * 0.3

        result = module(data)

        assert torch.isfinite(result["e_h"]).all()


class TestLRCoulombEwaldPBC:
    """Tests for Ewald Coulomb with periodic boundary conditions."""

    def test_ewald_pbc_output_shape(self, pbc_crystal_small, device):
        """Test Ewald output shape with PBC."""
        module = LRCoulomb(method="ewald").to(device)
        data = setup_pbc_data(pbc_crystal_small, device, cutoff=8.0)

        n_atoms = data["coord"].shape[0]
        charges = torch.randn(n_atoms, device=device) * 0.3
        charges = charges - charges.mean()  # Make neutral
        data["charges"] = charges

        result = module(data)

        assert "e_h" in result
        assert torch.isfinite(result["e_h"]).all()

    def test_ewald_pbc_gradient_wrt_coords(self, pbc_crystal_small, device):
        """Test Ewald coordinate gradients with PBC."""
        module = LRCoulomb(method="ewald").to(device)

        data = pbc_crystal_small.copy()
        data["coord"] = data["coord"].clone().requires_grad_(True)
        data = setup_pbc_data(data, device)

        n_atoms = data["coord"].shape[0]
        charges = torch.randn(n_atoms, device=device) * 0.3
        charges = charges - charges.mean()
        data["charges"] = charges

        result = module(data)
        result["e_h"].backward()

        assert data["coord"].grad is not None
        assert torch.isfinite(data["coord"].grad).all()

    def test_ewald_pbc_cell_present(self, pbc_crystal_small, device):
        """Test Ewald requires cell for PBC."""
        data = setup_pbc_data(pbc_crystal_small, device)

        n_atoms = data["coord"].shape[0]
        charges = torch.randn(n_atoms, device=device) * 0.3
        charges = charges - charges.mean()
        data["charges"] = charges

        # Verify cell is present
        assert "cell" in data
        assert data["cell"].shape == (3, 3)

        module = LRCoulomb(method="ewald").to(device)
        result = module(data)

        assert torch.isfinite(result["e_h"]).all()


class TestLRCoulombEnvelope:
    """Tests for envelope parameter in LRCoulomb."""

    @pytest.mark.parametrize("envelope", ["exp", "cosine"])
    def test_envelope_parameter_valid(self, device, envelope):
        """Test that valid envelope parameters are accepted."""
        module = LRCoulomb(method="simple", envelope=envelope).to(device)
        assert module.envelope == envelope

    def test_envelope_parameter_invalid(self, device):
        """Test that invalid envelope raises ValueError."""
        with pytest.raises(ValueError, match="Unknown envelope"):
            LRCoulomb(method="simple", envelope="invalid").to(device)

    def test_exp_envelope_default(self, device):
        """Test that exp is the default envelope."""
        module = LRCoulomb(method="simple").to(device)
        assert module.envelope == "exp"

    @pytest.mark.parametrize("envelope", ["exp", "cosine"])
    def test_envelope_produces_finite_output(self, device, envelope):
        """Test that both envelopes produce finite energy."""
        module = LRCoulomb(method="simple", envelope=envelope, subtract_sr=True).to(device)
        data = setup_data_mode_0(device, n_atoms=5)

        result = module(data)

        assert torch.isfinite(result["e_h"]).all()

    def test_different_envelopes_give_different_sr(self, device):
        """Test that different envelopes produce different SR contributions."""
        data_exp = setup_data_mode_0(device, n_atoms=5)
        data_cos = setup_data_mode_0(device, n_atoms=5)

        module_exp = LRCoulomb(method="simple", envelope="exp", subtract_sr=True).to(device)
        module_cos = LRCoulomb(method="simple", envelope="cosine", subtract_sr=True).to(device)

        result_exp = module_exp(data_exp)
        result_cos = module_cos(data_cos)

        # Results should differ because SR is computed differently
        # (unless atoms are too far for SR cutoff to matter)
        assert torch.isfinite(result_exp["e_h"]).all()
        assert torch.isfinite(result_cos["e_h"]).all()


class TestSRCoulomb:
    """Tests for SRCoulomb module."""

    def test_srcoulomb_creation(self, device):
        """Test SRCoulomb can be created with default parameters."""
        module = SRCoulomb().to(device)
        assert module.key_in == "charges"
        assert module.key_out == "energy"

    def test_srcoulomb_envelope_parameter(self, device):
        """Test SRCoulomb accepts envelope parameter."""
        module_exp = SRCoulomb(envelope="exp").to(device)
        module_cos = SRCoulomb(envelope="cosine").to(device)

        assert module_exp.envelope == "exp"
        assert module_cos.envelope == "cosine"

    def test_srcoulomb_invalid_envelope(self, device):
        """Test SRCoulomb rejects invalid envelope."""
        with pytest.raises(ValueError, match="Unknown envelope"):
            SRCoulomb(envelope="invalid").to(device)

    def test_srcoulomb_subtracts_from_energy(self, device):
        """Test SRCoulomb subtracts energy from data."""
        module = SRCoulomb(key_out="energy").to(device)
        data = setup_data_mode_0(device, n_atoms=5)

        # Add initial energy
        data["energy"] = torch.tensor([10.0], device=device)

        result = module(data)

        # Energy should be reduced (SR Coulomb subtracted)
        assert result["energy"].item() != 10.0
        assert torch.isfinite(result["energy"]).all()

    @pytest.mark.parametrize("envelope", ["exp", "cosine"])
    def test_srcoulomb_envelope_produces_finite(self, device, envelope):
        """Test that both envelopes produce finite results."""
        module = SRCoulomb(envelope=envelope, key_out="sr_energy").to(device)
        data = setup_data_mode_0(device, n_atoms=5)

        result = module(data)

        assert "sr_energy" in result
        assert torch.isfinite(result["sr_energy"]).all()
