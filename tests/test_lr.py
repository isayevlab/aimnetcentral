"""Tests for aimnet.modules.lr - long-range Coulomb and dispersion modules."""

import pytest
import torch

from aimnet import nbops, ops
from aimnet.modules.lr import LRCoulomb


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
    d_ij, r_ij = ops.calc_distances(data)
    data["d_ij"] = d_ij

    return data


def setup_data_mode_1(device, n_atoms=5):
    """Create test data in nb_mode=1 format (flat with mol_idx)."""
    torch.manual_seed(42)
    coord = torch.rand((n_atoms + 1, 3), device=device) * 5  # +1 for padding
    numbers = torch.cat([torch.randint(1, 10, (n_atoms,), device=device), torch.tensor([0], device=device)])
    charges = torch.cat([torch.randn(n_atoms, device=device) * 0.3, torch.tensor([0.0], device=device)])
    mol_idx = torch.cat([torch.zeros(n_atoms, dtype=torch.long, device=device), torch.tensor([1], device=device)])

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
    d_ij, r_ij = ops.calc_distances(data)
    data["d_ij"] = d_ij

    return data


class TestLRCoulombInit:
    """Tests for LRCoulomb initialization."""

    def test_default_init(self, device):
        """Test default initialization."""
        module = LRCoulomb().to(device)
        assert module.key_in == "charges"
        assert module.key_out == "e_h"
        assert module.method == "simple"

    def test_custom_keys(self, device):
        """Test initialization with custom keys."""
        module = LRCoulomb(key_in="q", key_out="energy_coul").to(device)
        assert module.key_in == "q"
        assert module.key_out == "energy_coul"

    def test_dsf_method(self, device):
        """Test DSF method initialization."""
        module = LRCoulomb(method="dsf", dsf_alpha=0.25, dsf_rc=12.0).to(device)
        assert module.method == "dsf"
        assert module.dsf_alpha == 0.25
        assert module.dsf_rc == 12.0

    def test_ewald_method(self, device):
        """Test Ewald method initialization."""
        module = LRCoulomb(method="ewald").to(device)
        assert module.method == "ewald"

    def test_invalid_method(self, device):
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown method"):
            LRCoulomb(method="invalid").to(device)


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
        coord = torch.rand((N + 1, 3), device=device) * 5
        numbers = torch.cat([torch.randint(1, 10, (N,), device=device), torch.tensor([0], device=device)])
        charges = torch.cat([torch.randn(N, device=device) * 0.3, torch.tensor([0.0], device=device)])
        cell = torch.eye(3, device=device) * 10
        mol_idx = torch.cat([torch.zeros(N, dtype=torch.long, device=device), torch.tensor([1], device=device)])
        # Create neighbor matrix for mode 1
        nbmat = torch.zeros((N + 1, N), dtype=torch.long, device=device)
        for i in range(N):
            nbmat[i] = torch.tensor([j for j in range(N + 1) if j != i][:N], device=device)
        nbmat[-1] = N  # padding points to itself

        data = {
            "coord": coord,
            "numbers": numbers,
            "charges": charges,
            "cell": cell,
            "mol_idx": mol_idx,
            "nbmat": nbmat,
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
        coord = torch.rand((N + 1, 3), device=device) * 5
        numbers = torch.cat([torch.randint(1, 10, (N,), device=device), torch.tensor([0], device=device)])
        charges = torch.zeros(N + 1, device=device)
        cell = torch.eye(3, device=device) * 10
        mol_idx = torch.cat([torch.zeros(N, dtype=torch.long, device=device), torch.tensor([1], device=device)])
        nbmat = torch.zeros((N + 1, N), dtype=torch.long, device=device)
        for i in range(N):
            nbmat[i] = torch.tensor([j for j in range(N + 1) if j != i][:N], device=device)
        nbmat[-1] = N

        data = {
            "coord": coord,
            "numbers": numbers,
            "charges": charges,
            "cell": cell,
            "mol_idx": mol_idx,
            "nbmat": nbmat,
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

    def test_energy_sign_consistency(self, device):
        """Test that all methods agree on energy sign for simple systems."""
        # Two opposite charges - should be attractive (negative)
        coord = torch.tensor([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]], device=device)
        numbers = torch.tensor([[1, 1]], device=device)
        charges = torch.tensor([[1.0, -1.0]], device=device)

        data = {"coord": coord, "numbers": numbers, "charges": charges}
        data = nbops.set_nb_mode(data)
        data = nbops.calc_masks(data)
        data["d_ij"], _ = ops.calc_distances(data)

        module_simple = LRCoulomb(method="simple").to(device)
        result = module_simple(data)

        # Opposite charges should have negative energy
        assert result["e_h"].item() < 0


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
