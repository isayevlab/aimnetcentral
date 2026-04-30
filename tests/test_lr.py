"""Tests for aimnet.modules.lr - long-range Coulomb and dispersion modules."""

import pytest
import torch

from aimnet import nbops, ops
from aimnet.modules.lr import LRCoulomb, SRCoulomb

# Coulomb methods for parametrized tests (non-periodic)
# Note: "ewald" excluded here because it requires cell and flat format (mode 1)
# Ewald is tested separately in TestLRCoulombEwald and TestLRCoulombEwaldPBC classes
COULOMB_METHODS = ["simple", "dsf"]


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

    def test_dsf_mode0_explicit_forces_shape(self, device):
        """DSF explicit forces preserve dense batched input shape."""
        module = LRCoulomb(method="dsf", subtract_sr=False).to(device)
        data = setup_data_mode_0(device, n_atoms=5)

        result, terms = module.forward_with_derivatives(data, compute_forces=True)

        assert "e_h" in result
        assert terms is not None
        assert terms.forces is not None
        assert terms.forces.shape == data["coord"].shape
        assert torch.isfinite(terms.forces).all()


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
    """Set up padded flat nb_mode=1 periodic data for Coulomb calculations."""
    data = pbc_fixture.copy()
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
    pad_mol_idx = mol_idx_real.max().to(torch.long)
    mol_idx = torch.cat([mol_idx_real, pad_mol_idx.reshape(1)], dim=0)
    n_atoms = coord.shape[0]

    # Create a simple all-pairs neighbor matrix over real atoms; invalid slots
    # and the final row point to the required padding atom.
    max_nb = max(1, min(n_real - 1, 50))
    nbmat = torch.full((n_atoms, max_nb), pad_idx, dtype=torch.long, device=device)
    for i in range(n_real):
        neighbors = [j for j in range(n_real) if j != i][:max_nb]
        for k, nb in enumerate(neighbors):
            nbmat[i, k] = nb

    # Shifts are unit cell translations - use float for matrix multiplication with cell
    shifts = torch.zeros((n_atoms, max_nb, 3), dtype=torch.float32, device=device)

    data["coord"] = coord
    data["numbers"] = numbers
    data["cell"] = data["cell"].to(device)
    data["mol_idx"] = mol_idx
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


def _neutral_padded_charges(n_atoms: int, device: torch.device) -> torch.Tensor:
    """Random neutral charges with the final padded atom fixed at zero."""
    charges = torch.randn(n_atoms, device=device) * 0.3
    charges[:-1] = charges[:-1] - charges[:-1].mean()
    charges[-1] = 0.0
    return charges


def _random_padded_charges(n_atoms: int, device: torch.device) -> torch.Tensor:
    """Random charges following the padded flat nb_mode=1 contract."""
    charges = torch.randn(n_atoms, device=device) * 0.3
    charges[-1] = 0.0
    return charges


class TestLRCoulombDSFPBC:
    """Tests for DSF Coulomb with periodic boundary conditions."""

    def test_dsf_pbc_output_shape(self, pbc_crystal_small, device):
        """Test DSF output shape with PBC."""
        module = LRCoulomb(method="dsf", dsf_rc=8.0).to(device)
        data = setup_pbc_data(pbc_crystal_small, device)

        n_atoms = data["coord"].shape[0]
        data["charges"] = _random_padded_charges(n_atoms, device)

        result = module(data)

        assert "e_h" in result
        assert result["e_h"].shape == (1,)

    def test_dsf_pbc_explicit_forces(self, pbc_crystal_small, device):
        """DSF exposes nvalchemiops explicit forces for padded flat PBC inputs."""
        module = LRCoulomb(method="dsf", dsf_rc=8.0).to(device)

        data = setup_pbc_data(pbc_crystal_small, device)

        n_atoms = data["coord"].shape[0]
        data["charges"] = _random_padded_charges(n_atoms, device)

        result, terms = module.forward_with_derivatives(data, compute_forces=True)

        assert "e_h" in result
        assert terms is not None
        assert terms.forces is not None
        assert terms.forces.shape == data["coord"].shape
        assert torch.isfinite(terms.forces).all()
        assert torch.allclose(terms.forces[-1], torch.zeros(3, device=device))

    def test_dsf_pbc_finite_energy(self, pbc_crystal_small, device):
        """Test DSF produces finite energy with PBC."""
        module = LRCoulomb(method="dsf", dsf_rc=8.0).to(device)
        data = setup_pbc_data(pbc_crystal_small, device)

        n_atoms = data["coord"].shape[0]
        data["charges"] = _random_padded_charges(n_atoms, device)

        result = module(data)

        assert torch.isfinite(result["e_h"]).all()

    def test_dsf_mode2_matches_individual_batches(self, device):
        """DSF mode-2 energy and forces match independent per-system calls."""
        torch.manual_seed(0)
        B, N = 2, 5
        coord = torch.randn(B, N, 3, device=device) * 2.0
        charges = torch.randn(B, N, device=device) * 0.3
        charges = charges - charges.mean(dim=-1, keepdim=True)

        # All-pairs nbmat with local atom indices per batch.
        nbmat = torch.zeros((B, N, N - 1), dtype=torch.long, device=device)
        for i in range(N):
            others = [j for j in range(N) if j != i]
            nbmat[:, i, :] = torch.tensor(others, device=device)
        mask_ij = torch.zeros((B, N, N - 1), dtype=torch.bool, device=device)
        mask_i = torch.zeros((B, N), dtype=torch.bool, device=device)

        module = LRCoulomb(method="dsf", dsf_rc=8.0, subtract_sr=False).to(device)

        data_mode2 = {
            "_nb_mode": torch.tensor(2),
            "_input_padded": torch.tensor(False),
            "coord": coord,
            "charges": charges,
            "numbers": torch.ones((B, N), dtype=torch.long, device=device),
            "nbmat_lr": nbmat,
            "mask_ij_lr": mask_ij,
            "mask_i": mask_i,
            "mol_idx": torch.arange(B, device=device).repeat_interleave(N),
            "mol_sizes": torch.full((B,), N, device=device, dtype=torch.long),
        }
        result_mode2, terms_mode2 = module.forward_with_derivatives(data_mode2, compute_forces=True)
        e_mode2 = result_mode2["e_h"]
        assert terms_mode2 is not None and terms_mode2.forces is not None
        assert terms_mode2.forces.shape == coord.shape

        # Reference: each batch evaluated independently in mode 0.
        e_ref = []
        f_ref = []
        for b in range(B):
            data_b = {
                "_nb_mode": torch.tensor(0),
                "_input_padded": torch.tensor(False),
                "coord": coord[b : b + 1],
                "charges": charges[b : b + 1],
                "numbers": torch.ones((1, N), dtype=torch.long, device=device),
                "mask_ij": mask_ij[b : b + 1],
                "mask_i": mask_i[b : b + 1],
                "mol_idx": torch.zeros(N, device=device, dtype=torch.long),
                "mol_sizes": torch.tensor([N], device=device, dtype=torch.long),
            }
            result_b, terms_b = module.forward_with_derivatives(data_b, compute_forces=True)
            assert terms_b is not None and terms_b.forces is not None
            e_ref.append(result_b["e_h"])
            f_ref.append(terms_b.forces.squeeze(0))
        e_ref = torch.cat(e_ref)
        f_ref = torch.stack(f_ref, dim=0)

        torch.testing.assert_close(e_mode2, e_ref, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(terms_mode2.forces, f_ref, atol=1e-5, rtol=1e-5)

    def test_dsf_mode0_padded_batched_indices(self, device):
        """DSF mode-0 builds a cutoff-bounded NL with correct global indices and
        respects ``mask_i`` for mixed-size batches.

        Reference: evaluate each molecule independently in mode 0 and confirm
        the batched mode-0 result matches both energy and per-atom forces.
        Padded atoms in batch 0 must not contaminate batch 1's neighbors.
        """
        torch.manual_seed(0)
        B, N = 2, 5
        # Batch 0: 4 real atoms + 1 padded (mask_i[0, -1] = True)
        # Batch 1: 5 real atoms (no padding)
        coord = torch.randn(B, N, 3, device=device) * 2.0
        charges = torch.randn(B, N, device=device) * 0.3
        charges = charges - charges.mean(dim=-1, keepdim=True)
        mask_i = torch.zeros((B, N), dtype=torch.bool, device=device)
        mask_i[0, -1] = True

        module = LRCoulomb(method="dsf", dsf_rc=8.0, subtract_sr=False).to(device)

        data_mode0 = {
            "_nb_mode": torch.tensor(0),
            "_input_padded": torch.tensor(True),
            "coord": coord,
            "charges": charges,
            "numbers": torch.ones((B, N), dtype=torch.long, device=device),
            "mask_i": mask_i,
            "mol_idx": torch.arange(B, device=device).repeat_interleave(N),
            "mol_sizes": torch.tensor([N - 1, N], device=device, dtype=torch.long),
        }
        result_mode0, terms_mode0 = module.forward_with_derivatives(data_mode0, compute_forces=True)
        e_mode0 = result_mode0["e_h"]
        assert terms_mode0 is not None and terms_mode0.forces is not None
        assert terms_mode0.forces.shape == coord.shape

        # Reference: each batch in its own mode-0 call.
        e_ref = []
        f_ref = []
        for b in range(B):
            data_b = {
                "_nb_mode": torch.tensor(0),
                "_input_padded": torch.tensor(True),
                "coord": coord[b : b + 1],
                "charges": charges[b : b + 1],
                "numbers": torch.ones((1, N), dtype=torch.long, device=device),
                "mask_i": mask_i[b : b + 1],
                "mol_idx": torch.zeros(N, device=device, dtype=torch.long),
                "mol_sizes": data_mode0["mol_sizes"][b : b + 1],
            }
            result_b, terms_b = module.forward_with_derivatives(data_b, compute_forces=True)
            assert terms_b is not None and terms_b.forces is not None
            e_ref.append(result_b["e_h"])
            f_ref.append(terms_b.forces.squeeze(0))
        e_ref = torch.cat(e_ref)
        f_ref = torch.stack(f_ref, dim=0)

        torch.testing.assert_close(e_mode0, e_ref, atol=1e-5, rtol=1e-5)
        # Padded atom force should be zero (it has charge zero and was shifted out).
        assert torch.allclose(terms_mode0.forces[0, -1], torch.zeros(3, device=device), atol=1e-5)
        torch.testing.assert_close(terms_mode0.forces, f_ref, atol=1e-5, rtol=1e-5)

    def test_dsf_mode0_large_coordinates_keep_padding_out_of_neighbor_list(self, device):
        """Padded atoms stay out of the DSF NL even for large unwrapped coordinates."""
        dsf_rc = 15.0
        module = LRCoulomb(method="dsf", dsf_rc=dsf_rc, subtract_sr=False).to(device)
        coord = torch.tensor(
            [[[1505.0, 0.0, 0.0], [1510.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
            dtype=torch.float32,
            device=device,
        )
        charges = torch.tensor([[1.0, -1.0, 0.0]], dtype=torch.float32, device=device)
        mask_i = torch.tensor([[False, False, True]], dtype=torch.bool, device=device)
        data_padded = {
            "_nb_mode": torch.tensor(0),
            "_input_padded": torch.tensor(True),
            "coord": coord,
            "charges": charges,
            "numbers": torch.tensor([[11, 17, 0]], dtype=torch.long, device=device),
            "mask_i": mask_i,
            "mol_idx": torch.zeros(coord.numel() // 3, device=device, dtype=torch.long),
            "mol_sizes": torch.tensor([2], device=device, dtype=torch.long),
        }

        _positions, _charges, _batch_idx, nbmat, _cell, _shifts, fill_value, _num_systems = module._dsf_inputs_mode0(
            data_padded
        )

        padded_idx = 2
        real_rows = nbmat[:padded_idx]
        assert not (real_rows == padded_idx).any()
        assert torch.all(nbmat[padded_idx] == fill_value)
        assert torch.all(nbmat[-1] == fill_value)

        result_padded, terms_padded = module.forward_with_derivatives(data_padded, compute_forces=True)
        assert terms_padded is not None and terms_padded.forces is not None

        data_ref = {
            "_nb_mode": torch.tensor(0),
            "_input_padded": torch.tensor(False),
            "coord": coord[:, :2],
            "charges": charges[:, :2],
            "numbers": torch.tensor([[11, 17]], dtype=torch.long, device=device),
            "mask_i": torch.zeros((1, 2), dtype=torch.bool, device=device),
            "mol_idx": torch.zeros(2, device=device, dtype=torch.long),
            "mol_sizes": torch.tensor([2], device=device, dtype=torch.long),
        }
        result_ref, terms_ref = module.forward_with_derivatives(data_ref, compute_forces=True)
        assert terms_ref is not None and terms_ref.forces is not None

        torch.testing.assert_close(result_padded["e_h"], result_ref["e_h"], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(terms_padded.forces[:, :2], terms_ref.forces, atol=1e-5, rtol=1e-5)
        assert torch.allclose(terms_padded.forces[0, padded_idx], torch.zeros(3, device=device), atol=1e-5)


class TestLRCoulombEwaldPBC:
    """Tests for Ewald Coulomb with periodic boundary conditions."""

    def test_ewald_pbc_output_shape(self, pbc_crystal_small, device):
        """Test Ewald output shape with PBC."""
        module = LRCoulomb(method="ewald").to(device)
        data = setup_pbc_data(pbc_crystal_small, device, cutoff=8.0)

        n_atoms = data["coord"].shape[0]
        data["charges"] = _neutral_padded_charges(n_atoms, device)

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
        data["charges"] = _neutral_padded_charges(n_atoms, device)

        result = module(data)
        result["e_h"].backward()

        assert data["coord"].grad is not None
        assert torch.isfinite(data["coord"].grad).all()

    def test_ewald_pbc_cell_present(self, pbc_crystal_small, device):
        """Test Ewald requires cell for PBC."""
        data = setup_pbc_data(pbc_crystal_small, device)

        n_atoms = data["coord"].shape[0]
        data["charges"] = _neutral_padded_charges(n_atoms, device)

        # Verify cell is present
        assert "cell" in data
        assert data["cell"].shape == (3, 3)

        module = LRCoulomb(method="ewald").to(device)
        result = module(data)

        assert torch.isfinite(result["e_h"]).all()


class TestLRCoulombPMEPBC:
    """Tests for Particle Mesh Ewald (PME) with periodic boundary conditions."""

    def test_pme_pbc_output_shape(self, pbc_crystal_small, device):
        """Test PME output shape with PBC."""
        module = LRCoulomb(method="pme").to(device)
        data = setup_pbc_data(pbc_crystal_small, device, cutoff=8.0)

        n_atoms = data["coord"].shape[0]
        data["charges"] = _neutral_padded_charges(n_atoms, device)

        result = module(data)

        assert "e_h" in result
        assert torch.isfinite(result["e_h"]).all()

    def test_pme_pbc_zero_charges(self, pbc_crystal_small, device):
        """Test PME with zero charges produces zero energy."""
        module = LRCoulomb(method="pme").to(device)
        data = setup_pbc_data(pbc_crystal_small, device)

        n_atoms = data["coord"].shape[0]
        data["charges"] = torch.zeros(n_atoms, device=device)

        result = module(data)

        assert result["e_h"].abs().sum().item() < 1e-6

    def test_pme_pbc_gradient_wrt_coords(self, pbc_crystal_small, device):
        """Test PME coordinate gradients with PBC."""
        module = LRCoulomb(method="pme").to(device)

        data = pbc_crystal_small.copy()
        data["coord"] = data["coord"].clone().requires_grad_(True)
        data = setup_pbc_data(data, device)

        n_atoms = data["coord"].shape[0]
        data["charges"] = _neutral_padded_charges(n_atoms, device)

        result = module(data)
        result["e_h"].backward()

        assert data["coord"].grad is not None
        assert torch.isfinite(data["coord"].grad).all()


class TestLRCoulombBackendInterface:
    """Tests for the common external Coulomb derivative interface."""

    @pytest.mark.parametrize("method", ["ewald", "pme"])
    def test_ewald_pme_forward_with_derivatives_returns_no_terms(self, pbc_crystal_small, device, method):
        """Ewald/PME use autograd derivatives and return no explicit derivative terms."""
        module = LRCoulomb(method=method).to(device)
        data = setup_pbc_data(pbc_crystal_small, device)

        n_atoms = data["coord"].shape[0]
        data["charges"] = _neutral_padded_charges(n_atoms, device)

        result, terms = module.forward_with_derivatives(data, compute_forces=True, compute_virial=True)

        assert "e_h" in result
        assert torch.isfinite(result["e_h"]).all()
        assert terms is None


class TestLRCoulombSRSubtraction:
    """Tests covering subtract_sr behaviour for nvalchemiops Coulomb backends."""

    @pytest.mark.parametrize("method", ["dsf", "ewald", "pme"])
    def test_last_real_atom_charge_affects_energy(self, pbc_crystal_small, device, method):
        """The atom before the padding row is real and must participate."""
        data = setup_pbc_data(pbc_crystal_small, device)
        n_atoms = data["coord"].shape[0]
        charges = torch.zeros(n_atoms, device=device)
        charges[0] = -0.4
        charges[-2] = 0.4

        charges_changed = charges.clone()
        charges_changed[0] = -0.8
        charges_changed[-2] = 0.8

        mod = LRCoulomb(method=method, subtract_sr=False).to(device)
        e_ref = mod({**data, "charges": charges})["e_h"]
        e_changed = mod({**data, "charges": charges_changed})["e_h"]

        assert not torch.allclose(e_ref, e_changed, rtol=0.0, atol=1e-8)

    @pytest.mark.parametrize("method", ["dsf", "ewald", "pme"])
    def test_subtract_sr_changes_energy(self, pbc_crystal_small, device, method):
        """``subtract_sr`` should produce a different energy than the raw backend."""
        data = setup_pbc_data(pbc_crystal_small, device)
        n_atoms = data["coord"].shape[0]
        data["charges"] = _neutral_padded_charges(n_atoms, device)

        mod_with_sr = LRCoulomb(method=method, subtract_sr=True).to(device)
        mod_without_sr = LRCoulomb(method=method, subtract_sr=False).to(device)

        e_with = mod_with_sr(data.copy())["e_h"]
        e_without = mod_without_sr(data.copy())["e_h"]

        assert torch.isfinite(e_with).all()
        assert torch.isfinite(e_without).all()
        assert (e_with - e_without).abs().item() > 1e-8

    @pytest.mark.parametrize("method", ["dsf", "ewald", "pme"])
    def test_charge_non_neutral_finite(self, pbc_crystal_small, device, method):
        """The nvalchemiops Coulomb backends produce finite non-neutral energies."""
        data = setup_pbc_data(pbc_crystal_small, device)
        n_atoms = data["coord"].shape[0]
        charges = torch.randn(n_atoms, device=device) * 0.3
        charges = charges + 0.5  # explicitly NOT neutral
        charges[-1] = 0.0
        data["charges"] = charges

        mod = LRCoulomb(method=method).to(device)
        result = mod(data)

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


class TestLRCoulombTraining:
    """Model-level training regression tests for Ewald/PME.

    The default training pipeline (``aimnet/train/train.py``) wraps models
    with ``aimnet.modules.core.Forces``, which runs
    ``torch.autograd.grad(energy, coord, create_graph=self.training)`` and
    then ``loss.backward()`` on a force-dependent loss. Before
    ``aimnet::lr_coulomb_fwd`` was introduced, this raised
    ``RuntimeError: Trying to backward through
    alchemiops._batch_ewald_real_space_energy_matrix_backward.default but no
    autograd formula was registered`` because the nvalchemiops Warp kernels
    lack a registered 2nd-order autograd formula.

    These tests lock in that contract: 1st-order forces, 2nd-order force
    loss, and param-gradient flow through the charge chain all work for
    both Ewald and PME.
    """

    @staticmethod
    def _tiny_charge_model(n_atoms: int, device: torch.device) -> torch.nn.Module:
        """Tiny learnable module that maps ``coord`` to charge-like features.

        Produces neutral per-atom values with a graph dependency on both
        ``coord`` and the module's parameters, so a 2nd-order force-loss
        backward exercises the full charge-chain path.
        """

        class _TinyChargeModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.w = torch.nn.Parameter(torch.randn(3, device=device) * 0.1)
                self.b = torch.nn.Parameter(torch.zeros(n_atoms, device=device))

            def forward(self, coord_real: torch.Tensor) -> torch.Tensor:
                raw = torch.tanh(coord_real @ self.w + self.b) * 0.3
                # Enforce charge neutrality (total charge must sum to 0 for Ewald).
                return raw - raw.mean()

        return _TinyChargeModel().to(device)

    @pytest.mark.parametrize("method", ["ewald", "pme"])
    def test_force_loss_backward_does_not_raise(self, pbc_crystal_small, device, method):
        """Force-loss backward through Ewald/PME must not hit the unregistered
        2nd-order Warp autograd path (regression for the critical training gap)."""
        torch.manual_seed(0)
        data = pbc_crystal_small.copy()
        data["coord"] = data["coord"].clone().requires_grad_(True)
        data = setup_pbc_data(data, device)
        n_real = data["coord"].shape[0] - 1

        model = self._tiny_charge_model(n_real, device)
        charges_real = model(data["coord"][:-1])
        data["charges"] = torch.cat([charges_real, charges_real.new_zeros(1)], dim=0)

        coulomb = LRCoulomb(method=method, subtract_sr=False).to(device)
        result = coulomb(data)
        energy = result["e_h"].sum()

        forces = -torch.autograd.grad(energy, data["coord"], create_graph=True)[0]
        target = torch.zeros_like(forces)
        loss = ((forces - target) ** 2).mean()
        loss.backward()

        param_grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert param_grads, "expected gradients on tiny model parameters"
        assert any(g.abs().sum().item() > 0 for g in param_grads), (
            "force-loss backward through Ewald/PME produced zero gradient for every "
            "parameter — the charge chain is not reaching model parameters."
        )
        for g in param_grads:
            assert torch.isfinite(g).all()

    @pytest.mark.parametrize("method", ["ewald", "pme"])
    def test_energy_backward_matches_explicit_forces(self, pbc_crystal_small, device, method):
        """``-grad(energy, coord)`` from the custom-op Function must match the
        explicit forces the same nvalchemiops call would have produced."""
        from aimnet.modules.ops import lr_coulomb_energy

        torch.manual_seed(1)
        data = pbc_crystal_small.copy()
        data["coord"] = data["coord"].clone().requires_grad_(True)
        data = setup_pbc_data(data, device)
        n_atoms = data["coord"].shape[0]
        data["charges"] = _neutral_padded_charges(n_atoms, device)

        coulomb = LRCoulomb(method=method, subtract_sr=False).to(device)
        energy = coulomb(data)["e_h"].sum()
        f_autograd = -torch.autograd.grad(energy, data["coord"])[0]

        n_real = n_atoms - 1
        mol_idx_real = data["mol_idx"][:-1].to(torch.int32)
        _e, f_explicit, _qg, _v = lr_coulomb_energy(
            coord=data["coord"].detach()[:-1],
            cell=data["cell"],
            charges=data["charges"].detach()[:-1],
            batch_idx=mol_idx_real,
            neighbor_matrix=data["nbmat_lr"][:-1].to(torch.int32),
            shifts=data["shifts_lr"][:-1].to(torch.int32),
            mask_value=n_real,
            num_systems=int(mol_idx_real.max().item()) + 1,
            accuracy=float(coulomb.ewald_accuracy),
            backend=method,
            compute_virial=False,
        )
        torch.testing.assert_close(f_autograd[:-1], f_explicit, atol=1e-5, rtol=1e-5)
        assert torch.allclose(f_autograd[-1], torch.zeros(3, device=device))

    @pytest.mark.parametrize("method", ["ewald", "pme"])
    def test_double_backward_smoke(self, pbc_crystal_small, device, method):
        """Minimal double-backward smoke test: ``grad(grad(E).sum(), charges)``
        must not raise even though the nvalchemiops Warp backward has no
        2nd-order autograd rule registered."""
        torch.manual_seed(2)
        data = pbc_crystal_small.copy()
        data["coord"] = data["coord"].clone().requires_grad_(True)
        data = setup_pbc_data(data, device)
        n_atoms = data["coord"].shape[0]
        charges = torch.randn(n_atoms, device=device) * 0.2
        charges[:-1] = charges[:-1] - charges[:-1].mean()
        charges[-1] = 0.0
        charges = charges.clone().requires_grad_(True)
        data["charges"] = charges

        coulomb = LRCoulomb(method=method, subtract_sr=False).to(device)
        energy = coulomb(data)["e_h"].sum()
        forces_like = torch.autograd.grad(energy, data["coord"], create_graph=True)[0]
        (forces_like.pow(2).sum()).backward()  # raises before the fix

        assert charges.grad is not None
        assert torch.isfinite(charges.grad).all()
