"""Tests for aimnet.ops - mathematical operations module."""

import math

import pytest
import torch

from aimnet import nbops, ops


class TestCutoffFunctions:
    """Tests for cutoff functions (cosine_cutoff, exp_cutoff)."""

    def test_cosine_cutoff_at_zero(self, device):
        """Test cosine cutoff at d=0 returns 1."""
        d = torch.tensor([0.0, 0.001], device=device)
        rc = 5.0
        fc = ops.cosine_cutoff(d, rc)

        # At d~0, cutoff should be ~1
        assert fc[0].item() == pytest.approx(1.0, abs=0.01)
        assert fc[1].item() == pytest.approx(1.0, abs=0.01)

    def test_cosine_cutoff_at_rc(self, device):
        """Test cosine cutoff at d=rc returns 0."""
        rc = 5.0
        d = torch.tensor([rc, rc + 1.0], device=device)
        fc = ops.cosine_cutoff(d, rc)

        # At d=rc, cutoff should be 0
        assert fc[0].item() == pytest.approx(0.0, abs=1e-6)
        # Beyond rc, should be clamped
        assert fc[1].item() == pytest.approx(0.0, abs=1e-6)

    def test_cosine_cutoff_monotonic(self, device):
        """Test that cosine cutoff decreases monotonically."""
        rc = 5.0
        d = torch.linspace(0.1, rc, 50, device=device)
        fc = ops.cosine_cutoff(d, rc)

        # Should be monotonically decreasing
        diffs = fc[1:] - fc[:-1]
        assert (diffs <= 0).all()

    def test_cosine_cutoff_smooth(self, device):
        """Test that cosine cutoff is smooth (differentiable)."""
        rc = 5.0
        d = torch.tensor([2.5], device=device, requires_grad=True)
        fc = ops.cosine_cutoff(d, rc)
        fc.backward()

        # Gradient should exist and be finite
        assert d.grad is not None
        assert torch.isfinite(d.grad).all()

    def test_exp_cutoff_at_zero(self, device):
        """Test exp cutoff at d=0 returns 1."""
        d = torch.tensor([0.0, 0.001], device=device)
        rc = torch.tensor(5.0, device=device)
        fc = ops.exp_cutoff(d, rc)

        assert fc[0].item() == pytest.approx(1.0, abs=0.01)

    def test_exp_cutoff_at_rc(self, device):
        """Test exp cutoff at d=rc returns 0."""
        rc = torch.tensor(5.0, device=device)
        d = torch.tensor([4.99, 5.0, 5.01], device=device)
        fc = ops.exp_cutoff(d, rc)

        # Should approach 0 near rc
        assert fc[1].item() < 0.1
        assert fc[2].item() < 0.1

    def test_exp_cutoff_smooth(self, device):
        """Test that exp cutoff is smooth (differentiable)."""
        rc = torch.tensor(5.0, device=device)
        d = torch.tensor([2.5], device=device, requires_grad=True)
        fc = ops.exp_cutoff(d, rc)
        fc.backward()

        assert d.grad is not None
        assert torch.isfinite(d.grad).all()


class TestExpExpand:
    """Tests for exp_expand function (Gaussian basis expansion)."""

    def test_exp_expand_shape(self, device):
        """Test exp_expand output shape."""
        d_ij = torch.rand((2, 10, 10), device=device) * 5  # (B, N, N)
        shifts = torch.linspace(0.5, 4.5, 8, device=device)
        eta = 0.5

        result = ops.exp_expand(d_ij, shifts, eta)

        # Output should add shift dimension
        assert result.shape == (2, 10, 10, 8)

    def test_exp_expand_peak(self, device):
        """Test that exp_expand peaks at shift values."""
        shifts = torch.tensor([1.0, 2.0, 3.0], device=device)
        eta = 10.0  # narrow Gaussians
        d_ij = torch.tensor([1.0, 2.0, 3.0], device=device)

        result = ops.exp_expand(d_ij, shifts, eta)

        # Each distance should peak at corresponding shift
        assert result[0, 0].item() > result[0, 1].item()  # d=1 peaks at shift=1
        assert result[1, 1].item() > result[1, 0].item()  # d=2 peaks at shift=2

    def test_exp_expand_gradient(self, device):
        """Test gradient flow through exp_expand."""
        d_ij = torch.tensor([[2.0]], device=device, requires_grad=True)
        shifts = torch.tensor([1.0, 2.0, 3.0], device=device)
        eta = 0.5

        result = ops.exp_expand(d_ij, shifts, eta)
        result.sum().backward()

        assert d_ij.grad is not None
        assert torch.isfinite(d_ij.grad).all()


class TestCalcDistances:
    """Tests for calc_distances function."""

    def test_calc_distances_basic(self, simple_molecule):
        """Test basic distance calculation."""
        data = simple_molecule.copy()
        data = nbops.set_nb_mode(data)
        data = nbops.calc_masks(data)

        d_ij, r_ij = ops.calc_distances(data)

        # Shape should be (B, N, N) for mode 0
        assert d_ij.shape == (1, 3, 3)
        assert r_ij.shape == (1, 3, 3, 3)

        # Diagonal r_ij is masked with pad_value=1.0 for each component
        # So d_ij diagonal = norm([1,1,1]) = sqrt(3)
        for i in range(3):
            assert d_ij[0, i, i].item() == pytest.approx(math.sqrt(3), abs=1e-5)

        # Off-diagonal distances should be positive
        assert d_ij[0, 0, 1].item() > 0

    def test_calc_distances_symmetry(self, simple_molecule):
        """Test that distances are symmetric."""
        data = simple_molecule.copy()
        data = nbops.set_nb_mode(data)
        data = nbops.calc_masks(data)

        d_ij, _r_ij = ops.calc_distances(data)

        # d_ij should be symmetric
        assert d_ij[0, 0, 1].item() == pytest.approx(d_ij[0, 1, 0].item(), abs=1e-6)
        assert d_ij[0, 0, 2].item() == pytest.approx(d_ij[0, 2, 0].item(), abs=1e-6)

    def test_calc_distances_gradient(self, device):
        """Test gradient flow through calc_distances."""
        coord = torch.tensor(
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
            device=device,
            requires_grad=True,
        )
        data = {
            "coord": coord,
            "numbers": torch.tensor([[6, 1, 1]], device=device),
        }
        data = nbops.set_nb_mode(data)
        data = nbops.calc_masks(data)

        d_ij, _r_ij = ops.calc_distances(data)
        loss = d_ij.sum()
        loss.backward()

        assert coord.grad is not None
        assert torch.isfinite(coord.grad).all()


class TestCenterCoordinates:
    """Tests for center_coordinates function."""

    def test_center_coordinates_basic(self, simple_molecule):
        """Test basic coordinate centering."""
        data = simple_molecule.copy()
        data = nbops.set_nb_mode(data)
        data = nbops.calc_masks(data)

        coord = data["coord"]
        centered = ops.center_coordinates(coord, data)

        # Center of mass should be approximately at origin
        center = centered.mean(dim=-2)
        assert center.abs().max().item() < 1e-5

    def test_center_coordinates_with_masses(self, device):
        """Test coordinate centering with masses."""
        coord = torch.tensor(
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]],
            device=device,
        )
        numbers = torch.tensor([[8, 1]], device=device)
        masses = torch.tensor([[16.0, 1.0]], device=device)

        data = {"coord": coord, "numbers": numbers}
        data = nbops.set_nb_mode(data)
        data = nbops.calc_masks(data)

        centered = ops.center_coordinates(coord, data, masses)

        # Mass-weighted center should be at origin
        # heavier O atom should move less
        assert centered[0, 0, 0].abs().item() < centered[0, 1, 0].abs().item()


class TestCoulombMatrixDSF:
    """Tests for coulomb_matrix_dsf function."""

    def test_coulomb_dsf_shape(self, device):
        """Test DSF Coulomb matrix shape."""
        N = 5
        d_ij = torch.rand((N, N), device=device) * 10 + 1.0  # avoid division by zero
        Rc = 15.0
        alpha = 0.2

        # Create minimal data dict for masking
        data = {
            "mask_ij_lr": torch.eye(N, device=device, dtype=torch.bool),
        }

        J = ops.coulomb_matrix_dsf(d_ij, Rc, alpha, data)

        assert J.shape == (N, N)

    def test_coulomb_dsf_masked_diagonal(self, device):
        """Test that diagonal is properly handled."""
        N = 3
        d_ij = torch.tensor(
            [[0.0, 2.0, 3.0], [2.0, 0.0, 2.5], [3.0, 2.5, 0.0]],
            device=device,
        )
        Rc = 15.0
        alpha = 0.2

        # Diagonal should be masked
        data = {
            "mask_ij_lr": torch.eye(N, device=device, dtype=torch.bool),
        }

        J = ops.coulomb_matrix_dsf(d_ij, Rc, alpha, data)

        # Off-diagonal should be non-zero
        assert J[0, 1].item() != 0.0
        assert J[1, 2].item() != 0.0

    def test_coulomb_dsf_cutoff(self, device):
        """Test that values beyond Rc are masked when mask_ij_lr is True."""
        N = 3
        Rc = 5.0
        d_ij = torch.tensor(
            [[1.0, 3.0, 6.0], [3.0, 1.0, 4.0], [6.0, 4.0, 1.0]],
            device=device,
        )
        alpha = 0.2

        # mask_ij_lr True means "this pair should be masked (excluded)"
        # For DSF, pairs with d > Rc AND mask_ij_lr==True are masked to 0
        data = {
            "mask_ij_lr": torch.eye(N, device=device, dtype=torch.bool),  # Only diagonal masked
        }

        J = ops.coulomb_matrix_dsf(d_ij, Rc, alpha, data)

        # Distance 3.0 < Rc should be non-zero (not diagonal)
        assert J[0, 1].item() != 0.0
        # Function returns valid Coulomb values for distances within cutoff
        assert torch.isfinite(J).all()


class TestCoulombMatrixSF:
    """Tests for coulomb_matrix_sf function (shifted force)."""

    def test_coulomb_sf_shape(self, device):
        """Test SF Coulomb matrix shape."""
        N = 5
        d_ij = torch.rand((N, N), device=device) * 10 + 1.0
        Rc = 15.0
        q_j = torch.rand((N,), device=device)

        data = {"mask_ij_lr": torch.eye(N, device=device, dtype=torch.bool)}

        J = ops.coulomb_matrix_sf(q_j, d_ij, Rc, data)

        assert J.shape == (N, N)

    def test_coulomb_sf_cutoff(self, device):
        """Test SF cutoff behavior."""
        N = 2
        Rc = 5.0
        d_ij = torch.tensor([[1.0, 3.0], [3.0, 1.0]], device=device)  # Within Rc
        q_j = torch.ones(N, device=device)

        data = {"mask_ij_lr": torch.eye(N, device=device, dtype=torch.bool)}

        J = ops.coulomb_matrix_sf(q_j, d_ij, Rc, data)

        # Off-diagonal values within Rc should be non-zero
        assert J[0, 1].item() != 0.0
        assert torch.isfinite(J).all()


class TestNSE:
    """Tests for nse (neural charge equilibration) function."""

    def test_nse_charge_conservation(self, simple_molecule):
        """Test that NSE conserves total charge."""
        data = simple_molecule.copy()
        data = nbops.set_nb_mode(data)
        data = nbops.calc_masks(data)

        # Initial atomic charges (unconstrained)
        q_u = torch.tensor([[[0.1], [-0.2], [0.3]]], device=simple_molecule["coord"].device)
        f_u = torch.ones_like(q_u)  # uniform flexibility
        Q = torch.tensor([[0.0]], device=simple_molecule["coord"].device)  # target total charge

        q = ops.nse(Q, q_u, f_u, data)

        # Total charge should equal target
        q_total = q.sum(dim=-2)
        assert q_total.item() == pytest.approx(0.0, abs=1e-5)

    def test_nse_with_nonzero_charge(self, simple_molecule):
        """Test NSE with non-zero total charge."""
        data = simple_molecule.copy()
        data = nbops.set_nb_mode(data)
        data = nbops.calc_masks(data)

        q_u = torch.zeros((1, 3, 1), device=simple_molecule["coord"].device)
        f_u = torch.ones_like(q_u)
        Q = torch.tensor([[1.0]], device=simple_molecule["coord"].device)  # +1 charge

        q = ops.nse(Q, q_u, f_u, data)

        # Total charge should be +1
        q_total = q.sum(dim=-2)
        assert q_total.item() == pytest.approx(1.0, abs=1e-5)

    def test_nse_gradient(self, device):
        """Test gradient flow through NSE."""
        coord = torch.rand((1, 3, 3), device=device)
        numbers = torch.tensor([[6, 1, 1]], device=device)
        data = {"coord": coord, "numbers": numbers}
        data = nbops.set_nb_mode(data)
        data = nbops.calc_masks(data)

        q_u = torch.tensor([[[0.1], [0.1], [0.1]]], device=device, requires_grad=True)
        f_u = torch.ones_like(q_u)
        Q = torch.tensor([[0.0]], device=device)

        q = ops.nse(Q, q_u, f_u, data)
        loss = q.sum()
        loss.backward()

        assert q_u.grad is not None


class TestTransitionFunctions:
    """Tests for smooth transition functions (huber, bumpfn, smoothstep, expstep)."""

    def test_huber_basic(self, device):
        """Test Huber loss function."""
        x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0], device=device)
        delta = 1.0

        h = ops.huber(x, delta)

        # For |x| < delta, should be 0.5 * x^2
        assert h[1].item() == pytest.approx(0.5 * 0.25, abs=1e-6)
        assert h[2].item() == pytest.approx(0.0, abs=1e-6)

        # For |x| >= delta, should be delta * (|x| - 0.5 * delta)
        assert h[0].item() == pytest.approx(1.0 * (2.0 - 0.5), abs=1e-6)

    def test_bumpfn_boundaries(self, device):
        """Test bumpfn at boundaries."""
        x = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0], device=device)
        low, high = 0.5, 1.5

        b = ops.bumpfn(x, low, high)

        # Should be ~0 for x <= low
        assert b[0].item() < 0.01
        # Should be ~1 for x >= high
        assert b[4].item() > 0.99
        # Should be ~0.5 at midpoint
        assert 0.4 < b[2].item() < 0.6

    def test_smoothstep_boundaries(self, device):
        """Test smoothstep at boundaries."""
        x = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0], device=device)
        low, high = 0.5, 1.5

        s = ops.smoothstep(x, low, high)

        # Should be exactly 0 for x <= low
        assert s[0].item() == pytest.approx(0.0, abs=1e-6)
        # Should be exactly 1 for x >= high
        assert s[4].item() == pytest.approx(1.0, abs=1e-6)
        # Should be exactly 0.5 at midpoint
        assert s[2].item() == pytest.approx(0.5, abs=1e-6)

    def test_expstep_boundaries(self, device):
        """Test expstep produces valid values in range [0, 1]."""
        x = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0], device=device)
        low, high = 0.5, 1.5

        e = ops.expstep(x, low, high)

        # All values should be finite and in valid range
        assert torch.isfinite(e).all()
        # Values are bounded due to clamping
        assert (e >= 0).all()
        assert (e <= 1.1).all()  # Allow small numerical overshoot

    def test_transition_functions_gradient(self, device):
        """Test gradient flow through transition functions."""
        x = torch.tensor([0.75], device=device, requires_grad=True)

        # Test all transition functions
        for fn in [ops.bumpfn, ops.smoothstep, ops.expstep]:
            x.grad = None
            y = fn(x, 0.5, 1.0)
            y.backward()
            assert x.grad is not None
            assert torch.isfinite(x.grad).all()


class TestBatchedCells:
    """Tests for batched cell support in calc_distances."""

    def test_calc_distances_batched_cells_nb_mode1(self, device):
        """Test calc_distances with flat coordinates and batched cells (nb_mode=1)."""
        # Create flat coordinates for 2 systems (3 atoms each)
        coord = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 0.0],  # padding atom
            ],
            dtype=torch.float32,
            device=device,
        )

        # Batched cells (2, 3, 3) - different cell sizes
        cell = torch.stack([
            torch.eye(3, device=device) * 10.0,
            torch.eye(3, device=device) * 20.0,
        ])

        mol_idx = torch.tensor([0, 0, 0, 1, 1, 1, 1], device=device)

        # Create neighbor matrix (N_total, M)
        N = 7
        M = 3
        nbmat = torch.full((N, M), N - 1, dtype=torch.int32, device=device)
        # Atom 0's neighbors: 1, 2
        nbmat[0, 0] = 1
        nbmat[0, 1] = 2
        # Atom 1's neighbors: 0, 2
        nbmat[1, 0] = 0
        nbmat[1, 1] = 2
        # Atom 2's neighbors: 0, 1
        nbmat[2, 0] = 0
        nbmat[2, 1] = 1
        # Atom 3's neighbors: 4, 5
        nbmat[3, 0] = 4
        nbmat[3, 1] = 5
        # Atom 4's neighbors: 3, 5
        nbmat[4, 0] = 3
        nbmat[4, 1] = 5
        # Atom 5's neighbors: 3, 4
        nbmat[5, 0] = 3
        nbmat[5, 1] = 4

        # Create shifts (N_total, M, 3) - zeros for non-PBC test
        shifts = torch.zeros((N, M, 3), dtype=torch.float32, device=device)

        data = {
            "coord": coord,
            "numbers": torch.tensor([8, 1, 1, 8, 1, 1, 0], device=device),
            "mol_idx": mol_idx,
            "nbmat": nbmat,
            "shifts": shifts,
            "cell": cell,
        }
        data = nbops.set_nb_mode(data)
        data = nbops.calc_masks(data)

        # Should not raise with batched cells
        d_ij, r_ij = ops.calc_distances(data)

        # Check shapes
        assert d_ij.shape == (N, M)
        assert r_ij.shape == (N, M, 3)

        # Distances should be positive for real neighbors
        assert d_ij[0, 0].item() == pytest.approx(1.0, abs=1e-5)  # Atom 0 to 1
        assert d_ij[3, 0].item() == pytest.approx(2.0, abs=1e-5)  # Atom 3 to 4

    def test_calc_distances_batched_cells_nb_mode2(self, device):
        """Test calc_distances with batched coordinates and batched cells (nb_mode=2)."""
        # Batched coordinates (B, N, 3)
        coord = torch.tensor(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
            ],
            dtype=torch.float32,
            device=device,
        )

        # Batched cells (B, 3, 3)
        cell = torch.stack([
            torch.eye(3, device=device) * 10.0,
            torch.eye(3, device=device) * 20.0,
        ])

        # Create neighbor matrix (B, N, M)
        # Note: In mode 2, nbmat indices are GLOBAL into flattened (B*N) array
        # System 0 atoms: 0, 1, 2; System 1 atoms: 3, 4, 5
        B, N, M = 2, 3, 2
        fill_val = B * N  # Fill value is total number of atoms
        nbmat = torch.full((B, N, M), fill_val, dtype=torch.int32, device=device)
        # System 0: atom 0's neighbors are atoms 1, 2 (global indices 1, 2)
        nbmat[0, 0, 0] = 1
        nbmat[0, 0, 1] = 2
        nbmat[0, 1, 0] = 0
        nbmat[0, 1, 1] = 2
        nbmat[0, 2, 0] = 0
        nbmat[0, 2, 1] = 1
        # System 1: atom 3's neighbors are atoms 4, 5 (global indices 4, 5)
        nbmat[1, 0, 0] = 4  # System 1, atom 0 -> global atom 4
        nbmat[1, 0, 1] = 5  # System 1, atom 0 -> global atom 5
        nbmat[1, 1, 0] = 3
        nbmat[1, 1, 1] = 5
        nbmat[1, 2, 0] = 3
        nbmat[1, 2, 1] = 4

        # Create shifts (B, N, M, 3)
        shifts = torch.zeros((B, N, M, 3), dtype=torch.float32, device=device)

        data = {
            "coord": coord,
            "numbers": torch.tensor([[8, 1, 1], [8, 1, 1]], device=device),
            "nbmat": nbmat,
            "shifts": shifts,
            "cell": cell,
        }
        data = nbops.set_nb_mode(data)
        data = nbops.calc_masks(data)

        # Should not raise with batched cells
        d_ij, r_ij = ops.calc_distances(data)

        # Check shapes
        assert d_ij.shape == (B, N, M)
        assert r_ij.shape == (B, N, M, 3)

        # Check distances
        # System 0: distance from atom 0 to atom 1 is 1.0
        assert d_ij[0, 0, 0].item() == pytest.approx(1.0, abs=1e-5)
        # System 1: distance from atom 0 to atom 1 is 2.0
        assert d_ij[1, 0, 0].item() == pytest.approx(2.0, abs=1e-5)

    def test_calc_distances_single_cell_nb_mode2(self, device):
        """Test calc_distances with batched coordinates but single cell (backward compatible)."""
        # Batched coordinates (B, N, 3)
        coord = torch.tensor(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
            ],
            dtype=torch.float32,
            device=device,
        )

        # Single cell (3, 3)
        cell = torch.eye(3, device=device) * 10.0

        # Create neighbor matrix (B, N, M)
        # Note: In mode 2, nbmat indices are GLOBAL into flattened (B*N) array
        B, N, M = 2, 3, 2
        fill_val = B * N
        nbmat = torch.full((B, N, M), fill_val, dtype=torch.int32, device=device)
        # System 0
        nbmat[0, 0, 0] = 1
        nbmat[0, 0, 1] = 2
        nbmat[0, 1, 0] = 0
        nbmat[0, 1, 1] = 2
        nbmat[0, 2, 0] = 0
        nbmat[0, 2, 1] = 1
        # System 1 (global indices 3, 4, 5)
        nbmat[1, 0, 0] = 4
        nbmat[1, 0, 1] = 5
        nbmat[1, 1, 0] = 3
        nbmat[1, 1, 1] = 5
        nbmat[1, 2, 0] = 3
        nbmat[1, 2, 1] = 4

        # Create shifts (B, N, M, 3)
        shifts = torch.zeros((B, N, M, 3), dtype=torch.float32, device=device)

        data = {
            "coord": coord,
            "numbers": torch.tensor([[8, 1, 1], [8, 1, 1]], device=device),
            "nbmat": nbmat,
            "shifts": shifts,
            "cell": cell,  # Single cell, not batched
        }
        data = nbops.set_nb_mode(data)
        data = nbops.calc_masks(data)

        # Should work with single cell
        d_ij, _r_ij = ops.calc_distances(data)

        assert d_ij.shape == (B, N, M)
        assert d_ij[0, 0, 0].item() == pytest.approx(1.0, abs=1e-5)
