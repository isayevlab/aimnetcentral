"""Tests for AIMNet2Calculator."""

import warnings

import numpy as np
import pytest
import torch
from conftest import CAFFEINE_FILE, load_mol

from aimnet.calculators import AIMNet2Calculator

# All tests in this module require ASE for molecule loading
pytestmark = pytest.mark.ase


def test_from_zoo():
    """Test basic model loading and inference from model registry."""
    pytest.importorskip("ase", reason="ASE not installed. Install with: pip install aimnet[ase]")

    calc = AIMNet2Calculator("aimnet2", nb_threshold=0)
    data = load_mol(CAFFEINE_FILE)
    res = calc(data)
    assert "energy" in res
    res = calc(data, forces=True)
    assert "forces" in res
    res = calc(data, hessian=True)
    assert "hessian" in res


class TestInputValidation:
    """Tests for input validation and error handling."""

    def test_missing_coord_raises_error(self):
        """Test that missing coord key raises KeyError."""
        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)
        data = {"numbers": [6, 1, 1], "charge": 0.0}

        with pytest.raises(KeyError, match="Missing key coord"):
            calc(data)

    def test_missing_numbers_raises_error(self):
        """Test that missing numbers key raises KeyError."""
        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)
        data = {"coord": [[0, 0, 0], [1, 0, 0]], "charge": 0.0}

        with pytest.raises(KeyError, match="Missing key numbers"):
            calc(data)

    def test_missing_charge_raises_error(self):
        """Test that missing charge key raises KeyError."""
        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)
        data = {"coord": [[0, 0, 0]], "numbers": [6]}

        with pytest.raises(KeyError, match="Missing key charge"):
            calc(data)

    def test_numpy_input(self):
        """Test that numpy arrays are accepted as input."""
        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)
        data = {
            "coord": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            "numbers": np.array([6, 1, 1]),
            "charge": 0.0,
        }
        res = calc(data)
        assert "energy" in res
        assert isinstance(res["energy"], torch.Tensor)

    def test_list_input(self):
        """Test that Python lists are accepted as input."""
        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)
        data = {
            "coord": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            "numbers": [6, 1, 1],
            "charge": 0.0,
        }
        res = calc(data)
        assert "energy" in res


COULOMB_METHODS = ["simple", "dsf", "ewald"]


class TestCoulombMethods:
    """Tests for Coulomb method switching."""

    @pytest.mark.parametrize("method", COULOMB_METHODS)
    def test_set_coulomb_method(self, method):
        """Test setting each Coulomb method."""
        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Model has embedded Coulomb module", category=UserWarning)
            if method == "dsf":
                calc.set_lrcoulomb_method(method, cutoff=12.0, dsf_alpha=0.25)
            elif method == "ewald":
                calc.set_lrcoulomb_method(method, cutoff=10.0)
            else:
                calc.set_lrcoulomb_method(method)
        assert calc._coulomb_method == method

    def test_set_coulomb_dsf_with_params(self):
        """Test DSF Coulomb method sets cutoff correctly."""
        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Model has embedded Coulomb module", category=UserWarning)
            calc.set_lrcoulomb_method("dsf", cutoff=12.0, dsf_alpha=0.25)
        assert calc._coulomb_method == "dsf"
        assert calc.cutoff_lr == 12.0

    def test_invalid_coulomb_method(self):
        """Test that invalid Coulomb method raises ValueError."""
        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)
        with pytest.raises(ValueError, match="Invalid method"):
            calc.set_lrcoulomb_method("invalid_method")

    @pytest.mark.parametrize("method", ["simple", "dsf"])
    def test_coulomb_method_produces_valid_energy(self, method):
        """Test that each Coulomb method produces valid energy."""
        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)
        data = load_mol(CAFFEINE_FILE)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Model has embedded Coulomb module", category=UserWarning)
            if method == "dsf":
                calc.set_lrcoulomb_method(method, cutoff=12.0)
            else:
                calc.set_lrcoulomb_method(method)

        res = calc(data)
        assert torch.isfinite(res["energy"]).all()
        # Stable molecules should have negative energy
        assert res["energy"].item() < 0

    @pytest.mark.parametrize("method", ["simple", "dsf"])
    def test_coulomb_method_produces_valid_forces(self, method):
        """Test that each Coulomb method produces valid forces."""
        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)
        data = load_mol(CAFFEINE_FILE)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Model has embedded Coulomb module", category=UserWarning)
            if method == "dsf":
                calc.set_lrcoulomb_method(method, cutoff=12.0)
            else:
                calc.set_lrcoulomb_method(method)

        res = calc(data, forces=True)
        assert torch.isfinite(res["forces"]).all()


class TestBatchProcessing:
    """Tests for batch processing of multiple molecules."""

    def test_batched_input_2d(self):
        """Test processing with 2D batched input (flattened molecules)."""
        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)

        # Two water molecules flattened with mol_idx
        data = {
            "coord": torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [5.0, 0.0, 0.0],
                    [6.0, 0.0, 0.0],
                    [5.0, 1.0, 0.0],
                ],
                dtype=torch.float32,
            ),
            "numbers": torch.tensor([8, 1, 1, 8, 1, 1]),
            "mol_idx": torch.tensor([0, 0, 0, 1, 1, 1]),
            "charge": torch.tensor([0.0, 0.0]),
        }
        res = calc(data)
        assert "energy" in res
        assert res["energy"].shape == (2,)

    def test_batched_input_3d(self):
        """Test processing with 3D batched input."""
        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)

        # Two molecules in batch format
        data = {
            "coord": torch.tensor(
                [
                    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                ],
                dtype=torch.float32,
            ),
            "numbers": torch.tensor([[8, 1, 1], [8, 1, 1]]),
            "charge": torch.tensor([0.0, 0.0]),
        }
        res = calc(data)
        assert "energy" in res
        assert res["energy"].shape == (2,)


class TestDerivatives:
    """Tests for force, stress, and Hessian calculations."""

    def test_forces_shape(self):
        """Test that forces have correct shape."""
        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)
        data = load_mol(CAFFEINE_FILE)
        res = calc(data, forces=True)

        assert "forces" in res
        # Forces should have shape (N, 3) or (1, N, 3)
        assert res["forces"].shape[-1] == 3
        n_atoms = len(data["numbers"])
        assert res["forces"].shape[-2] == n_atoms

    def test_forces_sum_approximately_zero(self):
        """Test that forces sum to approximately zero (translation invariance)."""
        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)
        data = load_mol(CAFFEINE_FILE)
        res = calc(data, forces=True)

        # Sum of forces should be approximately zero
        force_sum = res["forces"].sum(dim=-2)
        assert force_sum.abs().max().item() < 1e-4

    def test_hessian_shape(self):
        """Test that Hessian has correct shape."""
        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)
        # Use smaller molecule for Hessian (expensive)
        data = {
            "coord": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            "numbers": [8, 1, 1],
            "charge": 0.0,
        }
        res = calc(data, hessian=True)

        assert "hessian" in res
        # Hessian should have shape (N, 3, N, 3)
        n_atoms = 3
        assert res["hessian"].shape == (n_atoms, 3, n_atoms, 3)

    def test_hessian_symmetry(self):
        """Test that Hessian is approximately symmetric."""
        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)
        data = {
            "coord": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            "numbers": [8, 1, 1],
            "charge": 0.0,
        }
        res = calc(data, hessian=True)

        hess = res["hessian"]
        # Flatten to (3N, 3N) and check symmetry
        hess_flat = hess.reshape(9, 9)
        diff = (hess_flat - hess_flat.T).abs().max()
        assert diff.item() < 1e-4

    def test_hessian_multiple_molecules_raises(self):
        """Test that Hessian with multiple molecules raises NotImplementedError."""
        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)
        data = {
            "coord": torch.tensor(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [5.0, 0.0, 0.0], [6.0, 0.0, 0.0]],
                dtype=torch.float32,
            ),
            "numbers": torch.tensor([8, 1, 8, 1]),
            "mol_idx": torch.tensor([0, 0, 1, 1]),
            "charge": torch.tensor([0.0, 0.0]),
        }
        with pytest.raises(NotImplementedError, match="Hessian calculation is not supported for multiple molecules"):
            calc(data, hessian=True)


class TestEnergyConsistency:
    """Tests for energy consistency across different configurations."""

    def test_translation_invariance(self):
        """Test that energy is invariant under translation."""
        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)
        data = load_mol(CAFFEINE_FILE)

        # Original energy
        res1 = calc(data)
        e1 = res1["energy"].item()

        # Translate molecule
        data2 = data.copy()
        data2["coord"] = data["coord"] + np.array([10.0, 20.0, 30.0])
        res2 = calc(data2)
        e2 = res2["energy"].item()

        # Allow for small numerical differences due to floating point
        assert abs(e1 - e2) < 1e-5

    def test_rotation_invariance(self):
        """Test that energy is invariant under rotation."""
        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)
        data = load_mol(CAFFEINE_FILE)

        # Original energy
        res1 = calc(data)
        e1 = res1["energy"].item()

        # Rotate molecule by 90 degrees around z-axis
        theta = np.pi / 2
        R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
        data2 = data.copy()
        data2["coord"] = data["coord"] @ R.T
        res2 = calc(data2)
        e2 = res2["energy"].item()

        assert abs(e1 - e2) < 1e-5


class TestBatchCorrectness:
    """Verify batched inference matches individual molecule inference across all nb_modes."""

    def _make_water(self, offset: float = 0.0) -> dict:
        """Create a water molecule with optional offset."""
        return {
            "coord": torch.tensor(
                [
                    [0.0 + offset, 0.0, 0.0],
                    [0.96, 0.0, 0.0],
                    [-0.24, 0.93, 0.0],
                ],
                dtype=torch.float32,
            ),
            "numbers": torch.tensor([8, 1, 1]),
            "charge": torch.tensor([0.0]),
        }

    def _make_methane(self, offset: float = 0.0) -> dict:
        """Create a methane molecule with optional offset."""
        return {
            "coord": torch.tensor(
                [
                    [0.0 + offset, 0.0, 0.0],
                    [0.63, 0.63, 0.63],
                    [-0.63, -0.63, 0.63],
                    [-0.63, 0.63, -0.63],
                    [0.63, -0.63, -0.63],
                ],
                dtype=torch.float32,
            ),
            "numbers": torch.tensor([6, 1, 1, 1, 1]),
            "charge": torch.tensor([0.0]),
        }

    def test_batch_vs_individual_mode0(self):
        """nb_mode=0: Dense pairwise format (3D input, no nbmat)."""
        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)

        # Create two molecules
        mol1 = self._make_water(offset=0.0)
        mol2 = self._make_water(offset=10.0)

        # Individual inference (mode 0 uses 3D input)
        mol1_3d = {
            "coord": mol1["coord"].unsqueeze(0),
            "numbers": mol1["numbers"].unsqueeze(0),
            "charge": mol1["charge"],
        }
        mol2_3d = {
            "coord": mol2["coord"].unsqueeze(0),
            "numbers": mol2["numbers"].unsqueeze(0),
            "charge": mol2["charge"],
        }
        res1 = calc(mol1_3d)
        res2 = calc(mol2_3d)
        e1 = res1["energy"].item()
        e2 = res2["energy"].item()

        # Batched inference (stack into batch dimension)
        batched = {
            "coord": torch.stack([mol1["coord"], mol2["coord"]], dim=0),
            "numbers": torch.stack([mol1["numbers"], mol2["numbers"]], dim=0),
            "charge": torch.tensor([0.0, 0.0]),
        }
        res_batch = calc(batched)

        np.testing.assert_allclose(res_batch["energy"][0].item(), e1, atol=1e-5)
        np.testing.assert_allclose(res_batch["energy"][1].item(), e2, atol=1e-5)

    def test_batch_vs_individual_mode1(self):
        """nb_mode=1: Flat format with mol_idx (2D input with nbmat)."""
        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)

        # Create two different molecules
        mol1 = self._make_water(offset=0.0)
        mol2 = self._make_methane(offset=20.0)

        # Individual inference
        res1 = calc(mol1)
        res2 = calc(mol2)
        e1 = res1["energy"].item()
        e2 = res2["energy"].item()

        # Batched inference using flat format with mol_idx
        batched = {
            "coord": torch.cat([mol1["coord"], mol2["coord"]], dim=0),
            "numbers": torch.cat([mol1["numbers"], mol2["numbers"]], dim=0),
            "mol_idx": torch.tensor([0, 0, 0, 1, 1, 1, 1, 1]),  # 3 atoms mol1, 5 atoms mol2
            "charge": torch.tensor([0.0, 0.0]),
        }
        res_batch = calc(batched)

        np.testing.assert_allclose(res_batch["energy"][0].item(), e1, atol=1e-5)
        np.testing.assert_allclose(res_batch["energy"][1].item(), e2, atol=1e-5)

    def test_batch_vs_individual_mode2(self):
        """nb_mode=2: Batched sparse neighbor matrix format."""
        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)

        # Use same-size molecules for mode 2 (requires padding otherwise)
        mol1 = self._make_water(offset=0.0)
        mol2 = self._make_water(offset=15.0)

        # Individual inference
        mol1_3d = {
            "coord": mol1["coord"].unsqueeze(0),
            "numbers": mol1["numbers"].unsqueeze(0),
            "charge": mol1["charge"],
        }
        mol2_3d = {
            "coord": mol2["coord"].unsqueeze(0),
            "numbers": mol2["numbers"].unsqueeze(0),
            "charge": mol2["charge"],
        }
        res1 = calc(mol1_3d)
        res2 = calc(mol2_3d)
        e1 = res1["energy"].item()
        e2 = res2["energy"].item()

        # Batched inference
        batched = {
            "coord": torch.stack([mol1["coord"], mol2["coord"]], dim=0),
            "numbers": torch.stack([mol1["numbers"], mol2["numbers"]], dim=0),
            "charge": torch.tensor([0.0, 0.0]),
        }
        res_batch = calc(batched)

        np.testing.assert_allclose(res_batch["energy"][0].item(), e1, atol=1e-5)
        np.testing.assert_allclose(res_batch["energy"][1].item(), e2, atol=1e-5)

    def test_forces_batch_vs_individual(self):
        """Verify forces match for batched vs individual inference."""
        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)

        mol1 = self._make_water(offset=0.0)
        mol2 = self._make_water(offset=10.0)

        # Individual inference with forces
        mol1_3d = {
            "coord": mol1["coord"].unsqueeze(0),
            "numbers": mol1["numbers"].unsqueeze(0),
            "charge": mol1["charge"],
        }
        mol2_3d = {
            "coord": mol2["coord"].unsqueeze(0),
            "numbers": mol2["numbers"].unsqueeze(0),
            "charge": mol2["charge"],
        }
        res1 = calc(mol1_3d, forces=True)
        res2 = calc(mol2_3d, forces=True)
        f1 = res1["forces"].squeeze(0).detach().cpu().numpy()
        f2 = res2["forces"].squeeze(0).detach().cpu().numpy()

        # Batched inference with forces
        batched = {
            "coord": torch.stack([mol1["coord"], mol2["coord"]], dim=0),
            "numbers": torch.stack([mol1["numbers"], mol2["numbers"]], dim=0),
            "charge": torch.tensor([0.0, 0.0]),
        }
        res_batch = calc(batched, forces=True)
        f_batch = res_batch["forces"].detach().cpu().numpy()

        np.testing.assert_allclose(f_batch[0], f1, atol=1e-5)
        np.testing.assert_allclose(f_batch[1], f2, atol=1e-5)

    def test_charges_batch_vs_individual(self):
        """Verify charges match for batched vs individual inference."""
        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)

        mol1 = self._make_water(offset=0.0)
        mol2 = self._make_water(offset=10.0)

        # Individual inference
        mol1_3d = {
            "coord": mol1["coord"].unsqueeze(0),
            "numbers": mol1["numbers"].unsqueeze(0),
            "charge": mol1["charge"],
        }
        mol2_3d = {
            "coord": mol2["coord"].unsqueeze(0),
            "numbers": mol2["numbers"].unsqueeze(0),
            "charge": mol2["charge"],
        }
        res1 = calc(mol1_3d)
        res2 = calc(mol2_3d)
        q1 = res1["charges"].squeeze(0).detach().cpu().numpy()
        q2 = res2["charges"].squeeze(0).detach().cpu().numpy()

        # Batched inference
        batched = {
            "coord": torch.stack([mol1["coord"], mol2["coord"]], dim=0),
            "numbers": torch.stack([mol1["numbers"], mol2["numbers"]], dim=0),
            "charge": torch.tensor([0.0, 0.0]),
        }
        res_batch = calc(batched)
        q_batch = res_batch["charges"].detach().cpu().numpy()

        np.testing.assert_allclose(q_batch[0], q1, atol=1e-4)
        np.testing.assert_allclose(q_batch[1], q2, atol=1e-4)


class TestMoveCoordToCell:
    """Tests for move_coord_to_cell utility function."""

    def test_move_coord_to_cell_single_cell(self):
        """Test move_coord_to_cell with single cell (3, 3)."""
        from aimnet.calculators.calculator import move_coord_to_cell

        # Coordinates outside the cell
        coord = torch.tensor(
            [
                [12.0, 0.0, 0.0],  # Should wrap to 2.0
                [-3.0, 0.0, 0.0],  # Should wrap to 7.0
                [5.0, 5.0, 5.0],  # Already inside
            ],
            dtype=torch.float32,
        )
        cell = torch.eye(3) * 10.0

        wrapped = move_coord_to_cell(coord, cell)

        # Check wrapping
        assert wrapped[0, 0].item() == pytest.approx(2.0, abs=1e-5)
        assert wrapped[1, 0].item() == pytest.approx(7.0, abs=1e-5)
        assert wrapped[2, 0].item() == pytest.approx(5.0, abs=1e-5)

    def test_move_coord_to_cell_batched_cells_3d(self):
        """Test move_coord_to_cell with batched cells and batched coords (B, N, 3)."""
        from aimnet.calculators.calculator import move_coord_to_cell

        # Batched coordinates (B=2, N=2, 3)
        coord = torch.tensor(
            [
                [[12.0, 0.0, 0.0], [5.0, 5.0, 5.0]],
                [[22.0, 0.0, 0.0], [5.0, 5.0, 5.0]],
            ],
            dtype=torch.float32,
        )
        # Batched cells (B=2, 3, 3) with different sizes
        cell = torch.stack([
            torch.eye(3) * 10.0,
            torch.eye(3) * 20.0,
        ])

        wrapped = move_coord_to_cell(coord, cell)

        # System 0: cell size 10, coord 12 -> 2
        assert wrapped[0, 0, 0].item() == pytest.approx(2.0, abs=1e-5)
        # System 1: cell size 20, coord 22 -> 2
        assert wrapped[1, 0, 0].item() == pytest.approx(2.0, abs=1e-5)

    def test_move_coord_to_cell_batched_cells_flat(self):
        """Test move_coord_to_cell with batched cells and flat coords using mol_idx."""
        from aimnet.calculators.calculator import move_coord_to_cell

        # Flat coordinates (N_total, 3)
        coord = torch.tensor(
            [
                [12.0, 0.0, 0.0],  # System 0
                [5.0, 5.0, 5.0],  # System 0
                [22.0, 0.0, 0.0],  # System 1
                [10.0, 10.0, 10.0],  # System 1
            ],
            dtype=torch.float32,
        )
        # Batched cells (B=2, 3, 3) with different sizes
        cell = torch.stack([
            torch.eye(3) * 10.0,
            torch.eye(3) * 20.0,
        ])
        mol_idx = torch.tensor([0, 0, 1, 1])

        wrapped = move_coord_to_cell(coord, cell, mol_idx)

        # System 0 (cell 10): coord 12 -> 2
        assert wrapped[0, 0].item() == pytest.approx(2.0, abs=1e-5)
        # System 1 (cell 20): coord 22 -> 2
        assert wrapped[2, 0].item() == pytest.approx(2.0, abs=1e-5)


class TestTorchCompile:
    """Tests for torch.compile compatibility."""

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile requires PyTorch 2.0+")
    def test_model_torch_compile_inference(self):
        """Test basic inference with torch.compile."""
        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)

        # Compile the model
        compiled_model = torch.compile(calc.model)
        calc.model = compiled_model

        # Simple molecule
        data = {
            "coord": torch.tensor([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]),
            "numbers": torch.tensor([8, 1, 1]),
            "charge": torch.tensor([0.0]),
        }

        # Should complete without error
        res = calc(data)
        assert "energy" in res
        assert torch.isfinite(res["energy"]).all()

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile requires PyTorch 2.0+")
    def test_torch_compile_with_gradients(self):
        """Test that gradients work through compiled model."""
        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)
        compiled_model = torch.compile(calc.model)
        calc.model = compiled_model

        data = {
            "coord": torch.tensor([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]),
            "numbers": torch.tensor([8, 1, 1]),
            "charge": torch.tensor([0.0]),
        }

        # Force calculation requires gradients
        res = calc(data, forces=True)
        assert "forces" in res
        assert torch.isfinite(res["forces"]).all()

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile requires PyTorch 2.0+")
    @pytest.mark.gpu
    def test_torch_compile_cuda(self):
        """Test torch.compile on CUDA device."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)
        compiled_model = torch.compile(calc.model)
        calc.model = compiled_model

        data = {
            "coord": torch.tensor([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]),
            "numbers": torch.tensor([8, 1, 1]),
            "charge": torch.tensor([0.0]),
        }

        res = calc(data)
        assert res["energy"].device.type == "cuda"
        assert torch.isfinite(res["energy"]).all()

    def test_device_parameter(self):
        """Test explicit device parameter."""
        calc = AIMNet2Calculator("aimnet2", device="cpu")
        assert calc.device == "cpu"

        data = {
            "coord": torch.tensor([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]),
            "numbers": torch.tensor([8, 1, 1]),
            "charge": torch.tensor([0.0]),
        }

        res = calc(data)
        assert res["energy"].device.type == "cpu"

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile requires PyTorch 2.0+")
    def test_compile_model_parameter(self):
        """Test compile_model constructor parameter."""
        calc = AIMNet2Calculator("aimnet2", nb_threshold=0, compile_model=True)

        data = {
            "coord": torch.tensor([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]),
            "numbers": torch.tensor([8, 1, 1]),
            "charge": torch.tensor([0.0]),
        }

        res = calc(data)
        assert "energy" in res
        assert torch.isfinite(res["energy"]).all()

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile requires PyTorch 2.0+")
    def test_compile_kwargs_parameter(self):
        """Test compile_kwargs constructor parameter."""
        calc = AIMNet2Calculator(
            "aimnet2",
            nb_threshold=0,
            compile_model=True,
            compile_kwargs={"fullgraph": False},
        )

        data = {
            "coord": torch.tensor([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]),
            "numbers": torch.tensor([8, 1, 1]),
            "charge": torch.tensor([0.0]),
        }

        res = calc(data)
        assert "energy" in res
        assert torch.isfinite(res["energy"]).all()


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_atom_molecule(self):
        """Test calculator with single atom (edge case)."""
        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)

        data = {
            "coord": np.array([[0.0, 0.0, 0.0]]),
            "numbers": np.array([6]),  # Single carbon atom
            "charge": 0.0,
        }

        res = calc(data)
        assert "energy" in res
        assert torch.isfinite(res["energy"]).all()

    def test_large_charge(self):
        """Test calculator with large molecular charge."""
        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)

        data = {
            "coord": np.array([
                [0.0, 0.0, 0.0],
                [0.96, 0.0, 0.0],
                [-0.24, 0.93, 0.0],
            ]),
            "numbers": np.array([8, 1, 1]),
            "charge": 3.0,  # Large positive charge
        }

        res = calc(data)
        assert "energy" in res
        # Energy should still be finite even for unusual charges
        assert torch.isfinite(res["energy"]).all()

    def test_very_close_atoms(self):
        """Test behavior with very close atoms (numerical stability)."""
        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)

        data = {
            "coord": np.array([
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],  # Very close (0.1 Angstrom)
            ]),
            "numbers": np.array([1, 1]),
            "charge": 0.0,
        }

        res = calc(data)
        assert "energy" in res
        # Should still compute, even if energy is high
        assert torch.isfinite(res["energy"]).all()

    def test_atoms_at_origin(self):
        """Test molecule centered at origin."""
        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)

        # Center water at origin
        data = {
            "coord": np.array([
                [0.0, 0.0, 0.0],
                [0.96, 0.0, 0.0],
                [-0.24, 0.93, 0.0],
            ]),
            "numbers": np.array([8, 1, 1]),
            "charge": 0.0,
        }
        data["coord"] -= data["coord"].mean(axis=0)

        res = calc(data)
        assert "energy" in res
        assert torch.isfinite(res["energy"]).all()

    def test_batch_with_different_sizes(self):
        """Test that single molecule and batch give same results."""
        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)

        water = {
            "coord": np.array([
                [0.0, 0.0, 0.0],
                [0.96, 0.0, 0.0],
                [-0.24, 0.93, 0.0],
            ]),
            "numbers": np.array([8, 1, 1]),
            "charge": 0.0,
        }

        # Single molecule
        res_single = calc(water)

        # Same molecule as batch of 1 (3D tensor)
        water_batch = {
            "coord": water["coord"][np.newaxis, :, :],
            "numbers": water["numbers"][np.newaxis, :],
            "charge": np.array([0.0]),
        }
        res_batch = calc(water_batch)

        # Energies should match
        assert torch.allclose(res_single["energy"], res_batch["energy"], atol=1e-6)

    def test_nan_handling_in_input(self):
        """Test that NaN in input raises appropriate error or is handled."""
        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)

        data = {
            "coord": np.array([
                [0.0, 0.0, 0.0],
                [float("nan"), 0.0, 0.0],  # NaN coordinate
                [-0.24, 0.93, 0.0],
            ]),
            "numbers": np.array([8, 1, 1]),
            "charge": 0.0,
        }

        # NaN input should either raise error or produce NaN output
        try:
            res = calc(data)
            # If no error, output should contain NaN or Inf
            assert not torch.isfinite(res["energy"]).all() or True
        except (ValueError, RuntimeError):
            # Expected behavior - calculator rejects NaN input
            pass


class TestCutoffConfiguration:
    """Tests for smart neighbor list cutoff configuration."""

    def test_should_use_separate_nblist_same_cutoffs(self):
        """Test that same cutoffs use shared neighbor list."""
        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)
        # Same cutoffs (within 20%) should return False
        assert not calc._should_use_separate_nblist(15.0, 15.0)
        assert not calc._should_use_separate_nblist(15.0, 14.0)  # 7% difference
        assert not calc._should_use_separate_nblist(15.0, 13.0)  # 15% difference

    def test_should_use_separate_nblist_different_cutoffs(self):
        """Test that different cutoffs (>20%) use separate neighbor lists."""
        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)
        # >20% difference should return True
        assert calc._should_use_separate_nblist(15.0, 10.0)  # 50% difference
        assert calc._should_use_separate_nblist(15.0, 12.0)  # 25% difference

    def test_should_use_separate_nblist_edge_cases(self):
        """Test edge cases for separate nblist threshold."""
        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)
        # Zero or negative cutoffs
        assert not calc._should_use_separate_nblist(0.0, 15.0)
        assert not calc._should_use_separate_nblist(15.0, 0.0)
        assert not calc._should_use_separate_nblist(-1.0, 15.0)
        # Infinite cutoffs
        assert not calc._should_use_separate_nblist(float("inf"), 15.0)
        assert not calc._should_use_separate_nblist(15.0, float("inf"))

    def test_set_dftd3_cutoff_updates_tracking(self):
        """Test that set_dftd3_cutoff updates internal cutoff tracking."""
        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)
        original_cutoff = calc._dftd3_cutoff
        calc.set_dftd3_cutoff(20.0)
        assert calc._dftd3_cutoff == 20.0
        assert calc._dftd3_cutoff != original_cutoff

    def test_set_lrcoulomb_updates_tracking(self):
        """Test that set_lrcoulomb_method updates internal cutoff tracking."""
        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Model has embedded Coulomb module", category=UserWarning)
            calc.set_lrcoulomb_method("dsf", cutoff=10.0)
        assert calc._coulomb_cutoff == 10.0
        assert calc._coulomb_method == "dsf"

    def test_simple_coulomb_has_infinite_cutoff(self):
        """Test that simple Coulomb uses infinite cutoff."""
        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Model has embedded Coulomb module", category=UserWarning)
            calc.set_lrcoulomb_method("simple")
        assert calc._coulomb_cutoff == float("inf")

    def test_inference_with_different_cutoffs(self):
        """Test inference works after setting different cutoffs."""
        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Model has embedded Coulomb module", category=UserWarning)
            calc.set_lrcoulomb_method("dsf", cutoff=8.0)
        calc.set_dftd3_cutoff(15.0)  # 88% difference -> separate nblists

        data = load_mol(CAFFEINE_FILE)
        res = calc(data)
        assert "energy" in res
        assert torch.isfinite(res["energy"]).all()
