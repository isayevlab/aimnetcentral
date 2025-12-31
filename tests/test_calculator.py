import os

import numpy as np
import pytest
import torch

from aimnet.calculators import AIMNet2Calculator

file = os.path.join(os.path.dirname(__file__), "data", "caffeine.xyz")


def load_mol(filepath):
    """Helper to load molecule from xyz file."""
    pytest.importorskip("ase", reason="ASE not installed")
    import ase.io

    atoms = ase.io.read(filepath)
    data = {
        "coord": atoms.get_positions(),
        "numbers": atoms.get_atomic_numbers(),
        "charge": 0.0,
    }
    return data


@pytest.mark.ase
def test_from_zoo():
    """Test basic model loading and inference from model registry."""
    pytest.importorskip("ase", reason="ASE not installed. Install with: pip install aimnet[ase]")

    calc = AIMNet2Calculator("aimnet2", nb_threshold=0)
    data = load_mol(file)
    res = calc(data)
    assert "energy" in res
    res = calc(data, forces=True)
    assert "forces" in res
    res = calc(data, hessian=True)
    assert "hessian" in res


class TestInputValidation:
    """Tests for input validation and error handling."""

    @pytest.mark.ase
    def test_missing_coord_raises_error(self):
        """Test that missing coord key raises KeyError."""
        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)
        data = {"numbers": [6, 1, 1], "charge": 0.0}

        with pytest.raises(KeyError, match="Missing key coord"):
            calc(data)

    @pytest.mark.ase
    def test_missing_numbers_raises_error(self):
        """Test that missing numbers key raises KeyError."""
        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)
        data = {"coord": [[0, 0, 0], [1, 0, 0]], "charge": 0.0}

        with pytest.raises(KeyError, match="Missing key numbers"):
            calc(data)

    @pytest.mark.ase
    def test_missing_charge_raises_error(self):
        """Test that missing charge key raises KeyError."""
        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)
        data = {"coord": [[0, 0, 0]], "numbers": [6]}

        with pytest.raises(KeyError, match="Missing key charge"):
            calc(data)

    @pytest.mark.ase
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

    @pytest.mark.ase
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


class TestCoulombMethods:
    """Tests for Coulomb method switching."""

    @pytest.mark.ase
    def test_set_coulomb_simple(self):
        """Test setting simple Coulomb method."""
        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)
        calc.set_lrcoulomb_method("simple")
        assert calc._coulomb_method == "simple"

    @pytest.mark.ase
    def test_set_coulomb_dsf(self):
        """Test setting DSF Coulomb method."""
        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)
        calc.set_lrcoulomb_method("dsf", cutoff=12.0, dsf_alpha=0.25)
        assert calc._coulomb_method == "dsf"
        assert calc.cutoff_lr == 12.0

    @pytest.mark.ase
    def test_set_coulomb_ewald(self):
        """Test setting Ewald Coulomb method."""
        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)
        calc.set_lrcoulomb_method("ewald", cutoff=10.0)
        assert calc._coulomb_method == "ewald"

    @pytest.mark.ase
    def test_invalid_coulomb_method(self):
        """Test that invalid Coulomb method raises ValueError."""
        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)
        with pytest.raises(ValueError, match="Invalid method"):
            calc.set_lrcoulomb_method("invalid_method")

    @pytest.mark.ase
    def test_coulomb_method_both_produce_valid_energy(self):
        """Test that both simple and DSF Coulomb methods produce valid energies."""
        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)
        data = load_mol(file)

        # Get energy with simple method
        calc.set_lrcoulomb_method("simple")
        res_simple = calc(data)
        assert torch.isfinite(res_simple["energy"]).all()

        # Get energy with DSF method
        calc.set_lrcoulomb_method("dsf")
        res_dsf = calc(data)
        assert torch.isfinite(res_dsf["energy"]).all()

        # Both should produce negative energies for stable molecules
        assert res_simple["energy"].item() < 0
        assert res_dsf["energy"].item() < 0


class TestBatchProcessing:
    """Tests for batch processing of multiple molecules."""

    @pytest.mark.ase
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

    @pytest.mark.ase
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

    @pytest.mark.ase
    def test_forces_shape(self):
        """Test that forces have correct shape."""
        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)
        data = load_mol(file)
        res = calc(data, forces=True)

        assert "forces" in res
        # Forces should have shape (N, 3) or (1, N, 3)
        assert res["forces"].shape[-1] == 3
        n_atoms = len(data["numbers"])
        assert res["forces"].shape[-2] == n_atoms

    @pytest.mark.ase
    def test_forces_sum_approximately_zero(self):
        """Test that forces sum to approximately zero (translation invariance)."""
        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)
        data = load_mol(file)
        res = calc(data, forces=True)

        # Sum of forces should be approximately zero
        force_sum = res["forces"].sum(dim=-2)
        assert force_sum.abs().max().item() < 1e-4

    @pytest.mark.ase
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

    @pytest.mark.ase
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

    @pytest.mark.ase
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

    @pytest.mark.ase
    def test_translation_invariance(self):
        """Test that energy is invariant under translation."""
        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)
        data = load_mol(file)

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

    @pytest.mark.ase
    def test_rotation_invariance(self):
        """Test that energy is invariant under rotation."""
        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)
        data = load_mol(file)

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
