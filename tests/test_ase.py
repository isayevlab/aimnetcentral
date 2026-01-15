"""Tests for ASE calculator interface."""

import numpy as np
import pytest
from conftest import CAFFEINE_FILE, CIF_FILE_2

# All tests in this module require ASE
pytestmark = pytest.mark.ase

MODELS = ("aimnet2", "aimnet2_b973c")


class TestBasicCalculator:
    """Basic ASE calculator tests."""

    def test_energy_calculation(self):
        """Test that energy calculation works."""
        pytest.importorskip("ase", reason="ASE not installed")
        from ase.io import read

        from aimnet.calculators import AIMNet2ASE

        for model in MODELS:
            atoms = read(CAFFEINE_FILE)
            atoms.calc = AIMNet2ASE(model)
            e = atoms.get_potential_energy()
            assert isinstance(e, float)
            assert np.isfinite(e)

    def test_charges(self):
        """Test that charge calculation works."""
        pytest.importorskip("ase", reason="ASE not installed")
        from ase.io import read

        from aimnet.calculators import AIMNet2ASE

        atoms = read(CAFFEINE_FILE)
        atoms.calc = AIMNet2ASE("aimnet2")

        q = atoms.get_charges()
        assert q.shape == (len(atoms),)
        assert np.isfinite(q).all()

    def test_dipole_moment(self):
        """Test that dipole moment calculation works."""
        pytest.importorskip("ase", reason="ASE not installed")
        from ase.io import read

        from aimnet.calculators import AIMNet2ASE

        atoms = read(CAFFEINE_FILE)
        atoms.calc = AIMNet2ASE("aimnet2")

        dm = atoms.get_dipole_moment()
        assert dm.shape == (3,)
        assert np.isfinite(dm).all()


class TestForces:
    """Tests for force calculations."""

    def test_forces_shape(self):
        """Test that forces have correct shape."""
        pytest.importorskip("ase", reason="ASE not installed")
        from ase.io import read

        from aimnet.calculators import AIMNet2ASE

        atoms = read(CAFFEINE_FILE)
        atoms.calc = AIMNet2ASE("aimnet2")

        f = atoms.get_forces()
        assert f.shape == (len(atoms), 3)
        assert np.isfinite(f).all()

    def test_forces_sum_nearly_zero(self):
        """Test that total force is nearly zero (Newton's third law)."""
        pytest.importorskip("ase", reason="ASE not installed")
        from ase.io import read

        from aimnet.calculators import AIMNet2ASE

        atoms = read(CAFFEINE_FILE)
        atoms.calc = AIMNet2ASE("aimnet2")

        f = atoms.get_forces()
        total_force = np.sum(f, axis=0)
        # Total force should be nearly zero
        assert np.allclose(total_force, 0, atol=1e-4)


class TestPBC:
    """Tests for periodic boundary conditions."""

    def test_pbc_energy(self):
        """Test energy calculation for periodic system."""
        pytest.importorskip("ase", reason="ASE not installed")
        from ase.io import read

        from aimnet.calculators import AIMNet2ASE

        atoms = read(CIF_FILE_2)
        atoms.calc = AIMNet2ASE("aimnet2")

        e = atoms.get_potential_energy()
        assert isinstance(e, float)
        assert np.isfinite(e)

    def test_pbc_forces(self):
        """Test force calculation for periodic system."""
        pytest.importorskip("ase", reason="ASE not installed")
        from ase.io import read

        from aimnet.calculators import AIMNet2ASE

        atoms = read(CIF_FILE_2)
        atoms.calc = AIMNet2ASE("aimnet2")

        f = atoms.get_forces()
        assert f.shape == (len(atoms), 3)
        assert np.isfinite(f).all()

    def test_pbc_stress_tensor(self):
        """Test stress tensor calculation for periodic system."""
        pytest.importorskip("ase", reason="ASE not installed")
        from ase.io import read

        from aimnet.calculators import AIMNet2ASE

        atoms = read(CIF_FILE_2)
        atoms.calc = AIMNet2ASE("aimnet2")

        # Get stress tensor (Voigt notation: xx, yy, zz, yz, xz, xy)
        stress = atoms.get_stress()
        assert stress.shape == (6,)
        assert np.isfinite(stress).all()

    def test_pbc_stress_volume_normalized(self):
        """Test that stress is volume normalized (units are pressure)."""
        pytest.importorskip("ase", reason="ASE not installed")
        from ase.io import read

        from aimnet.calculators import AIMNet2ASE

        atoms = read(CIF_FILE_2)
        atoms.calc = AIMNet2ASE("aimnet2")

        stress = atoms.get_stress()
        # Stress values should be reasonable (not extremely large)
        # Typical stress values are in GPa range (1e-4 to 1e-1 eV/Å³)
        assert np.abs(stress).max() < 10.0  # Sanity check


class TestOptimization:
    """Tests for geometry optimization."""

    def test_energy_decreases_on_optimization(self):
        """Test that energy decreases during optimization."""
        pytest.importorskip("ase", reason="ASE not installed")
        from ase.io import read
        from ase.optimize import BFGS

        from aimnet.calculators import AIMNet2ASE

        atoms = read(CAFFEINE_FILE)
        # Add small random displacement
        atoms.positions += np.random.randn(*atoms.positions.shape) * 0.1
        atoms.calc = AIMNet2ASE("aimnet2")

        e_initial = atoms.get_potential_energy()

        # Run a few optimization steps
        opt = BFGS(atoms, logfile=None)
        opt.run(fmax=0.5, steps=5)

        e_final = atoms.get_potential_energy()

        # Energy should decrease (or stay same if already at minimum)
        assert e_final <= e_initial + 1e-3


class TestMultipleModels:
    """Tests for multiple model support."""

    @pytest.mark.parametrize("model", MODELS)
    def test_model_gives_finite_energy(self, model):
        """Test each model gives finite energy."""
        pytest.importorskip("ase", reason="ASE not installed")
        from ase.io import read

        from aimnet.calculators import AIMNet2ASE

        atoms = read(CAFFEINE_FILE)
        atoms.calc = AIMNet2ASE(model)

        e = atoms.get_potential_energy()
        assert np.isfinite(e)

    @pytest.mark.parametrize("model", MODELS)
    def test_model_gives_finite_forces(self, model):
        """Test each model gives finite forces."""
        pytest.importorskip("ase", reason="ASE not installed")
        from ase.io import read

        from aimnet.calculators import AIMNet2ASE

        atoms = read(CAFFEINE_FILE)
        atoms.calc = AIMNet2ASE(model)

        f = atoms.get_forces()
        assert np.isfinite(f).all()
