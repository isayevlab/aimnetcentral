"""Tests for ASE calculator interface."""

import warnings

import numpy as np
import pytest
from conftest import CAFFEINE_FILE, CIF_SPIRO

# All tests in this module require ASE
pytestmark = pytest.mark.ase

MODELS = ("aimnet2", "aimnet2_b973c")
NSE_MODEL = "aimnet2nse"


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

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="crystal system.*monoclinic", category=UserWarning)
            atoms = read(CIF_SPIRO)
        atoms.calc = AIMNet2ASE("aimnet2")

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Switching to DSF Coulomb", category=UserWarning)
            warnings.filterwarnings("ignore", message="Model has embedded Coulomb module", category=UserWarning)
            e = atoms.get_potential_energy()
        assert isinstance(e, float)
        assert np.isfinite(e)

    def test_pbc_forces(self):
        """Test force calculation for periodic system."""
        pytest.importorskip("ase", reason="ASE not installed")
        from ase.io import read

        from aimnet.calculators import AIMNet2ASE

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="crystal system.*monoclinic", category=UserWarning)
            atoms = read(CIF_SPIRO)
        atoms.calc = AIMNet2ASE("aimnet2")

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Switching to DSF Coulomb", category=UserWarning)
            warnings.filterwarnings("ignore", message="Model has embedded Coulomb module", category=UserWarning)
            f = atoms.get_forces()
        assert f.shape == (len(atoms), 3)
        assert np.isfinite(f).all()

    def test_pbc_stress_tensor(self):
        """Test stress tensor calculation for periodic system."""
        pytest.importorskip("ase", reason="ASE not installed")
        from ase.io import read

        from aimnet.calculators import AIMNet2ASE

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="crystal system.*monoclinic", category=UserWarning)
            atoms = read(CIF_SPIRO)
        atoms.calc = AIMNet2ASE("aimnet2")

        # Get stress tensor (Voigt notation: xx, yy, zz, yz, xz, xy)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Switching to DSF Coulomb", category=UserWarning)
            warnings.filterwarnings("ignore", message="Model has embedded Coulomb module", category=UserWarning)
            stress = atoms.get_stress()
        assert stress.shape == (6,)
        assert np.isfinite(stress).all()

    def test_pbc_stress_volume_normalized(self):
        """Test that stress is volume normalized (units are pressure)."""
        pytest.importorskip("ase", reason="ASE not installed")
        from ase.io import read

        from aimnet.calculators import AIMNet2ASE

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="crystal system.*monoclinic", category=UserWarning)
            atoms = read(CIF_SPIRO)
        atoms.calc = AIMNet2ASE("aimnet2")

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Switching to DSF Coulomb", category=UserWarning)
            warnings.filterwarnings("ignore", message="Model has embedded Coulomb module", category=UserWarning)
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


class TestSpinCharges:
    """Tests for spin_charges support (NSE models)."""

    def test_is_nse_false_for_standard_model(self):
        """Standard models report is_nse=False."""
        from aimnet.calculators import AIMNet2Calculator

        calc = AIMNet2Calculator("aimnet2")
        assert calc.is_nse is False

    def test_spin_charges_not_in_implemented_properties_for_standard_model(self):
        """spin_charges must not be advertised for non-NSE instances."""
        pytest.importorskip("ase", reason="ASE not installed")
        from aimnet.calculators import AIMNet2ASE

        calc = AIMNet2ASE("aimnet2")
        assert "spin_charges" not in calc.implemented_properties

    def test_spin_charges_not_in_class_implemented_properties(self):
        """Class-level implemented_properties must never include spin_charges."""
        pytest.importorskip("ase", reason="ASE not installed")
        from aimnet.calculators import AIMNet2ASE

        assert "spin_charges" not in AIMNet2ASE.implemented_properties

    def test_get_spin_charges_raises_for_standard_model(self):
        """get_spin_charges() on a non-NSE model raises PropertyNotImplementedError."""
        pytest.importorskip("ase", reason="ASE not installed")
        from ase.calculators.calculator import PropertyNotImplementedError
        from ase.io import read

        from aimnet.calculators import AIMNet2ASE

        atoms = read(CAFFEINE_FILE)
        atoms.calc = AIMNet2ASE("aimnet2")
        atoms.get_potential_energy()

        with pytest.raises(PropertyNotImplementedError, match="aimnet2nse"):
            atoms.calc.get_spin_charges()

    def test_is_nse_true_for_nse_model(self):
        """NSE model reports is_nse=True."""
        from aimnet.calculators import AIMNet2Calculator

        calc = AIMNet2Calculator(NSE_MODEL)
        assert calc.is_nse is True

    def test_spin_charges_in_implemented_properties_for_nse_model(self):
        """spin_charges is advertised for NSE instances only."""
        pytest.importorskip("ase", reason="ASE not installed")
        from aimnet.calculators import AIMNet2ASE

        calc = AIMNet2ASE(NSE_MODEL)
        assert "spin_charges" in calc.implemented_properties
        # Class-level list must remain unmodified
        assert "spin_charges" not in AIMNet2ASE.implemented_properties

    def test_spin_charges_shape_and_dtype(self):
        """get_spin_charges() returns float32 array of shape (N,) for NSE model."""
        pytest.importorskip("ase", reason="ASE not installed")
        from ase.io import read

        from aimnet.calculators import AIMNet2ASE

        atoms = read(CAFFEINE_FILE)
        atoms.calc = AIMNet2ASE(NSE_MODEL, mult=1)
        atoms.get_potential_energy()

        sc = atoms.calc.get_spin_charges()
        assert sc.shape == (len(atoms),)
        assert sc.dtype == np.float32
        assert np.isfinite(sc).all()

    def test_spin_charges_sum_equals_mult_minus_one(self):
        """For a doublet (mult=2), spin_charges must sum to 1.0."""
        pytest.importorskip("ase", reason="ASE not installed")
        from ase import Atoms

        from aimnet.calculators import AIMNet2ASE

        # H2 radical cation: 1 unpaired electron
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
        atoms.calc = AIMNet2ASE(NSE_MODEL, charge=1, mult=2)
        atoms.get_potential_energy()

        sc = atoms.calc.get_spin_charges()
        assert sc.shape == (2,)
        assert np.isfinite(sc).all()
        assert abs(sc.sum() - 1.0) < 1e-3


class TestAtomsInfoChargeSpin:
    """Tests for charge/spin multiplicity in Atoms.info."""

    def test_atoms_info_updates_charge_and_spin_and_triggers_recalc(self):
        """Changing info['charge'] or info['spin'] must invalidate ASE cache and recalculate."""
        pytest.importorskip("ase", reason="ASE not installed")
        from ase import Atoms

        from aimnet.calculators import AIMNet2ASE

        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
        atoms.calc = AIMNet2ASE(NSE_MODEL, charge=0, mult=1)

        # Initial calculation
        atoms.info["charge"] = 1
        atoms.info["spin"] = 2
        e1 = atoms.get_potential_energy()

        assert atoms.calc.charge == 1
        assert atoms.calc.mult == 2
        assert float(atoms.calc._t_charge) == pytest.approx(1.0)
        assert float(atoms.calc._t_mult) == pytest.approx(2.0)

        # Invalidate cache by changing only atoms.info in-place
        atoms.info["charge"] = -1
        atoms.info["mult"] = 1  # Using 'mult' key also works
        assert "info" in atoms.calc.check_state(atoms)

        e2 = atoms.get_potential_energy()
        assert e2 != e1

        assert atoms.calc.charge == -1
        assert atoms.calc.mult == 1
        assert float(atoms.calc._t_charge) == pytest.approx(-1.0)
        assert float(atoms.calc._t_mult) == pytest.approx(1.0)

    def test_atoms_info_none_uses_cached_charge_and_spin(self):
        """Null values in info must fall back to the calculator's current state."""
        pytest.importorskip("ase", reason="ASE not installed")
        from ase import Atoms

        from aimnet.calculators import AIMNet2ASE

        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
        atoms.info["charge"] = None
        atoms.info["spin"] = None
        atoms.calc = AIMNet2ASE(NSE_MODEL, charge=1, mult=2)

        atoms.get_potential_energy()

        assert atoms.calc.charge == 1
        assert atoms.calc.mult == 2
        assert float(atoms.calc._t_charge) == pytest.approx(1.0)
        assert float(atoms.calc._t_mult) == pytest.approx(2.0)
        assert abs(atoms.calc.get_spin_charges().sum() - 1.0) < 1e-3

    def test_robustness_with_tensors_in_info(self):
        """Dictionary comparison must not crash when info contains complex types like Tensors or Arrays."""
        pytest.importorskip("ase", reason="ASE not installed")
        import torch
        from ase import Atoms

        from aimnet.calculators import AIMNet2ASE

        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
        # Adding a large tensor to info would cause dictionary comparison to fail in Python
        atoms.info["data"] = torch.randn(10, 10)
        atoms.calc = AIMNet2ASE(NSE_MODEL)

        # First calculation (must not crash)
        atoms.get_potential_energy()

        # Update unrelated key in info
        atoms.info["step"] = 1
        # Should NOT trigger recalculation as charge/mult did not change
        assert "info" not in atoms.calc.check_state(atoms)

        # Update charge in info
        atoms.info["charge"] = 1.0
        # MUST trigger recalculation
        assert "info" in atoms.calc.check_state(atoms)
        atoms.get_potential_energy()

        assert atoms.calc.charge == 1.0
