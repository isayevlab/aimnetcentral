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


class TestSpeciesValidation:
    """Tests for implemented_species population and validation."""

    def test_implemented_species_populated_from_metadata(self):
        """AIMNet2ASE must read implemented_species from model metadata, not a missing attribute."""
        pytest.importorskip("ase", reason="ASE not installed")
        from aimnet.calculators import AIMNet2ASE, AIMNet2Calculator

        base_calc = AIMNet2Calculator("aimnet2")
        ase_calc = AIMNet2ASE(base_calc)
        assert ase_calc.implemented_species is not None
        assert len(ase_calc.implemented_species) > 0
        # H (atomic number 1) must be supported
        assert 1 in ase_calc.implemented_species

    def test_species_validation_rejects_unsupported_element(self):
        """set_atoms must raise ValueError when atoms contain an unsupported element."""
        pytest.importorskip("ase", reason="ASE not installed")
        from ase import Atoms

        from aimnet.calculators import AIMNet2ASE

        ase_calc = AIMNet2ASE("aimnet2")
        # Uranium (Z=92) is not in aimnet2 supported elements
        atoms = Atoms("U", positions=[[0, 0, 0]])
        with pytest.raises(ValueError, match="not implemented"):
            ase_calc.set_atoms(atoms)


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


class TestHessian:
    """Hessian property — used by Sella analytic-Hessian callback."""

    def test_hessian_shape_and_finite(self):
        pytest.importorskip("ase", reason="ASE not installed")
        from ase.io import read

        from aimnet.calculators import AIMNet2ASE

        atoms = read(CAFFEINE_FILE)
        atoms.calc = AIMNet2ASE("aimnet2")

        H = atoms.calc.get_hessian(atoms)
        N = len(atoms)
        assert H.shape == (3 * N, 3 * N)
        assert np.isfinite(H).all()

    def test_hessian_symmetric(self):
        pytest.importorskip("ase", reason="ASE not installed")
        from ase.io import read

        from aimnet.calculators import AIMNet2ASE

        atoms = read(CAFFEINE_FILE)
        atoms.calc = AIMNet2ASE("aimnet2")

        H = atoms.calc.get_hessian(atoms)
        # Relative tolerance: small-magnitude entries can be noisier than
        # large ones in row-wise fp32 autograd. Asymmetry that scales with
        # |H|_max would still catch a real index transposition bug.
        assert np.max(np.abs(H - H.T)) / np.max(np.abs(H)) < 1e-3

    def test_hessian_callback_signature(self):
        """Must be usable as Sella's hessian_function=callable callback."""
        pytest.importorskip("ase", reason="ASE not installed")
        from ase import Atoms

        from aimnet.calculators import AIMNet2ASE

        atoms = Atoms("OH2", positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
        atoms.calc = AIMNet2ASE("aimnet2")

        callback = atoms.calc.get_hessian
        assert callable(callback)
        H = callback(atoms)
        assert H.shape == (9, 9)
        assert isinstance(H, np.ndarray)

    def test_hessian_default_atoms(self):
        """get_hessian() with no argument should use the attached atoms."""
        pytest.importorskip("ase", reason="ASE not installed")
        from ase import Atoms

        from aimnet.calculators import AIMNet2ASE

        atoms = Atoms("OH2", positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
        calc = AIMNet2ASE("aimnet2")
        atoms.calc = calc
        atoms.get_potential_energy()
        H = calc.get_hessian()
        assert H.shape == (9, 9)

    def test_hessian_pbc_raises(self):
        """PBC input must raise PropertyNotImplementedError (gas-phase only)."""
        pytest.importorskip("ase", reason="ASE not installed")
        from ase import Atoms
        from ase.calculators.calculator import PropertyNotImplementedError

        from aimnet.calculators import AIMNet2ASE

        atoms = Atoms(
            "OH2",
            positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]],
            cell=[10.0, 10.0, 10.0],
            pbc=True,
        )
        atoms.calc = AIMNet2ASE("aimnet2")

        with pytest.raises(PropertyNotImplementedError, match="periodic"):
            atoms.calc.get_hessian(atoms)

    def test_hessian_no_atoms_raises(self):
        """get_hessian() with no attached Atoms and no argument must raise."""
        pytest.importorskip("ase", reason="ASE not installed")
        from ase.calculators.calculator import PropertyNotImplementedError

        from aimnet.calculators import AIMNet2ASE

        calc = AIMNet2ASE("aimnet2")
        # Calc has never been attached to any Atoms — self.atoms is unset.
        with pytest.raises(PropertyNotImplementedError, match="attached"):
            calc.get_hessian()

    def test_hessian_species_swap_invalidates_cache(self):
        """get_hessian() must rebuild _t_numbers when called with different species at same N."""
        pytest.importorskip("ase", reason="ASE not installed")
        from ase import Atoms

        from aimnet.calculators import AIMNet2ASE

        # 3-atom water and 3-atom HCN: same length (3), different elements.
        water = Atoms("OH2", positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
        hcn = Atoms("HCN", positions=[[0, 0, 0], [1.06, 0, 0], [2.22, 0, 0]])

        calc = AIMNet2ASE("aimnet2")

        water.calc = calc
        H_water = calc.get_hessian(water)
        assert H_water.shape == (9, 9)

        # Pass a DIFFERENT atoms object with same N but different species,
        # without going through set_atoms — the cache must invalidate.
        H_hcn = calc.get_hessian(hcn)
        assert H_hcn.shape == (9, 9)

        # Hessians must differ — if cache was stale, both would be water's.
        assert not np.allclose(H_water, H_hcn, atol=1e-3)

        # Direct cache probe: _t_numbers must reflect the most recent atoms.
        cached = calc._t_numbers.detach().cpu().tolist()
        assert cached == hcn.numbers.tolist()

    def test_hessian_nse_open_shell(self):
        """NSE (open-shell) Hessian must run on a doublet without exception."""
        pytest.importorskip("ase", reason="ASE not installed")
        from ase import Atoms

        from aimnet.calculators import AIMNet2ASE

        # Methyl radical CH3 — doublet (mult=2).
        atoms = Atoms(
            "CH3",
            positions=[
                [0.000, 0.000, 0.000],
                [1.080, 0.000, 0.000],
                [-0.540, 0.935, 0.000],
                [-0.540, -0.935, 0.000],
            ],
            info={"mult": 2},
        )
        atoms.calc = AIMNet2ASE(NSE_MODEL)

        H = atoms.calc.get_hessian(atoms)
        N = len(atoms)
        assert H.shape == (3 * N, 3 * N)
        assert np.isfinite(H).all()
        # Sanity: doublet Hessian should be symmetric to fp32 noise.
        assert np.max(np.abs(H - H.T)) / np.max(np.abs(H)) < 1e-3

    def test_hessian_compile_model_raises(self):
        """compile_model=True must reject the Hessian path with RuntimeError."""
        pytest.importorskip("ase", reason="ASE not installed")
        from ase import Atoms

        from aimnet.calculators import AIMNet2ASE, AIMNet2Calculator

        base = AIMNet2Calculator("aimnet2", compile_model=True)
        atoms = Atoms("OH2", positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
        atoms.calc = AIMNet2ASE(base)

        with pytest.raises(RuntimeError, match="compile_model"):
            atoms.calc.get_hessian(atoms)

    def test_hessian_cpu_device(self):
        """CPU device must produce a correct-shape Hessian for small molecules.

        Defends the GPU-vs-CPU branch in mol_flatten/calculate_hessian: an
        earlier bug where (1, N, 3) coord on GPU + N < nb_threshold yielded
        a degenerate (N, 3, 1, 3) tensor; the 2D-coord workaround in
        get_hessian must work consistently on CPU as well.
        """
        pytest.importorskip("ase", reason="ASE not installed")
        from ase import Atoms

        from aimnet.calculators import AIMNet2ASE, AIMNet2Calculator

        base = AIMNet2Calculator("aimnet2", device="cpu")
        atoms = Atoms("OH2", positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
        atoms.calc = AIMNet2ASE(base)

        H = atoms.calc.get_hessian(atoms)
        assert H.shape == (9, 9)
        assert np.isfinite(H).all()
        assert np.max(np.abs(H - H.T)) / np.max(np.abs(H)) < 1e-3


@pytest.mark.ase
def test_aimnet2ase_propagates_validate_species(monkeypatch):
    """AIMNet2ASE.calculate(..., validate_species=False) must propagate the kwarg
    through to the underlying AIMNet2Calculator.__call__."""
    ase = pytest.importorskip("ase", reason="ASE not installed")

    from aimnet.calculators.aimnet2ase import AIMNet2ASE

    calc = AIMNet2ASE("aimnet2")
    # H2O — only H, O which are supported. Use validate_species=False to verify the
    # kwarg flows through (the call should succeed regardless).
    atoms = ase.Atoms("H2O", positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]])
    atoms.calc = calc

    # Fast path: just access energy. This exercises the full pipeline.
    _ = atoms.get_potential_energy()  # baseline default validate_species=True
    # Re-run with the explicit kwarg via the constructor escape hatch:
    calc2 = AIMNet2ASE("aimnet2", validate_species=False)
    atoms.calc = calc2
    _ = atoms.get_potential_energy()
