"""Tests for the harmonic vibrational analysis helper (CPU only, no model weights)."""

import math

import numpy as np
import pytest
import torch

from aimnet.calculators.vibrations import (
    VibrationalAnalysis,
    analyze_hessian,
    is_linear_molecule,
    masses_amu,
    translation_rotation_basis,
    vibrational_analysis,
)

WATER = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
WATER_NUMBERS = [8, 1, 1]
CO2 = np.array([[-1.16, 0.0, 0.0], [0.0, 0.0, 0.0], [1.16, 0.0, 0.0]])
CO2_NUMBERS = [8, 6, 8]

# CODATA 2018, written out independently of the module so the conversion is cross-checked.
_E = 1.602176634e-19
_AMU = 1.66053906660e-27
_C = 299792458.0


def _wavenumber_cm1(omega2_ev_per_a2_amu: float) -> float:
    """Wavenumber for a mass-weighted Hessian eigenvalue via omega [rad/s] = sqrt(k / mu)."""
    omega = math.sqrt(omega2_ev_per_a2_amu * _E / _AMU) * 1e10
    return omega / (2.0 * math.pi * _C * 100.0)


def _mass_weighted_isotropic_hessian(masses: np.ndarray, k: float = 0.02) -> np.ndarray:
    """Cartesian Hessian whose mass-weighted form is ``k * I``: every vibrational mode has eigenvalue ``k``."""
    return k * np.diag(np.repeat(masses, 3))


def _vibrational_subspace(positions: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Orthonormal columns spanning the complement of the rigid-body motions in mass-weighted coordinates."""
    basis = translation_rotation_basis(positions, masses, is_linear_molecule(positions, masses))
    projector = np.eye(3 * len(masses)) - basis.T @ basis
    eigenvalues, eigenvectors = np.linalg.eigh(projector)
    return eigenvectors[:, eigenvalues > 0.5]


def test_masses_amu_uses_bundled_table():
    masses = masses_amu([1, 8])
    assert masses.dtype == np.float64 and masses.shape == (2,)
    assert masses == pytest.approx([1.008, 15.999], rel=1e-6)
    assert masses_amu(torch.tensor(WATER_NUMBERS)).shape == (3,)


def test_isotropic_cartesian_well_without_projection_gives_per_atom_frequencies():
    masses = masses_amu(WATER_NUMBERS)
    result = analyze_hessian(0.02 * np.eye(9), WATER, masses, project_tr=False)
    assert isinstance(result, VibrationalAnalysis)
    assert result.n_tr_removed == 0 and result.frequencies_cm1.shape == (9,)
    expected = sorted(_wavenumber_cm1(0.02 / m) for m in masses for _ in range(3))
    np.testing.assert_allclose(result.frequencies_cm1, expected, rtol=1e-8)
    assert result.n_imaginary == 0 and np.all(np.diff(result.frequencies_cm1) >= 0)
    assert result.modes.shape == (9, 3, 3)
    np.testing.assert_allclose(np.linalg.norm(result.modes.reshape(9, -1), axis=1), 1.0)


def test_projection_leaves_three_identical_real_modes_for_water():
    masses = masses_amu(WATER_NUMBERS)
    result = analyze_hessian(_mass_weighted_isotropic_hessian(masses), WATER, masses)
    assert not result.is_linear and result.n_tr_removed == 6
    assert result.frequencies_cm1.shape == (3,) and result.n_imaginary == 0
    np.testing.assert_allclose(result.frequencies_cm1, _wavenumber_cm1(0.02), rtol=1e-9)
    assert result.modes.shape == (3, 3, 3)
    np.testing.assert_allclose(np.linalg.norm(result.modes.reshape(3, -1), axis=1), 1.0)
    # The Cartesian modes carry no rigid-body component once mass-weighted.
    basis = translation_rotation_basis(WATER, masses, is_linear=False)
    mass_weighted_modes = (result.modes * np.sqrt(masses)[None, :, None]).reshape(3, -1)
    np.testing.assert_allclose(basis @ mass_weighted_modes.T, 0.0, atol=1e-10)
    # A plain Cartesian 0.02 * I well is not isotropic after mass weighting but still yields three real modes.
    plain = analyze_hessian(0.02 * np.eye(9), WATER, masses)
    assert plain.frequencies_cm1.shape == (3,) and plain.n_imaginary == 0


def test_translation_rotation_basis_is_orthonormal():
    masses = masses_amu(WATER_NUMBERS)
    basis = translation_rotation_basis(WATER, masses, is_linear=False)
    assert basis.shape == (6, 9)
    np.testing.assert_allclose(basis @ basis.T, np.eye(6), atol=1e-10)
    co2_basis = translation_rotation_basis(CO2, masses_amu(CO2_NUMBERS), is_linear=True)
    assert co2_basis.shape == (5, 9)
    np.testing.assert_allclose(co2_basis @ co2_basis.T, np.eye(5), atol=1e-10)


def test_linear_molecule_detection_and_mode_count():
    masses = masses_amu(CO2_NUMBERS)
    assert is_linear_molecule(CO2, masses)
    assert not is_linear_molecule(WATER, masses_amu(WATER_NUMBERS))
    assert is_linear_molecule(np.array([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]]), masses_amu([1, 1]))
    assert is_linear_molecule(np.zeros((1, 3)), masses_amu([18]))
    result = analyze_hessian(_mass_weighted_isotropic_hessian(masses), CO2, masses)
    assert result.is_linear and result.n_tr_removed == 5
    assert result.frequencies_cm1.shape == (4,) and result.modes.shape == (4, 3, 3)
    assert result.n_imaginary == 0


def test_negative_curvature_along_a_vibrational_direction_gives_one_imaginary_mode():
    masses = masses_amu(WATER_NUMBERS)
    vib = _vibrational_subspace(WATER, masses)
    projector = vib @ vib.T
    h_mw = 0.02 * projector - 0.07 * np.outer(vib[:, 0], vib[:, 0])
    sqrt_m = np.sqrt(np.repeat(masses, 3))
    hessian = h_mw * sqrt_m[:, None] * sqrt_m[None, :]
    result = analyze_hessian(hessian, WATER, masses)
    assert result.frequencies_cm1.shape == (3,) and result.n_imaginary == 1
    assert result.frequencies_cm1[0] < 0 < result.frequencies_cm1[1]
    assert np.all(np.diff(result.frequencies_cm1) >= 0)
    assert result.frequencies_cm1[0] == pytest.approx(-_wavenumber_cm1(0.05), rel=1e-9)
    assert result.energies_ev[0] < 0 and np.all(result.energies_ev[1:] > 0)


def test_diatomic_matches_analytic_frequency():
    k = 6.0  # eV/A^2
    positions = np.array([[0.0, 0.0, 0.0], [0.92, 0.0, 0.0]])
    masses = masses_amu([1, 9])
    bond = np.array([1.0, 0.0, 0.0])
    block = k * np.outer(bond, bond)
    hessian = np.block([[block, -block], [-block, block]])
    result = analyze_hessian(hessian, positions, masses)
    assert result.is_linear and result.n_tr_removed == 5 and result.frequencies_cm1.shape == (1,)
    reduced_mass = masses[0] * masses[1] / masses.sum()
    expected = _wavenumber_cm1(k / reduced_mass)
    assert result.frequencies_cm1[0] == pytest.approx(expected, rel=1e-6)
    # The stretch is along the bond and the light atom moves in inverse proportion to its mass.
    mode = result.modes[0]
    np.testing.assert_allclose(mode[:, 1:], 0.0, atol=1e-12)
    assert abs(mode[0, 0] / mode[1, 0]) == pytest.approx(masses[1] / masses[0], rel=1e-8)
    unprojected = analyze_hessian(hessian, positions, masses, project_tr=False)
    assert unprojected.frequencies_cm1.shape == (6,)
    assert unprojected.frequencies_cm1[-1] == pytest.approx(expected, rel=1e-6)


def test_accepts_calculator_layout_and_torch_tensors():
    masses = masses_amu(WATER_NUMBERS)
    hessian = _mass_weighted_isotropic_hessian(masses) + 1e-3 * np.eye(9)
    reference = analyze_hessian(hessian, WATER, masses)
    for candidate in (hessian.reshape(3, 3, 3, 3), torch.tensor(hessian.reshape(3, 3, 3, 3))):
        result = analyze_hessian(candidate, WATER, masses)
        np.testing.assert_allclose(result.frequencies_cm1, reference.frequencies_cm1)
    with pytest.raises(ValueError, match="shape"):
        analyze_hessian(np.eye(6), WATER, masses)
    with pytest.raises(ValueError, match="atoms"):
        analyze_hessian(hessian, WATER[:2], masses)


def test_vibrational_analysis_wraps_calculator_eval():
    masses = masses_amu(WATER_NUMBERS)
    hessian = _mass_weighted_isotropic_hessian(masses)
    calls = []

    class FakeCalculator:
        def eval(self, data, **kwargs):
            calls.append(kwargs)
            return {"hessian": torch.tensor(hessian.reshape(3, 3, 3, 3), dtype=torch.float32)}

    data = {"coord": WATER, "numbers": torch.tensor(WATER_NUMBERS), "charge": 0.0}
    result = vibrational_analysis(FakeCalculator(), data)
    assert calls == [{"hessian": True}]
    expected = analyze_hessian(hessian.astype(np.float32), WATER, masses)
    np.testing.assert_allclose(result.frequencies_cm1, expected.frequencies_cm1)
    assert result.n_tr_removed == 6 and result.n_imaginary == 0
