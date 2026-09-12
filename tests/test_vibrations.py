"""Tests for the harmonic vibrational analysis helper.

Everything is CPU-only synthetic data except the one ``weights``/``ase``-marked
cross-check of the calculator wrapper against ASE on a real model.
"""

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


# --- Rigid-body basis must span the rigid motions of a near-linear molecule --
#
# The isotropic fixtures above cannot see a wrong basis: every subspace of the
# right dimension has the same eigenvalue. The tests below use a noisy,
# axis-aligned linear molecule -- the geometry every optimizer and file parser
# actually produces. The span test checks the basis directly against raw rigid
# vectors built here; the orientation test takes its vibrational subspace from
# the module on the exact geometry (where the projector is unambiguous) and
# checks the frequencies stay put under rotation plus noise; the Wilson GF tests
# are fully independent of the module, deriving the reference from internal
# force constants.


def _rigid_rotation_vector(positions: np.ndarray, masses: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Raw (un-orthonormalized) mass-weighted rigid rotation about ``axis`` through the center of mass."""
    r = positions - (positions * masses[:, None]).sum(axis=0) / masses.sum()
    return (np.cross(axis, r) * np.sqrt(masses)[:, None]).ravel()


@pytest.mark.parametrize("axis_index", [0, 1, 2])
def test_basis_spans_rotations_of_a_noisy_axis_aligned_linear_molecule(axis_index: int):
    rng = np.random.default_rng(axis_index)
    positions = np.zeros((3, 3))
    positions[:, axis_index] = [-1.16, 0.0, 1.16]
    positions += 1e-6 * rng.standard_normal((3, 3))
    masses = masses_amu(CO2_NUMBERS)
    basis = translation_rotation_basis(positions, masses, is_linear=True)
    assert basis.shape == (5, 9)
    np.testing.assert_allclose(basis @ basis.T, np.eye(5), atol=1e-10)
    sqrt_m = np.sqrt(masses)
    rigid = {f"translation {'xyz'[k]}": np.outer(sqrt_m, np.eye(3)[k]).ravel() for k in range(3)}
    for k in range(3):
        if k == axis_index:
            continue  # the near-null rotation about the molecular axis is the one to drop
        rigid[f"rotation {'xyz'[k]}"] = _rigid_rotation_vector(positions, masses, np.eye(3)[k])
    for name, v in rigid.items():
        residual = np.linalg.norm(v - basis.T @ (basis @ v))
        assert residual < 1e-9 * np.linalg.norm(v), f"{name} not in span, residual {residual:.2e}"


def test_frequencies_of_a_noisy_linear_molecule_are_orientation_independent():
    """A structured Hessian with distinct eigenvalues on the vibrational subspace,
    rotated to lie along x, y and z with 1e-6 Å off-axis noise on the positions,
    must give the same four frequencies in every orientation."""
    masses = masses_amu(CO2_NUMBERS)
    sqrt_m = np.sqrt(np.repeat(masses, 3))
    # Reference eigenvalues in eV A^-2 amu^-1: a degenerate bend pair, the symmetric and the asymmetric stretch.
    eigenvalues = np.array([0.6, 0.6, 2.5, 7.0])
    expected = np.array([_wavenumber_cm1(k) for k in eigenvalues])
    # Build the mass-weighted Hessian for the exactly linear geometry along x, where
    # the vibrational subspace can be taken from the projector without ambiguity.
    vib = _vibrational_subspace(CO2, masses)
    assert vib.shape[1] == 4
    h_mw = vib @ np.diag(eigenvalues) @ vib.T
    hessian_x = h_mw * sqrt_m[:, None] * sqrt_m[None, :]

    rotations = {
        "x": np.eye(3),
        "y": np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),  # proper rotation taking x to -y
        "z": np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]),  # proper rotation taking x to -z
    }
    rng = np.random.default_rng(7)
    for name, rot in rotations.items():
        positions = CO2 @ rot.T + 1e-6 * rng.standard_normal((3, 3))
        big = np.kron(np.eye(len(masses)), rot)  # rotate each atom's Cartesian block
        hessian = big @ hessian_x @ big.T
        result = analyze_hessian(hessian, positions, masses)
        assert result.is_linear and result.n_tr_removed == 5, name
        np.testing.assert_allclose(result.frequencies_cm1, expected, rtol=1e-6, err_msg=f"orientation {name}")


_MDYN_PER_A = 6.241509074  # 1 mdyn/A in eV/A^2


def _stretch_row(positions: np.ndarray, i: int, j: int) -> np.ndarray:
    """Wilson B-matrix row for the bond length |r_i - r_j|."""
    e = positions[i] - positions[j]
    e = e / np.linalg.norm(e)
    row = np.zeros(positions.size)
    row[3 * i : 3 * i + 3] = e
    row[3 * j : 3 * j + 3] = -e
    return row


def _bend_row(positions: np.ndarray, i: int, c: int, j: int) -> np.ndarray:
    """Wilson B-matrix row for the valence angle i-c-j (Wilson, Decius and Cross)."""
    e1 = positions[i] - positions[c]
    e2 = positions[j] - positions[c]
    r1, r2 = np.linalg.norm(e1), np.linalg.norm(e2)
    e1, e2 = e1 / r1, e2 / r2
    cos = e1 @ e2
    sin = math.sqrt(1.0 - cos * cos)
    s_i = (cos * e1 - e2) / (r1 * sin)
    s_j = (cos * e2 - e1) / (r2 * sin)
    row = np.zeros(positions.size)
    row[3 * i : 3 * i + 3] = s_i
    row[3 * j : 3 * j + 3] = s_j
    row[3 * c : 3 * c + 3] = -s_i - s_j
    return row


def _gf_wavenumbers(b_matrix: np.ndarray, force_constants: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Wilson GF frequencies in cm^-1 for internal-coordinate force constants, independent of the module."""
    g_matrix = b_matrix @ np.diag(np.repeat(1.0 / masses, 3)) @ b_matrix.T
    eigenvalues = np.sort(np.linalg.eigvals(g_matrix @ force_constants).real)
    return np.array([_wavenumber_cm1(k) for k in eigenvalues])


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


@pytest.mark.parametrize("orientation", ["x", "y", "z", "random"])
def test_linear_triatomic_matches_wilson_gf_reference(orientation: str):
    """O-C-O with two stretches (coupled) and a degenerate linear bend, built from
    internal force constants via the Wilson B matrix, must reproduce the GF
    frequencies in every orientation with off-axis noise on the positions."""
    d = 1.16
    masses = masses_amu(CO2_NUMBERS)
    positions = np.array([[-d, 0.0, 0.0], [0.0, 0.0, 0.0], [d, 0.0, 0.0]])
    b_matrix = np.zeros((4, 9))
    b_matrix[0] = _stretch_row(positions, 0, 1)
    b_matrix[1] = _stretch_row(positions, 2, 1)
    b_matrix[2, [1, 4, 7]] = [1 / d, -2 / d, 1 / d]  # linear bend in y
    b_matrix[3, [2, 5, 8]] = [1 / d, -2 / d, 1 / d]  # linear bend in z
    force_constants = np.diag([16.0, 16.0, 0.62, 0.62]) * _MDYN_PER_A
    force_constants[0, 1] = force_constants[1, 0] = 1.3 * _MDYN_PER_A
    hessian = b_matrix.T @ force_constants @ b_matrix
    expected = _gf_wavenumbers(b_matrix, force_constants, masses)

    rng = np.random.default_rng(11)
    rot = {
        "x": np.eye(3),
        "y": np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        "z": np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]),
        "random": _random_rotation(rng),
    }[orientation]
    big = np.kron(np.eye(len(masses)), rot)
    result = analyze_hessian(big @ hessian @ big.T, positions @ rot.T + 1e-6 * rng.standard_normal((3, 3)), masses)
    assert result.is_linear and result.n_tr_removed == 5
    np.testing.assert_allclose(result.frequencies_cm1, expected, rtol=1e-6)


def test_bent_triatomic_outside_tolerance_keeps_all_three_modes():
    """A bent stationary point well outside the linearity tolerance (170 degrees) is
    treated as non-linear: six rigid modes removed, three GF frequencies reproduced."""
    d, half = 1.16, math.radians(180.0 - 170.0) / 2
    masses = masses_amu(CO2_NUMBERS)
    positions = np.array([
        [-d * math.cos(half), d * math.sin(half), 0.0],
        [0.0, 0.0, 0.0],
        [d * math.cos(half), d * math.sin(half), 0.0],
    ])
    b_matrix = np.array([_stretch_row(positions, 0, 1), _stretch_row(positions, 2, 1), _bend_row(positions, 0, 1, 2)])
    force_constants = np.diag([16.0, 16.0, 0.62]) * _MDYN_PER_A
    force_constants[0, 1] = force_constants[1, 0] = 1.3 * _MDYN_PER_A
    hessian = b_matrix.T @ force_constants @ b_matrix
    expected = _gf_wavenumbers(b_matrix, force_constants, masses)

    rot = _random_rotation(np.random.default_rng(5))
    big = np.kron(np.eye(len(masses)), rot)
    result = analyze_hessian(big @ hessian @ big.T, positions @ rot.T, masses)
    assert not result.is_linear and result.n_tr_removed == 6
    np.testing.assert_allclose(result.frequencies_cm1, expected, rtol=1e-6)


def test_single_atom_has_no_vibrational_modes():
    result = analyze_hessian(1e-6 * np.eye(3), np.zeros((1, 3)), masses_amu([8]))
    assert result.n_tr_removed == 3
    assert result.frequencies_cm1.shape == (0,) and result.modes.shape == (0, 1, 3)
    assert result.n_imaginary == 0


@pytest.mark.weights
@pytest.mark.ase
def test_vibrational_analysis_matches_ase_on_a_real_model(model_calculator):
    """The wrapper's frequencies from a real calculator Hessian agree with ASE's
    ``VibrationsData`` on the three vibrational modes of water. ASE does not project
    out translations and rotations, so only its three largest modes are compared."""
    from ase import Atoms
    from ase.vibrations import VibrationsData

    positions = np.array([[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]])
    numbers = np.array([8, 1, 1])
    data = {
        "coord": torch.tensor(positions, dtype=torch.float32),
        "numbers": torch.tensor(numbers),
        "charge": torch.tensor(0.0),
    }
    vib = vibrational_analysis(model_calculator, dict(data))
    assert vib.n_tr_removed == 6 and vib.frequencies_cm1.shape == (3,)
    assert vib.n_imaginary == 0

    hessian = model_calculator.eval(dict(data), hessian=True)["hessian"].detach().cpu().numpy()
    ase_freqs = VibrationsData(Atoms(numbers=numbers, positions=positions), hessian).get_frequencies()
    np.testing.assert_allclose(vib.frequencies_cm1, np.sort(ase_freqs.real)[-3:], atol=1.0)


# --- Input validation, batch guards, and the embedded-dispersion warning -----


def test_masses_amu_rejects_padding_negative_and_out_of_range_numbers():
    np.testing.assert_allclose(masses_amu([1, 118]), [1.008, 294.21398926], rtol=1e-6)
    assert masses_amu([]).shape == (0,)
    for bad in ([0], [119], [-1]):
        with pytest.raises(ValueError, match=r"1\.\.118"):
            masses_amu(bad)
    with pytest.raises(ValueError, match=r"1\.\.118, got 0\.\.8"):
        masses_amu([8, 0])


def test_analyze_hessian_rejects_nonfinite_inputs_and_bad_masses():
    masses = masses_amu(WATER_NUMBERS)
    hessian = _mass_weighted_isotropic_hessian(masses)
    with pytest.raises(ValueError, match="NaN or Inf"):
        analyze_hessian(np.full((9, 9), np.nan), WATER, masses)
    with pytest.raises(ValueError, match="finite and positive"):
        analyze_hessian(hessian, WATER, np.array([16.0, 0.0, 1.0]))
    with pytest.raises(ValueError, match="finite and positive"):
        analyze_hessian(hessian, WATER, np.array([16.0, np.nan, 1.0]))


def test_analyze_hessian_accepts_tensors_for_every_input():
    masses = masses_amu(WATER_NUMBERS)
    hessian = _mass_weighted_isotropic_hessian(masses)
    reference = analyze_hessian(hessian, WATER, masses)
    result = analyze_hessian(
        torch.tensor(hessian, dtype=torch.float32),
        torch.tensor(WATER, requires_grad=True),
        torch.tensor(masses),
    )
    np.testing.assert_allclose(result.frequencies_cm1, reference.frequencies_cm1, rtol=1e-5)


def test_result_is_frozen():
    import dataclasses

    result = analyze_hessian(0.02 * np.eye(9), WATER, masses_amu(WATER_NUMBERS))
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.is_linear = True  # type: ignore[misc]


class _FakeCalculator:
    """Minimal stand-in: returns a fixed isotropic water Hessian for any input."""

    def __init__(self, model=None):
        self.model = model
        masses = masses_amu(WATER_NUMBERS)
        self._hessian = torch.tensor(_mass_weighted_isotropic_hessian(masses).reshape(3, 3, 3, 3), dtype=torch.float32)

    def eval(self, data, **kwargs):
        return {"hessian": self._hessian}


def _water_data(coord):
    return {"coord": coord, "numbers": torch.tensor(WATER_NUMBERS), "charge": 0.0}


def test_vibrational_analysis_rejects_batches_up_front():
    with pytest.raises(ValueError, match="one structure"):
        vibrational_analysis(_FakeCalculator(), _water_data(np.stack([WATER, WATER])))
    flat = _water_data(np.concatenate([WATER, WATER]))
    flat["mol_idx"] = torch.tensor([0, 0, 0, 1, 1, 1])
    with pytest.raises(ValueError, match="mol_idx"):
        vibrational_analysis(_FakeCalculator(), flat)
    # A leading batch dimension of one is a single structure.
    assert vibrational_analysis(_FakeCalculator(), _water_data(WATER[None])).n_tr_removed == 6


def test_vibrational_analysis_warns_only_for_embedded_tabulated_dftd3():
    import warnings

    from torch import nn

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        vibrational_analysis(_FakeCalculator(), _water_data(WATER))  # no model attribute: silent
        d3ts_model = nn.Module()
        d3ts_model.add_module("d3ts", nn.Identity())
        vibrational_analysis(_FakeCalculator(d3ts_model), _water_data(WATER))  # D3TS differentiates correctly
    dftd3_model = nn.Module()
    dftd3_model.add_module("dftd3", nn.Identity())
    with pytest.warns(UserWarning, match="tabulated DFT-D3"):
        vibrational_analysis(_FakeCalculator(dftd3_model), _water_data(WATER))
