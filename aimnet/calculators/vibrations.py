"""Harmonic vibrational analysis of a Cartesian Hessian.

Mass-weights the Hessian, projects out the rigid-body translations and
rotations (five for a linear molecule, six otherwise), diagonalizes, and
converts the eigenvalues to wavenumbers. Imaginary modes are reported as
negative wavenumbers, so they are never confused with the near-zero
rigid-body block that the ``sorted(freqs)[6:]`` recipe relies on.

The input is the dense Hessian returned by
``AIMNet2Calculator.eval(data, hessian=True)["hessian"]`` -- shape
``(N, 3, N, 3)`` in eV/Å^2 for coordinates in Å, assembled by
:func:`aimnet.calculators.derivatives.calculate_hessian` -- or its
``(3N, 3N)`` reshape as returned by
:meth:`aimnet.calculators.aimnet2ase.AIMNet2ASE.get_hessian`.

Only numpy and the mass table bundled in :mod:`aimnet.constants` are used, so
the module works with or without the ``ase`` extra; ASE users can pass
``atoms.get_positions()`` and ``atoms.get_masses()`` directly.
"""

import math
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from aimnet.constants import get_masses
from aimnet.models.utils import has_externalizable_dftd3

if TYPE_CHECKING:
    from .calculator import AIMNet2Calculator

# CODATA 2018 recommended values (https://physics.nist.gov/cuu/Constants/).
# e, h and c are exact since the 2019 SI revision.
_E = 1.602176634e-19  # elementary charge, C
_HBAR = 6.62607015e-34 / (2.0 * math.pi)  # reduced Planck constant, J s
_AMU = 1.66053906660e-27  # unified atomic mass unit, kg
_C = 299792458.0  # speed of light in vacuum, m/s

# hbar*omega in eV for a mass-weighted Hessian eigenvalue omega^2 in eV A^-2 amu^-1.
_EV_PER_SQRT_EIGENVALUE = _HBAR * 1e10 / math.sqrt(_E * _AMU)
# Energy of a photon with wavenumber 1 cm^-1, in eV (``ase.units.create_units("2018")["invcm"]``).
_INVCM = 100.0 * 2.0 * math.pi * _HBAR * _C / _E


@dataclass(frozen=True, eq=False)
class VibrationalAnalysis:
    """Result of :func:`analyze_hessian`.

    Attributes:
        frequencies_cm1: Harmonic wavenumbers in cm^-1, shape ``(n_modes,)``, in ascending order of the
            mass-weighted Hessian eigenvalue. Imaginary modes are reported as negative values.
        energies_ev: Vibrational quanta ``h*nu`` in eV with the same ordering and sign convention.
        modes: Unit-norm Cartesian displacement vectors, shape ``(n_modes, N, 3)``.
        is_linear: Whether the geometry was detected as linear.
        n_tr_removed: Number of rigid-body modes projected out: 0 when ``project_tr=False``,
            otherwise 3 for a single atom, 5 for a linear molecule and 6 for any other.
    """

    frequencies_cm1: np.ndarray
    energies_ev: np.ndarray
    modes: np.ndarray
    is_linear: bool
    n_tr_removed: int

    @property
    def n_imaginary(self) -> int:
        """Number of imaginary modes (negative wavenumbers)."""
        return int(np.sum(self.frequencies_cm1 < 0))


def masses_amu(atomic_numbers: Any) -> np.ndarray:
    """Standard atomic masses in amu for ``atomic_numbers``.

    Uses the table bundled in :func:`aimnet.constants.get_masses` (the values of
    ``ase.data.atomic_masses`` stored in single precision), so no ASE install is
    required. Pass ``atoms.get_masses()`` to :func:`analyze_hessian` instead when
    ASE is available and full double precision matters.

    Args:
        atomic_numbers: Integer array-like of shape ``(N,)``.

    Returns:
        Array of shape ``(N,)`` with dtype float64.
    """
    numbers = np.asarray(atomic_numbers, dtype=np.int64).reshape(-1)
    table = get_masses().double().numpy()
    if numbers.size and (numbers.min() < 1 or numbers.max() >= len(table)):
        # Z=0 is the calculator's padding value (mass 0); negative Z would wrap around the table.
        raise ValueError(f"atomic numbers must be in 1..{len(table) - 1}, got {numbers.min()}..{numbers.max()}")
    return table[numbers]


def is_linear_molecule(positions: np.ndarray, masses: np.ndarray, tol: float = 1e-4) -> bool:
    """Return ``True`` when the smallest principal moment of inertia is negligible.

    This is the single place that decides how many rigid-body vectors
    :func:`translation_rotation_basis` keeps (5 or 6); the basis itself applies
    no second threshold. The default ``tol`` treats a triatomic within about two
    degrees of 180° (each bond about one degree off the axis) as linear. Either
    misclassification is wrong, so choose ``tol`` for the molecule at hand:
    flagging a linear molecule non-linear projects out one bend, because the
    sixth rigid-body vector is the axial rotation of the slightly bent geometry
    and coincides with a bend of the linear reference; flagging a slightly bent
    minimum linear leaves one rigid rotation in the vibrational space as a
    spurious near-zero mode. Optimized linear molecules land far inside the
    default (measured ``I_min/I_max`` below 1e-7 at ``fmax=0.01``).

    Args:
        positions: Cartesian coordinates, shape ``(N, 3)``, in Å.
        masses: Atomic masses, shape ``(N,)``, in amu.
        tol: Ratio of the smallest to the largest principal moment below which
            the molecule is treated as linear. Atoms and diatomics are always linear.
    """
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    masses = np.asarray(masses, dtype=np.float64).reshape(-1)
    if len(masses) <= 2:
        return True
    r = positions - (positions * masses[:, None]).sum(axis=0) / masses.sum()
    inertia = np.eye(3) * np.sum(masses * np.einsum("ij,ij->i", r, r)) - np.einsum("i,ij,ik->jk", masses, r, r)
    moments = np.sort(np.linalg.eigvalsh(inertia))
    return bool(moments[0] < tol * max(moments[-1], 1e-12))


def translation_rotation_basis(positions: np.ndarray, masses: np.ndarray, is_linear: bool) -> np.ndarray:
    """Orthonormal basis of rigid-body motions in mass-weighted Cartesian coordinates.

    The three translations and three infinitesimal rotations about the center of
    mass are scaled by ``sqrt(m)`` and orthonormalized with a QR decomposition.
    For a linear molecule the rotation about the molecular axis is a null
    vector and is dropped.

    Args:
        positions: Cartesian coordinates, shape ``(N, 3)``, in Å.
        masses: Atomic masses, shape ``(N,)``, in amu.
        is_linear: Result of :func:`is_linear_molecule`.

    Returns:
        Array of shape ``(k, 3N)`` with orthonormal rows, ``k = 5`` for a linear
        molecule and ``6`` otherwise (fewer only for a single atom).
    """
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    masses = np.asarray(masses, dtype=np.float64).reshape(-1)
    n = len(masses)
    sqm = np.sqrt(masses)
    r = positions - (positions * masses[:, None]).sum(axis=0) / masses.sum()
    vectors = []
    for k in range(3):
        t = np.zeros((n, 3))
        t[:, k] = sqm
        vectors.append(t.ravel())
    for k in range(3):
        axis = np.zeros(3)
        axis[k] = 1.0
        vectors.append((np.cross(axis, r) * sqm[:, None]).ravel())
    # SVD, not QR: singular values come out descending, so keeping the leading
    # columns always discards the near-null axial rotation of a (near-)linear
    # molecule, whatever its lab orientation. Unpivoted QR normalizes that tiny
    # column into noise and contaminates the rotations that follow it, which
    # for an x- or y-aligned linear molecule silently corrupts a bend.
    # The singular values are sqrt(M) (three times) and sqrt(I_k), so "smallest
    # is the axial rotation" assumes I_min < M, i.e. a mass-weighted radius under
    # ~100 A at the default tolerance -- far beyond any dense-Hessian system.
    u, _, _ = np.linalg.svd(np.array(vectors).T, full_matrices=False)
    expected = 3 if n == 1 else (5 if is_linear else 6)
    return u[:, :expected].T


def _to_numpy(value: Any) -> np.ndarray:
    """Float64 numpy view of an array-like, detaching and moving torch tensors off the device first."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _as_square_hessian(hessian: Any, n: int) -> np.ndarray:
    """Return ``hessian`` as a float64 ``(3N, 3N)`` array, accepting the ``(N, 3, N, 3)`` layout too."""
    h = _to_numpy(hessian)
    if h.shape not in ((3 * n, 3 * n), (n, 3, n, 3)):
        raise ValueError(f"Expected a Hessian of shape (3N, 3N) or (N, 3, N, 3) with N={n}, got {h.shape}")
    if not np.isfinite(h).all():
        raise ValueError("Hessian contains NaN or Inf")
    return h.reshape(3 * n, 3 * n)


def analyze_hessian(
    hessian: Any,
    positions: np.ndarray,
    masses: np.ndarray,
    *,
    project_tr: bool = True,
    linear_tol: float = 1e-4,
) -> VibrationalAnalysis:
    """Harmonic frequencies and normal modes from a Cartesian Hessian.

    The Hessian is symmetrized, mass-weighted, optionally projected onto the
    complement of the rigid-body translations and rotations, and diagonalized.
    With projection the ``n_tr_removed`` eigenvalues closest to zero are the
    rigid-body null space and are discarded; the remaining modes are returned
    in ascending order of eigenvalue, so imaginary modes (negative eigenvalue,
    reported as negative wavenumber) come first.

    Args:
        hessian: Cartesian Hessian in eV/Å^2 with shape ``(3N, 3N)`` or ``(N, 3, N, 3)``, as a numpy
            array or torch tensor. This is the layout and unit of
            ``AIMNet2Calculator.eval(data, hessian=True)["hessian"]``.
        positions: Cartesian coordinates, shape ``(N, 3)``, in Å; numpy array or torch tensor.
        masses: Atomic masses, shape ``(N,)``, in amu (see :func:`masses_amu`); numpy array or torch tensor.
        project_tr: Project out translations and rotations. When ``False`` all ``3N`` modes are
            returned, including the near-zero rigid-body ones.
        linear_tol: Passed to :func:`is_linear_molecule`; see there for what each
            misclassification costs.

    Returns:
        :class:`VibrationalAnalysis` with ``3N - n_tr_removed`` modes.
    """
    positions = _to_numpy(positions).reshape(-1, 3)
    masses = _to_numpy(masses).reshape(-1)
    n = len(masses)
    if positions.shape[0] != n:
        raise ValueError(f"positions has {positions.shape[0]} atoms but masses has {n}")
    if not np.isfinite(masses).all() or np.any(masses <= 0):
        raise ValueError("masses must be finite and positive")
    h = _as_square_hessian(hessian, n)
    h = 0.5 * (h + h.T)
    inv_sqrt_m = np.repeat(1.0 / np.sqrt(masses), 3)
    h_mw = h * inv_sqrt_m[:, None] * inv_sqrt_m[None, :]
    linear = is_linear_molecule(positions, masses, tol=linear_tol)
    removed = 0
    if project_tr:
        basis = translation_rotation_basis(positions, masses, linear)
        projector = np.eye(3 * n) - basis.T @ basis
        h_mw = projector @ h_mw @ projector
        h_mw = 0.5 * (h_mw + h_mw.T)
        removed = basis.shape[0]
    omega2, vecs = np.linalg.eigh(h_mw)
    if removed:
        order = np.argsort(np.abs(omega2))
        keep = np.sort(order[removed:])  # drop the null space, keep ascending eigenvalue order
        omega2, vecs = omega2[keep], vecs[:, keep]
    energies = _EV_PER_SQRT_EIGENVALUE * np.sqrt(np.abs(omega2)) * np.where(omega2 < 0, -1.0, 1.0)
    modes = (vecs * inv_sqrt_m[:, None]).T.reshape(-1, n, 3)
    norms = np.linalg.norm(modes.reshape(len(modes), 3 * n), axis=1)  # explicit width: 0 modes for one atom
    modes = modes / np.where(norms > 0, norms, 1.0)[:, None, None]
    return VibrationalAnalysis(
        frequencies_cm1=energies / _INVCM,
        energies_ev=energies,
        modes=modes,
        is_linear=linear,
        n_tr_removed=removed,
    )


def vibrational_analysis(
    calc: "AIMNet2Calculator", data: dict[str, Any], *, project_tr: bool = True, linear_tol: float = 1e-4
) -> VibrationalAnalysis:
    """Compute the Hessian of a single structure with ``calc.eval(data, hessian=True)`` and analyze it.

    Args:
        calc: An :class:`~aimnet.calculators.AIMNet2Calculator`.
        data: Single-structure input as accepted by :meth:`AIMNet2Calculator.eval`, with ``coord`` of
            shape ``(N, 3)`` in Å and ``numbers`` of shape ``(N,)``.
        project_tr: Passed to :func:`analyze_hessian`.
    """
    coord = torch.as_tensor(data["coord"]).detach().cpu()
    if coord.ndim == 3 and coord.shape[0] > 1:
        raise ValueError("vibrational_analysis handles one structure at a time; loop over the batch")
    mol_idx = data.get("mol_idx")
    if mol_idx is not None and torch.as_tensor(mol_idx).numel() and int(torch.as_tensor(mol_idx).max()) > 0:
        # A flat multi-molecule input makes the calculator return one Hessian per
        # molecule; mirror its split check here instead of failing after the work.
        raise ValueError("vibrational_analysis handles one structure at a time; split the input by mol_idx")
    model = getattr(calc, "model", None)
    if model is not None and has_externalizable_dftd3(model):
        # Only the tabulated DFT-D3/D3BJ module is affected: its energy enters
        # autograd through a first-order-only Function, so the calculator's
        # Hessian currently omits its curvature and low-frequency modes of
        # dispersion-bound systems come out too soft. D3TS is plain torch and
        # differentiates correctly; registry models externalize dispersion and
        # are unaffected.
        warnings.warn(
            "This model embeds a tabulated DFT-D3 (D3BJ) module whose curvature the calculator's "
            "Hessian currently omits; low-frequency modes of dispersion-bound systems will be off.",
            stacklevel=2,
        )
    numbers = torch.as_tensor(data["numbers"]).detach().cpu().numpy().reshape(-1)
    hessian = calc.eval(data, hessian=True)["hessian"]
    return analyze_hessian(
        hessian, coord.numpy().reshape(-1, 3), masses_amu(numbers), project_tr=project_tr, linear_tol=linear_tol
    )
