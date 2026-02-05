"""Shared pytest fixtures for AIMNet2 tests."""

import os
import warnings

import pytest
import torch
from torch import Tensor

# =============================================================================
# Test Data Paths
# =============================================================================

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CAFFEINE_FILE = os.path.join(DATA_DIR, "caffeine.xyz")
CIF_PHENYLGLYCINIUM = os.path.join(DATA_DIR, "1100172.cif")  # Phenylglycinium nitrate (~50 atoms)
CIF_SPIRO = os.path.join(DATA_DIR, "2000054.cif")  # Spiro compound (~28 atoms)

# =============================================================================
# Tolerance Constants
# =============================================================================

ENERGY_ATOL = 1e-5  # Absolute tolerance for energy comparisons
FORCE_ATOL = 1e-5  # Absolute tolerance for force comparisons
CHARGE_ATOL = 1e-4  # Absolute tolerance for charge comparisons
GRAD_RTOL = 1e-4  # Relative tolerance for gradient checks


# =============================================================================
# Helper Functions
# =============================================================================


def load_mol(filepath: str) -> dict:
    """Load molecule from xyz or cif file.

    Parameters
    ----------
    filepath : str
        Path to the xyz or cif file.

    Returns
    -------
    dict
        Dictionary with 'coord', 'numbers', 'charge' keys.
        For periodic structures, also includes 'cell' and 'pbc'.
    """
    pytest.importorskip("ase", reason="ASE not installed")
    import ase.io

    atoms = ase.io.read(filepath)
    data = {
        "coord": atoms.get_positions(),
        "numbers": atoms.get_atomic_numbers(),
        "charge": 0.0,
    }
    # Include cell info for periodic structures
    if atoms.pbc.any():
        data["cell"] = atoms.get_cell().array
        data["pbc"] = atoms.pbc
    return data


def add_dftd3_keys(
    data: dict[str, Tensor],
    device: torch.device | None = None,
    cutoff: float = 15.0,
) -> dict[str, Tensor]:
    """Add DFTD3 neighbor matrix keys for molecular systems.

    This is a convenience wrapper around add_lr_keys for DFTD3-specific usage.

    Parameters
    ----------
    data : dict
        Data dictionary containing coord.
    device : torch.device, optional
        Device to create tensors on. If None, inferred from data["coord"].
    cutoff : float
        Cutoff distance for neighbor list.

    Returns
    -------
    dict
        Updated data dictionary with DFTD3 neighbor matrix keys.
    """
    if device is None:
        device = data["coord"].device if hasattr(data["coord"], "device") else torch.device("cpu")
    return add_lr_keys(data, device, module_type="dftd3", cutoff=cutoff)


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@pytest.fixture
def device():
    """Fixture providing the best available device."""
    return get_device()


@pytest.fixture
def requires_gpu():
    """Skip test if GPU is not available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


@pytest.fixture
def simple_molecule(device) -> dict[str, Tensor]:
    """Simple water molecule for basic tests (H2O)."""
    # Water molecule: O at origin, H atoms around it
    coord = torch.tensor(
        [
            [0.0000, 0.0000, 0.1173],  # O
            [0.0000, 0.7572, -0.4692],  # H
            [0.0000, -0.7572, -0.4692],  # H
        ],
        device=device,
        dtype=torch.float32,
    )
    numbers = torch.tensor([8, 1, 1], device=device)
    return {
        "coord": coord.unsqueeze(0),  # (1, 3, 3)
        "numbers": numbers.unsqueeze(0),  # (1, 3)
        "charge": torch.tensor([0.0], device=device),
    }


@pytest.fixture
def water_molecule(device) -> dict[str, Tensor]:
    """Water molecule in flat format (for calculator tests)."""
    coord = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [-0.24, 0.93, 0.0],
        ],
        device=device,
        dtype=torch.float32,
    )
    numbers = torch.tensor([8, 1, 1], device=device)
    return {
        "coord": coord,
        "numbers": numbers,
        "charge": torch.tensor([0.0], device=device),
    }


@pytest.fixture
def methane_molecule(device) -> dict[str, Tensor]:
    """Methane molecule (CH4) in flat format."""
    coord = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # C
            [0.63, 0.63, 0.63],  # H
            [-0.63, -0.63, 0.63],  # H
            [-0.63, 0.63, -0.63],  # H
            [0.63, -0.63, -0.63],  # H
        ],
        device=device,
        dtype=torch.float32,
    )
    numbers = torch.tensor([6, 1, 1, 1, 1], device=device)
    return {
        "coord": coord,
        "numbers": numbers,
        "charge": torch.tensor([0.0], device=device),
    }


@pytest.fixture
def pbc_crystal_small(device) -> dict[str, Tensor]:
    """Small periodic crystal (spiro compound, ~28 atoms) from CIF file."""
    pytest.importorskip("ase", reason="ASE required for loading CIF files")
    import ase.io

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="crystal system.*monoclinic", category=UserWarning)
        atoms = ase.io.read(CIF_SPIRO)
    n_atoms = len(atoms)
    return {
        "coord": torch.tensor(atoms.get_positions(), device=device, dtype=torch.float32),
        "numbers": torch.tensor(atoms.get_atomic_numbers(), device=device),
        "cell": torch.tensor(atoms.get_cell().array, device=device, dtype=torch.float32),
        "pbc": torch.tensor([True, True, True], device=device),
        "charge": torch.tensor([0.0], device=device),
        "mol_idx": torch.zeros(n_atoms, dtype=torch.long, device=device),
    }


@pytest.fixture
def pbc_crystal_large(device) -> dict[str, Tensor]:
    """Larger periodic crystal (phenylglycinium nitrate, ~50 atoms) from CIF file."""
    pytest.importorskip("ase", reason="ASE required for loading CIF files")
    import ase.io

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="crystal system.*monoclinic", category=UserWarning)
        atoms = ase.io.read(CIF_PHENYLGLYCINIUM)
    n_atoms = len(atoms)
    return {
        "coord": torch.tensor(atoms.get_positions(), device=device, dtype=torch.float32),
        "numbers": torch.tensor(atoms.get_atomic_numbers(), device=device),
        "cell": torch.tensor(atoms.get_cell().array, device=device, dtype=torch.float32),
        "pbc": torch.tensor([True, True, True], device=device),
        "charge": torch.tensor([0.0], device=device),
        "mol_idx": torch.zeros(n_atoms, dtype=torch.long, device=device),
    }


@pytest.fixture
def simple_molecule_flat(device) -> dict[str, Tensor]:
    """Simple water molecule for nb_mode=1 tests (flat tensor format)."""
    coord = torch.tensor(
        [
            [0.0000, 0.0000, 0.1173],  # O
            [0.0000, 0.7572, -0.4692],  # H
            [0.0000, -0.7572, -0.4692],  # H
            [0.0000, 0.0000, 0.0000],  # padding
        ],
        device=device,
        dtype=torch.float32,
    )
    numbers = torch.tensor([8, 1, 1, 0], device=device)
    mol_idx = torch.tensor([0, 0, 0, 1], device=device)  # last atom is padding "molecule"
    return {
        "coord": coord,  # (4, 3) - flat format
        "numbers": numbers,  # (4,)
        "mol_idx": mol_idx,
        "charge": torch.tensor([0.0], device=device),
    }


@pytest.fixture
def padded_batch(device) -> dict[str, Tensor]:
    """Batch of 2 molecules with padding (H2O and H2)."""
    # Mol 1: H2O, Mol 2: H2 (padded to 3 atoms)
    coord = torch.tensor(
        [
            # Water
            [
                [0.0000, 0.0000, 0.1173],  # O
                [0.0000, 0.7572, -0.4692],  # H
                [0.0000, -0.7572, -0.4692],  # H
            ],
            # H2 (with padding)
            [
                [0.0000, 0.0000, 0.0000],  # H
                [0.7414, 0.0000, 0.0000],  # H
                [0.0000, 0.0000, 0.0000],  # padding
            ],
        ],
        device=device,
        dtype=torch.float32,
    )
    numbers = torch.tensor(
        [
            [8, 1, 1],  # Water
            [1, 1, 0],  # H2 + padding
        ],
        device=device,
    )
    return {
        "coord": coord,  # (2, 3, 3)
        "numbers": numbers,  # (2, 3)
        "charge": torch.tensor([0.0, 0.0], device=device),
    }


@pytest.fixture
def caffeine_data(device) -> dict[str, Tensor]:
    """Caffeine molecule loaded from xyz file."""
    pytest.importorskip("ase", reason="ASE required for loading xyz files")
    import ase.io

    atoms = ase.io.read(CAFFEINE_FILE)
    data = {
        "coord": torch.tensor(atoms.get_positions(), device=device, dtype=torch.float32).unsqueeze(0),
        "numbers": torch.tensor(atoms.get_atomic_numbers(), device=device).unsqueeze(0),
        "charge": torch.tensor([0.0], device=device),
    }
    return data


@pytest.fixture
def random_coords_100(device) -> tuple[Tensor, Tensor, float, float]:
    """Random 100 atom coordinates for neighbor list tests."""
    torch.manual_seed(42)
    coord = torch.rand((100, 3), device=device) * 10  # 10 Angstrom box
    dmat = torch.cdist(coord, coord)
    dmat[torch.eye(100, dtype=torch.bool, device=device)] = float("inf")
    # Set cutoffs based on quantiles
    dmat_flat = dmat[dmat < float("inf")]
    cutoff1 = torch.quantile(dmat_flat, 0.3).item()
    cutoff2 = torch.quantile(dmat_flat, 0.6).item()
    return coord, dmat, cutoff1, cutoff2


@pytest.fixture
def model_calculator():
    """AIMNet2Calculator instance for integration tests."""
    pytest.importorskip("ase", reason="ASE required for calculator tests")
    from aimnet.calculators import AIMNet2Calculator

    return AIMNet2Calculator("aimnet2", nb_threshold=0)


@pytest.fixture
def pbc_cell(device) -> Tensor:
    """Simple cubic unit cell for PBC tests."""
    return torch.eye(3, device=device) * 10.0  # 10 Angstrom cubic cell


# =============================================================================
# Long-Range Module Test Helpers
# =============================================================================


def add_lr_keys(
    data: dict[str, Tensor],
    device: torch.device,
    module_type: str = "dftd3",
    cutoff: float = 15.0,
) -> dict[str, Tensor]:
    """Add neighbor matrix keys for long-range modules (DFTD3, Coulomb).

    This is a unified helper for adding nbmat_*, shifts_*, cutoff_* keys
    for both DFTD3 and Coulomb modules.

    Parameters
    ----------
    data : dict
        Data dictionary containing coord (and optionally cell for periodic systems).
    device : torch.device
        Device to create tensors on.
    module_type : str
        Either "dftd3" or "coulomb" - determines key naming.
    cutoff : float
        Cutoff distance for neighbor list.

    Returns
    -------
    dict
        Updated data dictionary with neighbor matrix keys.
    """
    coord = data["coord"]

    # Determine key suffix
    suffix = "_dftd3" if module_type == "dftd3" else "_coulomb"

    if coord.ndim == 3:
        # Batched mode (B, N, 3)
        B, N = coord.shape[:2]
        max_nb = N - 1
        nbmat = torch.zeros((B, N, max_nb), dtype=torch.long, device=device)
        for b in range(B):
            for i in range(N):
                neighbors = [j for j in range(N) if j != i]
                for k, nb in enumerate(neighbors[:max_nb]):
                    nbmat[b, i, k] = nb
        shifts = torch.zeros((B, N, max_nb, 3), dtype=torch.int32, device=device)
    else:
        # Flat mode (N, 3)
        N = coord.shape[0]
        max_nb = N - 1
        nbmat = torch.zeros((N, max_nb), dtype=torch.long, device=device)
        for i in range(N):
            neighbors = [j for j in range(N) if j != i]
            for k, nb in enumerate(neighbors[:max_nb]):
                nbmat[i, k] = nb
        shifts = torch.zeros((N, max_nb, 3), dtype=torch.int32, device=device)

    data[f"nbmat{suffix}"] = nbmat
    data[f"shifts{suffix}"] = shifts
    data[f"cutoff{suffix}"] = torch.tensor(cutoff, device=device)

    # Also set nbmat_lr as fallback for resolve_suffix
    if "nbmat_lr" not in data:
        data["nbmat_lr"] = nbmat
        data["shifts_lr"] = shifts

    return data


def create_nacl_crystal(
    device: torch.device,
    n_cells: int = 2,
    lattice_constant: float = 5.64,
) -> dict[str, Tensor]:
    """Create a NaCl crystal supercell for testing periodic electrostatics.

    Parameters
    ----------
    device : torch.device
        Device to create tensors on.
    n_cells : int
        Number of unit cells in each direction.
    lattice_constant : float
        NaCl lattice constant in Angstroms.

    Returns
    -------
    dict
        Data dictionary with coord, charges, cell, and neighbor info.
    """
    from aimnet import nbops

    # NaCl rock salt structure: Na+ at (0,0,0), Cl- at (0.5,0.5,0.5)
    base_positions = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    base_charges = [1.0, -1.0]
    base_numbers = [11, 17]  # Na, Cl

    positions = []
    charges = []
    numbers = []
    for i in range(n_cells):
        for j in range(n_cells):
            for k in range(n_cells):
                offset = [i, j, k]
                for pos, charge, num in zip(base_positions, base_charges, base_numbers, strict=False):
                    positions.append([(p + o) * lattice_constant for p, o in zip(pos, offset, strict=False)])
                    charges.append(charge)
                    numbers.append(num)

    n_atoms = len(positions)
    coord = torch.tensor(positions, dtype=torch.float32, device=device)
    charges_tensor = torch.tensor(charges, dtype=torch.float32, device=device)
    numbers_tensor = torch.tensor(numbers, dtype=torch.long, device=device)
    cell = torch.eye(3, dtype=torch.float32, device=device) * lattice_constant * n_cells
    mol_idx = torch.zeros(n_atoms, dtype=torch.long, device=device)

    # Create neighbor matrix (all pairs within molecule)
    max_nb = n_atoms - 1
    nbmat = torch.zeros((n_atoms, max_nb), dtype=torch.long, device=device)
    nbmat_lr = torch.zeros((n_atoms, max_nb), dtype=torch.long, device=device)
    for i in range(n_atoms):
        neighbors = [j for j in range(n_atoms) if j != i]
        for k, nb in enumerate(neighbors[:max_nb]):
            nbmat[i, k] = nb
            nbmat_lr[i, k] = nb

    # Create shifts (no periodic images in this simple test)
    shifts = torch.zeros((n_atoms, max_nb, 3), dtype=torch.float32, device=device)
    shifts_lr = torch.zeros((n_atoms, max_nb, 3), dtype=torch.int32, device=device)

    data = {
        "coord": coord,
        "charges": charges_tensor,
        "numbers": numbers_tensor,
        "cell": cell,
        "mol_idx": mol_idx,
        "nbmat": nbmat,
        "nbmat_lr": nbmat_lr,
        "shifts": shifts,
        "shifts_lr": shifts_lr,
        "nbmat_coulomb": nbmat_lr,
        "shifts_coulomb": shifts_lr,
        "cutoff_coulomb": torch.tensor(8.0, device=device),
    }
    data = nbops.set_nb_mode(data)
    data = nbops.calc_masks(data)

    return data


def setup_simple_periodic_system(
    device: torch.device,
    n_atoms: int = 4,
) -> dict[str, Tensor]:
    """Create a simple periodic system for testing Ewald.

    Parameters
    ----------
    device : torch.device
        Device to create tensors on.
    n_atoms : int
        Number of atoms in the system.

    Returns
    -------
    dict
        Data dictionary with coord, charges, cell, and neighbor info.
    """
    from aimnet import nbops

    torch.manual_seed(42)

    # Random positions in a cubic cell
    coord = torch.rand((n_atoms, 3), device=device) * 10  # 10 Angstrom box
    charges = torch.tensor([1.0, -1.0, 0.5, -0.5], device=device)[:n_atoms]
    numbers = torch.tensor([6, 1, 7, 8], dtype=torch.long, device=device)[:n_atoms]  # C, H, N, O
    cell = torch.eye(3, device=device) * 10
    mol_idx = torch.zeros(n_atoms, dtype=torch.long, device=device)

    # Create neighbor matrix
    max_nb = n_atoms - 1
    nbmat = torch.zeros((n_atoms, max_nb), dtype=torch.long, device=device)
    nbmat_lr = torch.zeros((n_atoms, max_nb), dtype=torch.long, device=device)
    for i in range(n_atoms):
        neighbors = [j for j in range(n_atoms) if j != i]
        for k, nb in enumerate(neighbors[:max_nb]):
            nbmat[i, k] = nb
            nbmat_lr[i, k] = nb

    shifts = torch.zeros((n_atoms, max_nb, 3), dtype=torch.float32, device=device)
    shifts_lr = torch.zeros((n_atoms, max_nb, 3), dtype=torch.int32, device=device)

    data = {
        "coord": coord,
        "charges": charges,
        "numbers": numbers,
        "cell": cell,
        "mol_idx": mol_idx,
        "nbmat": nbmat,
        "nbmat_lr": nbmat_lr,
        "shifts": shifts,
        "shifts_lr": shifts_lr,
        "nbmat_coulomb": nbmat_lr,
        "shifts_coulomb": shifts_lr,
        "cutoff_coulomb": torch.tensor(8.0, device=device),
    }
    data = nbops.set_nb_mode(data)
    data = nbops.calc_masks(data)

    return data


@pytest.fixture
def nacl_crystal(device) -> dict[str, Tensor]:
    """NaCl crystal fixture for periodic electrostatics tests."""
    return create_nacl_crystal(device)


@pytest.fixture
def simple_periodic_system(device) -> dict[str, Tensor]:
    """Simple periodic system fixture for Ewald tests."""
    return setup_simple_periodic_system(device)
