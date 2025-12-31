"""Shared pytest fixtures for AIMNet2 tests."""

import os

import pytest
import torch
from torch import Tensor

# Test data directory
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CAFFEINE_FILE = os.path.join(DATA_DIR, "caffeine.xyz")


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
