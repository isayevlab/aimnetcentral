"""Tests for the TorchSim wrapper around AIMNet2Calculator."""

import warnings

import pytest

ts = pytest.importorskip("torch_sim")
pytest.importorskip("ase")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from ase import Atoms  # noqa: E402
from ase.io import read  # noqa: E402
from conftest import CAFFEINE_FILE, CIF_SPIRO  # noqa: E402

from aimnet.calculators import AIMNet2Calculator, AIMNet2TorchSim  # noqa: E402

pytestmark = pytest.mark.torch_sim


class _FakeAIMNet2Calculator:
    def __init__(self, *, is_nse: bool = False) -> None:
        self.device = "cpu"
        self.is_nse = is_nse
        self.last_data = None
        self.last_forces = None
        self.last_stress = None
        self.last_validate_species = None

    @property
    def metadata(self):
        return {"family": "fake"}

    def __call__(self, data, *, forces: bool, stress: bool, validate_species: bool = True):
        self.last_data = data
        self.last_forces = forces
        self.last_stress = stress
        self.last_validate_species = validate_species
        data["coord"].requires_grad_(True)
        n_systems = int(data["mol_idx"].max().item()) + 1
        results = {
            "energy": torch.zeros(n_systems, requires_grad=True),
            "charges": torch.zeros(data["coord"].shape[0], requires_grad=True),
        }
        if forces:
            results["forces"] = torch.zeros_like(data["coord"], requires_grad=True)
        if self.is_nse:
            results["spin_charges"] = torch.zeros(data["coord"].shape[0], requires_grad=True)
        if stress:
            results["stress"] = torch.zeros((n_systems, 3, 3), requires_grad=True)
        return results


def _read_spiro():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="crystal system.*monoclinic", category=UserWarning)
        return read(CIF_SPIRO)


def test_forward_clones_positions_and_detaches_outputs():
    atoms = read(CAFFEINE_FILE)
    base_calc = _FakeAIMNet2Calculator()
    wrapper = AIMNet2TorchSim(base_calc)
    state = ts.io.atoms_to_state([atoms], device=wrapper.device, dtype=wrapper.dtype)
    original_positions = state.positions.clone()

    results = wrapper(state)

    assert not state.positions.requires_grad
    assert torch.equal(state.positions, original_positions)
    assert base_calc.last_data["coord"].data_ptr() != state.positions.data_ptr()
    assert torch.equal(base_calc.last_data["charge"], torch.zeros(1))
    assert results["partial_charges"].data_ptr() == results["charges"].data_ptr()
    assert all(not value.requires_grad for value in results.values())


def test_system_extras_map_charge_and_multiplicity():
    atoms = Atoms(
        "CH3",
        positions=[
            [0.0, 0.0, 0.0],
            [1.079, 0.0, 0.0],
            [-0.5395, 0.9344, 0.0],
            [-0.5395, -0.9344, 0.0],
        ],
        info={"charge": 0, "mult": 2},
    )
    base_calc = _FakeAIMNet2Calculator(is_nse=True)
    wrapper = AIMNet2TorchSim(base_calc)
    state = ts.io.atoms_to_state(
        [atoms],
        device=wrapper.device,
        dtype=wrapper.dtype,
        system_extras_map={"charge": "charge", "mult": "mult"},
    )

    results = wrapper(state)

    assert torch.equal(base_calc.last_data["charge"], torch.tensor([0.0]))
    assert torch.equal(base_calc.last_data["mult"], torch.tensor([2.0]))
    assert "spin_charges" in wrapper.implemented_properties
    assert "spin_charges" in results


def test_spin_extra_falls_back_to_multiplicity():
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]], info={"spin": 2})
    base_calc = _FakeAIMNet2Calculator(is_nse=True)
    wrapper = AIMNet2TorchSim(base_calc)
    state = ts.io.atoms_to_state(
        [atoms],
        device=wrapper.device,
        dtype=wrapper.dtype,
        system_extras_map={"spin": "spin"},
    )

    wrapper(state)

    assert torch.equal(base_calc.last_data["mult"], torch.tensor([2.0]))


def test_pbc_state_forwards_cell_pbc_and_stress_request():
    atoms = _read_spiro()
    base_calc = _FakeAIMNet2Calculator()
    wrapper = AIMNet2TorchSim(base_calc, compute_stress=True)
    state = ts.io.atoms_to_state([atoms], device=wrapper.device, dtype=wrapper.dtype)

    results = wrapper(state)

    assert base_calc.last_data["cell"].shape == (1, 3, 3)
    assert base_calc.last_data["pbc"].any()
    assert "stress" in wrapper.implemented_properties
    assert results["stress"].shape == (1, 3, 3)


def test_compute_toggles_update_properties_and_forward_request():
    atoms = read(CAFFEINE_FILE)
    base_calc = _FakeAIMNet2Calculator()
    wrapper = AIMNet2TorchSim(base_calc, compute_forces=False, validate_species=False)
    state = ts.io.atoms_to_state([atoms], device=wrapper.device, dtype=torch.float64)

    results = wrapper(state)

    assert "forces" not in wrapper.implemented_properties
    assert "forces" not in results
    assert base_calc.last_forces is False
    assert base_calc.last_validate_species is False
    assert base_calc.last_data["coord"].dtype == torch.float32

    wrapper.compute_forces = True
    wrapper.compute_stress = True
    assert "forces" in wrapper.implemented_properties
    assert "stress" in wrapper.implemented_properties


def test_stress_without_periodic_cell_raises_clear_error():
    atoms = read(CAFFEINE_FILE)
    wrapper = AIMNet2TorchSim(_FakeAIMNet2Calculator(), compute_stress=True)
    state = ts.io.atoms_to_state([atoms], device=wrapper.device, dtype=wrapper.dtype)

    with pytest.raises(ValueError, match="stress calculation requires a periodic TorchSim state"):
        wrapper(state)


def test_system_extra_shape_validation():
    atoms = read(CAFFEINE_FILE)
    wrapper = AIMNet2TorchSim(_FakeAIMNet2Calculator())
    state = ts.io.atoms_to_state([atoms], device=wrapper.device, dtype=wrapper.dtype)
    state = ts.SimState.from_state(state, charge=torch.zeros(state.n_atoms))

    with pytest.raises(ValueError, match="one value per system"):
        wrapper(state)


def test_default_properties_and_metadata():
    wrapper = AIMNet2TorchSim(_FakeAIMNet2Calculator())

    assert wrapper.implemented_properties == ["energy", "forces", "charges", "partial_charges"]
    assert wrapper.metadata == {"family": "fake"}
    assert wrapper.base_calc is not None


@pytest.mark.slow
def test_static_energy_smoke():
    atoms = read(CAFFEINE_FILE)
    base_calc = AIMNet2Calculator("aimnet2")
    state = ts.static(system=atoms, model=AIMNet2TorchSim(base_calc))

    energy = state[0]["potential_energy"].item()
    forces = state[0]["forces"].cpu().numpy()
    assert isinstance(energy, float)
    assert np.isfinite(energy)
    assert forces.shape == (len(atoms), 3)
    assert np.isfinite(forces).all()


@pytest.mark.slow
def test_periodic_stress_smoke():
    atoms = _read_spiro()
    base_calc = AIMNet2Calculator("aimnet2")
    results = ts.static(system=atoms, model=AIMNet2TorchSim(base_calc, compute_stress=True))

    stress = results[0]["stress"].cpu().numpy()
    assert stress.shape == (1, 3, 3)
    assert np.isfinite(stress).all()
