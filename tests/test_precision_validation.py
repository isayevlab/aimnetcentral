import importlib.util

import numpy as np
import pytest
import torch

from aimnet.validation.precision_metrics import (
    accuracy_pass,
    force_metrics,
    md_drift_metrics,
    output_metrics,
)
from aimnet.validation.stability import atoms_from_data, run_torchsim_static, stability_pass


def test_output_metrics_report_energy_force_charge():
    strict = {
        "energy": torch.tensor([1.0]),
        "forces": torch.zeros(2, 3),
        "charges": torch.tensor([0.25, -0.25]),
    }
    result = {
        "energy": torch.tensor([1.00005]),
        "forces": torch.ones(2, 3) * 1.0e-4,
        "charges": torch.tensor([0.250001, -0.250001]),
    }

    metrics = output_metrics(strict, result, natoms=2, target_charge=0.0)

    assert metrics["energy_mae_ev"] == pytest.approx(5.0e-5, rel=2e-3)
    assert metrics["force_rmse_ev_per_a"] == pytest.approx(1.0e-4, rel=1e-5)
    assert metrics["charge_sum_max_e"] == pytest.approx(0.0, abs=1e-7)
    assert accuracy_pass(metrics)


def test_force_metrics_include_tail_and_cosine():
    strict = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    result = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.9, 0.1]])

    metrics = force_metrics(result, strict)

    assert metrics["force_max_ev_per_a"] == pytest.approx(0.1)
    assert metrics["force_norm_max_ev_per_a"] > 0.1
    assert metrics["force_cosine_min"] < 1.0


def test_md_drift_metrics_fit_linear_slope():
    energy = torch.tensor([0.0, 0.01, 0.02, 0.03], dtype=torch.float64)

    metrics = md_drift_metrics(energy, natoms=2, timestep_fs=1.0)

    assert metrics["n_steps"] == 4
    assert metrics["total_energy_drift_ev_per_ps"] == pytest.approx(10.0)
    assert metrics["total_energy_drift_ev_per_atom_ps"] == pytest.approx(5.0)
    assert metrics["finite"] is True


def test_atoms_from_data_preserves_cell_charge_and_pbc():
    pytest.importorskip("ase")
    data = {
        "coord": [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0]],
        "numbers": [8, 1],
        "charge": 1.0,
        "cell": np.eye(3) * 8.0,
        "pbc": [True, False, True],
    }

    atoms = atoms_from_data(data)

    assert len(atoms) == 2
    assert atoms.info["charge"] == 1.0
    assert atoms.pbc.tolist() == [True, False, True]
    assert atoms.cell.lengths()[0] == pytest.approx(8.0)


def test_stability_pass_compares_nve_drift_to_strict():
    strict = {"backend": "ase_nve", "skipped": False, "finite": True, "total_energy_drift_ev_per_atom_ps": 1.0}
    good = {"backend": "ase_nve", "skipped": False, "finite": True, "total_energy_drift_ev_per_atom_ps": 1.1}
    bad = {"backend": "ase_nve", "skipped": False, "finite": True, "total_energy_drift_ev_per_atom_ps": 2.0}

    assert stability_pass(good, strict, drift_ratio=1.2) is True
    assert stability_pass(bad, strict, drift_ratio=1.2) is False
    assert stability_pass({"backend": "torchsim_static", "skipped": True}, strict) is None


def test_torchsim_static_skips_when_optional_dependency_missing():
    if importlib.util.find_spec("torch_sim") is not None:
        pytest.skip("TorchSim is installed in this environment")

    result = run_torchsim_static(
        model="aimnet2",
        precision="strict",
        data={"coord": [[0.0, 0.0, 0.0]], "numbers": [1], "charge": 0.0},
        device="cpu",
    )

    assert result["skipped"] is True
    assert result["backend"] == "torchsim_static"
