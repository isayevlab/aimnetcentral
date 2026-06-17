"""Optional ASE/TorchSim stability checks for precision modes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, cast

import numpy as np
import torch

from aimnet.calculators import AIMNet2Calculator
from aimnet.validation.precision_metrics import md_drift_metrics


def atoms_from_data(data: Mapping[str, Any]):
    try:
        from ase import Atoms
    except ImportError as exc:
        raise RuntimeError("ASE is required for atoms_from_data") from exc

    atoms = Atoms(
        numbers=np.asarray(data["numbers"], dtype=np.int64),
        positions=np.asarray(data["coord"], dtype=np.float64),
    )
    if data.get("cell") is not None:
        atoms.set_cell(np.asarray(data["cell"], dtype=np.float64))
        atoms.pbc = np.asarray(data.get("pbc", [True, True, True]), dtype=bool)
    if data.get("charge") is not None:
        atoms.info["charge"] = _scalar_or_first(data["charge"])
    if data.get("mult") is not None:
        atoms.info["mult"] = _scalar_or_first(data["mult"])
    return atoms


def run_ase_geometry_optimization(
    *,
    model: str,
    precision: str,
    data: Mapping[str, Any],
    device: str,
    max_steps: int,
    fmax: float,
    coulomb: Mapping[str, Any] | None = None,
    compile_model: bool = False,
) -> dict[str, Any]:
    try:
        from ase.optimize import BFGS

        from aimnet.calculators import AIMNet2ASE
    except ImportError as exc:
        return {"backend": "ase_geo", "skipped": True, "reason": str(exc)}

    atoms = atoms_from_data(data)
    calc = _make_calc(model, precision, device, coulomb, compile_model=compile_model)
    atoms.calc = AIMNet2ASE(calc)
    initial_energy = float(atoms.get_potential_energy())
    opt = BFGS(atoms, logfile=None)
    opt.run(fmax=fmax, steps=max_steps)
    forces = atoms.get_forces()
    final_energy = float(atoms.get_potential_energy())
    final_fmax = float(np.linalg.norm(forces, axis=1).max())
    finite = bool(np.isfinite(final_energy) and np.isfinite(forces).all())
    return {
        "backend": "ase_geo",
        "skipped": False,
        "steps": int(opt.nsteps),
        "converged": bool(final_fmax <= fmax),
        "initial_energy_ev": initial_energy,
        "final_energy_ev": final_energy,
        "final_fmax_ev_per_a": final_fmax,
        "finite": finite,
    }


def run_ase_nve(
    *,
    model: str,
    precision: str,
    data: Mapping[str, Any],
    device: str,
    steps: int,
    timestep_fs: float,
    temperature_k: float,
    coulomb: Mapping[str, Any] | None = None,
    compile_model: bool = False,
    seed: int = 0,
) -> dict[str, Any]:
    try:
        from ase import units
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
        from ase.md.verlet import VelocityVerlet

        from aimnet.calculators import AIMNet2ASE
    except ImportError as exc:
        return {"backend": "ase_nve", "skipped": True, "reason": str(exc)}

    atoms = atoms_from_data(data)
    atoms.calc = AIMNet2ASE(_make_calc(model, precision, device, coulomb, compile_model=compile_model))
    rng = np.random.default_rng(seed)
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_k, rng=rng)
    Stationary(atoms)
    ZeroRotation(atoms)

    energies: list[float] = []
    max_force = 0.0

    def record() -> None:
        nonlocal max_force
        forces = atoms.get_forces()
        energies.append(float(atoms.get_potential_energy() + atoms.get_kinetic_energy()))
        max_force = max(max_force, float(np.linalg.norm(forces, axis=1).max()))

    dyn = VelocityVerlet(atoms, timestep=timestep_fs * units.fs)
    dyn.attach(record, interval=1)
    record()
    dyn.run(steps)
    energy_t = torch.tensor(energies, dtype=torch.float64)
    metrics: dict[str, Any] = dict(md_drift_metrics(energy_t, natoms=len(atoms), timestep_fs=timestep_fs))
    metrics.update(
        {
            "backend": "ase_nve",
            "skipped": False,
            "temperature_k": temperature_k,
            "timestep_fs": timestep_fs,
            "max_force_ev_per_a": max_force,
        }
    )
    return metrics


def run_torchsim_static(
    *,
    model: str,
    precision: str,
    data: Mapping[str, Any],
    device: str,
    coulomb: Mapping[str, Any] | None = None,
    compile_model: bool = False,
) -> dict[str, Any]:
    try:
        import torch_sim as ts

        from aimnet.calculators import AIMNet2TorchSim
    except ImportError as exc:
        return {"backend": "torchsim_static", "skipped": True, "reason": str(exc)}

    atoms = atoms_from_data(data)
    calc = _make_calc(model, precision, device, coulomb, compile_model=compile_model)
    results = ts.static(system=atoms, model=AIMNet2TorchSim(calc))
    row = results[0]
    return {
        "backend": "torchsim_static",
        "skipped": False,
        "energy_ev": float(row["potential_energy"].detach().cpu().item()),
        "forces_finite": bool(torch.isfinite(row["forces"]).all().item()),
    }


def stability_pass(mode: Mapping[str, Any], strict: Mapping[str, Any] | None = None, *, drift_ratio: float = 1.2) -> bool | None:
    if mode.get("skipped"):
        return None
    if not mode.get("finite", mode.get("forces_finite", True)):
        return False
    if mode.get("backend") == "ase_nve" and strict is not None and not strict.get("skipped"):
        strict_drift = abs(float(strict.get("total_energy_drift_ev_per_atom_ps") or 0.0))
        mode_drift = abs(float(mode.get("total_energy_drift_ev_per_atom_ps") or 0.0))
        return mode_drift <= strict_drift * drift_ratio + 1.0e-6
    return True


def _make_calc(
    model: str,
    precision: str,
    device: str,
    coulomb: Mapping[str, Any] | None,
    *,
    compile_model: bool,
) -> AIMNet2Calculator:
    calc = AIMNet2Calculator(model, device=device, nb_threshold=0, precision=precision, compile_model=compile_model)
    if coulomb:
        method = cast(Literal["simple", "dsf", "ewald", "pme"], str(coulomb.get("method", "dsf")))
        kwargs = {k: v for k, v in coulomb.items() if k != "method"}
        calc.set_lrcoulomb_method(method, **kwargs)
    return calc


def _scalar_or_first(value: Any) -> float:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    return float(arr[0])
