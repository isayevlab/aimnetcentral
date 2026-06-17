"""Local comparison metrics for optimization benchmark outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


@dataclass(frozen=True)
class AccuracyGates:
    energy_mae_ev: float = 1.0e-4
    energy_mae_ev_per_atom: float = 1.0e-6
    force_rmse_ev_per_a: float = 1.0e-3
    force_max_ev_per_a: float = 1.0e-2
    charge_sum_e: float = 1.0e-5
    charge_sum_per_atom_e: float = 1.0e-7
    stress_max: float = 1.0e-3
    hessian_symmetry_max: float = 1.0e-4


DEFAULT_ACCURACY_GATES = AccuracyGates()


def tensor_mae(a: Tensor | None, b: Tensor | None) -> float | None:
    if a is None or b is None:
        return None
    return float((a.detach() - b.detach().to(a.device, a.dtype)).abs().mean().item())


def tensor_rmse(a: Tensor | None, b: Tensor | None) -> float | None:
    if a is None or b is None:
        return None
    diff = a.detach() - b.detach().to(a.device, a.dtype)
    return float(diff.pow(2).mean().sqrt().item())


def tensor_max_abs(a: Tensor | None, b: Tensor | None) -> float | None:
    if a is None or b is None:
        return None
    return float((a.detach() - b.detach().to(a.device, a.dtype)).abs().max().item())


def energy_metrics(
    mode_energy: Tensor | None,
    strict_energy: Tensor | None,
    natoms: int | Tensor | None = None,
) -> dict[str, float | None]:
    mae = tensor_mae(mode_energy, strict_energy)
    max_abs = tensor_max_abs(mode_energy, strict_energy)
    per_atom = None
    if mae is not None and natoms is not None:
        n = int(natoms.item()) if isinstance(natoms, Tensor) else int(natoms)
        per_atom = mae / max(n, 1)
    return {"energy_mae_ev": mae, "energy_max_ev": max_abs, "energy_per_atom_mae_ev": per_atom}


def force_metrics(mode_forces: Tensor | None, strict_forces: Tensor | None) -> dict[str, float | None]:
    if mode_forces is None or strict_forces is None:
        return {
            "force_component_mae_ev_per_a": None,
            "force_rmse_ev_per_a": None,
            "force_max_ev_per_a": None,
            "force_norm_max_ev_per_a": None,
            "force_cosine_min": None,
        }
    mode = mode_forces.detach()
    strict = strict_forces.detach().to(mode.device, mode.dtype)
    diff = mode - strict
    norm_diff = diff.reshape(-1, 3).norm(dim=-1)
    mode_norm = mode.reshape(-1, 3).norm(dim=-1)
    strict_norm = strict.reshape(-1, 3).norm(dim=-1)
    cosine = (mode.reshape(-1, 3) * strict.reshape(-1, 3)).sum(dim=-1) / (mode_norm * strict_norm + 1.0e-12)
    return {
        "force_component_mae_ev_per_a": float(diff.abs().mean().item()),
        "force_rmse_ev_per_a": float(diff.pow(2).mean().sqrt().item()),
        "force_max_ev_per_a": float(diff.abs().max().item()),
        "force_norm_max_ev_per_a": float(norm_diff.max().item()),
        "force_cosine_min": float(cosine.min().item()),
    }


def charge_metrics(
    mode_charges: Tensor | None,
    strict_charges: Tensor | None,
    target_charge: Any,
    natoms: int | Tensor | None = None,
) -> dict[str, float | None]:
    charge_sum = charge_sum_error(mode_charges, target_charge)
    charge_sum_per_atom = None
    if charge_sum is not None and natoms is not None:
        n = int(natoms.item()) if isinstance(natoms, Tensor) else int(natoms)
        charge_sum_per_atom = charge_sum / max(n, 1)
    return {
        "charge_mae_e": tensor_mae(mode_charges, strict_charges),
        "charge_max_e": tensor_max_abs(mode_charges, strict_charges),
        "charge_sum_max_e": charge_sum,
        "charge_sum_per_atom_e": charge_sum_per_atom,
    }


def charge_sum_error(charges: Tensor | None, target_charge: Any) -> float | None:
    if charges is None or target_charge is None:
        return None
    q = charges.detach()
    if q.ndim == 1:
        sums = q.sum().reshape(1)
    elif q.ndim == 2:
        sums = q.sum(dim=1)
    else:
        sums = q.flatten(1).sum(dim=1)
    target = torch.as_tensor(target_charge, device=q.device, dtype=q.dtype).reshape(-1)
    if target.numel() == 1 and sums.numel() > 1:
        target = target.expand_as(sums)
    return float((sums - target).abs().max().item())


def stress_metrics(mode_stress: Tensor | None, strict_stress: Tensor | None) -> dict[str, float | None]:
    symmetry = None
    if mode_stress is not None:
        stress = mode_stress.detach()
        symmetry = float((stress - stress.mT).abs().max().item())
    return {"stress_max": tensor_max_abs(mode_stress, strict_stress), "stress_symmetry_max": symmetry}


def hessian_metrics(mode_hessian: Tensor | None, strict_hessian: Tensor | None) -> dict[str, float | None]:
    if mode_hessian is None:
        return {"hessian_max": None, "hessian_symmetry_max": None, "hessian_sum_rule_max": None}
    hessian = mode_hessian.detach()
    sym = None
    sum_rule = None
    if hessian.ndim == 4:
        flat = hessian.reshape(hessian.shape[0] * 3, hessian.shape[2] * 3)
        sym = float((flat - flat.T).abs().max().item())
        sum_rule = float(hessian.sum(dim=2).abs().max().item())
    return {
        "hessian_max": tensor_max_abs(mode_hessian, strict_hessian),
        "hessian_symmetry_max": sym,
        "hessian_sum_rule_max": sum_rule,
    }


def output_metrics(
    strict: dict[str, Tensor],
    result: dict[str, Tensor],
    *,
    natoms: int | Tensor | None = None,
    target_charge: Any = None,
) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {}
    metrics.update(energy_metrics(result.get("energy"), strict.get("energy"), natoms))
    metrics.update(force_metrics(result.get("forces"), strict.get("forces")))
    metrics.update(charge_metrics(result.get("charges"), strict.get("charges"), target_charge, natoms))
    metrics.update(stress_metrics(result.get("stress"), strict.get("stress")))
    metrics.update(hessian_metrics(result.get("hessian"), strict.get("hessian")))
    return metrics


def accuracy_pass(metrics: dict[str, float | None], gates: AccuracyGates = DEFAULT_ACCURACY_GATES) -> bool:
    checks = (
        ("force_rmse_ev_per_a", gates.force_rmse_ev_per_a),
        ("force_max_ev_per_a", gates.force_max_ev_per_a),
        ("stress_max", gates.stress_max),
        ("hessian_symmetry_max", gates.hessian_symmetry_max),
    )
    for name, limit in checks:
        value = metrics[name]
        if value is not None and value > limit:
            return False
    energy_mae = metrics["energy_mae_ev"]
    energy_per_atom = metrics.get("energy_per_atom_mae_ev")
    if energy_mae is not None and energy_mae > gates.energy_mae_ev:
        if energy_per_atom is None or energy_per_atom > gates.energy_mae_ev_per_atom:
            return False
    charge_sum = metrics["charge_sum_max_e"]
    charge_sum_per_atom = metrics.get("charge_sum_per_atom_e")
    if charge_sum is not None and charge_sum > gates.charge_sum_e:
        if charge_sum_per_atom is None or charge_sum_per_atom > gates.charge_sum_per_atom_e:
            return False
    return True


def md_drift_metrics(total_energy_ev: Tensor, *, natoms: int, timestep_fs: float) -> dict[str, float | None]:
    energy = total_energy_ev.detach().double().flatten()
    if energy.numel() < 2:
        return {
            "n_steps": int(energy.numel()),
            "total_energy_drift_ev_per_ps": None,
            "total_energy_drift_ev_per_atom_ps": None,
            "total_energy_std_ev": None,
            "finite": bool(torch.isfinite(energy).all().item()),
        }
    time_ps = torch.arange(energy.numel(), device=energy.device, dtype=torch.float64) * (timestep_fs / 1000.0)
    x = time_ps - time_ps.mean()
    y = energy - energy.mean()
    denom = x.pow(2).sum()
    slope = torch.tensor(0.0, dtype=torch.float64, device=energy.device) if denom == 0 else (x * y).sum() / denom
    drift = float(slope.item())
    return {
        "n_steps": int(energy.numel()),
        "total_energy_drift_ev_per_ps": drift,
        "total_energy_drift_ev_per_atom_ps": drift / max(natoms, 1),
        "total_energy_std_ev": float(energy.std(unbiased=False).item()),
        "finite": bool(torch.isfinite(energy).all().item()),
    }
