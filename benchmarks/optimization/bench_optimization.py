from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aimnet.calculators import AIMNet2Calculator  # noqa: E402
from local_metrics import accuracy_pass, output_metrics  # noqa: E402
from local_stability import (  # noqa: E402
    run_ase_geometry_optimization,
    run_ase_nve,
    run_torchsim_static,
    stability_pass,
)


@dataclass(frozen=True)
class ExecutionMode:
    name: str
    nb_threshold: int
    resident_input: bool = False
    cache_static: bool = False
    neighbor_skin: float = 0.0


def main() -> int:
    args = parse_args()
    args.modes = expand_modes(args.modes)

    workloads = load_workloads(args.workloads)
    if args.list_workloads:
        for name in workloads:
            print(name)
        return 0

    selected = args.workload or list(workloads)
    unknown = [name for name in selected if name not in workloads]
    if unknown:
        raise SystemExit(f"Unknown workload(s): {', '.join(unknown)}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    hardware = detect_hardware(args.device)
    software = detect_software()
    git = git_metadata()
    command = [Path(sys.argv[0]).name, *sys.argv[1:]]

    with output.open("a", encoding="utf-8") as fh:
        for workload_name in selected:
            workload = workloads[workload_name]
            data = build_data(workload)
            request = dict(workload.get("request", {}))
            strict_ref = run_once(args, workload, data, request, mode_config("strict", args))
            strict_timing = None
            strict_stability: dict[str, dict[str, Any]] = {}

            for mode_name in args.modes:
                mode = mode_config(mode_name, args)
                if not mode_supported(mode, args.device):
                    row = skipped_row(
                        args,
                        workload_name,
                        mode,
                        hardware,
                        software,
                        git,
                        command,
                        "execution mode is unsupported on device",
                    )
                    fh.write(json.dumps(row) + "\n")
                    continue

                result, timing = run_timed(args, workload, data, request, mode)
                metrics = accuracy_metrics(strict_ref, result, data)
                stability = run_stability_checks(args, workload, data, mode)
                strict_path = effective_execution_path(data, request, mode_config("strict", args), args.device)
                mode_path = effective_execution_path(data, request, mode, args.device)
                equivalent_to_strict = mode.name != "strict" and not mode.resident_input and mode_path == strict_path
                if mode.name == "strict":
                    strict_timing = timing
                    strict_stability = stability
                speedup = None
                if strict_timing is not None and mode.name != "strict":
                    speedup = strict_timing["median_ms"] / timing["median_ms"] if timing["median_ms"] > 0 else None
                gates = gates_for(
                    metrics,
                    speedup,
                    args.min_speedup,
                    stability,
                    strict_stability,
                    speedup_reportable=not equivalent_to_strict,
                )
                row = result_row(
                    args=args,
                    workload_name=workload_name,
                    workload=workload,
                    mode=mode,
                    hardware=hardware,
                    software=software,
                    git=git,
                    command=command,
                    timing=timing,
                    accuracy=metrics,
                    stability=stability,
                    gates=gates,
                    speedup=speedup,
                    data=data,
                    request=request,
                    effective_path=mode_path,
                    equivalent_to_strict=equivalent_to_strict,
                )
                fh.write(json.dumps(row) + "\n")
                fh.flush()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AIMNet GPU optimization benchmark runner")
    parser.add_argument("--model", default="aimnet2")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["strict", "forced-sparse"],
        help="Execution modes: strict, dense-default, forced-sparse, gpu-resident, static-cache, neighbor-skin, or all",
    )
    parser.add_argument("--workloads", default=str(ROOT / "benchmarks/optimization/workloads.yaml"))
    parser.add_argument("--workload", action="append", help="Run one workload; may be supplied multiple times")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--output", default=str(ROOT / "results/optimization_bench.jsonl"))
    parser.add_argument(
        "--nb-threshold",
        type=int,
        default=120,
        help="Default dense/sparse threshold. forced-sparse uses 0 regardless of this value.",
    )
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument(
        "--neighbor-skin",
        type=float,
        default=0.5,
        help="Skin distance in Angstroms for the neighbor-skin execution mode.",
    )
    parser.add_argument("--min-speedup", type=float, default=1.05)
    parser.add_argument(
        "--stability",
        nargs="+",
        choices=["ase-geo", "ase-nve", "torchsim-static"],
        default=[],
        help="Optional stability checks to run per mode",
    )
    parser.add_argument("--stability-steps", type=int, default=10)
    parser.add_argument("--stability-fmax", type=float, default=0.05)
    parser.add_argument("--md-timestep-fs", type=float, default=0.5)
    parser.add_argument("--md-temperature-k", type=float, default=300.0)
    parser.add_argument("--list-workloads", action="store_true")
    return parser.parse_args()


def expand_modes(modes: list[str]) -> list[str]:
    if modes == ["all"]:
        return ["strict", "dense-default", "forced-sparse", "gpu-resident", "static-cache", "neighbor-skin"]
    allowed = {"strict", "dense-default", "forced-sparse", "gpu-resident", "static-cache", "neighbor-skin"}
    unknown = [mode for mode in modes if mode not in allowed]
    if unknown:
        raise SystemExit(f"Unknown execution mode(s): {', '.join(unknown)}")
    return modes


def mode_config(name: str, args: argparse.Namespace) -> ExecutionMode:
    if name == "forced-sparse":
        return ExecutionMode(name=name, nb_threshold=0)
    if name == "gpu-resident":
        return ExecutionMode(name=name, nb_threshold=args.nb_threshold, resident_input=True)
    if name == "static-cache":
        return ExecutionMode(name=name, nb_threshold=args.nb_threshold, resident_input=True, cache_static=True)
    if name == "neighbor-skin":
        return ExecutionMode(
            name=name,
            nb_threshold=args.nb_threshold,
            resident_input=True,
            neighbor_skin=float(args.neighbor_skin),
        )
    if name in {"strict", "dense-default"}:
        return ExecutionMode(name=name, nb_threshold=args.nb_threshold)
    raise ValueError(f"Unknown execution mode: {name}")


def mode_supported(mode: ExecutionMode, device: str) -> bool:
    if mode.resident_input or mode.neighbor_skin > 0.0:
        return torch.device(device).type == "cuda" and torch.cuda.is_available()
    return True


def effective_execution_path(
    data: dict[str, Any],
    request: dict[str, Any],
    mode: ExecutionMode,
    device: str,
) -> str:
    """Best-effort label for the calculator path a benchmark mode will exercise."""
    coord = torch.as_tensor(data["coord"])
    hessian = bool(request.get("hessian", False))
    if hessian:
        return "sparse-hessian"
    if data.get("cell") is not None:
        return "sparse-pbc"
    if "nbmat" in data:
        return "caller-nbmat"
    if torch.device(device).type == "cuda" and mode.nb_threshold > 0:
        if coord.ndim == 2 and "mol_idx" not in data and coord.shape[0] <= mode.nb_threshold:
            return "dense-single"
        if coord.ndim == 3 and coord.shape[1] <= mode.nb_threshold:
            return "dense-batch"
    if mode.neighbor_skin > 0.0:
        return "sparse-neighbor-skin"
    return "sparse"


def load_workloads(path: str | Path) -> dict[str, dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh) or {}
    workloads = doc.get("workloads")
    if not isinstance(workloads, dict):
        raise TypeError("workload file must contain a top-level 'workloads' mapping")
    return workloads


def build_data(workload: dict[str, Any]) -> dict[str, Any]:
    if "data" in workload:
        return dict(workload["data"])
    if "extxyz" in workload:
        return load_extxyz(ROOT / workload["extxyz"], float(workload.get("charge", 0.0)))
    if "generate" in workload:
        return generate_data(workload["generate"])
    raise ValueError("workload must define data, extxyz, or generate")


def load_extxyz(path: Path, charge: float) -> dict[str, Any]:
    try:
        import ase.io
    except ImportError as exc:
        raise RuntimeError(f"ASE is required to load {path}") from exc
    atoms = ase.io.read(path)
    return {
        "coord": atoms.get_positions(),
        "numbers": atoms.get_atomic_numbers(),
        "charge": charge,
    }


def generate_data(spec: dict[str, Any]) -> dict[str, Any]:
    if spec.get("kind") == "water_cluster":
        return generate_water_cluster(spec)
    if spec.get("kind") != "organic_cluster":
        raise ValueError(f"unknown generator kind: {spec.get('kind')}")
    gen = torch.Generator().manual_seed(int(spec.get("seed", 0)))
    atoms = int(spec["atoms"])
    box = float(spec.get("box", 20.0))
    coord = torch.rand((atoms, 3), generator=gen) * box
    pool = torch.tensor([1, 1, 1, 1, 6, 6, 7, 8], dtype=torch.long)
    numbers = pool[torch.randint(0, len(pool), (atoms,), generator=gen)]
    return {"coord": coord, "numbers": numbers, "charge": float(spec.get("charge", 0.0))}


def generate_water_cluster(spec: dict[str, Any]) -> dict[str, Any]:
    molecules = int(spec["molecules"])
    spacing = float(spec.get("spacing", 3.2))
    jitter = float(spec.get("jitter", 0.05))
    seed = int(spec.get("seed", 0))
    gen = torch.Generator().manual_seed(seed)
    side = int(torch.ceil(torch.tensor(float(molecules) ** (1.0 / 3.0))).item())
    water_coord = torch.tensor(
        [
            [0.0000, 0.0000, 0.0000],
            [0.9572, 0.0000, 0.0000],
            [-0.2390, 0.9270, 0.0000],
        ],
        dtype=torch.float32,
    )
    water_numbers = torch.tensor([8, 1, 1], dtype=torch.long)
    coords = []
    numbers = []
    for i in range(molecules):
        ix = i % side
        iy = (i // side) % side
        iz = i // (side * side)
        origin = torch.tensor([ix, iy, iz], dtype=torch.float32) * spacing
        offset = (torch.rand((1, 3), generator=gen) - 0.5) * jitter
        coords.append(water_coord + origin + offset)
        numbers.append(water_numbers)
    coord = torch.cat(coords, dim=0)
    nums = torch.cat(numbers, dim=0)
    return {"coord": coord, "numbers": nums, "charge": float(spec.get("charge", 0.0))}


def make_calc(args: argparse.Namespace, workload: dict[str, Any], mode: ExecutionMode) -> AIMNet2Calculator:
    calc = AIMNet2Calculator(
        args.model,
        device=args.device,
        nb_threshold=mode.nb_threshold,
        compile_model=args.compile_model,
        cache_static=mode.cache_static,
        neighbor_skin=mode.neighbor_skin,
    )
    coulomb = workload.get("coulomb")
    if coulomb:
        method = coulomb.get("method", "dsf")
        kwargs = {k: v for k, v in coulomb.items() if k != "method"}
        calc.set_lrcoulomb_method(method, **kwargs)
    return calc


def run_once(
    args: argparse.Namespace,
    workload: dict[str, Any],
    data: dict[str, Any],
    request: dict[str, Any],
    mode: ExecutionMode,
) -> dict[str, torch.Tensor]:
    calc = make_calc(args, workload, mode)
    input_data = resident_data(data, args.device) if mode.resident_input else clone_data(data)
    with torch.inference_mode(mode=not any(request.get(k, False) for k in ("forces", "stress", "hessian"))):
        return calc(input_data, **request)


def run_timed(
    args: argparse.Namespace,
    workload: dict[str, Any],
    data: dict[str, Any],
    request: dict[str, Any],
    mode: ExecutionMode,
) -> tuple[dict[str, torch.Tensor], dict[str, float | int | str | bool]]:
    calc = make_calc(args, workload, mode)
    input_data = resident_data(data, args.device) if mode.resident_input else data
    for _ in range(args.warmup):
        calc(timing_input(input_data, mode), **request)
    sync(args.device)
    times: list[float] = []
    result: dict[str, torch.Tensor] | None = None
    for _ in range(args.repeat):
        sync(args.device)
        start = time.perf_counter()
        result = calc(timing_input(input_data, mode), **request)
        sync(args.device)
        times.append((time.perf_counter() - start) * 1000.0)
    assert result is not None
    return result, {
        "warmup": args.warmup,
        "repeat": args.repeat,
        "median_ms": statistics.median(times),
        "p95_ms": percentile(times, 0.95),
        "min_ms": min(times),
        "max_ms": max(times),
        "compile_ms": 0.0,
        "input_mode": "gpu-resident" if mode.resident_input else "end-to-end",
        "resident_input": mode.resident_input,
        "cache_static": mode.cache_static,
        "neighbor_skin": mode.neighbor_skin,
    }


def timing_input(data: dict[str, Any], mode: ExecutionMode) -> dict[str, Any]:
    return data if mode.resident_input else clone_data(data)


def sync(device: str) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(torch.device(device))


def clone_data(data: dict[str, Any]) -> dict[str, Any]:
    cloned = {}
    for key, value in data.items():
        cloned[key] = value.clone() if isinstance(value, torch.Tensor) else value
    return cloned


def resident_data(data: dict[str, Any], device: str) -> dict[str, Any]:
    ret: dict[str, Any] = {}
    for key, value in data.items():
        if value is None:
            ret[key] = None
        elif isinstance(value, torch.Tensor):
            ret[key] = value.to(device)
        elif key == "pbc":
            ret[key] = torch.as_tensor(value, device=device, dtype=torch.bool)
        elif key.startswith(("numbers", "mol_idx", "nbmat")):
            ret[key] = torch.as_tensor(value, device=device, dtype=torch.int64)
        else:
            ret[key] = torch.as_tensor(value, device=device, dtype=torch.float32)
    return ret


def accuracy_metrics(
    strict: dict[str, torch.Tensor],
    result: dict[str, torch.Tensor],
    data: dict[str, Any],
) -> dict[str, float | None]:
    natoms = len(data["numbers"]) if "numbers" in data else None
    return output_metrics(strict, result, natoms=natoms, target_charge=data.get("charge"))


def run_stability_checks(
    args: argparse.Namespace,
    workload: dict[str, Any],
    data: dict[str, Any],
    mode: ExecutionMode,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    coulomb = workload.get("coulomb")
    if "ase-geo" in args.stability:
        results["ase_geo"] = run_ase_geometry_optimization(
            model=args.model,
            data=data,
            device=args.device,
            nb_threshold=mode.nb_threshold,
            neighbor_skin=mode.neighbor_skin,
            max_steps=args.stability_steps,
            fmax=args.stability_fmax,
            coulomb=coulomb,
            compile_model=args.compile_model,
        )
    if "ase-nve" in args.stability:
        results["ase_nve"] = run_ase_nve(
            model=args.model,
            data=data,
            device=args.device,
            nb_threshold=mode.nb_threshold,
            neighbor_skin=mode.neighbor_skin,
            steps=args.stability_steps,
            timestep_fs=args.md_timestep_fs,
            temperature_k=args.md_temperature_k,
            coulomb=coulomb,
            compile_model=args.compile_model,
        )
    if "torchsim-static" in args.stability:
        results["torchsim_static"] = run_torchsim_static(
            model=args.model,
            data=data,
            device=args.device,
            nb_threshold=mode.nb_threshold,
            neighbor_skin=mode.neighbor_skin,
            coulomb=coulomb,
            compile_model=args.compile_model,
        )
    return results


def gates_for(
    metrics: dict[str, float | None],
    speedup: float | None,
    min_speedup: float,
    stability: dict[str, dict[str, Any]],
    strict_stability: dict[str, dict[str, Any]],
    *,
    speedup_reportable: bool = True,
) -> dict[str, bool | None]:
    is_accuracy_pass = accuracy_pass(metrics)
    stability_values = [stability_pass(result, strict_stability.get(name)) for name, result in stability.items()]
    is_stability_pass = None
    if stability_values and any(value is not None for value in stability_values):
        is_stability_pass = all(value is not False for value in stability_values)
    can_report_speedup = bool(
        speedup_reportable
        and is_accuracy_pass
        and is_stability_pass is not False
        and speedup is not None
    )
    meaningful_speedup = None if not can_report_speedup else bool(speedup >= min_speedup)
    return {
        "accuracy_pass": is_accuracy_pass,
        "stability_pass": is_stability_pass,
        "meaningful_speedup": meaningful_speedup,
        "speedup_reportable": can_report_speedup,
    }


def result_row(
    *,
    args: argparse.Namespace,
    workload_name: str,
    workload: dict[str, Any],
    mode: ExecutionMode,
    hardware: dict[str, Any],
    software: dict[str, str],
    git: dict[str, Any],
    command: list[str],
    timing: dict[str, float | int | str | bool],
    accuracy: dict[str, float | None],
    stability: dict[str, dict[str, Any]],
    gates: dict[str, bool | None],
    speedup: float | None,
    data: dict[str, Any],
    request: dict[str, Any],
    effective_path: str,
    equivalent_to_strict: bool,
) -> dict[str, Any]:
    atoms = len(data["numbers"]) if hasattr(data.get("numbers"), "__len__") else None
    median_ms = float(timing["median_ms"])
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "git": git,
        "command": command,
        "device": hardware,
        "software": software,
        "model": {"name": args.model},
        "mode": mode.name,
        "workload": workload_name,
        "description": workload.get("description"),
        "request": request,
        "execution": {
            "nb_threshold": mode.nb_threshold,
            "compile_model": bool(args.compile_model),
            "resident_input": mode.resident_input,
            "cache_static": mode.cache_static,
            "neighbor_skin": mode.neighbor_skin,
            "effective_path": effective_path,
            "equivalent_to_strict": equivalent_to_strict,
        },
        "timing": timing,
        "throughput": {
            "atoms_per_second": None if atoms is None else atoms / (median_ms / 1000.0),
            "steps_per_second": 1000.0 / median_ms,
        },
        "accuracy_vs_strict": accuracy,
        "stability": stability,
        "gates": gates,
        "speedup": speedup,
        "min_speedup": args.min_speedup,
    }


def skipped_row(
    args: argparse.Namespace,
    workload_name: str,
    mode: ExecutionMode,
    hardware: dict[str, Any],
    software: dict[str, str],
    git: dict[str, Any],
    command: list[str],
    reason: str,
) -> dict[str, Any]:
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "git": git,
        "command": command,
        "device": hardware,
        "software": software,
        "model": {"name": args.model},
        "mode": mode.name,
        "workload": workload_name,
        "execution": {
            "nb_threshold": mode.nb_threshold,
            "compile_model": bool(args.compile_model),
            "resident_input": mode.resident_input,
            "cache_static": mode.cache_static,
            "neighbor_skin": mode.neighbor_skin,
        },
        "skipped": True,
        "reason": reason,
    }


def detect_hardware(device: str) -> dict[str, Any]:
    dev = torch.device(device)
    info: dict[str, Any] = {"requested": device, "type": dev.type, "gpus": []}
    if dev.type == "cuda" and torch.cuda.is_available():
        idx = dev.index if dev.index is not None else torch.cuda.current_device()
        for i in range(torch.cuda.device_count()):
            major, minor = torch.cuda.get_device_capability(i)
            name = torch.cuda.get_device_name(i)
            info["gpus"].append(
                {
                    "index": i,
                    "name": name,
                    "compute_capability": f"{major}.{minor}",
                    "class": gpu_class(name, major, minor),
                }
            )
        info["active_index"] = idx
        info["name"] = torch.cuda.get_device_name(idx)
        major, minor = torch.cuda.get_device_capability(idx)
        info["compute_capability"] = f"{major}.{minor}"
        info["class"] = gpu_class(info["name"], major, minor)
    return info


def gpu_class(name: str, major: int, minor: int) -> str:
    lowered = name.lower()
    if "blackwell" in lowered or "b200" in lowered or "gb200" in lowered or major >= 10:
        return "blackwell"
    if "h100" in lowered or "h200" in lowered or "gh200" in lowered or major == 9:
        return "hopper"
    if "l40" in lowered or "ada" in lowered or (major == 8 and minor == 9):
        return "ada"
    if major == 8:
        return "ampere"
    return "unknown"


def detect_software() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": str(torch.version.cuda),
        "nvalchemiops": package_version("nvalchemi-toolkit-ops"),
        "warp_lang": package_version("warp-lang"),
        "aimnet": package_version("aimnet"),
    }


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not-installed"


def git_metadata() -> dict[str, Any]:
    def run_git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()

    try:
        full_sha = run_git("rev-parse", "HEAD")
        branch = run_git("branch", "--show-current") or "detached"
        dirty = bool(run_git("status", "--porcelain"))
        return {
            "commit": full_sha,
            "short_commit": full_sha[:7],
            "branch": branch,
            "dirty": dirty,
        }
    except Exception:
        return {"commit": "unknown", "short_commit": "unknown", "branch": "unknown", "dirty": None}


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[idx]


if __name__ == "__main__":
    raise SystemExit(main())
