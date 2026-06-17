from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
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
from aimnet.precision import PrecisionPolicy, supports_bf16, supports_fp8, supports_tf32  # noqa: E402
from aimnet.validation.precision_metrics import accuracy_pass, output_metrics  # noqa: E402
from aimnet.validation.stability import (  # noqa: E402
    run_ase_geometry_optimization,
    run_ase_nve,
    run_torchsim_static,
    stability_pass,
)


def main() -> int:
    args = parse_args()
    args.modes = expand_modes(args.modes, args.device, args.fp8_backend)
    if args.nvtx:
        os.environ["AIMNET_NVTX"] = "1"

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
    commit = git_commit()

    with output.open("a", encoding="utf-8") as fh:
        for workload_name in selected:
            workload = workloads[workload_name]
            data = build_data(workload)
            request = dict(workload.get("request", {}))
            strict_ref = run_once(args, workload, data, request, "strict")
            strict_timing = None
            strict_stability: dict[str, dict[str, Any]] = {}
            for mode in args.modes:
                if not mode_supported(mode, args.device, args.fp8_backend):
                    row = skipped_row(args, workload_name, mode, hardware, software, commit, "unsupported mode on device")
                    fh.write(json.dumps(row) + "\n")
                    continue
                result, timing = run_timed(args, workload, data, request, mode)
                metrics = accuracy_metrics(strict_ref, result, data)
                stability = run_stability_checks(args, workload, data, mode)
                if mode == "strict":
                    strict_timing = timing
                    strict_stability = stability
                speedup = None
                if strict_timing is not None and mode != "strict":
                    speedup = strict_timing["median_ms"] / timing["median_ms"] if timing["median_ms"] > 0 else None
                gates = gates_for(metrics, speedup, args.min_speedup, stability, strict_stability)
                row = result_row(
                    args=args,
                    workload_name=workload_name,
                    workload=workload,
                    mode=mode,
                    hardware=hardware,
                    software=software,
                    commit=commit,
                    timing=timing,
                    accuracy=metrics,
                    stability=stability,
                    gates=gates,
                    speedup=speedup,
                    data=data,
                    request=request,
                )
                fh.write(json.dumps(row) + "\n")
                fh.flush()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AIMNet precision benchmark runner")
    parser.add_argument("--model", default="aimnet2")
    parser.add_argument("--modes", nargs="+", default=["auto"], help="Precision modes, or 'auto' for this device")
    parser.add_argument("--workloads", default=str(ROOT / "benchmarks/precision/workloads.yaml"))
    parser.add_argument("--workload", action="append", help="Run one workload; may be supplied multiple times")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--output", default=str(ROOT / "results/precision_bench.jsonl"))
    parser.add_argument(
        "--nb-threshold",
        type=int,
        default=120,
        help="AIMNet2Calculator dense/sparse threshold. Use 0 to force sparse neighbor-list execution.",
    )
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--min-speedup", type=float, default=1.15)
    parser.add_argument("--fp8-backend", choices=["none", "torchao", "transformer_engine", "custom"], default="none")
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
    parser.add_argument("--nvtx", action="store_true")
    parser.add_argument("--list-workloads", action="store_true")
    return parser.parse_args()


def expand_modes(modes: list[str], device: str, fp8_backend: str) -> list[str]:
    if modes != ["auto"]:
        return modes
    selected = ["strict"]
    dev = torch.device(device)
    if dev.type == "cuda" and supports_tf32(dev):
        selected.append("tf32_learned")
    if dev.type == "cuda" and supports_bf16(dev):
        selected.append("bf16_learned")
    if dev.type == "cuda" and supports_fp8(dev, fp8_backend):  # type: ignore[arg-type]
        selected.append("fp8_learned_experimental")
    return selected


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


def make_calc(args: argparse.Namespace, workload: dict[str, Any], mode: str) -> AIMNet2Calculator:
    precision: str | PrecisionPolicy
    if mode == "fp8_learned_experimental":
        precision = PrecisionPolicy(mode="fp8_learned_experimental", fp8_backend=args.fp8_backend)
    else:
        precision = mode
    calc = AIMNet2Calculator(
        args.model,
        device=args.device,
        nb_threshold=args.nb_threshold,
        compile_model=args.compile_model,
        precision=precision,
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
    mode: str,
) -> dict[str, torch.Tensor]:
    calc = make_calc(args, workload, mode)
    with torch.inference_mode(mode=not any(request.get(k, False) for k in ("forces", "stress", "hessian"))):
        return calc(clone_data(data), **request)


def run_timed(
    args: argparse.Namespace,
    workload: dict[str, Any],
    data: dict[str, Any],
    request: dict[str, Any],
    mode: str,
) -> tuple[dict[str, torch.Tensor], dict[str, float | int]]:
    calc = make_calc(args, workload, mode)
    for _ in range(args.warmup):
        calc(clone_data(data), **request)
    sync(args.device)
    times: list[float] = []
    result: dict[str, torch.Tensor] | None = None
    for _ in range(args.repeat):
        sync(args.device)
        start = time.perf_counter()
        result = calc(clone_data(data), **request)
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
    }


def sync(device: str) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(torch.device(device))


def clone_data(data: dict[str, Any]) -> dict[str, Any]:
    cloned = {}
    for key, value in data.items():
        cloned[key] = value.clone() if isinstance(value, torch.Tensor) else value
    return cloned


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
    mode: str,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    coulomb = workload.get("coulomb")
    if "ase-geo" in args.stability:
        results["ase_geo"] = run_ase_geometry_optimization(
            model=args.model,
            precision=mode,
            data=data,
            device=args.device,
            max_steps=args.stability_steps,
            fmax=args.stability_fmax,
            coulomb=coulomb,
            compile_model=args.compile_model,
        )
    if "ase-nve" in args.stability:
        results["ase_nve"] = run_ase_nve(
            model=args.model,
            precision=mode,
            data=data,
            device=args.device,
            steps=args.stability_steps,
            timestep_fs=args.md_timestep_fs,
            temperature_k=args.md_temperature_k,
            coulomb=coulomb,
            compile_model=args.compile_model,
        )
    if "torchsim-static" in args.stability:
        results["torchsim_static"] = run_torchsim_static(
            model=args.model,
            precision=mode,
            data=data,
            device=args.device,
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
) -> dict[str, bool | None]:
    is_accuracy_pass = accuracy_pass(metrics)
    stability_values = [
        stability_pass(result, strict_stability.get(name)) for name, result in stability.items()
    ]
    is_stability_pass = None
    if stability_values and any(value is not None for value in stability_values):
        is_stability_pass = all(value is not False for value in stability_values)
    meaningful_speedup = None if speedup is None else bool(speedup >= min_speedup)
    return {
        "accuracy_pass": is_accuracy_pass,
        "stability_pass": is_stability_pass,
        "meaningful_speedup": meaningful_speedup,
        "speedup_reportable": bool(is_accuracy_pass and is_stability_pass is not False and (speedup is not None)),
    }


def result_row(
    *,
    args: argparse.Namespace,
    workload_name: str,
    workload: dict[str, Any],
    mode: str,
    hardware: dict[str, Any],
    software: dict[str, str],
    commit: str,
    timing: dict[str, float | int],
    accuracy: dict[str, float | None],
    stability: dict[str, dict[str, Any]],
    gates: dict[str, bool | None],
    speedup: float | None,
    data: dict[str, Any],
    request: dict[str, Any],
) -> dict[str, Any]:
    atoms = len(data["numbers"]) if hasattr(data.get("numbers"), "__len__") else None
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "commit": commit,
        "device": hardware,
        "software": software,
        "model": {"name": args.model},
        "policy": mode,
        "workload": workload_name,
        "description": workload.get("description"),
        "request": request,
        "execution": {
            "nb_threshold": args.nb_threshold,
            "compile_model": bool(args.compile_model),
        },
        "timing": timing,
        "throughput": {
            "atoms_per_second": None if atoms is None else atoms / (float(timing["median_ms"]) / 1000.0),
            "steps_per_second": 1000.0 / float(timing["median_ms"]),
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
    mode: str,
    hardware: dict[str, Any],
    software: dict[str, str],
    commit: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "commit": commit,
        "device": hardware,
        "software": software,
        "model": {"name": args.model},
        "policy": mode,
        "workload": workload_name,
        "skipped": True,
        "reason": reason,
    }


def mode_supported(mode: str, device: str, fp8_backend: str) -> bool:
    dev = torch.device(device)
    if mode in {"strict", "emulated_reductions"}:
        return True
    if mode == "tf32_learned":
        return dev.type == "cuda" and supports_tf32(dev)
    if mode in {"bf16_learned", "fp16_learned"}:
        return dev.type == "cuda" and (mode == "fp16_learned" or supports_bf16(dev))
    if mode == "fp8_learned_experimental":
        return dev.type == "cuda" and supports_fp8(dev, fp8_backend)  # type: ignore[arg-type]
    return False


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


def git_commit() -> str:
    try:
        git_dir = _git_dir(ROOT)
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        if head.startswith("ref: "):
            ref_path = git_dir / head.removeprefix("ref: ").strip()
            return ref_path.read_text(encoding="utf-8").strip()[:7]
        return head[:7]
    except Exception:
        return "unknown"


def _git_dir(root: Path) -> Path:
    git_path = root / ".git"
    if git_path.is_dir():
        return git_path
    text = git_path.read_text(encoding="utf-8").strip()
    if text.startswith("gitdir: "):
        path = Path(text.removeprefix("gitdir: ").strip())
        return path if path.is_absolute() else root / path
    return git_path


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[idx]


if __name__ == "__main__":
    raise SystemExit(main())
