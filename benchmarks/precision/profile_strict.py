from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_BENCH = importlib.util.spec_from_file_location("bench_precision", Path(__file__).with_name("bench_precision.py"))
if _BENCH is None or _BENCH.loader is None:
    raise ImportError("Could not load bench_precision.py")
bench_precision = importlib.util.module_from_spec(_BENCH)
_BENCH.loader.exec_module(bench_precision)

from aimnet.calculators import AIMNet2Calculator  # noqa: E402


def main() -> int:
    args = parse_args()
    workloads = bench_precision.load_workloads(args.workloads)
    if args.workload not in workloads:
        raise SystemExit(f"Unknown workload: {args.workload}")
    workload = workloads[args.workload]
    data = bench_precision.build_data(workload)
    request = dict(workload.get("request", {}))
    calc = AIMNet2Calculator(
        args.model,
        device=args.device,
        nb_threshold=args.nb_threshold,
        compile_model=args.compile_model,
        precision="strict",
    )
    coulomb = workload.get("coulomb")
    if coulomb:
        method = coulomb.get("method", "dsf")
        kwargs = {k: v for k, v in coulomb.items() if k != "method"}
        calc.set_lrcoulomb_method(method, **kwargs)

    timing = run_timing(calc, data, request, args.device, args.warmup, args.repeat)
    profile = run_profile(calc, data, request, args.device, args.profile_steps, args.trace)
    row = {
        "timestamp": datetime.now(UTC).isoformat(),
        "commit": bench_precision.git_commit(),
        "device": bench_precision.detect_hardware(args.device),
        "software": bench_precision.detect_software(),
        "model": args.model,
        "workload": args.workload,
        "policy": "strict",
        "execution": {
            "nb_threshold": args.nb_threshold,
            "compile_model": bool(args.compile_model),
        },
        "request": request,
        "timing": timing,
        "profile": profile,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(row, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile one AIMNet strict-mode workload.")
    parser.add_argument("--model", default="aimnet2")
    parser.add_argument("--workloads", default=str(ROOT / "benchmarks/precision/workloads.yaml"))
    parser.add_argument("--workload", default="caffeine_forces")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--nb-threshold", type=int, default=120)
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--profile-steps", type=int, default=5)
    parser.add_argument("--trace", help="Optional Chrome trace output path.")
    parser.add_argument("--output", default=str(ROOT / "results/profile_strict.json"))
    return parser.parse_args()


def run_timing(
    calc: AIMNet2Calculator,
    data: dict[str, Any],
    request: dict[str, Any],
    device: str,
    warmup: int,
    repeat: int,
) -> dict[str, float | int]:
    for _ in range(warmup):
        calc(bench_precision.clone_data(data), **request)
    bench_precision.sync(device)
    times = []
    for _ in range(repeat):
        bench_precision.sync(device)
        start = time.perf_counter()
        calc(bench_precision.clone_data(data), **request)
        bench_precision.sync(device)
        times.append((time.perf_counter() - start) * 1000.0)
    return {
        "warmup": warmup,
        "repeat": repeat,
        "median_ms": statistics.median(times),
        "p95_ms": percentile(times, 0.95),
        "min_ms": min(times),
        "max_ms": max(times),
    }


def run_profile(
    calc: AIMNet2Calculator,
    data: dict[str, Any],
    request: dict[str, Any],
    device: str,
    steps: int,
    trace: str | None,
) -> dict[str, Any]:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.device(device).type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    bench_precision.sync(device)
    with torch.profiler.profile(activities=activities, record_shapes=True, with_stack=False) as prof:
        for _ in range(steps):
            calc(bench_precision.clone_data(data), **request)
            bench_precision.sync(device)
            prof.step()
    if trace:
        trace_path = Path(trace)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(trace_path))
    events = prof.key_averages(group_by_input_shape=False)
    return {
        "steps": steps,
        "trace": trace,
        "top_cuda": top_events(events, "cuda_time_total", "self_cuda_time_total", "device_time_total"),
        "top_self_cuda": top_events(events, "self_cuda_time_total", "self_device_time_total"),
        "top_cpu": top_events(events, "self_cpu_time_total"),
    }


def top_events(events: Any, *metric_names: str, limit: int = 15) -> list[dict[str, Any]]:
    ranked = sorted(events, key=lambda event: metric_value(event, *metric_names), reverse=True)
    result = []
    for event in ranked[:limit]:
        result.append(
            {
                "key": str(event.key),
                "count": int(getattr(event, "count", 0)),
                "metric_us": metric_value(event, *metric_names),
                "self_cpu_us": float(getattr(event, "self_cpu_time_total", 0.0)),
                "cpu_us": float(getattr(event, "cpu_time_total", 0.0)),
                "self_cuda_us": metric_value(event, "self_cuda_time_total", "self_device_time_total"),
                "cuda_us": metric_value(event, "cuda_time_total", "device_time_total"),
            }
        )
    return result


def metric_value(event: Any, *names: str) -> float:
    for name in names:
        if hasattr(event, name):
            return float(getattr(event, name))
    return 0.0


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[idx]


if __name__ == "__main__":
    raise SystemExit(main())
