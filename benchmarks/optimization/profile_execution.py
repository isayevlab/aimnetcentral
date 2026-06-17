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

_BENCH = importlib.util.spec_from_file_location("bench_optimization", Path(__file__).with_name("bench_optimization.py"))
if _BENCH is None or _BENCH.loader is None:
    raise ImportError("Could not load bench_optimization.py")
bench_optimization = importlib.util.module_from_spec(_BENCH)
sys.modules[_BENCH.name] = bench_optimization
_BENCH.loader.exec_module(bench_optimization)

from aimnet.calculators import AIMNet2Calculator  # noqa: E402


def main() -> int:
    args = parse_args()
    workloads = bench_optimization.load_workloads(args.workloads)
    if args.workload not in workloads:
        raise SystemExit(f"Unknown workload: {args.workload}")
    workload = workloads[args.workload]
    data = bench_optimization.build_data(workload)
    request = dict(workload.get("request", {}))
    mode = bench_optimization.mode_config(args.mode, args)
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

    input_data = bench_optimization.resident_data(data, args.device) if mode.resident_input else data
    timing = run_timing(calc, input_data, request, args.device, mode, args.warmup, args.repeat)
    profile = run_profile(calc, input_data, request, args.device, mode, args.profile_steps, args.trace)
    row = {
        "timestamp": datetime.now(UTC).isoformat(),
        "git": bench_optimization.git_metadata(),
        "command": [Path(sys.argv[0]).name, *sys.argv[1:]],
        "device": bench_optimization.detect_hardware(args.device),
        "software": bench_optimization.detect_software(),
        "model": args.model,
        "workload": args.workload,
        "mode": mode.name,
        "execution": {
            "nb_threshold": mode.nb_threshold,
            "compile_model": bool(args.compile_model),
            "resident_input": mode.resident_input,
            "cache_static": mode.cache_static,
            "neighbor_skin": mode.neighbor_skin,
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
    parser = argparse.ArgumentParser(description="Profile one AIMNet optimization workload.")
    parser.add_argument("--model", default="aimnet2")
    parser.add_argument("--workloads", default=str(ROOT / "benchmarks/optimization/workloads.yaml"))
    parser.add_argument("--workload", default="caffeine_forces")
    parser.add_argument(
        "--mode",
        choices=["strict", "dense-default", "forced-sparse", "gpu-resident", "static-cache", "neighbor-skin"],
        default="strict",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--nb-threshold", type=int, default=120)
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--neighbor-skin", type=float, default=0.5)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--profile-steps", type=int, default=5)
    parser.add_argument("--trace", help="Optional Chrome trace output path.")
    parser.add_argument("--output", default=str(ROOT / "results/profile_execution.json"))
    return parser.parse_args()


def run_timing(
    calc: AIMNet2Calculator,
    data: dict[str, Any],
    request: dict[str, Any],
    device: str,
    mode: Any,
    warmup: int,
    repeat: int,
) -> dict[str, float | int | str | bool]:
    for _ in range(warmup):
        calc(bench_optimization.timing_input(data, mode), **request)
    bench_optimization.sync(device)
    times = []
    for _ in range(repeat):
        bench_optimization.sync(device)
        start = time.perf_counter()
        calc(bench_optimization.timing_input(data, mode), **request)
        bench_optimization.sync(device)
        times.append((time.perf_counter() - start) * 1000.0)
    return {
        "warmup": warmup,
        "repeat": repeat,
        "median_ms": statistics.median(times),
        "p95_ms": percentile(times, 0.95),
        "min_ms": min(times),
        "max_ms": max(times),
        "input_mode": "gpu-resident" if mode.resident_input else "end-to-end",
        "resident_input": mode.resident_input,
        "cache_static": mode.cache_static,
        "neighbor_skin": mode.neighbor_skin,
    }


def run_profile(
    calc: AIMNet2Calculator,
    data: dict[str, Any],
    request: dict[str, Any],
    device: str,
    mode: Any,
    steps: int,
    trace: str | None,
) -> dict[str, Any]:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.device(device).type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    bench_optimization.sync(device)
    with torch.profiler.profile(activities=activities, record_shapes=True, with_stack=False) as prof:
        for _ in range(steps):
            calc(bench_optimization.timing_input(data, mode), **request)
            bench_optimization.sync(device)
            prof.step()
    if trace:
        trace_path = Path(trace)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(trace_path))
    events = prof.key_averages(group_by_input_shape=False)
    raw_events = list(prof.events())
    return {
        "steps": steps,
        "trace": trace,
        "event_counts": event_counts(raw_events),
        "top_cuda": top_events(events, "cuda_time_total", "self_cuda_time_total", "device_time_total"),
        "top_self_cuda": top_events(events, "self_cuda_time_total", "self_device_time_total"),
        "top_cpu": top_events(events, "self_cpu_time_total"),
    }


def event_counts(events: list[Any]) -> dict[str, int]:
    cuda_events = [event for event in events if "cuda" in str(getattr(event, "device_type", "")).lower()]
    cpu_launches = [
        event
        for event in events
        if str(getattr(event, "name", getattr(event, "key", ""))) in {"cudaLaunchKernel", "cudaLaunchKernel_ptsz"}
    ]
    memcpy_events = [
        event for event in cuda_events if "memcpy" in str(getattr(event, "name", getattr(event, "key", ""))).lower()
    ]
    return {
        "total_events": len(events),
        "cuda_events": len(cuda_events),
        "cuda_kernel_or_runtime_launch_events": len(cuda_events) + len(cpu_launches),
        "cpu_cuda_launch_events": len(cpu_launches),
        "cuda_memcpy_events": len(memcpy_events),
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
