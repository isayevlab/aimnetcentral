from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    args = parse_args()
    rows = read_rows(args.inputs)
    if not rows:
        raise SystemExit("No benchmark rows found.")
    report = render_markdown(rows)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(report, encoding="utf-8")
    print(report)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize AIMNet optimization benchmark JSONL results.")
    parser.add_argument("inputs", nargs="+", help="JSONL result files.")
    parser.add_argument("--output", help="Optional markdown output path.")
    return parser.parse_args()


def read_rows(paths: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def render_markdown(rows: list[dict[str, Any]]) -> str:
    complete = [r for r in rows if not r.get("skipped")]
    skipped = [r for r in rows if r.get("skipped")]
    device = first_value(complete, "device") or first_value(rows, "device") or {}
    software = first_value(complete, "software") or first_value(rows, "software") or {}
    workloads = sorted({str(r.get("workload")) for r in rows})
    modes = sorted({row_mode(r) for r in rows})
    non_strict = [r for r in complete if row_mode(r) != "strict"]
    strict = [r for r in complete if row_mode(r) == "strict"]
    accepted = [r for r in non_strict if accepted_row(r)]

    lines = [
        "# AIMNet Optimization Benchmark Summary",
        "",
        f"- Rows: {len(complete)} complete, {len(skipped)} skipped",
        f"- Device: {device.get('name', device.get('type', 'unknown'))} ({device.get('class', 'unknown')}, cc {device.get('compute_capability', 'n/a')})",
        f"- Torch/CUDA: {software.get('torch', 'unknown')} / {software.get('cuda', 'unknown')}",
        f"- AIMNet: {software.get('aimnet', 'unknown')}",
        f"- Workloads: {', '.join(workloads)}",
        f"- Modes: {', '.join(modes)}",
        "",
        "## Decision",
        "",
    ]
    if accepted:
        lines.append("Accepted non-strict execution-mode/workload combinations:")
        lines.extend(f"- {r['workload']} / {row_mode(r)}: speedup {fmt_speedup(r.get('speedup'))}" for r in accepted)
    else:
        lines.append("No non-strict execution mode met the combined accuracy, stability, and meaningful-speedup gates.")

    if strict:
        strict_pass = sum(
            bool(
                r.get("gates", {}).get("accuracy_pass")
                and r.get("gates", {}).get("stability_pass") is not False
            )
            for r in strict
        )
        lines.extend(["", f"Strict baseline pass rate: {strict_pass}/{len(strict)} workloads."])

    lines.extend(["", "## Result Matrix", "", matrix_table(complete)])

    if non_strict:
        lines.extend(["", "## Non-Strict Mode Aggregates", "", aggregate_table(non_strict)])
        lines.extend(["", "## Rejections", ""])
        for r in non_strict:
            if accepted_row(r):
                continue
            lines.append(f"- {r['workload']} / {row_mode(r)}: {rejection_reason(r)}")

    if skipped:
        lines.extend(["", "## Skipped", ""])
        for r in skipped:
            lines.append(f"- {r.get('workload')} / {row_mode(r)}: {r.get('reason', 'skipped')}")

    return "\n".join(lines) + "\n"


def first_value(rows: list[dict[str, Any]], key: str) -> Any:
    for row in rows:
        if key in row:
            return row[key]
    return None


def row_mode(row: dict[str, Any]) -> str:
    return str(row.get("mode", row.get("policy", "unknown")))


def accepted_row(row: dict[str, Any]) -> bool:
    gates = row.get("gates", {})
    return bool(
        gates.get("accuracy_pass")
        and gates.get("stability_pass") is not False
        and gates.get("meaningful_speedup")
    )


def matrix_table(rows: list[dict[str, Any]]) -> str:
    header = "| Workload | Mode | Median ms | Speedup | Accuracy | Stability | Meaningful |"
    sep = "| --- | --- | ---: | ---: | --- | --- | --- |"
    body = []
    for r in sorted(rows, key=lambda x: (str(x.get("workload")), row_mode(x))):
        gates = r.get("gates", {})
        timing = r.get("timing", {})
        body.append(
            "| "
            + " | ".join(
                [
                    str(r.get("workload")),
                    row_mode(r),
                    fmt_float(timing.get("median_ms"), digits=3),
                    fmt_speedup(r.get("speedup")),
                    fmt_bool(gates.get("accuracy_pass")),
                    fmt_bool(gates.get("stability_pass")),
                    fmt_bool(gates.get("meaningful_speedup")),
                ]
            )
            + " |"
        )
    return "\n".join([header, sep, *body])


def aggregate_table(rows: list[dict[str, Any]]) -> str:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row_mode(row)].append(row)
    header = "| Mode | Speedup min | Speedup median | Speedup max | Accuracy pass | Stability pass | Meaningful pass |"
    sep = "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"
    body = []
    for mode, vals in sorted(groups.items()):
        speedups = [
            float(r["speedup"])
            for r in vals
            if r.get("speedup") is not None and r.get("gates", {}).get("speedup_reportable")
        ]
        gates = [r.get("gates", {}) for r in vals]
        body.append(
            "| "
            + " | ".join(
                [
                    mode,
                    fmt_float(min(speedups), digits=3) if speedups else "n/a",
                    fmt_float(statistics.median(speedups), digits=3) if speedups else "n/a",
                    fmt_float(max(speedups), digits=3) if speedups else "n/a",
                    f"{sum(bool(g.get('accuracy_pass')) for g in gates)}/{len(vals)}",
                    f"{sum(g.get('stability_pass') is not False for g in gates)}/{len(vals)}",
                    f"{sum(bool(g.get('meaningful_speedup')) for g in gates)}/{len(vals)}",
                ]
            )
            + " |"
        )
    return "\n".join([header, sep, *body])


def rejection_reason(row: dict[str, Any]) -> str:
    gates = row.get("gates", {})
    reasons = []
    if not gates.get("accuracy_pass"):
        reasons.append("accuracy gate failed")
    if gates.get("stability_pass") is False:
        reasons.append("stability gate failed")
    unreportable_speedup = row.get("speedup") is not None and gates.get("speedup_reportable") is False
    if unreportable_speedup:
        reasons.append("same effective execution path as strict")
    if not unreportable_speedup and not gates.get("meaningful_speedup"):
        reasons.append(f"speedup {fmt_speedup(row.get('speedup'))} below threshold")
    return "; ".join(reasons) if reasons else "not accepted"


def fmt_bool(value: Any) -> str:
    if value is None:
        return "n/a"
    return "pass" if value else "fail"


def fmt_speedup(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}x"


def fmt_float(value: Any, *, digits: int) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


if __name__ == "__main__":
    raise SystemExit(main())
