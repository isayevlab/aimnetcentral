"""Compare per-torch-version GPU observables against a same-run baseline.

Pure stdlib. The orchestrator writes one ``<label>.json`` per torch version
(plus a ``status.json``); this module diffs every version's energies/forces
against the baseline version and renders a verdict matrix.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Row:
    label: str
    install: str
    gpu_suite: str
    max_de: float | str
    max_df: float | str
    verdict: str
    versions: dict | str


def load_results(results_dir) -> dict:
    """Load ``<label>.json`` result files from a directory (skips status.json)."""
    out = {}
    for p in sorted(Path(results_dir).glob("*.json")):
        if p.name == "status.json":
            continue
        out[p.stem] = json.loads(p.read_text())
    return out


def load_status(results_dir) -> dict:
    p = Path(results_dir) / "status.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _max_abs_force_diff(fa, fb) -> float:
    """Max elementwise |fa - fb| over arbitrarily nested equal-shaped lists."""
    m = 0.0
    stack = [(fa, fb)]
    while stack:
        a, b = stack.pop()
        if isinstance(a, list):
            for x, y in zip(a, b, strict=True):
                stack.append((x, y))
        else:
            m = max(m, abs(a - b))
    return m


def compare(results: dict, status: dict, *, baseline: str, energy_atol: float, force_atol: float):
    """Return ``(rows, ok, base_label)``. ``ok`` is False if any leg failed.

    A leg fails if it did not install, has no results, failed the gpu-suite,
    or drifted beyond tolerance vs the baseline.
    """
    base_label = next(
        (lbl for lbl, d in results.items() if str(d["versions"]["torch"]).startswith(baseline) or lbl == baseline),
        None,
    )
    if base_label is None:
        raise SystemExit(f"baseline '{baseline}' not found among results {sorted(results)}")
    base = results[base_label]

    rows: list[Row] = []
    ok = True
    for lbl in sorted(set(status) | set(results)):
        st = status.get(lbl, {})
        install = st.get("install", "fail")
        suite = st.get("gpu_suite", "skipped")
        if install != "ok":
            rows.append(Row(lbl, install, suite, "-", "-", "INSTALL-FAIL", st.get("versions", "")))
            ok = False
            continue
        if lbl not in results:
            rows.append(Row(lbl, install, suite, "-", "-", "NO-RESULTS", ""))
            ok = False
            continue
        d = results[lbl]
        quad = d["versions"]
        if lbl == base_label:
            rows.append(Row(lbl, install, suite, 0.0, 0.0, "BASELINE", quad))
            if suite == "fail":
                ok = False
            continue
        max_de = 0.0
        max_df = 0.0
        for name, obs in d["systems"].items():
            b = base["systems"][name]
            max_de = max(max_de, abs(obs["energy"] - b["energy"]))
            max_df = max(max_df, _max_abs_force_diff(obs["forces"], b["forces"]))
        verdict = "PASS"
        if suite == "fail":
            verdict = "SUITE-FAIL"
            ok = False
        if max_de > energy_atol or max_df > force_atol:
            verdict = "DRIFT"
            ok = False
        rows.append(Row(lbl, install, suite, max_de, max_df, verdict, quad))
    return rows, ok, base_label


def _fmt(v) -> str:
    return f"{v:.2e}" if isinstance(v, float) else str(v)


def render(rows, base_label: str) -> str:
    lines = [f"baseline = {base_label}", ""]
    header = f"{'torch':>7} {'install':>8} {'gpu-suite':>10} {'maxDE':>10} {'maxDF':>10}  verdict"
    lines.append(header)
    lines.append("-" * len(header))
    for r in rows:
        lines.append(
            f"{r.label:>7} {r.install:>8} {r.gpu_suite:>10} {_fmt(r.max_de):>10} {_fmt(r.max_df):>10}  {r.verdict}"
        )
    lines.append("")
    for r in rows:
        if isinstance(r.versions, dict):
            quad = " ".join(f"{k}={v}" for k, v in r.versions.items())
            lines.append(f"  {r.label}: {quad}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir", help="directory containing <label>.json and status.json")
    ap.add_argument("--baseline", default="2.9", help="torch version prefix used as the reference")
    ap.add_argument("--energy-atol", type=float, default=1e-5, help="Hartree")
    ap.add_argument("--force-atol", type=float, default=1e-4, help="Hartree/Angstrom")
    args = ap.parse_args(argv)

    results = load_results(args.results_dir)
    status = load_status(args.results_dir)
    rows, ok, base_label = compare(
        results, status, baseline=args.baseline, energy_atol=args.energy_atol, force_atol=args.force_atol
    )
    print(render(rows, base_label))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
