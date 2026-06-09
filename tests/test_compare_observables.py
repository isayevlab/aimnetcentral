"""CPU unit tests for the GPU-validation comparator (no GPU needed)."""

import json

import pytest

from aimnet.validation import compare_observables as C


def _results(label, torch_ver, energy, forces):
    return {
        "versions": {"torch": torch_ver, "triton": "x", "warp-lang": "y", "nvalchemi-toolkit-ops": "z"},
        "systems": {"water": {"energy": energy, "forces": forces}},
    }


BASE = _results("2.9", "2.9.1+cu126", -10.0, [[0.1, 0.2, 0.3]])


def test_passes_when_within_tolerance():
    results = {"2.9": BASE, "2.12": _results("2.12", "2.12.0+cu126", -10.0 + 1e-7, [[0.1, 0.2, 0.3 + 1e-6]])}
    status = {"2.9": {"install": "ok", "gpu_suite": "ok"}, "2.12": {"install": "ok", "gpu_suite": "ok"}}
    rows, ok, base = C.compare(results, status, baseline="2.9", energy_atol=1e-5, force_atol=1e-4)
    assert base == "2.9"
    assert ok is True
    verdicts = {r.label: r.verdict for r in rows}
    assert verdicts["2.9"] == "BASELINE"
    assert verdicts["2.12"] == "PASS"


def test_flags_energy_drift():
    results = {"2.9": BASE, "2.8": _results("2.8", "2.8.0+cu126", -10.0 + 1e-3, [[0.1, 0.2, 0.3]])}
    status = {"2.9": {"install": "ok", "gpu_suite": "ok"}, "2.8": {"install": "ok", "gpu_suite": "ok"}}
    rows, ok, _ = C.compare(results, status, baseline="2.9", energy_atol=1e-5, force_atol=1e-4)
    assert ok is False
    assert {r.label: r.verdict for r in rows}["2.8"] == "DRIFT"


def test_flags_force_drift():
    results = {"2.9": BASE, "2.8": _results("2.8", "2.8.0+cu126", -10.0, [[0.1, 0.2, 0.5]])}
    status = {"2.9": {"install": "ok", "gpu_suite": "ok"}, "2.8": {"install": "ok", "gpu_suite": "ok"}}
    _, ok, _ = C.compare(results, status, baseline="2.9", energy_atol=1e-5, force_atol=1e-4)
    assert ok is False


def test_install_failure_is_reported_not_crashed():
    results = {"2.9": BASE}
    status = {"2.9": {"install": "ok", "gpu_suite": "ok"}, "2.11": {"install": "fail", "gpu_suite": "skipped"}}
    rows, ok, _ = C.compare(results, status, baseline="2.9", energy_atol=1e-5, force_atol=1e-4)
    assert ok is False
    assert {r.label: r.verdict for r in rows}["2.11"] == "INSTALL-FAIL"


def test_missing_results_for_installed_leg():
    results = {"2.9": BASE}
    status = {"2.9": {"install": "ok", "gpu_suite": "ok"}, "2.10": {"install": "ok", "gpu_suite": "ok"}}
    rows, ok, _ = C.compare(results, status, baseline="2.9", energy_atol=1e-5, force_atol=1e-4)
    assert ok is False
    assert {r.label: r.verdict for r in rows}["2.10"] == "NO-RESULTS"


def test_missing_baseline_raises():
    results = {"2.12": _results("2.12", "2.12.0+cu126", -10.0, [[0.1, 0.2, 0.3]])}
    status = {"2.12": {"install": "ok", "gpu_suite": "ok"}}
    with pytest.raises(SystemExit):
        C.compare(results, status, baseline="2.9", energy_atol=1e-5, force_atol=1e-4)


def test_load_results_from_dir(tmp_path):
    (tmp_path / "2.9.json").write_text(json.dumps(BASE))
    (tmp_path / "status.json").write_text(json.dumps({"2.9": {"install": "ok", "gpu_suite": "ok"}}))
    loaded = C.load_results(tmp_path)
    assert "2.9" in loaded and "status" not in loaded


def test_gpu_observables_importable_and_guards_cuda(monkeypatch, tmp_path):
    import torch

    from aimnet.validation import gpu_observables as G

    # versions dict always includes torch, regardless of device
    assert "torch" in G.resolved_versions()

    # main() must refuse to run without CUDA rather than crash cryptically
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit):
        G.main(["--out", str(tmp_path / "x.json")])
