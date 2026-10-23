"""Requested CUDA-graph replay that cannot apply warns once per calculator and reason (no silent no-ops).

These tests run without a GPU: the calculator is CUDA-labelled with a stub model, as in ``test_cuda_graph_opt_in``,
and the dependency, platform and hardware probes are monkeypatched.
"""

import types
import warnings

import pytest
import torch

import aimnet.calculators._graph_options as graph_options
import aimnet.calculators.calculator as calculator_module
from aimnet.calculators import AIMNet2Calculator
from aimnet.calculators.cuda_graph import cache as cache_module
from aimnet.kernels import triton_support

GOOD_VERSIONS = {"nvalchemi-toolkit-ops": "0.5.0", "warp-lang": "1.16.0"}


@pytest.fixture
def stub_model(monkeypatch):
    model = torch.nn.Identity()
    monkeypatch.setattr(calculator_module, "resolve_model", lambda *args, **kwargs: (model, None, 5.0))
    monkeypatch.setattr(graph_options, "installed_versions", lambda: dict(GOOD_VERSIONS))
    monkeypatch.setattr(torch.version, "hip", None)
    return model


def _graph_warnings(record):
    return [str(w.message) for w in record if "CUDA-graph replay is not used" in str(w.message)]


@pytest.mark.parametrize(
    ("versions", "names"),
    [
        ({"nvalchemi-toolkit-ops": "0.4.1", "warp-lang": "1.16.0"}, ["nvalchemi-toolkit-ops 0.4.1"]),
        ({"nvalchemi-toolkit-ops": "0.5.0", "warp-lang": "1.15.0"}, ["warp-lang 1.15.0"]),
        ({"nvalchemi-toolkit-ops": None, "warp-lang": "1.13"}, ["nvalchemi-toolkit-ops unknown", "warp-lang 1.13"]),
    ],
)
def test_old_dependencies_refuse_capture(monkeypatch, stub_model, versions, names):
    monkeypatch.setattr(graph_options, "installed_versions", lambda: dict(versions))
    with pytest.warns(UserWarning, match="CUDA-graph replay is not used") as record:
        calc = AIMNet2Calculator(stub_model, device="cuda", cuda_graph=True)
    assert calc._cuda_graph is None
    (message,) = _graph_warnings(record)
    for name in names:
        assert name in message
    assert "nvalchemi-toolkit-ops>=0.5.0" in message  # names the fix


@pytest.mark.parametrize(
    ("text", "ok"), [("0.5.0", True), ("0.5", True), ("0.6.0.dev1", True), ("0.4.9", False), ("1.0", True)]
)
def test_dependency_threshold_parsing(monkeypatch, text, ok):
    monkeypatch.setattr(
        graph_options, "installed_versions", lambda: {"nvalchemi-toolkit-ops": text, "warp-lang": "1.16"}
    )
    assert (graph_options.cuda_graph_dependency_problem() is None) is ok


def test_rocm_refuses_capture(monkeypatch, stub_model):
    monkeypatch.setattr(torch.version, "hip", "6.2.0")
    with pytest.warns(UserWarning, match="ROCm"):
        calc = AIMNet2Calculator(stub_model, device="cuda", cuda_graph=True)
    assert calc._cuda_graph is None


def test_cpu_device_warns(stub_model):
    with pytest.warns(UserWarning, match="runs on cpu, not CUDA"):
        calc = AIMNet2Calculator(stub_model, device="cpu", cuda_graph=True)
    assert calc._cuda_graph is None


def test_deterministic_warns_at_construction(stub_model):
    with pytest.warns(UserWarning, match=r"deterministic=True") as record:
        calc = AIMNet2Calculator(stub_model, device="cuda", cuda_graph=True, deterministic=True)
    assert calc._cuda_graph is None
    assert len(_graph_warnings(record)) == 1


def test_unsupported_model_warns_at_construction(stub_model):
    with pytest.warns(UserWarning, match="unsupported model"):
        calc = AIMNet2Calculator(stub_model, device="cuda", cuda_graph=True)
    assert calc._cuda_graph is None


def test_disabled_graphs_do_not_warn(stub_model):
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        AIMNet2Calculator(stub_model, device="cuda", cuda_graph=False)
        AIMNet2Calculator(stub_model, device="cpu")
    assert not _graph_warnings(record)


def test_fallback_warning_is_once_per_reason(stub_model):
    calc = AIMNet2Calculator(stub_model, device="cpu")
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        for _ in range(3):
            calc._warn_cuda_graph_fallback("reason A")
        calc._warn_cuda_graph_fallback("reason B")
    assert _graph_warnings(record) == [
        "AIMNet2Calculator: CUDA-graph replay is not used (reason A); the regular code path runs instead.",
        "AIMNet2Calculator: CUDA-graph replay is not used (reason B); the regular code path runs instead.",
    ]
    assert record[0].filename == __file__  # attributed to the caller, not to aimnet internals


def _bare_cache(warned):
    """A cache whose dispatch is reached without a model: only the input checks run before the warning."""
    cache = object.__new__(cache_module.CUDAGraphCache)
    cache._checking = False
    cache._calc = types.SimpleNamespace(_warn_cuda_graph_fallback=warned.append)
    return cache


@pytest.mark.parametrize(
    ("extra", "reason"),
    [
        ({"mol_idx": torch.zeros(3, dtype=torch.long)}, "mol_idx"),
        ({"cell": torch.eye(3)}, "periodic cell"),
        ({"nbmat": torch.zeros(3, 2, dtype=torch.long)}, "nbmat"),
    ],
)
def test_call_level_input_reasons(extra, reason):
    warned = []
    cache = _bare_cache(warned)
    data = {"coord": torch.zeros(3, 3), "numbers": torch.ones(3, dtype=torch.long), "charge": 0.0, **extra}
    assert cache.eval(data, forces=True, validate_species=False) is None
    assert len(warned) == 1 and reason in warned[0]


def test_call_level_requires_grad_reason():
    warned = []
    cache = _bare_cache(warned)
    data = {"coord": torch.zeros(3, 3, requires_grad=True), "numbers": torch.ones(3, dtype=torch.long), "charge": 0.0}
    assert cache.eval(data, forces=True, validate_species=False) is None
    assert warned == ["input tensors require grad"]


def test_internal_reference_call_does_not_warn():
    warned = []
    cache = _bare_cache(warned)
    cache._checking = True
    data = {"coord": torch.zeros(3, 3), "numbers": torch.ones(3, dtype=torch.long), "charge": 0.0, "mol_idx": None}
    assert cache.eval(data, forces=True, validate_species=False) is None
    assert warned == []


def test_compile_precedence_warnings(stub_model):
    with pytest.warns(UserWarning) as record:
        AIMNet2Calculator(stub_model, device="cuda", compile_model=True, cuda_graph=True, inference_kernels=True)
    messages = [str(w.message) for w in record]
    assert any("compile_model=True with cuda_graph=True" in m for m in messages)
    assert any("inference_kernels=True has no effect on compiled calls" in m for m in messages)


@pytest.fixture
def fresh_triton_probe():
    triton_support._unsupported_reason.cache_clear()
    yield
    triton_support._unsupported_reason.cache_clear()


@pytest.mark.parametrize(("capability", "supported"), [((7, 5), False), ((8, 0), True), ((8, 9), True)])
def test_triton_kernels_require_compute_capability_8(monkeypatch, fresh_triton_probe, capability, supported):
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index=None: capability)
    monkeypatch.setattr(triton_support.importlib.util, "find_spec", lambda name: object())
    reason = triton_support.triton_unsupported_reason("cuda:0")
    assert (reason is None) is supported
    if not supported:
        assert "compute capability 7.5" in reason


def test_triton_kernels_refused_on_rocm(monkeypatch, fresh_triton_probe):
    monkeypatch.setattr(torch.version, "hip", "6.2.0")
    assert "ROCm" in triton_support.triton_unsupported_reason("cuda:0")


def test_low_capability_warns_for_inference_kernels(monkeypatch, stub_model, fresh_triton_probe):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index=None: (7, 0))
    with pytest.warns(UserWarning, match=r"Triton inference kernels are not used \(compute capability 7\.0"):
        AIMNet2Calculator(stub_model, device="cuda:0", inference_kernels=True)


def test_force_tolerance_rejects_differences_above_fp32_noise():
    ref = {"energy": torch.zeros(1), "charges": torch.zeros(1, 4), "forces": torch.ones(1, 4, 3)}
    same = (torch.zeros(1), ref["forces"] + 1e-5, torch.zeros(1, 4))
    cache_module.CUDAGraphCache._compare(*same, ref, forces=True, B=1, N=4)  # within noise: accepted
    off = (torch.zeros(1), ref["forces"] + 5e-4, torch.zeros(1, 4))
    with pytest.raises(RuntimeError, match="forces"):
        cache_module.CUDAGraphCache._compare(*off, ref, forces=True, B=1, N=4)
