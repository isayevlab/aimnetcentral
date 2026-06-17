import importlib.util
from pathlib import Path

import pytest
import torch
from torch import nn

from aimnet.calculators import AIMNet2Calculator
from aimnet.precision import (
    PrecisionPolicy,
    available_precision_modes,
    resolve_precision_policy,
    supports_bf16,
    supports_fp8,
    supports_tf32,
    torch_precision_context,
)


def test_resolve_precision_policy_presets():
    assert set(available_precision_modes()) == {
        "strict",
        "tf32_learned",
        "bf16_learned",
        "fp16_learned",
        "fp8_learned_experimental",
        "emulated_reductions",
    }
    strict = resolve_precision_policy(None)
    assert strict.mode == "strict"
    assert strict.learned_amp_dtype is None
    assert strict.allow_tf32_matmul is False

    tf32 = resolve_precision_policy("tf32_learned")
    assert tf32.mode == "tf32_learned"
    assert tf32.allow_tf32_matmul is True
    assert tf32.strict_long_range is True

    bf16 = resolve_precision_policy("bf16_learned")
    assert bf16.learned_amp_dtype is torch.bfloat16


def test_invalid_precision_policy_raises():
    with pytest.raises(ValueError, match="Unknown precision policy"):
        resolve_precision_policy("tf33")
    with pytest.raises(TypeError, match="precision must be"):
        resolve_precision_policy(123)  # type: ignore[arg-type]


def test_torch_precision_context_restores_state():
    before = _backend_state()
    policy = PrecisionPolicy(mode="tf32_learned", allow_tf32_matmul=True, allow_tf32_cudnn=True)

    with torch_precision_context(policy):
        assert torch.get_float32_matmul_precision() == "high"
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            assert torch.backends.cuda.matmul.allow_tf32 is True
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            assert torch.backends.cudnn.allow_tf32 is True

    assert _backend_state() == before


def test_strict_policy_disables_reduced_precision_inside_context():
    before = _backend_state()
    with torch_precision_context(resolve_precision_policy("strict")):
        assert torch.get_float32_matmul_precision() == "highest"
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            assert torch.backends.cuda.matmul.allow_tf32 is False
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            assert torch.backends.cudnn.allow_tf32 is False
    assert _backend_state() == before


def test_capability_probes_do_not_raise():
    assert isinstance(supports_tf32("cpu"), bool)
    assert isinstance(supports_bf16("cpu"), bool)
    assert supports_fp8("cpu", "none") is False
    assert isinstance(supports_fp8("cpu", "torchao"), bool)


class _RecordingModel(nn.Module):
    cutoff = 3.0

    def __init__(self):
        super().__init__()
        self.seen_precision: list[str] = []

    def forward(self, data):
        self.seen_precision.append(torch.get_float32_matmul_precision())
        data["energy"] = torch.zeros(1, device=data["coord"].device, dtype=data["coord"].dtype)
        return data


def test_calculator_accepts_precision_and_restores_backend_state():
    model = _RecordingModel()
    calc = AIMNet2Calculator(model, device="cpu", precision="tf32_learned")
    before = _backend_state()

    out = calc(
        {
            "coord": torch.tensor([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]),
            "numbers": torch.tensor([8, 1, 1]),
            "charge": 0.0,
        }
    )

    assert "energy" in out
    assert model.seen_precision == ["high"]
    assert _backend_state() == before


def test_benchmark_workloads_and_metrics_are_importable():
    bench = _load_bench_module()
    workloads = bench.load_workloads(Path(__file__).resolve().parents[1] / "benchmarks/precision/workloads.yaml")
    assert "water_forces" in workloads
    generated = bench.build_data(workloads["generated_organic_128"])
    assert len(generated["numbers"]) == 128
    large_workloads = bench.load_workloads(Path(__file__).resolve().parents[1] / "benchmarks/precision/large_workloads.yaml")
    large = bench.build_data(large_workloads["water_cluster_501"])
    assert len(large["numbers"]) == 501
    assert large["coord"].shape == (501, 3)
    assert bench.mode_supported("strict", "cpu", "none") is True

    ref = {"energy": torch.tensor([1.0]), "forces": torch.zeros(2, 3)}
    result = {"energy": torch.tensor([1.001]), "forces": torch.ones(2, 3) * 0.01}
    metrics = bench.accuracy_metrics(ref, result, {"charge": 0.0})
    assert metrics["energy_mae_ev"] == pytest.approx(0.001, rel=1e-4)
    assert metrics["force_rmse_ev_per_a"] == pytest.approx(0.01)


def _backend_state():
    state = {"matmul": torch.get_float32_matmul_precision()}
    for key, obj, attr in (
        ("cuda_matmul_allow_tf32", torch.backends.cuda.matmul, "allow_tf32"),
        ("cuda_matmul_fp32_precision", torch.backends.cuda.matmul, "fp32_precision"),
        ("cudnn_allow_tf32", torch.backends.cudnn, "allow_tf32"),
        ("cudnn_fp32_precision", torch.backends.cudnn, "fp32_precision"),
    ):
        if hasattr(obj, attr):
            state[key] = getattr(obj, attr)
    return state


def _load_bench_module():
    path = Path(__file__).resolve().parents[1] / "benchmarks/precision/bench_precision.py"
    spec = importlib.util.spec_from_file_location("bench_precision", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
