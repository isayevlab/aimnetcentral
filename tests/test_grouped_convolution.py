"""First-/higher-order contracts of shared per-pass dense convolutions."""

from types import MethodType

import pytest
import torch

from aimnet.models.aimnet2 import AIMNet2
from aimnet.modules import aev_inference
from aimnet.modules.aev import ConvSV
from aimnet.modules.core import AtomicSum, Output


@pytest.mark.parametrize(
    "flags,channels", [((False,), (2,)), ((True, False), (3, 2)), ((False, True, False), (1, 4, 2))]
)
def test_grouped_primitive_outputs_gradients_and_hvp(flags, channels):
    torch.manual_seed(1729)
    batch, shifts, neighbors, centers = 2, 3, 4, 5
    geometry = torch.randn(batch, shifts, neighbors, centers, 4, dtype=torch.float64, requires_grad=True)
    features = tuple(
        torch.randn(
            (batch, neighbors, channel, shifts) if flag else (batch, neighbors, channel),
            dtype=torch.float64,
            requires_grad=True,
        )
        for channel, flag in zip(channels, flags, strict=True)
    )
    reference = tuple(
        aev_inference.dense_convolution(feature, geometry, flag) for feature, flag in zip(features, flags, strict=True)
    )
    actual = aev_inference.dense_convolutions(features, geometry, flags)
    for result, expected in zip(actual, reference, strict=True):
        torch.testing.assert_close(result, expected, atol=1e-12, rtol=1e-12)
    inputs = (*features, geometry)
    directions = tuple(torch.randn_like(value) for value in inputs)

    def derivatives(outputs):
        loss = sum((index + 1) * value.sin().sum() for index, value in enumerate(outputs))
        gradients = torch.autograd.grad(loss, inputs, create_graph=True)
        hvp = torch.autograd.grad(
            sum((gradient * direction).sum() for gradient, direction in zip(gradients, directions, strict=True)),
            inputs,
        )
        return (*gradients, *hvp)

    for result, expected in zip(derivatives(actual), derivatives(reference), strict=True):
        torch.testing.assert_close(result, expected, atol=1e-10, rtol=1e-11)


def test_grouped_primitive_unused_output_and_repeated_backward():
    geometry = torch.randn(1, 2, 3, 3, 4, dtype=torch.float64, requires_grad=True)
    first = torch.randn(1, 3, 2, dtype=torch.float64, requires_grad=True)
    unused = torch.randn(1, 3, 4, 2, dtype=torch.float64, requires_grad=True)
    result, _ = aev_inference.dense_convolutions((first, unused), geometry, (False, True))
    expected = aev_inference.dense_convolution(first, geometry, False)
    reference = torch.autograd.grad(expected.square().sum(), (first, geometry))
    for _ in range(2):
        gradients = torch.autograd.grad(result.square().sum(), (first, geometry, unused), retain_graph=True)
        for actual, target in zip(gradients[:2], reference, strict=True):
            torch.testing.assert_close(actual, target, atol=1e-12, rtol=1e-12)
        assert torch.count_nonzero(gradients[2]) == 0


def _model(*, d2features=True, charge_channels=1, passes=3):
    torch.manual_seed(71)
    model = (
        AIMNet2(
            aev={"nshifts_s": 3},
            nfeature=2,
            d2features=d2features,
            ncomb_v=3,
            hidden=tuple([7] for _ in range(passes)),
            aim_size=4,
            num_charge_channels=charge_channels,
            outputs=[
                Output(mlp={"hidden": []}, n_in=4, n_out=1, key_in="aim", key_out="atomic_energy"),
                AtomicSum("atomic_energy", "energy"),
            ],
        )
        .double()
        .eval()
    )
    # The normal preparation metadata intentionally defaults to FP32; match it
    # to the FP64 weights for these high-precision derivative comparisons.
    model._required_keys_dtype = [
        torch.float64 if dtype == torch.float32 else dtype for dtype in model._required_keys_dtype
    ]
    model._optional_keys_dtype = [
        torch.float64 if dtype == torch.float32 else dtype for dtype in model._optional_keys_dtype
    ]
    return model


def _input():
    return {
        "coord": torch.randn(2, 5, 3, dtype=torch.float64, requires_grad=True),
        "numbers": torch.tensor([[6, 8, 1, 1, 0], [6, 7, 1, 0, 0]]),
        "charge": torch.tensor([0.0, 1.0], dtype=torch.float64, requires_grad=True),
        "mult": torch.tensor([1.0, 2.0], dtype=torch.float64),
    }


def _optimized(model, source):
    data = model.prepare_input(dict(source))
    data["_inference_kernels"] = torch.tensor(True)
    data["_aev_inference_constants"] = model.aev.prepare_inference()
    return model.forward_prepared(data)


def _same_outputs(actual, expected):
    for key in ("energy", "charges", "aim"):
        torch.testing.assert_close(actual[key], expected[key], atol=1e-10, rtol=1e-10)


def _count_grouped(monkeypatch):
    calls = []

    def instrument(original):
        def counted(*args):
            if len(args[0]) > 1:
                calls.append(tuple(feature.shape[2] for feature in args[0]))
            return original(*args)

        return counted

    for name in ("dense_convolutions", "dense_convolutions_carry"):
        monkeypatch.setattr(aev_inference, name, instrument(getattr(aev_inference, name)))
    return calls


def _forbid_grouped(monkeypatch):
    def forbidden(*args):
        pytest.fail("Hooks/custom methods/default execution must retain individual module calls")

    monkeypatch.setattr(aev_inference, "dense_convolutions", forbidden)
    monkeypatch.setattr(aev_inference, "dense_convolutions_carry", forbidden)


@pytest.mark.parametrize("d2features", [False, True])
@pytest.mark.parametrize("charge_channels", [1, 2])
@pytest.mark.parametrize("passes", [2, 4])
def test_padded_model_equations_and_derivatives(monkeypatch, d2features, charge_channels, passes):
    model = _model(d2features=d2features, charge_channels=charge_channels, passes=passes)
    source = _input()
    calls = _count_grouped(monkeypatch)
    expected = model(dict(source))
    assert not calls  # The default path remains opt-out.
    actual = _optimized(model, source)
    assert calls == [(2, charge_channels)] * (passes - 1)
    _same_outputs(actual, expected)
    inputs = (source["coord"], source["charge"], model.afv.weight, model.conv_a.agh, model.conv_q.agh)
    reference = torch.autograd.grad(expected["energy"].sum(), inputs, create_graph=True)
    gradients = torch.autograd.grad(actual["energy"].sum(), inputs, create_graph=True)
    for result, target in zip(gradients, reference, strict=True):
        torch.testing.assert_close(result, target, atol=1e-9, rtol=1e-9)
    direction = torch.randn_like(source["coord"])
    hvp_ref = torch.autograd.grad((reference[0] * direction).sum(), source["coord"])[0]
    hvp = torch.autograd.grad((gradients[0] * direction).sum(), source["coord"])[0]
    torch.testing.assert_close(hvp, hvp_ref, atol=1e-8, rtol=1e-8)
    assert torch.count_nonzero(gradients[0][source["numbers"] == 0]) == 0


@pytest.mark.parametrize("kind", ["forward", "forward_pre", "full_backward", "full_backward_pre"])
def test_local_convolution_hooks_keep_module_calls(monkeypatch, kind):
    model = _model()
    source = _input()
    calls = []

    def hook(module, *args):
        calls.append(module)

    handle = getattr(model.conv_q, f"register_{kind}_hook")(hook)
    _forbid_grouped(monkeypatch)
    try:
        expected = model(dict(source))
        torch.autograd.grad(expected["energy"].sum(), source["coord"])
        calls.clear()
        actual = _optimized(model, source)
        torch.autograd.grad(actual["energy"].sum(), source["coord"])
        _same_outputs(actual, expected)
        assert calls == [model.conv_q] * (len(model.mlps) - 1)
    finally:
        handle.remove()


@pytest.mark.parametrize("kind", ["forward", "forward_pre"])
def test_global_convolution_hooks_keep_module_calls(monkeypatch, kind):
    model = _model()
    source = _input()
    calls = []

    def hook(module, *args):
        if module is model.conv_a or module is model.conv_q:
            calls.append(module)

    handle = getattr(torch.nn.modules.module, f"register_module_{kind}_hook")(hook)
    _forbid_grouped(monkeypatch)
    try:
        expected = model(dict(source))
        calls.clear()
        actual = _optimized(model, source)
        _same_outputs(actual, expected)
        assert calls.count(model.conv_a) == len(model.mlps)
        assert calls.count(model.conv_q) == len(model.mlps) - 1
    finally:
        handle.remove()


@pytest.mark.parametrize("name", ["_prepare_in_a", "_prepare_in_q"])
def test_prepare_overrides_keep_canonical_dispatch(monkeypatch, name):
    model = _model()
    source = _input()
    calls = []
    original = getattr(model, name)

    def changed(self, data):
        calls.append(self)
        return 0.71 * original(data)

    monkeypatch.setattr(model, name, MethodType(changed, model))
    _forbid_grouped(monkeypatch)
    expected = model(dict(source))
    calls.clear()
    actual = _optimized(model, source)
    _same_outputs(actual, expected)
    assert len(calls) == len(model.mlps) - (name == "_prepare_in_q")


@pytest.mark.parametrize("owner,name", [(ConvSV, "__call__"), (torch.nn.Module, "_call_impl")])
def test_class_dispatch_overrides_keep_convolution_calls(monkeypatch, owner, name):
    model = _model()
    source = _input()
    unchanged = model(dict(source))
    original = getattr(owner, name)
    calls = []

    def changed(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if isinstance(self, ConvSV):
            calls.append(self)
            return result * 0.71
        return result

    monkeypatch.setattr(owner, name, changed)
    _forbid_grouped(monkeypatch)
    expected = model(dict(source))
    # The override must change actual computation, not only dispatch counters.
    assert not torch.equal(expected["energy"], unchanged["energy"])
    expected_calls = list(calls)
    assert expected_calls.count(model.conv_a) == len(model.mlps)
    assert expected_calls.count(model.conv_q) == len(model.mlps) - 1
    calls.clear()
    actual = _optimized(model, source)
    assert calls == expected_calls
    _same_outputs(actual, expected)
    reference = torch.autograd.grad(expected["energy"].sum(), source["coord"])[0]
    gradient = torch.autograd.grad(actual["energy"].sum(), source["coord"])[0]
    torch.testing.assert_close(gradient, reference, atol=1e-9, rtol=1e-9)


def test_aev_hook_replacement_is_consumed_by_grouped_path(monkeypatch):
    model = _model()
    source = _input()
    calls = _count_grouped(monkeypatch)

    def scale_geometry(module, args, output):
        output["g_sv"] = 0.37 * output["g_sv"]
        return output

    handle = model.aev.register_forward_hook(scale_geometry)
    try:
        expected = model(dict(source))
        actual = _optimized(model, source)
        assert len(calls) == len(model.mlps) - 1
        _same_outputs(actual, expected)
        reference = torch.autograd.grad(expected["energy"].sum(), source["coord"])[0]
        gradient = torch.autograd.grad(actual["energy"].sum(), source["coord"])[0]
        torch.testing.assert_close(gradient, reference, atol=1e-9, rtol=1e-9)
    finally:
        handle.remove()
