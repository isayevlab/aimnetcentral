"""CPU-only autograd and observable-state contracts of geometry continuation."""

from types import MethodType

import pytest
import torch

from aimnet.modules import aev_inference


def _reference(left, right):
    return torch.bmm(left, right), right


def _jets(loss, inputs, directions):
    first = torch.autograd.grad(loss, inputs, create_graph=True, retain_graph=True)
    second = torch.autograd.grad(
        sum((value * direction).sum() for value, direction in zip(first, directions, strict=True)),
        inputs,
        create_graph=True,
        retain_graph=True,
    )
    third = torch.autograd.grad(
        sum((value * direction).sum() for value, direction in zip(second, directions, strict=True)),
        inputs,
        retain_graph=True,
    )
    return (*first, *second, *third)


def _assert_jets(actual, expected, inputs):
    directions = tuple(torch.randn_like(value) for value in inputs)
    for value, reference in zip(_jets(actual, inputs, directions), _jets(expected, inputs, directions), strict=True):
        torch.testing.assert_close(value, reference, rtol=2e-10, atol=2e-10)


@pytest.mark.parametrize("noncontiguous", [False, True])
def test_bmm_carry_gradcheck_and_gradgradcheck(noncontiguous):
    torch.manual_seed(411)
    left = torch.randn(2, 2, 3, dtype=torch.float64, requires_grad=True)
    right = torch.randn(2, 3, 2, dtype=torch.float64, requires_grad=True)
    if noncontiguous:
        left = left.transpose(1, 2).contiguous().transpose(1, 2)
        right = right.transpose(1, 2).contiguous().transpose(1, 2)
        assert not left.is_contiguous() and not right.is_contiguous()
    primitive = aev_inference._BmmGeometryCarry.apply
    assert torch.autograd.gradcheck(primitive, (left, right))
    assert torch.autograd.gradgradcheck(primitive, (left, right))


def test_bmm_carry_complex_gradcheck_and_gradgradcheck():
    torch.manual_seed(418)
    left = torch.randn(1, 2, 3, dtype=torch.complex128, requires_grad=True)
    right = torch.randn(1, 3, 2, dtype=torch.complex128, requires_grad=True)
    primitive = aev_inference._BmmGeometryCarry.apply
    assert torch.autograd.gradcheck(primitive, (left, right))
    assert torch.autograd.gradgradcheck(primitive, (left, right))


@pytest.mark.parametrize("requires_grad", [(True, True), (True, False), (False, True), (False, False)])
def test_bmm_carry_selective_gradients_and_no_grad(requires_grad):
    torch.manual_seed(412)
    left = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=requires_grad[0])
    right = torch.randn(2, 4, 5, dtype=torch.float64, requires_grad=requires_grad[1])
    result, carry = aev_inference._BmmGeometryCarry.apply(left, right)
    expected, geometry = _reference(left, right)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    torch.testing.assert_close(carry, geometry, rtol=0, atol=0)
    inputs = tuple(value for value in (left, right) if value.requires_grad)
    if inputs:
        _assert_jets(result.sin().sum() + carry.cos().sum(), expected.sin().sum() + geometry.cos().sum(), inputs)
    else:
        assert not result.requires_grad and not carry.requires_grad


@pytest.mark.parametrize("used", ["contraction", "carry", "both"])
def test_unused_outputs_and_repeated_backward_do_not_mutate_gradients(used):
    torch.manual_seed(413)
    left = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
    right = torch.randn(2, 4, 5, dtype=torch.float64, requires_grad=True)
    actual = aev_inference._BmmGeometryCarry.apply(left, right)
    expected = _reference(left, right)

    def loss(outputs):
        if used == "contraction":
            return outputs[0].sin().sum()
        if used == "carry":
            return outputs[1].cos().sum()
        return outputs[0].sin().sum() + outputs[1].cos().sum()

    target = torch.autograd.grad(loss(expected), (left, right), allow_unused=True, retain_graph=True)
    previous = None
    for _ in range(3):
        gradients = torch.autograd.grad(loss(actual), (left, right), allow_unused=True, retain_graph=True)
        for value, reference in zip(gradients, target, strict=True):
            if reference is None:
                assert value is None or torch.count_nonzero(value) == 0
            else:
                torch.testing.assert_close(value, reference, rtol=1e-12, atol=1e-12)
        if previous is not None:
            for value, snapshot in previous:
                torch.testing.assert_close(value, snapshot, rtol=0, atol=0)
        previous = [(value, value.clone()) for value in gradients if value is not None]


def test_branched_carry_and_external_geometry_consumers_preserve_derivatives():
    torch.manual_seed(414)
    lefts = tuple(torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True) * 0.2 for _ in range(3))
    geometry = torch.randn(2, 4, 5, dtype=torch.float64, requires_grad=True) * 0.2

    def evaluate(primitive):
        first, first_geometry = primitive(lefts[0], geometry)
        second, second_geometry = primitive(lefts[1], first_geometry)
        third, third_geometry = primitive(lefts[2], first_geometry)
        return (
            first.sin().sum()
            + second.cos().sum()
            + third.square().sum()
            + first_geometry.sin().sum()
            + second_geometry.cos().sum()
            + third_geometry.square().sum()
            + geometry.exp().sum()
        )

    _assert_jets(evaluate(aev_inference._BmmGeometryCarry.apply), evaluate(_reference), (*lefts, geometry))


@pytest.mark.parametrize("passes", [2, 5])
def test_sequential_features_depend_on_previous_outputs_and_geometry(passes):
    """Charge/feature response must remain differentiable across the whole chain."""
    torch.manual_seed(415)
    features, geometry, weight = ((torch.randn(2, 3, 3, dtype=torch.float64) * 0.2).requires_grad_() for _ in range(3))

    def evaluate(primitive):
        current, carry = features, geometry
        loss = geometry.sin().sum()
        for index in range(passes):
            output, carry = primitive(current, carry)
            loss = loss + (index + 1) * output.sin().sum()
            current = torch.tanh(output) + 0.1 * torch.bmm(current, weight) + 0.1 * carry.square()
        return loss + carry.cos().sum()

    _assert_jets(evaluate(aev_inference._BmmGeometryCarry.apply), evaluate(_reference), (features, geometry, weight))


@pytest.mark.parametrize(
    "flags,channels", [((False,), (2,)), ((True,), (3,)), ((True, False), (3, 2)), ((False, True, False), (1, 4, 2))]
)
def test_dense_carry_helpers_outputs_and_mixed_higher_derivatives(flags, channels):
    torch.manual_seed(416)
    batch, shifts, neighbors, centers = 2, 2, 3, 4
    geometry = (torch.randn(batch, shifts, neighbors, centers, 4, dtype=torch.float64) * 0.2).requires_grad_()
    features = tuple(
        (
            torch.randn(
                (batch, neighbors, channel, shifts) if flag else (batch, neighbors, channel), dtype=torch.float64
            )
            * 0.2
        ).requires_grad_()
        for flag, channel in zip(flags, channels, strict=True)
    )
    actual, carry = aev_inference.dense_convolutions_carry(features, geometry, flags)
    expected = aev_inference.dense_convolutions(features, geometry, flags)
    torch.testing.assert_close(carry, geometry, rtol=0, atol=0)
    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, reference, rtol=0, atol=0)
    actual_loss = sum((index + 1) * value.sin().sum() for index, value in enumerate(actual)) + carry.cos().sum()
    expected_loss = sum((index + 1) * value.sin().sum() for index, value in enumerate(expected)) + geometry.cos().sum()
    _assert_jets(actual_loss, expected_loss, (*features, geometry))
    if len(features) == 1:
        single, single_carry = aev_inference.dense_convolution_carry(features[0], geometry, flags[0])
        torch.testing.assert_close(single, actual[0], rtol=0, atol=0)
        torch.testing.assert_close(single_carry, geometry, rtol=0, atol=0)


@pytest.mark.parametrize("flags", [(), (True, False)])
def test_dense_carry_rejects_mismatched_feature_flags(flags):
    geometry = torch.randn(2, 2, 3, 3, 4)
    features = (torch.randn(2, 3, 2),) if flags else ()
    with pytest.raises(ValueError, match="one d2features flag"):
        aev_inference.dense_convolutions_carry(features, geometry, flags)


def test_dense_carry_preserves_autocast_backward_outside_the_context():
    torch.manual_seed(417)
    geometry = torch.randn(2, 2, 3, 3, 4, requires_grad=True)
    features = torch.randn(2, 3, 2, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        (actual,), carry = aev_inference.dense_convolutions_carry((features,), geometry, (False,))
        expected = aev_inference.dense_convolution(features, geometry, False)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    reference = torch.autograd.grad(expected.float().square().sum() + geometry.square().sum(), (features, geometry))
    gradients = torch.autograd.grad(actual.float().square().sum() + carry.square().sum(), (features, geometry))
    for result, target in zip(gradients, reference, strict=True):
        torch.testing.assert_close(result, target, rtol=0, atol=0)


def test_custom_mid_loop_geometry_replacement_keeps_existing_semantics(monkeypatch):
    from test_grouped_convolution import _input, _model, _optimized, _same_outputs

    model = _model()
    source = _input()
    observed = {}
    aev_forward = model.aev.forward
    mlp_forward = model.mlps[0].forward

    def capture(self, data):
        result = aev_forward(data)
        observed["data"] = result
        return result

    def replace(self, features):
        result = mlp_forward(features)
        data = observed["data"]
        data["g_sv"] = 0.37 * data["g_sv"]
        return result

    monkeypatch.setattr(model.aev, "forward", MethodType(capture, model.aev))
    monkeypatch.setattr(model.mlps[0], "forward", MethodType(replace, model.mlps[0]))
    expected = model(dict(source))
    actual = _optimized(model, source)
    _same_outputs(actual, expected)
    assert "_geometry_carry" not in actual
    reference = torch.autograd.grad(expected["energy"].sum(), source["coord"])[0]
    gradient = torch.autograd.grad(actual["energy"].sum(), source["coord"])[0]
    torch.testing.assert_close(gradient, reference, rtol=1e-9, atol=1e-9)


@pytest.mark.slow
def test_model_retains_public_geometry_identity_and_total_gradient(monkeypatch):
    from test_grouped_convolution import _input, _model, _same_outputs

    model = _model()
    source = _input()
    expected = model(dict(source))
    data = model.prepare_input(dict(source))
    data["_inference_kernels"] = torch.tensor(True)
    data["_aev_inference_constants"] = model.aev.prepare_inference()
    original = aev_inference.dense_convolutions_carry
    observed = []

    def capture(*args):
        observed.append(data["g_sv"])
        return original(*args)

    monkeypatch.setattr(aev_inference, "dense_convolutions_carry", capture)
    actual = model.forward_prepared(data)
    assert len(observed) == len(model.mlps)
    assert all(value is actual["g_sv"] for value in observed)
    assert not any(key.startswith("_geometry_carry") for key in actual)
    _same_outputs(actual, expected)
    reference = torch.autograd.grad(expected["energy"].sum(), expected["g_sv"], create_graph=True)[0]
    gradient = torch.autograd.grad(actual["energy"].sum(), actual["g_sv"], create_graph=True)[0]
    torch.testing.assert_close(gradient, reference, rtol=1e-9, atol=1e-9)
    reference_mixed = torch.autograd.grad(reference.square().sum(), model.afv.weight)[0]
    mixed = torch.autograd.grad(gradient.square().sum(), model.afv.weight)[0]
    torch.testing.assert_close(mixed, reference_mixed, rtol=1e-8, atol=1e-8)


@pytest.mark.slow
def test_model_force_loss_parameter_gradients_preserve_charge_response():
    from test_grouped_convolution import _input, _model, _optimized

    model = _model()
    source = _input()
    expected = model(dict(source))
    actual = _optimized(model, source)
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)

    def force_loss(output):
        force = -torch.autograd.grad(output["energy"].sum(), source["coord"], create_graph=True)[0]
        return force.square().sum() + output["charges"].square().sum() + output["energy"].square().sum()

    reference = torch.autograd.grad(force_loss(expected), parameters)
    gradients = torch.autograd.grad(force_loss(actual), parameters)
    for result, target in zip(gradients, reference, strict=True):
        torch.testing.assert_close(result, target, rtol=1e-8, atol=1e-8)


def test_geometry_carry_state_removed_when_a_contraction_raises(monkeypatch):
    from test_grouped_convolution import _input, _model

    model = _model()
    data = model.prepare_input(_input())
    data["_inference_kernels"] = torch.tensor(True)
    data["_aev_inference_constants"] = model.aev.prepare_inference()

    def fail(*args):
        assert "_geometry_carry" in data
        raise RuntimeError("test contraction failure")

    monkeypatch.setattr(aev_inference, "dense_convolutions_carry", fail)
    with pytest.raises(RuntimeError, match="test contraction failure"):
        model.forward_prepared(data)
    assert "g_sv" in data
    assert not any(key.startswith("_geometry_carry") for key in data)


@pytest.mark.parametrize("mode", ["default", "single", "training"])
def test_unselected_model_paths_do_not_enter_geometry_carry(monkeypatch, mode):
    from test_grouped_convolution import _input, _model, _optimized

    model = _model()
    source = _input()

    def forbidden(*args):
        pytest.fail("Unselected model path must not enter geometry continuation")

    monkeypatch.setattr(aev_inference, "dense_convolutions_carry", forbidden)
    if mode == "default":
        output = model(source)
    else:
        if mode == "single":
            source = {key: value[:1] for key, value in source.items()}
        else:
            model.train()
        output = _optimized(model, source)
    assert not any(key.startswith("_geometry_carry") for key in output)
