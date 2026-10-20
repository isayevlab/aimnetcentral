"""Optional ConvSV finish kernel: layouts, derivatives and fallback contracts."""

import math

import pytest
import torch

pytest.importorskip("triton")

from aimnet.kernels.conv_finish_triton import _reference, conv_finish


@pytest.mark.parametrize("trainable_weights", [False, True])
@pytest.mark.parametrize("masked", [False, True])
def test_cpu_fallback_gradchecks(trainable_weights, masked):
    generator = torch.Generator().manual_seed(13)
    x = torch.randn(2, 2, 2, 3, 4, generator=generator, dtype=torch.float64, requires_grad=True)
    weights = torch.randn(2, 3, 2, generator=generator, dtype=torch.float64, requires_grad=trainable_weights)
    mask = torch.tensor([[False, True], [False, False]]) if masked else None

    def function(x, weights):
        return conv_finish(x, weights, mask)

    torch.testing.assert_close(function(x, weights), _reference(x, weights, mask), rtol=0, atol=0)
    assert torch.autograd.gradcheck(function, (x, weights))
    assert torch.autograd.gradgradcheck(function, (x, weights))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "centers,features,shifts,combinations", [(17, 16, 16, 12), (60, 16, 16, 16), (17, 1, 16, 16), (9, 3, 7, 5)]
)
def test_cuda_strides_values_and_first_through_third_derivatives(centers, features, shifts, combinations):
    generator = torch.Generator(device="cuda").manual_seed(13)
    native = torch.randn(2, shifts, features + 1, centers, 4, device="cuda", generator=generator) * 0.3
    # This models the holes between A and Q slices in a grouped contraction.
    x = native[:, :, :features].permute(0, 3, 2, 1, 4).detach().requires_grad_(True)
    weights = (
        torch.randn(features, combinations, shifts, device="cuda", generator=generator) / math.sqrt(shifts)
    ).transpose(1, 2)
    mask = (torch.arange(4 * centers, device="cuda").reshape(2, 2 * centers) % 3 == 0)[:, ::2]
    grad = torch.randn(2, centers, 2 * features * (shifts + combinations), device="cuda", generator=generator)[..., ::2]
    before = tuple(value.detach().clone() for value in (x, weights, mask, grad))
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        actual, expected = conv_finish(x, weights, mask), _reference(x, weights, mask)
        observed = torch.autograd.grad(actual, x, grad, retain_graph=True)[0]
        reference = torch.autograd.grad(expected, x, grad, retain_graph=True)[0]
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(observed, reference, rtol=2e-5, atol=5e-5)
        direction = torch.randn(x.shape, device="cuda", generator=generator)
        derivative = torch.autograd.grad(actual.sin().sum(), x, create_graph=True)[0]
        reference_derivative = torch.autograd.grad(expected.sin().sum(), x, create_graph=True)[0]
        for order in (2, 3):
            derivative = torch.autograd.grad((derivative * direction).sum(), x, create_graph=order < 3)[0]
            reference_derivative = torch.autograd.grad(
                (reference_derivative * direction).sum(), x, create_graph=order < 3
            )[0]
            torch.testing.assert_close(derivative, reference_derivative, rtol=5e-5, atol=5e-4)
        for original, current in zip(before, (x, weights, mask, grad), strict=True):
            torch.testing.assert_close(current, original, rtol=0, atol=0)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("which", ["input", "weight"])
def test_cuda_saved_tensor_mutation_detected(which):
    x = torch.randn(2, 7, 3, 5, 4, device="cuda", requires_grad=True)
    weights = torch.randn(3, 5, 7, device="cuda")
    output = conv_finish(x, weights, None)
    with torch.no_grad():
        (x if which == "input" else weights).add_(0.1)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        torch.autograd.grad(output.sum(), x)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("reason", ["single_batch", "trainable_weights", "large_basis", "autocast"])
def test_cuda_unsupported_contracts_use_torch(monkeypatch, reason):
    import aimnet.kernels.conv_finish_triton as implementation

    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported inputs dispatched the optimized kernel")

    monkeypatch.setattr(implementation._ConvFinish, "apply", forbidden)
    batch = 1 if reason == "single_batch" else 2
    shifts = 33 if reason == "large_basis" else 5
    x = torch.randn(batch, 7, 3, shifts, 4, device="cuda", requires_grad=True)
    weights = torch.randn(3, shifts, 7, device="cuda", requires_grad=reason == "trainable_weights")
    with torch.autocast("cuda", enabled=reason == "autocast"):
        actual, expected = conv_finish(x, weights, None), _reference(x, weights, None)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    requested = (x, weights) if weights.requires_grad else (x,)
    actual_grads = torch.autograd.grad(actual.sum(), requested)
    expected_grads = torch.autograd.grad(expected.sum(), requested)
    for observed, reference in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(observed, reference, rtol=0, atol=0)
