"""CUDA contracts for the optional out-of-place geometry-gradient accumulator."""

import sys

import pytest
import torch

from aimnet.modules import aev_inference

pytestmark = pytest.mark.gpu


@pytest.fixture
def kernel_module():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    pytest.importorskip("triton")
    import aimnet.kernels.geometry_baddbmm_triton as module

    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield module
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def _operands(inner=17, layout="contiguous", dtype=torch.float32):
    batch, rows, columns = 2, 31, 67
    options = {"device": "cuda", "dtype": dtype}
    if layout == "transposed":
        accum = torch.randn(batch, columns, rows, **options).transpose(1, 2)
        left = torch.randn(batch, inner, rows, **options).transpose(1, 2)
        right = torch.randn(batch, columns, inner, **options).transpose(1, 2)
    elif layout == "expanded":
        accum = torch.randn(1, rows, 1, **options).expand(batch, rows, columns)
        left = torch.randn(batch, 1, inner, **options).expand(batch, rows, inner)
        right = torch.randn(1, inner, columns, **options).expand(batch, inner, columns)
    else:
        accum = torch.randn(batch, rows, columns, **options)
        left = torch.randn(batch, rows, inner, **options)
        right = torch.randn(batch, inner, columns, **options)
    return accum, left, right


@pytest.mark.parametrize("inner", [1, 2, 15, 16, 17, 31, 32, 33, 64])
@pytest.mark.parametrize("layout", ["contiguous", "transposed", "expanded"])
def test_fused_accumulator_tail_tiles_strides_and_no_mutation(kernel_module, inner, layout):
    torch.manual_seed(924)
    operands = _operands(inner, layout)
    snapshots = tuple(value.clone() for value in operands)
    with torch.no_grad():
        expected = torch.baddbmm(*operands)
        actual = kernel_module.geometry_baddbmm(*operands)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-5)
    assert actual.is_contiguous()
    assert all(actual.data_ptr() != value.data_ptr() for value in operands)
    for value, snapshot in zip(operands, snapshots, strict=True):
        torch.testing.assert_close(value, snapshot, rtol=0, atol=0)


@pytest.mark.parametrize("inner,dtype", [(0, torch.float32), (65, torch.float32), (17, torch.float64)])
def test_unoptimized_operand_contracts_use_torch(kernel_module, monkeypatch, inner, dtype):
    class ForbiddenKernel:
        def __getitem__(self, grid):
            pytest.fail("Unsupported shape/dtype entered the Triton kernel")

    monkeypatch.setattr(kernel_module, "_geometry_baddbmm_kernel", ForbiddenKernel())
    operands = _operands(inner, dtype=dtype)
    with torch.no_grad():
        actual = kernel_module.geometry_baddbmm(*operands)
        expected = torch.baddbmm(*operands)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(0, 4, 5, 3), (2, 0, 5, 3), (2, 4, 0, 3)])
def test_empty_operand_shapes_keep_reference_semantics(kernel_module, shape):
    batch, rows, columns, inner = shape
    accum = torch.empty(batch, rows, columns, device="cuda")
    left = torch.empty(batch, rows, inner, device="cuda")
    right = torch.empty(batch, inner, columns, device="cuda")
    with torch.no_grad():
        actual = kernel_module.geometry_baddbmm(accum, left, right)
        expected = torch.baddbmm(accum, left, right)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_higher_derivative_calls_never_enter_raw_kernel(kernel_module, monkeypatch):
    class ForbiddenKernel:
        def __getitem__(self, grid):
            pytest.fail("Differentiable call entered the first-order Triton kernel")

    monkeypatch.setattr(kernel_module, "_geometry_baddbmm_kernel", ForbiddenKernel())
    inputs = tuple(value.requires_grad_() for value in _operands())
    actual = kernel_module.geometry_baddbmm(*inputs)
    expected = torch.baddbmm(*inputs)
    gradients = torch.autograd.grad(actual.sin().sum(), inputs, create_graph=True)
    reference = torch.autograd.grad(expected.sin().sum(), inputs, create_graph=True)
    directions = tuple(torch.randn_like(value) for value in inputs)

    def hvp(values):
        return torch.autograd.grad(
            sum((value * direction).sum() for value, direction in zip(values, directions, strict=True)), inputs
        )

    for value, target in zip((*gradients, *hvp(gradients)), (*reference, *hvp(reference)), strict=True):
        torch.testing.assert_close(value, target, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_direct_autocast_call_uses_reference_dtype_and_values(kernel_module, monkeypatch, dtype):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA BF16 autocast is unsupported")

    class ForbiddenKernel:
        def __getitem__(self, grid):
            pytest.fail("Autocast call entered the float32-only Triton kernel")

    monkeypatch.setattr(kernel_module, "_geometry_baddbmm_kernel", ForbiddenKernel())
    operands = _operands()
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        actual = kernel_module.geometry_baddbmm(*operands)
        expected = torch.baddbmm(*operands)
    assert actual.dtype == expected.dtype == dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_missing_optional_kernel_uses_reference_backward(kernel_module, monkeypatch):
    operands = _operands()
    monkeypatch.setitem(sys.modules, "aimnet.kernels.geometry_baddbmm_triton", None)
    with torch.no_grad():
        actual = aev_inference._geometry_baddbmm(*operands)
        expected = torch.baddbmm(*operands)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_graph_replay_consumes_changed_accumulator_and_matrix_inputs(kernel_module):
    operands = _operands()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.no_grad():
        for _ in range(3):
            kernel_module.geometry_baddbmm(*operands)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.no_grad(), torch.cuda.graph(graph, stream=stream):
        output = kernel_module.geometry_baddbmm(*operands)
    for index in range(3):
        with torch.no_grad():
            for operand in operands:
                operand.mul_(0.91).add_(0.03 * (index + 1))
            graph.replay()
            expected = torch.baddbmm(*operands)
        torch.testing.assert_close(output, expected, rtol=1e-5, atol=2e-5)


def test_retained_custom_backward_does_not_change_prior_results(kernel_module):
    torch.manual_seed(925)
    left = torch.randn(2, 3, 4, device="cuda", requires_grad=True)
    geometry = torch.randn(2, 4, 5, device="cuda", requires_grad=True)
    output, carry = aev_inference._BmmGeometryCarry.apply(left, geometry)
    loss = output.sin().sum() + carry.square().sum()
    expected = torch.autograd.grad(torch.bmm(left, geometry).sin().sum() + geometry.square().sum(), (left, geometry))
    previous = []
    for _ in range(3):
        gradients = torch.autograd.grad(loss, (left, geometry), retain_graph=True)
        for value, reference in zip(gradients, expected, strict=True):
            torch.testing.assert_close(value, reference, rtol=1e-5, atol=2e-5)
        for value, snapshot in previous:
            torch.testing.assert_close(value, snapshot, rtol=0, atol=0)
        previous = [(value, value.clone()) for value in gradients]
