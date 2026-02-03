# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
"""
Tests for conv_sv_2d_sp Warp kernels.

These tests directly call the warp kernel functions (not through ConvSV dispatch)
to enable testing on both CPU and CUDA devices.
"""

import pytest
import torch

# All tests in this module require GPU
pytestmark = pytest.mark.gpu

# Skip all tests if warp is not available
try:
    from aimnet.kernels.conv_sv_2d_sp_wp import conv_sv_2d_sp

    WARP_AVAILABLE = True
except ImportError:
    WARP_AVAILABLE = False
    conv_sv_2d_sp = None


def generate_valid_neighbor_idx(B: int, M: int, num_neighbors: int, device: str) -> torch.Tensor:
    """Generate properly structured neighbor indices with padding.

    The Warp kernel expects neighbor indices where:
    - Real neighbor indices (valid atom indices in [0, B-1)) come first
    - Padding indices (value = B-1, the last row) come at the end
    - No real indices appear after the first padding index

    Args:
        B: Number of atoms including padding row (valid indices are in [0, B-1))
        M: Maximum number of neighbors
        num_neighbors: Number of real neighbors per atom
        device: Device to create tensor on

    Returns:
        Index tensor of shape (B, M) with proper padding structure
    """
    padding_value = B - 1  # last row is padding
    # Start with all padding
    idx = torch.full((B, M), padding_value, device=device, dtype=torch.int64)
    # Fill in real neighbor indices at the beginning (valid range is [0, B-2])
    n = min(num_neighbors, M)
    if n > 0 and B > 1:
        idx[:, :n] = torch.randint(0, padding_value, (B, n), device=device, dtype=torch.int64)
    return idx


def reference_conv_sv_2d_sp_einsum(a, idx, g):
    """Reference implementation using PyTorch einsum.

    This implementation handles padding by masking out padded entries.
    Padding is indicated by idx >= B-1 (where B-1 is the padding row).

    Args:
        a: Input tensor of shape (B, A, G)
        idx: Index tensor of shape (B, M) with padding value B-1
        g: Gate tensor of shape (B, M, G, 4)

    Returns:
        Output tensor of shape (B, A, G, 4)
    """
    B, M = idx.shape
    padding_value = B - 1  # last row is padding
    # Create mask for valid (non-padding) entries: True where idx < padding_value
    valid_mask = (idx < padding_value).unsqueeze(-1).unsqueeze(-1)  # (B, M, 1, 1)
    # Zero out g for padded entries
    g_masked = g * valid_mask
    # Clamp idx to valid range for indexing (padding entries will be zeroed anyway)
    idx_clamped = idx.clamp(0, a.shape[0] - 1)
    # Select features based on neighbor indices
    a_selected = a.index_select(0, idx_clamped.flatten()).unflatten(0, (B, M))  # (B, M, A, G)
    # Einsum contraction
    output = torch.einsum("bmag,bmgd->bagd", a_selected, g_masked)
    # Zero out padding row (kernel doesn't process it)
    output[-1] = 0
    return output


@pytest.mark.skipif(not WARP_AVAILABLE, reason="Warp not available")
class TestConvSV2dSP:
    """Test class for conv_sv_2d_sp operations."""

    @pytest.fixture
    def test_data_cuda(self):
        """Standard test data on CUDA: B=8, A=16, G=12, M=10, D=4
        
        Note: The padding sentinel is B-1 (last row). Valid neighbor indices are in [0, B-2].
        The kernel uses `if idx >= padding_value: break` for padding detection.
        """
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        device = "cuda"
        B, A, G, M = 8, 16, 12, 10  # M >= B to ensure padding sentinel works

        a = torch.randn(B, A, G, device=device, dtype=torch.float32, requires_grad=True)
        # Use properly structured neighbor indices (all positions filled, no padding)
        idx = generate_valid_neighbor_idx(B, M, num_neighbors=M, device=device)
        g = torch.randn(B, M, G, 4, device=device, dtype=torch.float32, requires_grad=True)

        return a, idx, g

    @pytest.fixture
    def test_data_small_cuda(self):
        """Smaller test data for gradient tests on CUDA.
        
        Note: M must be >= B so that the padding sentinel (M) is never a valid batch index.
        """
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        device = "cuda"
        B, A, G, M = 2, 4, 3, 4  # M >= B to ensure padding sentinel works

        a = torch.randn(B, A, G, device=device, dtype=torch.float32, requires_grad=True)
        # Use properly structured neighbor indices (all positions filled, no padding)
        idx = generate_valid_neighbor_idx(B, M, num_neighbors=M, device=device)
        g = torch.randn(B, M, G, 4, device=device, dtype=torch.float32, requires_grad=True)

        return a, idx, g

    def test_forward_accuracy(self, test_data_cuda):
        """Test forward pass accuracy against einsum reference."""
        a, idx, g = test_data_cuda

        # Compute with Warp kernel
        output_warp = conv_sv_2d_sp(a.detach(), idx, g.detach())

        # Compute with einsum reference
        output_ref = reference_conv_sv_2d_sp_einsum(a.detach(), idx, g.detach())

        # Check accuracy
        assert torch.allclose(output_warp, output_ref, atol=1e-5, rtol=1e-4), (
            f"Forward pass accuracy failed. Max diff: {torch.max(torch.abs(output_warp - output_ref))}"
        )

    def test_backward_accuracy(self, test_data_cuda):
        """Test backward pass accuracy against PyTorch autograd."""
        a, idx, g = test_data_cuda

        # Reference computation using einsum with autograd
        a_ref = a.clone().detach().requires_grad_(True)
        g_ref = g.clone().detach().requires_grad_(True)
        output_ref = reference_conv_sv_2d_sp_einsum(a_ref, idx, g_ref)
        loss_ref = output_ref.sum()
        loss_ref.backward()

        # Warp computation
        a_warp = a.clone().detach().requires_grad_(True)
        g_warp = g.clone().detach().requires_grad_(True)
        output_warp = conv_sv_2d_sp(a_warp, idx, g_warp)
        loss_warp = output_warp.sum()
        loss_warp.backward()

        # Check forward accuracy
        assert torch.allclose(output_warp, output_ref, atol=1e-5, rtol=1e-4), (
            f"Forward pass accuracy failed. Max diff: {torch.max(torch.abs(output_warp - output_ref))}"
        )

        # Check gradient accuracy
        assert torch.allclose(a_warp.grad, a_ref.grad, atol=1e-4, rtol=1e-3), (
            f"Gradient w.r.t. 'a' accuracy failed. Max diff: {torch.max(torch.abs(a_warp.grad - a_ref.grad))}"
        )

        assert torch.allclose(g_warp.grad, g_ref.grad, atol=1e-4, rtol=1e-3), (
            f"Gradient w.r.t. 'g' accuracy failed. Max diff: {torch.max(torch.abs(g_warp.grad - g_ref.grad))}"
        )

    def test_double_backward_accuracy(self, test_data_small_cuda):
        """Test double backward pass accuracy."""
        a, idx, g = test_data_small_cuda

        # Reference gradients using PyTorch autograd
        a_ref = a.clone().detach().requires_grad_(True)
        g_ref = g.clone().detach().requires_grad_(True)
        output_ref = reference_conv_sv_2d_sp_einsum(a_ref, idx, g_ref)
        loss_ref = output_ref.sum()
        loss_ref.backward()

        # Warp gradients
        a_warp = a.clone().detach().requires_grad_(True)
        g_warp = g.clone().detach().requires_grad_(True)
        output_warp = conv_sv_2d_sp(a_warp, idx, g_warp)
        loss_warp = output_warp.sum()
        loss_warp.backward()

        # Check first-order gradient accuracy
        assert torch.allclose(a_warp.grad, a_ref.grad, atol=1e-4, rtol=1e-3), (
            f"First-order gradient w.r.t. 'a' failed. Max diff: {torch.max(torch.abs(a_warp.grad - a_ref.grad))}"
        )

        assert torch.allclose(g_warp.grad, g_ref.grad, atol=1e-4, rtol=1e-3), (
            f"First-order gradient w.r.t. 'g' failed. Max diff: {torch.max(torch.abs(g_warp.grad - g_ref.grad))}"
        )

        # Test second-order gradients (mixed partials)
        a_ref2 = a.clone().detach().requires_grad_(True)
        g_ref2 = g.clone().detach().requires_grad_(True)

        output_ref2 = reference_conv_sv_2d_sp_einsum(a_ref2, idx, g_ref2)
        grad_a_ref, grad_g_ref = torch.autograd.grad(
            output_ref2.sum(), [a_ref2, g_ref2], create_graph=True, retain_graph=True
        )

        # Second derivatives w.r.t. mixed partials
        grad2_a_g_ref = torch.autograd.grad(grad_a_ref.sum(), g_ref2, retain_graph=True)[0]
        grad2_g_a_ref = torch.autograd.grad(grad_g_ref.sum(), a_ref2, retain_graph=True)[0]

        # Warp second derivatives
        a_warp2 = a.clone().detach().requires_grad_(True)
        g_warp2 = g.clone().detach().requires_grad_(True)

        output_warp2 = conv_sv_2d_sp(a_warp2, idx, g_warp2)
        grad_a_warp, grad_g_warp = torch.autograd.grad(
            output_warp2.sum(), [a_warp2, g_warp2], create_graph=True, retain_graph=True
        )

        grad2_a_g_warp = torch.autograd.grad(grad_a_warp.sum(), g_warp2, retain_graph=True)[0]
        grad2_g_a_warp = torch.autograd.grad(grad_g_warp.sum(), a_warp2, retain_graph=True)[0]

        # Check second-order accuracy
        assert torch.allclose(grad2_a_g_warp, grad2_a_g_ref, atol=1e-4, rtol=1e-3), (
            f"Second-order gradient d²/dadg failed. Max diff: {torch.max(torch.abs(grad2_a_g_warp - grad2_a_g_ref))}"
        )

        assert torch.allclose(grad2_g_a_warp, grad2_g_a_ref, atol=1e-4, rtol=1e-3), (
            f"Second-order gradient d²/dgda failed. Max diff: {torch.max(torch.abs(grad2_g_a_warp - grad2_g_a_ref))}"
        )

    def test_different_shapes(self, test_data_cuda):
        """Test with various tensor shapes.
        
        Note: M must be >= B so that the padding sentinel (M) is never a valid batch index.
        """
        # Test with different batch sizes, ensuring M >= B
        for B in [1, 4, 8]:
            for M in [B, B + 2, B + 4]:  # M >= B ensures padding sentinel works
                A, G = 8, 6
                device = "cuda"

                a = torch.randn(B, A, G, device=device, dtype=torch.float32)
                # Use properly structured neighbor indices (all positions filled)
                idx = generate_valid_neighbor_idx(B, M, num_neighbors=M, device=device)
                g = torch.randn(B, M, G, 4, device=device, dtype=torch.float32)

                output_warp = conv_sv_2d_sp(a, idx, g)
                output_ref = reference_conv_sv_2d_sp_einsum(a, idx, g)

                assert torch.allclose(output_warp, output_ref, atol=1e-5, rtol=1e-4), (
                    f"Shape test failed for B={B}, M={M}. Max diff: {torch.max(torch.abs(output_warp - output_ref))}"
                )

    def test_op_registration(self, test_data_cuda):
        """Test that PyTorch ops are properly registered."""
        a, idx, g = test_data_cuda

        # Check that ops are registered under aimnet namespace
        assert hasattr(torch.ops, "aimnet"), "aimnet namespace not found in torch.ops"
        assert hasattr(torch.ops.aimnet, "conv_sv_2d_sp_fwd"), "conv_sv_2d_sp_fwd op not registered"

        # Test using ops directly
        output1 = torch.ops.aimnet.conv_sv_2d_sp_fwd(a.detach(), idx, g.detach())
        output2 = reference_conv_sv_2d_sp_einsum(a.detach(), idx, g.detach())

        assert torch.allclose(output1, output2, atol=1e-5, rtol=1e-4), (
            f"Op registration test failed. Max diff: {torch.max(torch.abs(output1 - output2))}"
        )

    def test_padding_behavior(self):
        """Test that padding (idx == M) is handled correctly.
        
        Note: M must be >= B so that the padding sentinel (M) is never a valid batch index.
        """
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        device = "cuda"
        B, A, G, M = 4, 8, 6, 8  # M >= B ensures padding sentinel works

        a = torch.randn(B, A, G, device=device, dtype=torch.float32)
        g = torch.randn(B, M, G, 4, device=device, dtype=torch.float32)

        # Test with various amounts of padding
        for num_neighbors in [1, 3, 5, M]:
            idx = generate_valid_neighbor_idx(B, M, num_neighbors=num_neighbors, device=device)

            output_warp = conv_sv_2d_sp(a, idx, g)
            output_ref = reference_conv_sv_2d_sp_einsum(a, idx, g)

            assert torch.allclose(output_warp, output_ref, atol=1e-5, rtol=1e-4), (
                f"Padding test failed for num_neighbors={num_neighbors}. "
                f"Max diff: {torch.max(torch.abs(output_warp - output_ref))}"
            )
