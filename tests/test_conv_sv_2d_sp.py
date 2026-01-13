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

# Skip all tests if warp is not available
try:
    from aimnet.kernels.conv_sv_2d_sp_wp import conv_sv_2d_sp
    WARP_AVAILABLE = True
except ImportError:
    WARP_AVAILABLE = False
    conv_sv_2d_sp = None


def reference_conv_sv_2d_sp_einsum(a, idx, g):
    """Reference implementation using PyTorch einsum.
    
    Args:
        a: Input tensor of shape (B, A, G)
        idx: Index tensor of shape (B, M)
        g: Gate tensor of shape (B, M, G, 4)
    
    Returns:
        Output tensor of shape (B, A, G, 4)
    """
    B, M = idx.shape
    # Select features based on neighbor indices
    a_selected = a.index_select(0, idx.flatten()).unflatten(0, (B, M))  # (B, M, A, G)
    # Einsum contraction
    output = torch.einsum('bmag,bmgd->bagd', a_selected, g)
    return output


@pytest.mark.skipif(not WARP_AVAILABLE, reason="Warp not available")
class TestConvSV2dSP:
    """Test class for conv_sv_2d_sp operations."""
    
    @pytest.fixture
    def test_data_cuda(self):
        """Standard test data on CUDA: B=8, A=16, G=12, M=6, D=4"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        device = 'cuda'
        B, A, G, M = 8, 16, 12, 6
        
        a = torch.randn(B, A, G, device=device, dtype=torch.float32, requires_grad=True)
        idx = torch.randint(0, B, (B, M), device=device, dtype=torch.int64)
        g = torch.randn(B, M, G, 4, device=device, dtype=torch.float32, requires_grad=True)
        
        return a, idx, g
    
    @pytest.fixture
    def test_data_small_cuda(self):
        """Smaller test data for gradient tests on CUDA."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        device = 'cuda'
        B, A, G, M = 2, 4, 3, 3
        
        a = torch.randn(B, A, G, device=device, dtype=torch.float32, requires_grad=True)
        idx = torch.randint(0, B, (B, M), device=device, dtype=torch.int64)
        g = torch.randn(B, M, G, 4, device=device, dtype=torch.float32, requires_grad=True)
        
        return a, idx, g
    
    @pytest.mark.gpu
    def test_forward_accuracy(self, test_data_cuda):
        """Test forward pass accuracy against einsum reference."""
        a, idx, g = test_data_cuda
        
        # Compute with Warp kernel
        output_warp = conv_sv_2d_sp(a.detach(), idx, g.detach())
        
        # Compute with einsum reference
        output_ref = reference_conv_sv_2d_sp_einsum(a.detach(), idx, g.detach())
        
        # Check accuracy
        assert torch.allclose(output_warp, output_ref, atol=1e-5, rtol=1e-4), \
            f"Forward pass accuracy failed. Max diff: {torch.max(torch.abs(output_warp - output_ref))}"
    
    @pytest.mark.gpu
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
        assert torch.allclose(output_warp, output_ref, atol=1e-5, rtol=1e-4), \
            f"Forward pass accuracy failed. Max diff: {torch.max(torch.abs(output_warp - output_ref))}"
        
        # Check gradient accuracy
        assert torch.allclose(a_warp.grad, a_ref.grad, atol=1e-4, rtol=1e-3), \
            f"Gradient w.r.t. 'a' accuracy failed. Max diff: {torch.max(torch.abs(a_warp.grad - a_ref.grad))}"
        
        assert torch.allclose(g_warp.grad, g_ref.grad, atol=1e-4, rtol=1e-3), \
            f"Gradient w.r.t. 'g' accuracy failed. Max diff: {torch.max(torch.abs(g_warp.grad - g_ref.grad))}"
    
    @pytest.mark.gpu
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
        assert torch.allclose(a_warp.grad, a_ref.grad, atol=1e-4, rtol=1e-3), \
            f"First-order gradient w.r.t. 'a' failed. Max diff: {torch.max(torch.abs(a_warp.grad - a_ref.grad))}"
        
        assert torch.allclose(g_warp.grad, g_ref.grad, atol=1e-4, rtol=1e-3), \
            f"First-order gradient w.r.t. 'g' failed. Max diff: {torch.max(torch.abs(g_warp.grad - g_ref.grad))}"
        
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
        assert torch.allclose(grad2_a_g_warp, grad2_a_g_ref, atol=1e-4, rtol=1e-3), \
            f"Second-order gradient d²/dadg failed. Max diff: {torch.max(torch.abs(grad2_a_g_warp - grad2_a_g_ref))}"
        
        assert torch.allclose(grad2_g_a_warp, grad2_g_a_ref, atol=1e-4, rtol=1e-3), \
            f"Second-order gradient d²/dgda failed. Max diff: {torch.max(torch.abs(grad2_g_a_warp - grad2_g_a_ref))}"
    
    @pytest.mark.gpu
    def test_different_shapes(self, test_data_cuda):
        """Test with various tensor shapes."""
        # Test with different batch sizes
        for B in [1, 4, 16]:
            for M in [2, 8, 16]:
                A, G = 8, 6
                device = 'cuda'
                
                a = torch.randn(B, A, G, device=device, dtype=torch.float32)
                idx = torch.randint(0, B, (B, M), device=device, dtype=torch.int64)
                g = torch.randn(B, M, G, 4, device=device, dtype=torch.float32)
                
                output_warp = conv_sv_2d_sp(a, idx, g)
                output_ref = reference_conv_sv_2d_sp_einsum(a, idx, g)
                
                assert torch.allclose(output_warp, output_ref, atol=1e-5, rtol=1e-4), \
                    f"Shape test failed for B={B}, M={M}. Max diff: {torch.max(torch.abs(output_warp - output_ref))}"
    
    @pytest.mark.gpu
    def test_op_registration(self, test_data_cuda):
        """Test that PyTorch ops are properly registered."""
        a, idx, g = test_data_cuda
        
        # Check that ops are registered under aimnet namespace
        assert hasattr(torch.ops, 'aimnet'), "aimnet namespace not found in torch.ops"
        assert hasattr(torch.ops.aimnet, 'conv_sv_2d_sp_fwd'), "conv_sv_2d_sp_fwd op not registered"
        
        # Test using ops directly
        output1 = torch.ops.aimnet.conv_sv_2d_sp_fwd(a.detach(), idx, g.detach())
        output2 = reference_conv_sv_2d_sp_einsum(a.detach(), idx, g.detach())
        
        assert torch.allclose(output1, output2, atol=1e-5, rtol=1e-4), \
            f"Op registration test failed. Max diff: {torch.max(torch.abs(output1 - output2))}"
