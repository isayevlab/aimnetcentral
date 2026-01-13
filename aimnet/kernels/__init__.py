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
"""AIMNet Kernels Package - GPU-accelerated operations using NVIDIA Warp."""

import torch

# Track warp availability
_warp_available = False

try:
    from .conv_sv_2d_sp_wp import conv_sv_2d_sp
    _warp_available = True
except ImportError:
    conv_sv_2d_sp = None  # type: ignore


def load_ops():
    """
    Load and register all custom ops for warp kernels.
    
    This function ensures that all warp-based custom operations are properly
    registered with PyTorch's operator registry.
    
    Should be called before using any of the custom kernels to ensure
    proper registration with the PyTorch dispatcher.
    
    Returns:
        list: Available ops that were registered.
    """
    global _warp_available
    
    available_ops = []
    
    # Import warp kernels to trigger registration
    try:
        from . import conv_sv_2d_sp_wp
        _warp_available = True
    except ImportError as e:
        print(f"Failed to load warp kernels: {e}")
        _warp_available = False
        return available_ops
    
    # Verify ops are available
    if hasattr(torch.ops, 'aimnet'):
        if hasattr(torch.ops.aimnet, 'conv_sv_2d_sp_fwd'):
            available_ops.append('aimnet::conv_sv_2d_sp_fwd')
        if hasattr(torch.ops.aimnet, 'conv_sv_2d_sp_bwd'):
            available_ops.append('aimnet::conv_sv_2d_sp_bwd')
        if hasattr(torch.ops.aimnet, 'conv_sv_2d_sp_bwd_bwd'):
            available_ops.append('aimnet::conv_sv_2d_sp_bwd_bwd')
    
    return available_ops


def is_warp_available() -> bool:
    """Check if warp kernels are available."""
    return _warp_available


__all__ = [
    'conv_sv_2d_sp',
    'load_ops',
    'is_warp_available',
]
