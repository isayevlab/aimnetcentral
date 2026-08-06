"""The AEV kernel gate must consult warp's CUDA availability, not only torch's.

A CUDA-enabled pytorch alongside a CPU-only warp-lang build is a solver-reachable
environment on conda-forge; the gate must fall back to the einsum path instead of
letting conv_sv_2d_sp raise inside forward.
"""

import pytest
import torch

import aimnet.kernels
import aimnet.modules.aev as aev_mod
from aimnet import nbops
from aimnet.modules.aev import ConvSV


def test_warp_cuda_available_is_bool():
    assert isinstance(aimnet.kernels.WARP_CUDA_AVAILABLE, bool)


def _generate_valid_neighbor_idx(b: int, m: int, num_neighbors: int, device: str) -> torch.Tensor:
    """Neighbor indices with real entries first and the padding sentinel (b - 1) after.

    Mirrors `generate_valid_neighbor_idx` in tests/test_conv_sv_2d_sp.py: the last
    batch row is a padding atom, and the Warp kernel skips computing its output row
    entirely (it launches over `b - 1` rows), so any einsum reference must be fed
    matching padding to agree with the kernel.
    """
    padding_value = b - 1
    idx = torch.full((b, m), padding_value, device=device, dtype=torch.int64)
    n = min(num_neighbors, m)
    if n > 0 and b > 1:
        idx[:, :n] = torch.randint(0, padding_value, (b, n), device=device, dtype=torch.int64)
    return idx


def _conv_inputs(device, dtype=torch.float32):
    """Mode-1 (padded flat) ConvSV.forward inputs, mirroring
    TestConvSVDispatch.convsv_data in tests/test_conv_sv_2d_sp.py: nchannel/nshifts_s
    are the (B, nchannel, nshifts_s) feature axes conv_sv_2d_sp contracts over, and
    nshifts_v must equal nshifts_s here for the vector-combination step in
    ConvSV.forward to line up (d2features requires nshifts_s == nshifts_v; see
    aimnet/models/aimnet2.py).
    """
    torch.manual_seed(0)
    b, nchannel, nshifts_s, m, ncomb_v = 4, 3, 4, 6, 2
    nshifts_v = nshifts_s

    idx = _generate_valid_neighbor_idx(b, m, num_neighbors=m - 1, device=device)
    idx[-1] = b - 1  # last atom is the padding row; it has no real neighbors
    a = torch.randn(b, nchannel, nshifts_s, device=device, dtype=dtype)
    g_sv = torch.randn(b, m, nshifts_s, 4, device=device, dtype=dtype)
    g_sv = g_sv * (idx < b - 1).view(b, m, 1, 1)

    conv = ConvSV(nshifts_s=nshifts_s, nchannel=nchannel, d2features=True, nshifts_v=nshifts_v, ncomb_v=ncomb_v).to(
        device=device, dtype=dtype
    )
    data = nbops.set_nb_mode({"g_sv": g_sv, "nbmat": idx})
    return conv, data, a


@pytest.mark.gpu
def test_cuda_forward_without_warp_cuda_uses_einsum_fallback(monkeypatch):
    """With warp CUDA reported unavailable, a fp32 CUDA forward must not raise."""
    monkeypatch.setattr(aev_mod, "WARP_CUDA_AVAILABLE", False)
    conv, data, a = _conv_inputs("cuda")
    with pytest.warns(RuntimeWarning, match="warp-lang has no CUDA support"):
        out = conv(data, a)
    assert out.device.type == "cuda"


@pytest.mark.gpu
def test_cuda_fallback_matches_kernel_path(monkeypatch):
    """Fallback einsum result must match the Warp kernel result."""
    conv, data, a = _conv_inputs("cuda")
    if not aimnet.kernels.WARP_CUDA_AVAILABLE:
        pytest.skip("warp has no CUDA here; kernel path unavailable")
    ref = conv(data, a)
    monkeypatch.setattr(aev_mod, "WARP_CUDA_AVAILABLE", False)
    with pytest.warns(RuntimeWarning):
        alt = conv(data, a)
    torch.testing.assert_close(ref, alt, rtol=1e-4, atol=1e-5)
