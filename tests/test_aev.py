"""Mode-2 ConvSV dispatch tests."""

import pytest
import torch

from aimnet import nbops
from aimnet.modules.aev import ConvSV


def _mode2_aev_data(device: str = "cpu"):
    B, N, M, C, G = 2, 4, 3, 2, 3
    nbmat = torch.tensor(
        [
            [[1, 2, 8], [0, 2, 8], [0, 1, 8], [8, 8, 8]],
            [[5, 6, 8], [4, 6, 8], [8, 8, 8], [8, 8, 8]],
        ],
        device=device,
        dtype=torch.int64,
    )
    a = torch.arange(B * N * C, device=device, dtype=torch.float64).reshape(B, N, C).requires_grad_()
    g_sv = torch.randn(B, N, M, G, 4, device=device, dtype=torch.float64)
    data = nbops.calc_masks(
        nbops.set_nb_mode({
            "nbmat": nbmat,
            "numbers": torch.tensor([[6, 1, 1, 0], [8, 1, 0, 0]], device=device),
            "g_sv": g_sv,
        })
    )
    return data, a


def _mode1_aev_data(*, a: torch.Tensor | None = None, g_sv: torch.Tensor | None = None):
    n_atoms, n_channels, n_basis, n_neighbors = 4, 2, 3, 3
    nbmat = torch.tensor([[1, 2, 3], [0, 2, 3], [0, 1, 3], [3, 3, 3]], dtype=torch.int32)
    if a is None:
        a = torch.randn(n_atoms, n_channels, dtype=torch.float64, requires_grad=True)
    if g_sv is None:
        g_sv = torch.randn(n_atoms, n_neighbors, n_basis, 4, dtype=torch.float64)
        g_sv = (g_sv * (nbmat < n_atoms - 1).view(n_atoms, n_neighbors, 1, 1)).requires_grad_()
    data = nbops.set_nb_mode({
        "g_sv": g_sv,
        "mol_idx": torch.tensor([0, 0, 0, 1]),
        "nbmat": nbmat,
    })
    return data, a, g_sv


def test_mode1_convsv_keeps_dummy_output_zero_and_backpropagates():
    data, a, g_sv = _mode1_aev_data()
    out = ConvSV(nshifts_s=3, nchannel=2, d2features=False).double()(data, a)

    assert torch.equal(out[-1], torch.zeros_like(out[-1]))
    assert torch.count_nonzero(out[:-1]) > 0
    grad_a, grad_g = torch.autograd.grad(out.square().sum(), (a, g_sv))
    assert torch.isfinite(grad_a).all()
    assert torch.isfinite(grad_g).all()


def test_mode1_convsv_does_not_remask_fresh_output(monkeypatch):
    data, a, _g_sv = _mode1_aev_data()
    calls = []
    original_mask_i = nbops.mask_i_

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return original_mask_i(*args, **kwargs)

    monkeypatch.setattr(nbops, "mask_i_", spy)

    ConvSV(nshifts_s=3, nchannel=2, d2features=False).double()(data, a)

    assert calls == []


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile requires PyTorch 2.0+")
def test_mode1_convsv_compile_matches_eager_forward_and_backward():
    conv = ConvSV(nshifts_s=3, nchannel=2, d2features=False).double()
    _data, a, g_sv = _mode1_aev_data()
    a_eager = a.detach().clone().requires_grad_()
    g_eager = g_sv.detach().clone().requires_grad_()
    eager_out = conv(_mode1_aev_data(a=a_eager, g_sv=g_eager)[0], a_eager)
    eager_loss = eager_out.square().sum()
    eager_grads = torch.autograd.grad(eager_loss, (a_eager, g_eager))

    a_compiled = a.detach().clone().requires_grad_()
    g_compiled = g_sv.detach().clone().requires_grad_()
    compiled = torch.compile(conv, backend="aot_eager")
    compiled_out = compiled(_mode1_aev_data(a=a_compiled, g_sv=g_compiled)[0], a_compiled)
    compiled_loss = compiled_out.square().sum()
    compiled_grads = torch.autograd.grad(compiled_loss, (a_compiled, g_compiled))

    torch.testing.assert_close(compiled_out, eager_out)
    torch.testing.assert_close(compiled_grads[0], eager_grads[0])
    torch.testing.assert_close(compiled_grads[1], eager_grads[1])


def test_global_mode2_convsv_d2false_forward():
    data, a = _mode2_aev_data()
    out = ConvSV(nshifts_s=3, nchannel=2, d2features=False).double()(data, a)
    assert out.shape[:2] == (2, 4)


def test_global_mode2_convsv_d2false_first_gradient():
    data, a = _mode2_aev_data()
    out = ConvSV(nshifts_s=3, nchannel=2, d2features=False).double()(data, a)
    assert torch.autograd.grad(out.sum(), a)[0].shape == a.shape


def test_global_mode2_convsv_d2false_hessian_vmap():
    data, a = _mode2_aev_data()
    conv = ConvSV(nshifts_s=3, nchannel=2, d2features=False).double()
    grad = torch.autograd.grad(conv(data, a).sum(), a, create_graph=True)[0]
    assert torch.autograd.grad(grad.sum(), a)[0].shape == a.shape


def test_global_mode2_convsv_d2false_padded_center():
    data, a = _mode2_aev_data()
    out = ConvSV(nshifts_s=3, nchannel=2, d2features=False).double()(data, a)
    assert torch.equal(out[0, 3], torch.zeros_like(out[0, 3]))
    assert torch.equal(out[1, 2], torch.zeros_like(out[1, 2]))


def test_global_mode2_convsv_d2false_cross_batch_gather():
    data, a = _mode2_aev_data()
    data["g_sv"].zero_()
    data["g_sv"][1, 0, 0, 0, 0] = 1
    out = ConvSV(nshifts_s=3, nchannel=2, d2features=False).double()(data, a)
    assert torch.equal(out[1, 0, 0], a[1, 1, 0])
    assert torch.equal(out[1, 0, 1:3], torch.zeros(2, dtype=out.dtype))


def test_global_mode2_convsv_rejects_noncontiguous_input():
    data, a = _mode2_aev_data()
    a = a.transpose(1, 2)
    with pytest.raises(ValueError, match="flatten"):
        ConvSV(nshifts_s=3, nchannel=2, d2features=False).double()(data, a)


@pytest.mark.gpu
def test_global_mode2_convsv_cuda_float64_fallback():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    data, a = _mode2_aev_data("cuda")
    a = a.unsqueeze(-1).expand(-1, -1, -1, 3).contiguous().requires_grad_()
    out = ConvSV(nshifts_s=3, nchannel=2, d2features=True).cuda().double()(data, a.cuda())
    assert out.device.type == "cuda"
