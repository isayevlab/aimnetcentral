"""Shared AEV/ConvSV kernels: graph support is not required for dispatch."""

import pytest
import torch

from aimnet import nbops
from aimnet.modules.aev import AEVSV, ConvSV
from aimnet.modules.aev_inference import _FirstOrderOnly


def _data(coord, *, padded=False):
    numbers = torch.ones(coord.shape[:2], dtype=torch.long, device=coord.device)
    if padded:
        numbers[0, -2:] = 0
        numbers[-1, -1:] = 0
    return nbops.calc_masks(nbops.set_nb_mode({"numbers": numbers, "coord": coord}))


def _optimized(aev, data):
    data["_inference_kernels"] = torch.tensor(True)
    data["_aev_inference_constants"] = aev.prepare_inference()
    return aev(data)


@pytest.mark.parametrize("d2features", [False, True])
@pytest.mark.parametrize("padded", [False, True])
def test_cpu_layout_forward_first_and_second_derivatives(d2features, padded):
    torch.manual_seed(42)
    aev = AEVSV(nshifts_s=4).double().eval()
    conv = ConvSV(nshifts_s=4, nchannel=3, d2features=d2features).double().eval()
    coord = torch.randn(2, 7, 3, dtype=torch.float64, requires_grad=True)
    shape = (2, 7, 3, 4) if d2features else (2, 7, 3)
    features = torch.randn(shape, dtype=torch.float64, requires_grad=True)
    regular = conv(aev(_data(coord, padded=padded)), features)
    optimized_data = _optimized(aev, _data(coord, padded=padded))
    optimized = conv(optimized_data, features)
    assert optimized_data["g_sv"]._base is not None
    torch.testing.assert_close(optimized, regular, rtol=1e-12, atol=1e-12)
    inputs = (coord, features, conv.agh)
    regular_grads = torch.autograd.grad(regular.square().sum(), inputs, create_graph=True)
    optimized_grads = torch.autograd.grad(optimized.square().sum(), inputs, create_graph=True)
    for reference, actual in zip(regular_grads, optimized_grads, strict=True):
        torch.testing.assert_close(actual, reference, rtol=1e-11, atol=1e-10)
    direction = torch.randn_like(coord)
    h_ref = torch.autograd.grad((regular_grads[0] * direction).sum(), coord)[0]
    h_opt = torch.autograd.grad((optimized_grads[0] * direction).sum(), coord)[0]
    torch.testing.assert_close(h_opt, h_ref, rtol=1e-10, atol=1e-9)
    if padded:
        assert torch.count_nonzero(optimized[0, -2:]) == 0
        assert torch.count_nonzero(optimized_grads[0][0, -2:]) == 0


def test_training_ignores_inference_dispatch_and_clears_stale_layout():
    aev = AEVSV(nshifts_s=4).eval()
    data = _optimized(aev, _data(torch.randn(2, 7, 3)))
    assert "g_L" in data
    aev.train()
    assert "g_L" not in aev(data)
    aev.eval()
    data = aev(data)
    assert "g_L" in data
    data["_inference_kernels"] = torch.tensor(False)
    assert "g_L" not in aev(data)


@pytest.mark.parametrize("name", ["rc_s", "eta_s", "shifts_s"])
def test_trainable_basis_constants_keep_canonical_autograd(name):
    aev = AEVSV(nshifts_s=4).double().eval()
    getattr(aev, name).requires_grad_(True)
    data = _optimized(aev, _data(torch.randn(2, 7, 3, dtype=torch.float64)))
    assert "g_L" not in data
    grad = torch.autograd.grad(data["g_sv"].sum(), getattr(aev, name))[0]
    assert torch.isfinite(grad).all()


def test_missing_inference_constants_is_explicit_error():
    aev = AEVSV(nshifts_s=4).eval()
    data = _data(torch.randn(2, 7, 3))
    data["_inference_kernels"] = torch.tensor(True)
    with pytest.raises(ValueError, match="prepare_inference"):
        aev(data)


def test_prepared_constants_are_fresh_snapshots():
    aev = AEVSV(nshifts_s=4).eval()
    before = aev.prepare_inference()
    with torch.no_grad():
        aev.rc_s.mul_(1.25)
    after = aev.prepare_inference()
    assert before.device.type == "cpu"
    assert before.shape == (3,)
    assert after[1] != before[1]
    assert after[2] != before[2]
    assert before[1].item() == 5.0


def test_first_order_kernel_guard_rejects_create_graph():
    x = torch.randn(4, requires_grad=True)
    out = _FirstOrderOnly.apply(x.square())
    with pytest.raises(RuntimeError, match="first-order inference only"):
        torch.autograd.grad(out.sum(), x, create_graph=True)


def test_cpu_dispatch_does_not_load_triton(monkeypatch):
    import aimnet.modules.aev_inference as helper

    def forbidden():
        raise AssertionError("CPU dispatched the optional Triton backend")

    monkeypatch.setattr(helper, "_triton_kernel", forbidden)
    _optimized(AEVSV(nshifts_s=4).eval(), _data(torch.randn(2, 7, 3)))


def test_replaced_geometry_from_forward_hook_is_not_ignored():
    aev = AEVSV(nshifts_s=4).double().eval()
    conv = ConvSV(nshifts_s=4, nchannel=3, d2features=False).double().eval()
    coord = torch.randn(2, 7, 3, dtype=torch.float64, requires_grad=True)
    features = torch.randn(2, 7, 3, dtype=torch.float64, requires_grad=True)

    def scale_geometry(module, args, output):
        output["g_sv"] = 0.37 * output["g_sv"]
        return output

    handle = aev.register_forward_hook(scale_geometry)
    try:
        reference = conv(aev(_data(coord)), features)
        actual = conv(_optimized(aev, _data(coord)), features)
        torch.testing.assert_close(actual, reference, rtol=1e-12, atol=1e-12)
        grad_ref = torch.autograd.grad(reference.square().sum(), (coord, features))
        grad = torch.autograd.grad(actual.square().sum(), (coord, features))
        for result, expected in zip(grad, grad_ref, strict=True):
            torch.testing.assert_close(result, expected, rtol=1e-11, atol=1e-10)
    finally:
        handle.remove()


@pytest.mark.parametrize("override", ["instance", "class", "subclass"])
def test_custom_aev_equations_keep_reference_dispatch(monkeypatch, override):
    original = AEVSV._calc_aev

    def changed(self, r_ij, d_ij, data):
        return 0.37 * original(self, r_ij, d_ij, data)

    if override == "subclass":

        class CustomAEV(AEVSV):
            _calc_aev = changed

        aev = CustomAEV(nshifts_s=4).eval()
    else:
        aev = AEVSV(nshifts_s=4).eval()
        if override == "instance":
            import types

            monkeypatch.setattr(aev, "_calc_aev", types.MethodType(changed, aev))
        else:
            monkeypatch.setattr(AEVSV, "_calc_aev", changed)
    coord = torch.randn(2, 7, 3)
    reference = aev(_data(coord))["g_sv"]
    optimized = _optimized(aev, _data(coord))
    assert "g_L" not in optimized
    torch.testing.assert_close(optimized["g_sv"], reference, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("activation", [torch.nn.GELU, torch.nn.SiLU, torch.nn.Tanh])
@pytest.mark.parametrize("charge_channels", [1, 2])
def test_canonical_and_prepared_model_share_equations_and_derivatives(activation, charge_channels):
    """The prepared entry is the same computation for varied MLPs and charge heads."""
    from aimnet.models.aimnet2 import AIMNet2
    from aimnet.modules.core import AtomicSum, Output

    torch.manual_seed(17)
    model = (
        AIMNet2(
            aev={"nshifts_s": 4},
            nfeature=3,
            d2features=True,
            ncomb_v=4,
            hidden=([8], [8], [8]),
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
    for mlp in model.mlps:
        for name, module in mlp.named_children():
            if isinstance(module, torch.nn.GELU):
                setattr(mlp, name, activation())
    # The native model's public defaults coerce inputs to FP32. This scientific
    # comparison explicitly sets matching FP64 preparation metadata as well as
    # FP64 weights, so it tests the shared entry at higher numerical precision.
    model._required_keys_dtype = [
        torch.float64 if dtype == torch.float32 else dtype for dtype in model._required_keys_dtype
    ]
    model._optional_keys_dtype = [
        torch.float64 if dtype == torch.float32 else dtype for dtype in model._optional_keys_dtype
    ]
    coord = torch.randn(2, 7, 3, dtype=torch.float64, requires_grad=True)
    base = {
        "coord": coord,
        "numbers": torch.tensor([[6, 8, 1, 1, 1, 0, 0], [6, 7, 1, 1, 1, 1, 0]]),
        "charge": torch.tensor([0.0, 1.0], dtype=torch.float64),
        "mult": torch.tensor([1.0, 2.0], dtype=torch.float64),
    }
    regular = model(dict(base))
    prepared = model.forward_prepared(model.prepare_input(dict(base)))
    optimized_input = model.prepare_input(dict(base))
    optimized_input["_inference_kernels"] = torch.tensor(True)
    optimized_input["_aev_inference_constants"] = model.aev.prepare_inference()
    optimized = model.forward_prepared(optimized_input)
    for key in ("energy", "charges", "aim"):
        assert torch.equal(regular[key], prepared[key]), key
        torch.testing.assert_close(optimized[key], regular[key], rtol=1e-10, atol=1e-10)
    direction = torch.randn_like(coord)
    reference_force = -torch.autograd.grad(regular["energy"].sum(), coord, create_graph=True)[0]
    reference_hvp = torch.autograd.grad((reference_force * direction).sum(), coord)[0]
    for result in (prepared, optimized):
        force = -torch.autograd.grad(result["energy"].sum(), coord, create_graph=True)[0]
        hvp = torch.autograd.grad((force * direction).sum(), coord)[0]
        torch.testing.assert_close(force, reference_force, rtol=1e-9, atol=1e-10)
        torch.testing.assert_close(hvp, reference_hvp, rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize("d2features", [False, True])
@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_shared_inference_output_and_coordinate_gradient(d2features, padded):
    torch.manual_seed(42)
    aev = AEVSV(nshifts_s=4).cuda().eval()
    conv = ConvSV(nshifts_s=4, nchannel=3, d2features=d2features).cuda().eval()
    coord = torch.randn(2, 20, 3, device="cuda", requires_grad=True)
    shape = (2, 20, 3, 4) if d2features else (2, 20, 3)
    features = torch.randn(shape, device="cuda", requires_grad=True)
    regular = conv(aev(_data(coord, padded=padded)), features)
    optimized = conv(_optimized(aev, _data(coord, padded=padded)), features)
    torch.testing.assert_close(optimized, regular, rtol=2e-5, atol=2e-5)
    regular_grads = torch.autograd.grad(regular.square().sum(), (coord, features, conv.agh))
    optimized_grads = torch.autograd.grad(optimized.square().sum(), (coord, features, conv.agh))
    for reference, actual in zip(regular_grads, optimized_grads, strict=True):
        torch.testing.assert_close(actual, reference, rtol=5e-5, atol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_missing_triton_uses_torch_layout(monkeypatch):
    import aimnet.modules.aev_inference as helper

    monkeypatch.setattr(helper, "_triton_kernel", lambda: None)
    aev = AEVSV(nshifts_s=4).cuda().eval()
    coord = torch.randn(2, 20, 3, device="cuda", requires_grad=True)
    reference = aev(_data(coord))["g_sv"]
    actual = _optimized(aev, _data(coord))["g_sv"]
    torch.testing.assert_close(actual, reference)
    grad = torch.autograd.grad(actual.square().sum(), coord, create_graph=True)[0]
    assert torch.isfinite(torch.autograd.grad(grad.square().sum(), coord)[0]).all()
