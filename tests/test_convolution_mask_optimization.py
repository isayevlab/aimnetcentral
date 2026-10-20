"""Fresh-output ConvSV masking must preserve higher derivatives and inputs."""

import pytest
import torch

from aimnet import nbops
from aimnet.modules.aev import AEVSV, ConvSV


def _mask_calls(monkeypatch):
    calls = []
    original = nbops.mask_i_

    def observed(value, data, mask_value=0.0, inplace=True):
        before = value.data_ptr()
        result = original(value, data, mask_value=mask_value, inplace=inplace)
        calls.append((inplace, before, result.data_ptr()))
        return result

    monkeypatch.setattr(nbops, "mask_i_", observed)
    return calls


@pytest.mark.parametrize("d2features", [False, True])
@pytest.mark.parametrize("padded", [False, True])
def test_dense_fresh_output_mask_matches_first_through_third_derivatives(monkeypatch, d2features, padded):
    from test_geometry_carry import _assert_jets

    torch.manual_seed(628)
    aev = AEVSV(nshifts_s=3).double().eval()
    convolution = ConvSV(nshifts_s=3, nchannel=2, d2features=d2features).double().eval()
    coordinates = torch.randn(2, 5, 3, dtype=torch.float64, requires_grad=True)
    features = torch.randn((2, 5, 2, 3) if d2features else (2, 5, 2), dtype=torch.float64, requires_grad=True)
    numbers = torch.tensor([[6, 8, 1, 1, 1], [6, 7, 1, 1, 1]])
    if padded:
        numbers[0, -1] = 0
        numbers[1, -2:] = 0
    data = aev(nbops.calc_masks(nbops.set_nb_mode({"numbers": numbers, "coord": coordinates})))
    geometry_snapshot = data["g_sv"].clone()
    features_snapshot = features.clone()
    calls = _mask_calls(monkeypatch)

    expected = convolution(data, features)
    # Keep the same contraction and AEV graph. This flag changes only the
    # output-masking policy because the reference data contains no g_L entry.
    actual = convolution({**data, "_inference_kernels": torch.tensor(True)}, features)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert [call[0] for call in calls] == [False, True]
    assert calls[1][1] == calls[1][2]
    if padded:
        assert calls[0][1] != calls[0][2]
        assert torch.count_nonzero(actual[numbers == 0]) == 0
    _assert_jets(actual.sin().sum(), expected.sin().sum(), (coordinates, features, convolution.agh))
    torch.testing.assert_close(data["g_sv"], geometry_snapshot, rtol=0, atol=0)
    torch.testing.assert_close(features, features_snapshot, rtol=0, atol=0)


@pytest.mark.parametrize("mode,enabled", [(0, False), (0, None), (2, False), (2, True)])
def test_default_and_mode2_mask_keep_existing_out_of_place_policy(monkeypatch, mode, enabled):
    torch.manual_seed(629)
    convolution = ConvSV(nshifts_s=3, nchannel=2, d2features=True).double().eval()
    contraction = torch.randn(2, 5, 2, 3, 4, dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[False, False, False, False, True], [False, False, False, True, True]])
    data = {"_nb_mode": torch.tensor(mode), "_input_padded": torch.tensor(True), "mask_i": mask}
    if enabled is not None:
        data["_inference_kernels"] = torch.tensor(enabled)
    calls = _mask_calls(monkeypatch)
    result = convolution._finish_convolution(data, contraction, mode)
    assert len(calls) == 1 and calls[0][0] is False
    assert calls[0][1] != calls[0][2]
    assert torch.count_nonzero(result[mask]) == 0
    gradient = torch.autograd.grad(result.sin().sum(), contraction)[0]
    assert torch.count_nonzero(gradient[mask]) == 0


def test_carry_keeps_public_latent_feature_derivative_relationships():
    from test_grouped_convolution import _input, _model, _optimized

    model = _model().requires_grad_(False)
    source = _input()
    expected = model(dict(source))
    actual = _optimized(model, source)
    reference = torch.autograd.grad(expected["energy"].sum(), expected["a"], retain_graph=True)[0]
    gradient = torch.autograd.grad(actual["energy"].sum(), actual["a"], retain_graph=True)[0]
    assert torch.count_nonzero(reference) > 0
    torch.testing.assert_close(gradient, reference, rtol=1e-10, atol=1e-10)
