import warnings

import numpy as np
import pytest
import torch

from aimnet.calculators import AIMNet2Calculator


def _set_method(calc, method):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Model has embedded Coulomb module", category=UserWarning)
        if method == "dsf":
            calc.set_lrcoulomb_method("dsf", cutoff=8.0)
        elif method in ("ewald", "pme"):
            calc.set_lrcoulomb_method(method)


def _dense_hessian_matmul(calc, data, v):
    H = calc({k: (val.clone() if isinstance(val, torch.Tensor) else val) for k, val in data.items()}, hessian=True)[
        "hessian"
    ]
    n = H.shape[0]
    Hmat = H.reshape(3 * n, 3 * n).double()
    # The calculator auto-selects its device (CPU or CUDA); align v with the
    # dense Hessian so the reference matmul runs regardless of placement.
    v = v.to(Hmat.device)
    return (Hmat @ v.reshape(-1).double()).reshape(n, 3)


@pytest.mark.parametrize("method", ["simple", "dsf"])
def test_hvp_matches_dense_nonperiodic(method):
    calc = AIMNet2Calculator("aimnet2", nb_threshold=0, needs_coulomb=(method != "simple"))
    calc.external_dftd3 = None
    if method != "simple":
        _set_method(calc, method)
    data = {
        "coord": np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]),
        "numbers": np.array([8, 1, 1]),
        "charge": 0.0,
    }
    torch.manual_seed(0)
    v = torch.randn(3, 3, dtype=torch.float64)
    hv = calc.hessian_vector_product(data, v)
    ref = _dense_hessian_matmul(calc, data, v)
    torch.testing.assert_close(hv.double().cpu(), ref.cpu(), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("method", ["ewald", "pme"])
def test_hvp_matches_dense_periodic(method):
    calc = AIMNet2Calculator("aimnet2", nb_threshold=0, needs_coulomb=True)
    calc.external_dftd3 = None
    _set_method(calc, method)
    data = {
        "coord": np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]),
        "numbers": np.array([8, 1, 1]),
        "charge": 0.0,
        "cell": np.eye(3) * 8.0,
    }
    torch.manual_seed(0)
    v = torch.randn(3, 3, dtype=torch.float64)
    hv = calc.hessian_vector_product(data, v)
    ref = _dense_hessian_matmul(calc, data, v)
    torch.testing.assert_close(hv.double().cpu(), ref.cpu(), rtol=5e-2, atol=5e-3)


def test_hvp_multiple_vectors_shape():
    calc = AIMNet2Calculator("aimnet2", nb_threshold=0)
    calc.external_dftd3 = None
    data = {
        "coord": np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]),
        "numbers": np.array([8, 1, 1]),
        "charge": 0.0,
    }
    torch.manual_seed(1)
    V = torch.randn(4, 3, 3, dtype=torch.float64)
    HV = calc.hessian_vector_product(data, V)
    assert HV.shape == (4, 3, 3)
    ref = torch.stack([_dense_hessian_matmul(calc, data, V[k]) for k in range(4)], 0)
    torch.testing.assert_close(HV.double().cpu(), ref.cpu(), rtol=1e-3, atol=1e-3)


def test_hvp_batched_input_raises():
    calc = AIMNet2Calculator("aimnet2", nb_threshold=1000)
    data = {
        "coord": torch.zeros(2, 3, 3),
        "numbers": torch.tensor([[8, 1, 1], [8, 1, 1]]),
        "charge": torch.zeros(2),
    }
    v = torch.zeros(3, 3)
    with pytest.raises((NotImplementedError, ValueError)):
        calc.hessian_vector_product(data, v)
