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


def _nblist_state(nblist):
    if nblist is None:
        return None
    return (nblist.cutoff, nblist.max_neighbors)


def _coulomb_state(calc):
    ext = None
    if calc.external_coulomb is not None:
        ext = (
            calc.external_coulomb.method,
            calc.external_coulomb.dsf_alpha,
            calc.external_coulomb.dsf_rc,
            calc.external_coulomb.ewald_accuracy,
        )
    return (
        calc._coulomb_method,
        calc._coulomb_cutoff,
        calc.cutoff_lr,
        _nblist_state(calc._nblist_lr),
        _nblist_state(calc._nblist_dftd3),
        _nblist_state(calc._nblist_coulomb),
        ext,
    )


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


def test_hvp_wrong_vector_shape_raises():
    calc = AIMNet2Calculator("aimnet2", nb_threshold=0)
    calc.external_dftd3 = None
    data = {
        "coord": np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]),
        "numbers": np.array([8, 1, 1]),
        "charge": 0.0,
    }
    with pytest.raises(ValueError):
        calc.hessian_vector_product(data, torch.zeros(5, 3))  # wrong N


@pytest.mark.parametrize("method", ["simple", "ewald"])
def test_hvp_matches_dense_with_dftd3(method):
    """HVP must include DFTD3 curvature (regression for dropped-D3 bug)."""
    calc = AIMNet2Calculator("aimnet2", nb_threshold=0, needs_coulomb=(method != "simple"))
    # NOTE: external_dftd3 is deliberately LEFT ATTACHED (the default).
    assert calc.external_dftd3 is not None, "expected default DFTD3 attached"
    if method != "simple":
        _set_method(calc, method)
    data = {
        "coord": np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]),
        "numbers": np.array([8, 1, 1]),
        "charge": 0.0,
    }
    if method != "simple":
        data["cell"] = np.eye(3) * 8.0
    torch.manual_seed(0)
    v = torch.randn(3, 3, dtype=torch.float64)
    hv = calc.hessian_vector_product(data, v)
    ref = _dense_hessian_matmul(calc, data, v)
    tol = {"rtol": 1e-3, "atol": 1e-3} if method == "simple" else {"rtol": 5e-2, "atol": 5e-3}
    torch.testing.assert_close(hv.double().cpu(), ref.cpu(), **tol)


def test_hvp_validates_unsupported_element():
    """HVP enforces the same species validation as eval (FIX 3)."""
    calc = AIMNet2Calculator("aimnet2", nb_threshold=0)
    impl = (calc.metadata or {}).get("implemented_species") or []
    if not impl:
        pytest.skip("model did not declare implemented_species")
    bad_z = next(z for z in range(1, 119) if z not in impl)
    data = {
        "coord": np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0]]),
        "numbers": np.array([bad_z, 1]),
        "charge": 0.0,
    }
    v = torch.zeros(2, 3, dtype=torch.float64)
    with pytest.raises(ValueError):
        calc.hessian_vector_product(data, v)
    # Bypass must not raise on the validation path.
    calc.hessian_vector_product(data, v, validate_species=False)


def test_hvp_periodic_returns_float64():
    calc = AIMNet2Calculator("aimnet2", nb_threshold=0, needs_coulomb=True)
    calc.external_dftd3 = None
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Model has embedded Coulomb module", category=UserWarning)
        calc.set_lrcoulomb_method("ewald")
    data = {
        "coord": np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]),
        "numbers": np.array([8, 1, 1]),
        "charge": 0.0,
        "cell": np.eye(3) * 8.0,
    }
    hv = calc.hessian_vector_product(data, torch.randn(3, 3, dtype=torch.float64))
    assert hv.dtype == torch.float64


def test_hvp_pbc_auto_switch_restores_full_coulomb_state():
    calc = AIMNet2Calculator("aimnet2", nb_threshold=0, needs_coulomb=True)
    calc.external_dftd3 = None
    calc.set_lrcoulomb_method("simple")
    before = _coulomb_state(calc)
    data = {
        "coord": np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]),
        "numbers": np.array([8, 1, 1]),
        "charge": 0.0,
        "cell": np.eye(3) * 8.0,
    }
    with pytest.warns(UserWarning, match="Switching to DSF Coulomb for PBC"):
        calc.hessian_vector_product(data, torch.randn(3, 3, dtype=torch.float64))
    assert _coulomb_state(calc) == before


def test_batched_hessian_pbc_auto_switch_restores_full_coulomb_state():
    calc = AIMNet2Calculator("aimnet2", nb_threshold=0, needs_coulomb=True)
    calc.external_dftd3 = None
    calc.set_lrcoulomb_method("simple")
    before = _coulomb_state(calc)
    data = {
        "coord": torch.tensor(
            [
                [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
                [[0.0, 0.0, 0.0], [0.97, 0.0, 0.0], [-0.25, 0.94, 0.0]],
            ],
            dtype=torch.float32,
        ),
        "numbers": torch.tensor([[8, 1, 1], [8, 1, 1]]),
        "charge": torch.tensor([0.0, 0.0]),
        "cell": torch.eye(3) * 8.0,
    }
    with pytest.warns(UserWarning, match="Switching to DSF Coulomb for PBC"):
        H = calc(data, hessian=True)["hessian"]
    assert H.shape == (2, 3, 3, 3, 3)
    assert _coulomb_state(calc) == before


def test_hvp_create_graph_contract():
    calc = AIMNet2Calculator("aimnet2", nb_threshold=0)
    calc.external_dftd3 = None
    data = {
        "coord": torch.tensor(
            [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
            dtype=torch.float32,
            requires_grad=True,
        ),
        "numbers": torch.tensor([8, 1, 1]),
        "charge": 0.0,
    }
    v = torch.randn(3, 3)

    hv_detached = calc.hessian_vector_product(data, v)
    assert not hv_detached.requires_grad

    hv_graph = calc.hessian_vector_product(data, v, create_graph=True)
    assert hv_graph.requires_grad
    assert hv_graph.grad_fn is not None
