"""CPU-only guard tests: no weights download, CUDA or graph capture required."""

import copy
import types

import pytest
import torch
from torch import nn
from torch.nn.modules import module as module_hooks

from aimnet.calculators.cuda_graph.contracts import ModelContract
from aimnet.models.aimnet2 import AIMNet2
from aimnet.modules.core import AtomicShift, AtomicSum, Output
from aimnet.modules.lr import DFTD3, LRCoulomb, SRCoulomb


@pytest.fixture(scope="module")
def _base_model():
    return (
        AIMNet2(
            aev={"rc_s": 5.0, "nshifts_s": 4},
            nfeature=4,
            d2features=True,
            ncomb_v=4,
            hidden=([8], [8], [8]),
            aim_size=8,
            outputs={
                "energy_mlp": Output({"hidden": [8]}, 8, 1, "aim", "energy"),
                "atomic_shift": AtomicShift("energy", "energy"),
                "atomic_sum": AtomicSum("energy", "energy"),
                "srcoulomb": SRCoulomb(),
            },
        )
        .eval()
        .requires_grad_(False)
    )


@pytest.fixture
def contract(_base_model):
    # A fresh copy per test (tests mutate it); building the model once keeps the default run fast.
    m = copy.deepcopy(_base_model)
    return ModelContract(m, LRCoulomb(key_out="energy", subtract_sr=False), DFTD3(1.0, 0.4, 4.5))


def test_stable_hashable_snapshot(contract):
    a, b = contract.snapshot(), contract.snapshot()
    assert a is not None and a == b and hash(a) == hash(b)


@pytest.mark.parametrize("name", ["a1", "a2", "s6", "s8", "smoothing_on", "smoothing_off"])
def test_d3_options_invalidate(contract, name):
    before = contract.snapshot()
    setattr(contract.dftd3, name, getattr(contract.dftd3, name) + 0.1)
    assert contract.snapshot() is not None and contract.snapshot() != before


@pytest.mark.parametrize("path", ["aev.eta_s", "aev.rc_s", "outputs.srcoulomb.rc", "afv.weight"])
def test_inplace_versions_invalidate(contract, path):
    before = contract.snapshot()
    obj = contract.model
    for name in path.split("."):
        obj = getattr(obj, name)
    with torch.no_grad():
        obj.add_(0.1)
    assert contract.snapshot() is not None and contract.snapshot() != before


def test_state_dict_and_parameter_replacement(contract):
    before = contract.snapshot()
    contract.model.load_state_dict(contract.model.state_dict())
    assert contract.snapshot() != before
    before = contract.snapshot()
    contract.model.afv.weight = nn.Parameter(contract.model.afv.weight.detach().clone(), requires_grad=False)
    assert contract.snapshot() != before


def test_module_replacement_and_activation_option(contract):
    before = contract.snapshot()
    contract.model.mlps[0][1] = nn.GELU(approximate="tanh")
    after = contract.snapshot()
    assert after is not None and after != before
    contract.model.mlps[0][1].approximate = "none"
    assert contract.snapshot() != after


@pytest.mark.parametrize("method", ["forward", "_update_q", "_prepare_in_a", "prepare_input"])
def test_instance_override_rejected_after_capture(contract, method):
    assert contract.snapshot() is not None
    setattr(contract.model, method, types.MethodType(lambda self, *a, **kw: a[0], contract.model))
    assert contract.snapshot() is None
    delattr(contract.model, method)
    assert contract.snapshot() is not None


@pytest.mark.parametrize(
    "method",
    [
        "register_forward_hook",
        "register_forward_pre_hook",
        "register_full_backward_hook",
        "register_full_backward_pre_hook",
    ],
)
@pytest.mark.parametrize("target", ["model", "aev", "mlp"])
def test_hooks_rejected_after_capture(contract, method, target):
    assert contract.snapshot() is not None
    module = {"model": contract.model, "aev": contract.model.aev, "mlp": contract.model.mlps[0]}[target]
    handle = getattr(module, method)(lambda *args: None)
    try:
        assert contract.snapshot() is None
    finally:
        handle.remove()
    assert contract.snapshot() is not None


@pytest.mark.parametrize(
    "method",
    [
        "register_module_forward_hook",
        "register_module_forward_pre_hook",
        "register_module_full_backward_hook",
        "register_module_full_backward_pre_hook",
    ],
)
def test_global_hooks_rejected(contract, method):
    assert contract.snapshot() is not None
    handle = getattr(module_hooks, method)(lambda *args: None)
    try:
        assert contract.snapshot() is None
    finally:
        handle.remove()
    assert contract.snapshot() is not None


def test_class_forward_patch_rejected(contract, monkeypatch):
    original = nn.Linear.forward
    monkeypatch.setattr(nn.Linear, "forward", lambda self, x: original(self, x) + 0.1)
    assert contract.snapshot() is None


def test_subclass_replacement_rejected(contract):
    class ModifiedLinear(nn.Linear):
        pass

    contract.model.mlps[0][0].__class__ = ModifiedLinear
    assert contract.snapshot() is None


def test_unknown_output_rejected(contract):
    contract.model.outputs["custom_head"] = nn.Identity()
    assert contract.snapshot() is None


def test_training_rejected(contract):
    contract.model.train()
    assert contract.snapshot() is None


def test_parameter_derivative_request_rejected(contract):
    assert contract.snapshot() is not None
    contract.model.afv.weight.requires_grad_(True)
    assert contract.snapshot() is None


def test_noncanonical_output_order_rejected(contract):
    contract.model.outputs = nn.ModuleDict(dict(reversed(list(contract.model.outputs.items()))))
    assert contract.snapshot() is None


def test_noncanonical_output_keys_rejected(contract):
    contract.model.outputs.energy_mlp.key_in = "coord"
    assert contract.snapshot() is None


def test_no_device_value_reads(contract, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("must not read device tensor contents")

    monkeypatch.setattr(torch.Tensor, "item", fail)
    monkeypatch.setattr(torch.Tensor, "__float__", fail)
    monkeypatch.setattr(torch.Tensor, "__bool__", fail)
    assert contract.snapshot() is not None
