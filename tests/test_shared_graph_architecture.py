"""Shared model computation and neighbor ownership, without CUDA capture."""

import copy

import pytest
import torch
from torch import nn

from aimnet.calculators import AIMNet2Calculator
from aimnet.calculators.cuda_graph.cache import CUDAGraphCache
from aimnet.calculators.cuda_graph.contracts import ModelContract
from aimnet.calculators.cuda_graph.dense import StaticDenseAIMNet2
from aimnet.calculators.cuda_graph.sparse import StaticSparseAIMNet2
from aimnet.models.aimnet2 import AIMNet2
from aimnet.modules.core import AtomicSum, Output
from aimnet.modules.lr import DFTD3, LRCoulomb


@pytest.fixture(scope="module")
def _base_model():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(137)
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
                    "atomic_sum": AtomicSum("energy", "energy"),
                },
            )
            .eval()
            .requires_grad_(False)
        )


@pytest.fixture
def model(_base_model):
    """A fresh copy per test (tests mutate it); building the model once keeps the default run fast."""
    return copy.deepcopy(_base_model)


def _inputs(padded=False):
    data = {
        "coord": torch.tensor([
            [[0.0, 0.0, 0.0], [1.1, 0.1, 0.0], [0.0, 1.2, 0.1], [0.1, 0.0, 1.3]],
            [[0.2, 0.0, 0.1], [1.3, 0.0, 0.1], [0.2, 1.4, 0.0], [0.0, 0.2, 1.5]],
        ]),
        "numbers": torch.tensor([[6, 1, 1, 1], [7, 1, 1, 0 if padded else 1]]),
        "charge": torch.tensor([0.0, 1.0]),
    }
    return data


@pytest.mark.parametrize("activation", [nn.GELU, nn.SiLU])
@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("forces", [False, True])
def test_dense_adapter_uses_canonical_model_and_calculator(model, monkeypatch, activation, padded, forces):
    """Changing the model's nonlinearity affects both paths through one model body."""
    for mlp in model.mlps:
        for index, layer in enumerate(mlp):
            if isinstance(layer, nn.GELU):
                mlp[index] = activation()
    calc = AIMNet2Calculator(model, device="cpu", cuda_graph=False)
    data = _inputs(padded)
    engine = StaticDenseAIMNet2(calc, data["numbers"], data["charge"], padded=padded)
    calls = []
    compute_model = model.forward_prepared
    compute_calculator = calc._evaluate_prepared

    def model_spy(data):
        calls.append("model")
        return compute_model(data)

    def calculator_spy(data, **kwargs):
        calls.append(("calculator", kwargs.get("model_prepared", False)))
        return compute_calculator(data, **kwargs)

    monkeypatch.setattr(model, "forward_prepared", model_spy)
    monkeypatch.setattr(calc, "_evaluate_prepared", calculator_spy)
    energy, force, charges = engine(data["coord"]) if forces else engine.energy_only(data["coord"])
    assert calls == [("calculator", True), "model"]
    calls.clear()
    reference = calc._evaluate_prepared({key: value.clone() for key, value in data.items()}, forces=forces)
    assert calls == [("calculator", False), "model"]
    torch.testing.assert_close(energy, reference["energy"])
    torch.testing.assert_close(charges, reference["charges"])
    if forces:
        torch.testing.assert_close(force, reference["forces"])
    else:
        assert force is None


def test_forward_hook_preserves_regular_behavior_and_declines_capture(model, monkeypatch):
    calc = AIMNet2Calculator(model, device="cpu", cuda_graph=False)
    data = _inputs()
    reference = calc._evaluate_prepared({key: value.clone() for key, value in data.items()}, forces=True)
    assert ModelContract(model).snapshot() is not None

    def add_coordinate_term(module, args, output):
        output["energy"] = output["energy"] + 0.1 * output["coord"].sum(dim=(-1, -2))
        return output

    handle = model.register_forward_hook(add_coordinate_term)
    try:
        changed = calc._evaluate_prepared({key: value.clone() for key, value in data.items()}, forces=True)
        torch.testing.assert_close(changed["forces"], reference["forces"] - 0.1)
        assert ModelContract(model).snapshot() is None
        # Cache selection reads the device label but makes no CUDA allocations.
        monkeypatch.setattr(calc, "device", "cuda")
        assert CUDAGraphCache.create(calc, cuda_graph=True) is None
    finally:
        handle.remove()


def test_kernel_selection_changes_graph_layout_key(model):
    calc = AIMNet2Calculator(model, device="cpu", cuda_graph=False)
    cache = CUDAGraphCache(calc)
    stock = cache._settings()
    assert stock is not None
    calc.inference_kernels = True
    optimized = cache._settings()
    assert optimized is not None and optimized != stock
    calc.inference_kernels = False
    assert cache._settings() == stock


def test_d3_neighbor_cutoff_changes_graph_key_without_changing_smoothing(model):
    calc = AIMNet2Calculator(model, device="cpu", cuda_graph=False)
    calc.external_coulomb = None
    calc.external_dftd3 = DFTD3(1.0, 0.4, 4.5)
    calc.set_lr_cutoff(20.0)
    cache = CUDAGraphCache(calc)
    before = cache._settings()
    smoothing = calc.external_dftd3.smoothing_off
    calc.set_lr_cutoff(25.0)
    assert calc.external_dftd3.smoothing_off == smoothing
    assert before is not None and cache._settings() is not None
    assert cache._settings() != before


class _CPUNeighborMatrix:
    """Small reference list for checking graph-adapter list ownership on CPU."""

    def __init__(self, cutoff, batch_idx, batch_ptr, capacity, fill_value):
        self.cutoff, self.batch_idx, self.batch_ptr = cutoff, batch_idx, batch_ptr
        self.capacity, self.fill_value = capacity, fill_value
        self.matrix = torch.full((batch_idx.numel(), capacity), fill_value, dtype=torch.int32)
        self.counts = torch.zeros(batch_idx.numel(), dtype=torch.int32)

    def fill(self, coordinates):
        distance = torch.cdist(coordinates, coordinates)
        mask = (distance < self.cutoff) & (self.batch_idx[:, None] == self.batch_idx[None, :])
        mask.fill_diagonal_(False)
        self.matrix.fill_(self.fill_value)
        for row in range(coordinates.shape[0]):
            indices = torch.where(mask[row])[0]
            self.counts[row] = indices.numel()
            count = min(indices.numel(), self.capacity)
            self.matrix[row, :count] = indices[:count].to(torch.int32)
        return self.matrix, self.counts


@pytest.mark.parametrize("forces", [False, True])
def test_sparse_adapter_uses_canonical_model_and_calculator(model, monkeypatch, forces):
    import aimnet.calculators.cuda_graph.sparse as sparse_module

    monkeypatch.setattr(sparse_module, "ReusableNeighborMatrix", _CPUNeighborMatrix)
    calc = AIMNet2Calculator(model, device="cpu", cuda_graph=False)
    data = _inputs()
    engine = StaticSparseAIMNet2(calc, data["numbers"], data["charge"], data["coord"], capacity=4)
    calls = []
    compute_model = model.forward_prepared
    compute_calculator = calc._evaluate_prepared

    def model_spy(data):
        calls.append("model")
        return compute_model(data)

    def calculator_spy(data, **kwargs):
        calls.append(("calculator", kwargs.get("model_prepared", False)))
        return compute_calculator(data, **kwargs)

    monkeypatch.setattr(model, "forward_prepared", model_spy)
    monkeypatch.setattr(calc, "_evaluate_prepared", calculator_spy)
    energy, force, charges, overflow = engine(data["coord"]) if forces else engine.energy_only(data["coord"])
    assert calls == [("calculator", True), "model"]
    assert not bool(overflow)
    calls.clear()
    prepared, _ = engine.prepare(data["coord"])
    reference = calc._evaluate_prepared(prepared, forces=forces)
    assert calls == [("calculator", False), "model"]
    torch.testing.assert_close(energy, reference["energy"])
    torch.testing.assert_close(charges, reference["charges"][: engine.T].reshape(engine.B, engine.N))
    if forces:
        torch.testing.assert_close(force, reference["forces"][: engine.T].reshape(engine.B, engine.N, 3))
    else:
        assert force is None


@pytest.mark.parametrize(
    ("coulomb", "dispersion", "kernels", "all_pairs"),
    [
        (False, False, False, False),
        (False, False, True, False),
        (True, False, False, True),
        (True, False, True, False),
        (True, True, True, True),
        (False, True, True, False),
    ],
)
def test_sparse_all_pairs_owned_only_by_consumers(model, monkeypatch, coulomb, dispersion, kernels, all_pairs):
    import aimnet.calculators.cuda_graph.sparse as sparse_module

    monkeypatch.setattr(sparse_module, "ReusableNeighborMatrix", _CPUNeighborMatrix)
    calc = AIMNet2Calculator(model, device="cpu", cuda_graph=False, inference_kernels=kernels)
    calc.external_coulomb = LRCoulomb(key_out="energy", subtract_sr=False) if coulomb else None
    calc.external_dftd3 = DFTD3(1.0, 0.4, 4.5) if dispersion else None
    calc._dftd3_cutoff = 17.0
    coord = torch.tensor([[[0.0, 0.0, 0.0], [1.3, 0.0, 0.0], [40.0, 0.0, 0.0]]])
    engine = StaticSparseAIMNet2(calc, torch.tensor([[6, 8, 1]]), torch.tensor([0.0]), coord, capacity=4)
    data, overflow = engine.prepare(coord)
    assert not bool(overflow)
    assert (engine.nbmat_lr is not None) == all_pairs
    if coulomb and dispersion:
        # Preserve the regular full list for D3 CN, even beyond its smoothing range.
        assert 2 in data["nbmat_dftd3"][0].tolist()
        assert engine._d3_neighbors is None
    elif dispersion:
        assert engine._d3_neighbors.cutoff == calc._dftd3_cutoff
        assert 2 not in data["nbmat_dftd3"][0].tolist()
    else:
        assert "nbmat_dftd3" not in data
    if kernels and coulomb:
        assert "nbmat_coulomb" not in data
