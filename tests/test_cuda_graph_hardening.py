"""Regression tests for CUDA graph ownership, invalidation and public API contracts.

These compare against the unchanged regular calculator, not another copy of the
graph engine. In particular, correctness must survive returning to an *old*
layout after callers mutate their tensors, and changes invisible at the original
capture geometry must not be silently omitted at subsequent geometries.
"""

import gc
import types

import pytest
import torch

from aimnet.calculators import AIMNet2Calculator

pytestmark = [pytest.mark.gpu, pytest.mark.weights]

if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)


def _water(batch=1):
    coord = torch.tensor([[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.239987, 0.927297, 0.0]], device="cuda")
    numbers = torch.tensor([8, 1, 1], device="cuda")
    if batch > 1:
        coord = coord.unsqueeze(0).repeat(batch, 1, 1)
        numbers = numbers.unsqueeze(0).repeat(batch, 1)
    return {"coord": coord, "numbers": numbers, "charge": torch.zeros((), device="cuda")}


def _copy(data):
    return {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in data.items()}


def _assert_same(result, reference):
    assert result.keys() == reference.keys()
    for key in reference:
        assert result[key].shape == reference[key].shape, key
        assert result[key].dtype == reference[key].dtype, key
        assert result[key].device == reference[key].device, key
        atol, rtol = {"energy": (1e-4, 0.0), "charges": (1e-5, 0.0)}.get(key, (2e-4, 2e-5))
        torch.testing.assert_close(result[key], reference[key], atol=atol, rtol=rtol)


def _capture(calc, data, forces=True):
    for _ in range(3):
        result = calc(data, forces=forces)
    assert calc._cuda_graph is not None
    assert len(calc._cuda_graph._engines) > 0, "the test must exercise a genuinely captured graph"
    return result


@pytest.fixture
def pair_factory():
    pairs = []

    def make(sparse=False):
        threshold = 0 if sparse else 120
        graph = AIMNet2Calculator("aimnet2-2025", cuda_graph=True, nb_threshold=threshold)
        regular = AIMNet2Calculator("aimnet2-2025", cuda_graph=False, nb_threshold=threshold)
        pairs.append((graph, regular))
        return graph, regular

    yield make
    for graph, _ in pairs:
        if graph._cuda_graph is not None:
            graph._cuda_graph.clear()
    pairs.clear()
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.parametrize("sparse", [False, True], ids=["dense", "sparse"])
@pytest.mark.parametrize("batch,scalar", [(1, True), (3, True), (3, False)])
def test_mutable_charge_return_to_original_layout(pair_factory, sparse, batch, scalar):
    """A -> mutate caller Q to B -> fresh A must not replay B's Q through A's key."""
    graph, regular = pair_factory(sparse)
    data = _water(batch)
    if not scalar:
        data["charge"] = torch.zeros(batch, device="cuda")
    original = _copy(data)
    _capture(graph, data)
    data["charge"].add_(1.0)
    _assert_same(_capture(graph, data), regular(data, forces=True))
    result = graph(original, forces=True)
    _assert_same(result, regular(original, forces=True))
    torch.testing.assert_close(
        result["charges"].reshape(batch, -1).sum(-1), torch.zeros(batch, device="cuda"), atol=1e-5, rtol=0.0
    )


@pytest.mark.parametrize("sparse", [False, True], ids=["dense", "sparse"])
def test_mutable_species_return_to_original_layout(pair_factory, sparse):
    graph, regular = pair_factory(sparse)
    data = _water()
    original = _copy(data)
    _capture(graph, data)
    data["numbers"][[0, 1]] = data["numbers"][[1, 0]].clone()
    _assert_same(_capture(graph, data), regular(data, forces=True))
    _assert_same(graph(original, forces=True), regular(original, forces=True))


@pytest.mark.parametrize("budget_reduced", [False, True], ids=["normal-budget", "zero-budget"])
def test_sparse_rebuild_keeps_owned_charge(pair_factory, budget_reduced):
    """A capacity rebuild must use the layout's snapshot, not a mutated caller Q."""
    graph, regular = pair_factory(sparse=True)
    water = _water()
    lattice = torch.cartesian_prod(*(torch.arange(3, device="cuda") for _ in range(3))).float()

    def cluster(spacing):
        return {
            "coord": (water["coord"][None] + spacing * lattice[:, None]).reshape(-1, 3),
            "numbers": water["numbers"].repeat(len(lattice)),
            "charge": torch.zeros((), device="cuda"),
        }

    initial = cluster(4.2)
    original = _copy(initial)
    _capture(graph, initial)
    entry = next(iter(graph._cuda_graph._engines.values()))
    initial["charge"].add_(1.0)
    _capture(graph, initial)
    compressed = {**original, "coord": cluster(2.7)["coord"]}
    if budget_reduced:
        graph._cuda_graph.cache_memory_fraction = 0.0
        with pytest.warns(RuntimeWarning, match="rebuilt sparse engine exceeds the cache memory budget"):
            result = graph(compressed, forces=True)
        _assert_same(result, regular(compressed, forces=True))
        assert all(cached is not entry for cached in graph._cuda_graph._engines.values())
    else:
        _assert_same(graph(compressed, forces=True), regular(compressed, forces=True))
        assert entry.rebuilds == entry.runner.rebuilds
    assert entry.runner.rebuilds >= 1, "geometry must actually exercise the overflow/rebuild path"
    _assert_same(graph(original, forces=True), regular(original, forces=True))


@pytest.mark.parametrize("sparse", [False, True], ids=["dense", "sparse"])
def test_inference_tensors_for_layout_only(pair_factory, sparse):
    """Inference numbers/Q have no version counter; normal coordinates remain differentiable internally."""
    graph, regular = pair_factory(sparse)
    data = _water()
    with torch.inference_mode():
        data["numbers"] = data["numbers"].clone()
        data["charge"] = data["charge"].clone()
    for _ in range(3):
        _assert_same(graph(data, forces=True), regular(data, forces=True))
    with torch.inference_mode():
        data["charge"].add_(1.0)
    for _ in range(3):
        _assert_same(graph(data, forces=True), regular(data, forces=True))


def test_bfloat16_charge_input_matches_regular_cast(pair_factory):
    graph, regular = pair_factory()
    data = _water()
    data["charge"] = data["charge"].to(torch.bfloat16)
    for _ in range(3):
        _assert_same(graph(data, forces=True), regular(data, forces=True))


@pytest.mark.parametrize("already_captured", [False, True])
def test_charge_autograd_contract_survives_repeated_calls(pair_factory, already_captured):
    graph, regular = pair_factory()
    data = _water()
    if already_captured:
        _capture(graph, data, forces=False)
    data["charge"].requires_grad_(True)
    for _ in range(3):
        result, reference = graph(data), regular(data)
        _assert_same(result, reference)
        assert result["energy"].requires_grad
        actual = torch.autograd.grad(result["energy"].sum(), data["charge"])[0]
        expected = torch.autograd.grad(reference["energy"].sum(), data["charge"])[0]
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_model_parameter_autograd_after_capture_uses_regular_path(pair_factory):
    graph, regular = pair_factory()
    data = _water()
    _capture(graph, data, forces=False)
    graph_weight = graph.model.outputs.energy_mlp.mlp[4].weight
    reference_weight = regular.model.outputs.energy_mlp.mlp[4].weight
    graph_weight.requires_grad_(True)
    reference_weight.requires_grad_(True)
    for _ in range(3):
        result, reference = graph(data), regular(data)
        _assert_same(result, reference)
        assert result["energy"].requires_grad
        actual = torch.autograd.grad(result["energy"].sum(), graph_weight)[0]
        expected = torch.autograd.grad(reference["energy"].sum(), reference_weight)[0]
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("sparse", [False, True], ids=["dense", "sparse"])
@pytest.mark.parametrize("kind", ["d3_s8", "aev_eta", "parameter_replace", "parameter_inplace", "load_state_assign"])
def test_settings_and_weights_changed_after_capture(pair_factory, sparse, kind):
    graph, regular = pair_factory(sparse)
    # A pair gives D3 a meaningful intermolecular contribution without a large test system.
    data = _water()
    data["coord"] = torch.cat((data["coord"], data["coord"] + data["coord"].new_tensor([3.0, 0.4, 0.0])))
    data["numbers"] = data["numbers"].repeat(2)
    _capture(graph, data)
    for calc in (graph, regular):
        with torch.no_grad():
            layer = calc.model.mlps[0][0]
            if kind == "d3_s8":
                calc.external_dftd3.s8 *= 1.5
            elif kind == "aev_eta":
                calc.model.aev.eta_s *= 1.1
            elif kind == "parameter_replace":
                layer.weight = torch.nn.Parameter(
                    layer.weight.detach().clone() * 1.01, requires_grad=layer.weight.requires_grad
                )
            elif kind == "parameter_inplace":
                layer.weight.mul_(1.01)
            else:
                state = {key: value.detach().clone() for key, value in calc.model.state_dict().items()}
                state["mlps.0.0.weight"].mul_(1.01)
                calc.model.load_state_dict(state, assign=True)
    for _ in range(3):
        _assert_same(graph(data, forces=True), regular(data, forces=True))
    assert len(graph._cuda_graph._engines) > 0, "a supported updated model must recapture, not stay on regular forever"


@pytest.mark.parametrize("sparse", [False, True], ids=["dense", "sparse"])
def test_replaced_parameter_storage_outlives_a_discarded_replay(pair_factory, sparse):
    """A replay is launched before the contract check: it may only read storage the cache entry keeps alive."""
    graph, regular = pair_factory(sparse)
    data = _water()
    _capture(graph, data)
    (entry,) = graph._cuda_graph._engines.values()
    old_ptr = graph.model.mlps[0][0].weight.data_ptr()
    assert any(t.data_ptr() == old_ptr for t in entry.keepalive)
    for calc in (graph, regular):
        layer = calc.model.mlps[0][0]
        with torch.no_grad():
            layer.weight = torch.nn.Parameter(
                layer.weight.detach().clone() * 1.01, requires_grad=layer.weight.requires_grad
            )
    gc.collect()
    torch.cuda.empty_cache()
    for _ in range(3):
        _assert_same(graph(data, forces=True), regular(data, forces=True))
    torch.cuda.synchronize()


@pytest.mark.parametrize("sparse", [False, True], ids=["dense", "sparse"])
@pytest.mark.parametrize("forces", [False, True], ids=["energy-only", "energy-forces"])
def test_resize_roundtrip_is_checked_before_old_graph_launch(pair_factory, monkeypatch, sparse, forces):
    """A detached keepalive does not pin the address when its shared storage is resized.

    The final parameter shape/content are identical and valid. Ordinary resize_,
    not .data/raw-pointer mutation, reallocates the shared StorageImpl. Never replay
    the old graph, even speculatively. The spy prevents an unfixed test run from
    poisoning its CUDA context with a known dangling-pointer graph launch.
    """
    graph, regular = pair_factory(sparse)
    data = _water()
    layer = graph.model.mlps[0][0]
    original = layer.weight.detach().clone()
    shape = original.shape
    backing = torch.empty(8 * 1024 * 1024, device="cuda", dtype=torch.float32)  # dedicated 32 MiB allocation
    layer.weight = torch.nn.Parameter(backing[: original.numel()].view(shape), requires_grad=False)
    layer.weight.copy_(original)
    del backing
    _capture(graph, data, forces=forces)
    (entry,) = graph._cuda_graph._engines.values()
    old_ptr = layer.weight.data_ptr()
    assert any(t.data_ptr() == old_ptr for t in entry.keepalive)
    layer.weight.resize_(16 * 1024 * 1024)  # force 64 MiB storage; detached views share the reallocated storage
    layer.weight.resize_(shape)
    layer.weight.copy_(original)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    assert layer.weight.data_ptr() != old_ptr
    assert torch.equal(layer.weight, original)

    attempted_launches = []

    def forbidden_launch(*args, **kwargs):
        attempted_launches.append(True)
        raise AssertionError("captured storage moved: an old graph launch would access freed memory")

    monkeypatch.setattr(entry.runner, "launch", forbidden_launch)
    _assert_same(graph(data, forces=forces), regular(data, forces=forces))
    assert not attempted_launches, "storage safety must be checked before, not after, replay"
    for _ in range(3):
        _assert_same(graph(data, forces=forces), regular(data, forces=forces))
    assert graph._cuda_graph._engines, "the restored valid model must recapture and retain acceleration"
    assert all(cached is not entry for cached in graph._cuda_graph._engines.values())


@pytest.mark.parametrize("sparse", [False, True], ids=["dense", "sparse"])
@pytest.mark.parametrize("forces", [False, True], ids=["energy-only", "energy-forces"])
def test_external_d3_buffer_replacement_retains_old_storage(pair_factory, sparse, forces):
    graph, regular = pair_factory(sparse)
    data = _water()
    data["coord"] = torch.cat((data["coord"], data["coord"] + data["coord"].new_tensor([3.0, 0.4, 0.0])))
    data["numbers"] = data["numbers"].repeat(2)
    _capture(graph, data, forces=forces)
    (entry,) = graph._cuda_graph._engines.values()
    old_ptr = graph.external_dftd3.c6ab.data_ptr()
    assert any(t.data_ptr() == old_ptr for t in entry.keepalive)
    for calc in (graph, regular):
        calc.external_dftd3.c6ab = calc.external_dftd3.c6ab.detach().clone() * 1.01
    gc.collect()
    torch.cuda.empty_cache()
    for _ in range(3):
        _assert_same(graph(data, forces=forces), regular(data, forces=forces))
    assert graph._cuda_graph._engines, "a supported buffer replacement must recapture"
    assert all(cached is not entry for cached in graph._cuda_graph._engines.values())


def test_model_invalidation_discards_sparse_overflow_before_result(pair_factory, monkeypatch):
    """A stale speculative graph may overflow, but must not rebuild under the changed model."""
    graph, regular = pair_factory(sparse=True)
    water = _water()
    lattice = torch.cartesian_prod(*(torch.arange(3, device="cuda") for _ in range(3))).float()
    data = {
        "coord": (water["coord"][None] + 4.2 * lattice[:, None]).reshape(-1, 3),
        "numbers": water["numbers"].repeat(len(lattice)),
        "charge": torch.zeros((), device="cuda"),
    }
    _capture(graph, data)
    (entry,) = graph._cuda_graph._engines.values()
    for calc in (graph, regular):
        calc.external_dftd3.s8 *= 1.1
    compressed = {**data, "coord": (water["coord"][None] + 2.7 * lattice[:, None]).reshape(-1, 3)}
    attempted_results = []

    def forbidden_result():
        attempted_results.append(True)
        raise AssertionError("invalidated speculative result must not be read or trigger a rebuild")

    monkeypatch.setattr(entry.runner, "result", forbidden_result)
    _assert_same(graph(compressed, forces=True), regular(compressed, forces=True))
    assert not attempted_results
    assert entry.runner.rebuilds == 0
    assert bool(entry.runner.graph.overflow), "the chosen geometry must actually overflow the old graph"
    for _ in range(3):
        _assert_same(graph(compressed, forces=True), regular(compressed, forces=True))
    assert graph._cuda_graph._engines
    assert all(cached is not entry for cached in graph._cuda_graph._engines.values())


def _bond_length(data):
    coord = data["coord"]
    if coord.ndim == 3:
        return (coord[:, 0] - coord[:, 1]).norm(dim=-1).mean()
    return (coord[0] - coord[1]).norm()


def _moved(data):
    result = _copy(data)
    result["coord"][1, 0] += 0.4
    return result


@pytest.mark.parametrize("sparse", [False, True], ids=["dense", "sparse"])
@pytest.mark.parametrize("when", ["before", "after"])
@pytest.mark.parametrize("target", ["model", "aev"])
def test_hooks_invisible_at_capture_geometry_are_not_omitted(pair_factory, sparse, when, target):
    graph, regular = pair_factory(sparse)
    data = _water()
    r0 = float(_bond_length(data))
    if when == "after":
        _capture(graph, data)

    def extra_energy(module, inputs, output):
        return {**output, "energy": output["energy"] + (_bond_length(output) - r0).pow(4).double()}

    def changed_aev(module, inputs, output):
        return {**output, "g_sv": output["g_sv"] * (1.0 + (_bond_length(output) - r0).pow(4))}

    handles = []
    for calc in (graph, regular):
        module = calc.model if target == "model" else calc.model.aev
        handles.append(module.register_forward_hook(extra_energy if target == "model" else changed_aev))
    try:
        for _ in range(3):
            _assert_same(graph(data, forces=True), regular(data, forces=True))
        moved = _moved(data)
        for _ in range(3):
            _assert_same(graph(moved, forces=True), regular(moved, forces=True))
    finally:
        for handle in handles:
            handle.remove()


@pytest.mark.parametrize("kind", ["instance_forward", "subclass"])
@pytest.mark.parametrize("when", ["before", "after"])
def test_custom_forward_uses_regular_path(pair_factory, kind, when):
    graph, regular = pair_factory()
    data = _water()
    r0 = float(_bond_length(data))
    if when == "after":
        _capture(graph, data)
    base = type(graph.model)

    class ExtraEnergyModel(base):
        def forward(self, inputs):
            output = super().forward(inputs)
            return {**output, "energy": output["energy"] + (_bond_length(output) - r0).pow(4).double()}

    def new_forward(self, inputs):
        output = base.forward(self, inputs)
        return {**output, "energy": output["energy"] + (_bond_length(output) - r0).pow(4).double()}

    for calc in (graph, regular):
        if kind == "subclass":
            calc.model.__class__ = ExtraEnergyModel
        else:
            calc.model.forward = types.MethodType(new_forward, calc.model)
    for _ in range(3):
        _assert_same(graph(data, forces=True), regular(data, forces=True))
    moved = _moved(data)
    for _ in range(3):
        _assert_same(graph(moved, forces=True), regular(moved, forces=True))
