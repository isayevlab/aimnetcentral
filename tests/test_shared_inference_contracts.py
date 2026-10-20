"""Public-API contracts of the optional shared inference primitives."""

import pytest
import torch

from aimnet.calculators import AIMNet2Calculator

pytestmark = [pytest.mark.gpu, pytest.mark.weights]

if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)


def _waters(batch=1):
    water = torch.tensor([[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.24, 0.9273, 0.0]], device="cuda")
    coord = torch.cat((water, water + water.new_tensor([8.0, 0.0, 0.0]))).unsqueeze(0).repeat(batch, 1, 1)
    numbers = torch.tensor([[8, 1, 1, 8, 1, 1]], device="cuda").repeat(batch, 1)
    return {"coord": coord, "numbers": numbers, "charge": torch.zeros(batch, device="cuda")}


def _copy(data):
    return {key: value.detach().clone() for key, value in data.items()}


def _same(actual, reference):
    assert actual.keys() == reference.keys()
    for key in reference:
        assert actual[key].shape == reference[key].shape
        assert torch.isfinite(actual[key]).all(), key
        atol, rtol = {"energy": (2e-4, 0.0), "charges": (1e-5, 0.0)}.get(key, (2e-4, 2e-5))
        torch.testing.assert_close(actual[key], reference[key], atol=atol, rtol=rtol)


def _forbid_streaming(monkeypatch):
    import aimnet.kernels.coulomb_wp as coulomb

    def forbidden(*args, **kwargs):
        raise AssertionError("This input must retain reference Coulomb semantics")

    monkeypatch.setattr(coulomb, "full_coulomb_rows", forbidden)


@pytest.mark.parametrize("threshold", [0, 120], ids=["packed", "dense"])
def test_finite_coulomb_cutoff_never_uses_full_pair_streaming(monkeypatch, threshold):
    optimized = AIMNet2Calculator("aimnet2-2025", cuda_graph=False, inference_kernels=True, nb_threshold=threshold)
    reference = AIMNet2Calculator("aimnet2-2025", cuda_graph=False, inference_kernels=False, nb_threshold=threshold)
    for calc in (optimized, reference):
        calc.set_lr_cutoff(4.0)
    data = _waters()
    data["charge"].fill_(1.0)
    prepared = optimized.prepare_input(_copy(data))
    assert "_inference_atoms_per_system" not in prepared
    _forbid_streaming(monkeypatch)
    for step in range(2):
        moved = _copy(data)
        moved["coord"][:, 3:, 0] += step * 0.25
        _same(optimized(moved, forces=True), reference(moved, forces=True))


def test_rectangular_padded_packed_input_does_not_enable_streaming(monkeypatch):
    optimized = AIMNet2Calculator("aimnet2-2025", cuda_graph=False, inference_kernels=True, nb_threshold=0)
    reference = AIMNet2Calculator("aimnet2-2025", cuda_graph=False, inference_kernels=False, nb_threshold=0)
    data = _waters(batch=2)
    data["numbers"][1, -2:] = 0
    # Distinct placeholder coordinates keep this a comparison of the existing
    # packed-padding contract rather than a coincident-coordinate singularity.
    data["coord"][1, -2:] = data["coord"].new_tensor([[30.0, 0.0, 0.0], [33.0, 0.0, 0.0]])
    prepared = optimized.prepare_input(_copy(data))
    assert "_inference_atoms_per_system" not in prepared
    _forbid_streaming(monkeypatch)
    _same(optimized(data, forces=True), reference(data, forces=True))


@pytest.mark.parametrize("gradient_input", ["coord", "charge"])
def test_caller_gradient_requests_use_reference_autograd(monkeypatch, gradient_input):
    import aimnet.modules.aev_inference as aev

    optimized = AIMNet2Calculator("aimnet2-2025", cuda_graph=True, inference_kernels=True)
    reference = AIMNet2Calculator("aimnet2-2025", cuda_graph=False, inference_kernels=False)
    source = _waters()
    for _ in range(3):
        optimized(source, forces=True)
    assert optimized._cuda_graph._engines
    _forbid_streaming(monkeypatch)

    def forbidden():
        raise AssertionError("Caller gradients must not enter the first-order inference AEV")

    monkeypatch.setattr(aev, "_triton_kernel", forbidden)
    actual_input, reference_input = _copy(source), _copy(source)
    actual_input[gradient_input].requires_grad_()
    reference_input[gradient_input].requires_grad_()
    actual, expected = optimized(actual_input), reference(reference_input)
    _same(actual, expected)
    grad = torch.autograd.grad(actual["energy"].sum(), actual_input[gradient_input])[0]
    grad_ref = torch.autograd.grad(expected["energy"].sum(), reference_input[gradient_input])[0]
    torch.testing.assert_close(grad, grad_ref, atol=2e-4, rtol=2e-5)
    optimized._cuda_graph.clear()


@pytest.mark.parametrize("threshold", [0, 120], ids=["packed", "dense"])
def test_static_cache_does_not_freeze_d3_during_graph_replay(threshold):
    optimized = AIMNet2Calculator(
        "aimnet2-2025", cuda_graph=True, inference_kernels=True, cache_static=True, nb_threshold=threshold
    )
    reference = AIMNet2Calculator(
        "aimnet2-2025", cuda_graph=False, inference_kernels=False, cache_static=False, nb_threshold=threshold
    )
    data = _waters()
    assert optimized.external_dftd3 is not None
    try:
        for _ in range(3):
            _same(optimized(data, forces=True), reference(data, forces=True))
        assert len(optimized._cuda_graph._engines) == 1
        entry = next(iter(optimized._cuda_graph._engines.values()))
        initial = optimized(data, forces=True)
        energies = []
        for delta in (0.05, -0.08, 0.12):
            data["coord"][0, 1, 0] += delta
            actual = optimized(data, forces=True)
            _same(actual, reference(data, forces=True))
            assert next(iter(optimized._cuda_graph._engines.values())) is entry
            energies.append(actual["energy"])
        assert any(not torch.equal(value, initial["energy"]) for value in energies)
    finally:
        optimized._cuda_graph.clear()


def _two_caffeines(separation, packed=True):
    """One system of two caffeine molecules ``separation`` A apart along x: wider than DFT-D3's smoothing_off at 25 A."""
    from conftest import CAFFEINE_FILE, load_mol

    mol = load_mol(CAFFEINE_FILE)
    coord = torch.as_tensor(mol["coord"], dtype=torch.float32, device="cuda")
    data = {
        "coord": torch.cat((coord, coord + coord.new_tensor([separation, 0.0, 0.0]))),
        "numbers": torch.as_tensor(mol["numbers"], device="cuda").repeat(2),
        "charge": torch.zeros((), device="cuda"),
    }
    if not packed:
        data = {"coord": data["coord"][None], "numbers": data["numbers"][None], "charge": data["charge"].reshape(1)}
    return data


@pytest.mark.parametrize("packed", [True, False], ids=["N3", "1N3"])
def test_dense_graph_d3_neighbors_follow_the_regular_path_of_the_input_shape(packed):
    """The regular path packs ``[N, 3]`` input (nb_mode 1), where DFT-D3 shares the all-pairs long-range list; dense
    ``[1, N, 3]`` input (nb_mode 0) uses DFT-D3's own smoothing_off list. Coordination numbers count every listed pair,
    so a dense replay must use the list of its input shape, and the two shapes are separate layouts."""
    graph = AIMNet2Calculator("aimnet2-2025", cuda_graph=True)
    data = _two_caffeines(25.0, packed)
    try:
        for _ in range(2):
            graph(data, forces=True)
        (entry,) = graph._cuda_graph._engines.values()
        expected = graph._nblist_lr.cutoff if packed else graph.external_dftd3.smoothing_off
        assert entry.runner.engine._d3_neighbors.cutoff == expected
        other = _two_caffeines(25.0, not packed)
        for _ in range(2):
            graph(other, forces=True)
        assert len(graph._cuda_graph._engines) == 2
    finally:
        graph._cuda_graph.clear()


def test_packed_graph_replay_uses_the_regular_d3_list_numerically():
    """Replay of packed input equals the same dense computation with the regular packed path's D3 list; the
    smoothing_off list differs by ~6e-6 eV here (CN of pairs beyond 15 A), so an unfixed replay fails."""
    from aimnet.calculators.cuda_graph.cache import _packed_d3_cutoff
    from aimnet.calculators.cuda_graph.dense import StaticDenseAIMNet2

    graph = AIMNet2Calculator("aimnet2-2025", cuda_graph=True)
    data = _two_caffeines(25.0)
    try:
        for _ in range(2):
            graph(data, forces=True)
        moved = {
            **data,
            "coord": data["coord"]
            + 0.01 * torch.sin(torch.arange(data["coord"].numel(), device="cuda")).reshape_as(data["coord"]),
        }
        replay = graph(moved, forces=True)
        assert graph._cuda_graph.captures == 1
        engine = StaticDenseAIMNet2(
            graph, moved["numbers"][None], torch.zeros(1, device="cuda"), d3_cutoff=_packed_d3_cutoff(graph)
        )
        with torch.inference_mode(False):
            energy, forces, _ = engine.forward(moved["coord"][None].clone())
        torch.testing.assert_close(replay["energy"].double(), energy.double(), atol=1e-6, rtol=0.0)
        torch.testing.assert_close(replay["forces"], forces[0], atol=1e-6, rtol=0.0)
    finally:
        graph._cuda_graph.clear()


def test_inference_mode_first_call_keeps_forces_on_the_torch_fallback(monkeypatch):
    """AEV constants cached by a first call under ``torch.inference_mode()`` must not be inference tensors: the
    PyTorch fallback (no Triton) saves them for backward in a later forces call."""
    import aimnet.modules.aev_inference as aev_inference

    monkeypatch.setattr(aev_inference, "_triton_kernel", lambda: None)  # a CUDA installation without Triton
    optimized = AIMNet2Calculator("aimnet2-2025", cuda_graph=False, inference_kernels=True)
    reference = AIMNet2Calculator("aimnet2-2025", cuda_graph=False)
    data = _waters(batch=2)
    with torch.inference_mode():
        optimized(_copy(data), forces=False)
    assert not torch.is_inference(optimized._inference_aev_cache[1])
    _same(optimized(_copy(data), forces=True), reference(_copy(data), forces=True))
