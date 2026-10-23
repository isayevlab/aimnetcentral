"""Capture-time reference ordering and reuse of the validated graph output."""

import numpy as np
import pytest
import torch
from conftest import CAFFEINE_FILE, load_mol

from aimnet.calculators import AIMNet2Calculator
from aimnet.calculators.cuda_graph import dense

pytestmark = [pytest.mark.gpu, pytest.mark.weights]

if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)


@pytest.fixture
def pair():
    graph = AIMNet2Calculator("aimnet2-2025", cuda_graph=True)
    regular = AIMNet2Calculator("aimnet2-2025", cuda_graph=False)
    assert graph._cuda_graph is not None
    yield graph, regular
    graph._cuda_graph.clear()


@pytest.fixture
def batch():
    mol = load_mol(CAFFEINE_FILE)
    coord = np.stack([mol["coord"]] * 2)
    numbers = np.stack([mol["numbers"]] * 2)
    numbers[1, -2:] = 0
    coord[1, -2:] = 0.0
    return {"coord": coord, "numbers": numbers, "charge": np.zeros(2)}


def _assert_same(out, ref):
    assert out.keys() == ref.keys()
    for key in ref:
        atol, rtol = {"energy": (1e-3, 0.0), "charges": (1e-5, 0.0)}.get(key, (1e-4, 1e-5))
        torch.testing.assert_close(out[key], ref[key], atol=atol, rtol=rtol)


def test_reference_uses_capture_time_coordinates_species_and_charge(pair, batch, monkeypatch):
    graph, regular = pair
    cache = graph._cuda_graph
    graph(batch, forces=True)  # First sighting must not supply a stale reference for the next call.
    assert not cache._engines
    batch["coord"] += 0.02 * np.random.default_rng(41).normal(size=batch["coord"].shape)
    carbon = int(np.flatnonzero(batch["numbers"][0] == 6)[0])
    batch["numbers"][:, carbon] = 7
    batch["charge"][:] = [1.0, -1.0]
    expected = {key: np.array(value, copy=True) for key, value in batch.items()}
    reference = cache._reference
    seen = []

    def observed_reference(data, forces):
        seen.append({key: np.array(data[key], copy=True) for key in expected})
        return reference(data, forces)

    monkeypatch.setattr(cache, "_reference", observed_reference)
    _assert_same(graph(batch, forces=True), regular(batch, forces=True))
    assert len(cache._engines) == 1 and len(seen) == 1
    for key in expected:
        np.testing.assert_array_equal(seen[0][key], expected[key])


def test_reference_precedes_capture_and_cold_result_uses_one_replay(pair, batch, monkeypatch):
    graph, regular = pair
    cache = graph._cuda_graph
    graph(batch, forces=True)
    reference = cache._reference
    capture_begin, replay = torch.cuda.CUDAGraph.capture_begin, torch.cuda.CUDAGraph.replay
    events = []

    def observed_reference(data, forces):
        out = reference(data, forces)
        events.append("reference_complete")
        return out

    def observed_capture(self, *args, **kwargs):
        events.append("capture_begin")
        return capture_begin(self, *args, **kwargs)

    def observed_replay(self):
        events.append("replay")
        return replay(self)

    monkeypatch.setattr(cache, "_reference", observed_reference)
    monkeypatch.setattr(torch.cuda.CUDAGraph, "capture_begin", observed_capture)
    monkeypatch.setattr(torch.cuda.CUDAGraph, "replay", observed_replay)
    _assert_same(graph(batch, forces=True), regular(batch, forces=True))
    assert events == ["reference_complete", "capture_begin", "replay"]
    _assert_same(graph(batch, forces=True), regular(batch, forces=True))
    assert events == ["reference_complete", "capture_begin", "replay", "replay"]


def test_failed_capture_check_never_delivers_unchecked_output(pair, batch, monkeypatch):
    graph, regular = pair
    cache = graph._cuda_graph
    graph(batch, forces=True)
    forward = dense.StaticDenseAIMNet2.forward

    def incorrect_energy(self, coord):
        energy, forces, charges = forward(self, coord)
        return energy + 1.0, forces, charges

    monkeypatch.setattr(dense.StaticDenseAIMNet2, "forward", incorrect_energy)
    with pytest.warns(RuntimeWarning, match="disagrees with the regular path"):
        _assert_same(graph(batch, forces=True), regular(batch, forces=True))
    assert not cache._engines and cache._gen is None
    captures = cache.captures
    _assert_same(graph(batch, forces=True), regular(batch, forces=True))
    assert cache.captures == captures  # A failed layout remains regular, not repeatedly captured.


def test_validated_cold_output_survives_replay_clear_and_recapture(pair, batch):
    graph, regular = pair
    graph(batch, forces=True)
    output = graph(batch, forces=True)  # The new graph's validated output is returned directly, but cloned.
    _assert_same(output, regular(batch, forces=True))
    saved = {key: value.clone() for key, value in output.items()}
    batch["coord"] += 0.02 * np.random.default_rng(42).normal(size=batch["coord"].shape)
    _assert_same(graph(batch, forces=True), regular(batch, forces=True))
    graph._cuda_graph.clear()
    for _ in range(2):
        _assert_same(graph(batch, forces=True), regular(batch, forces=True))
    assert len(graph._cuda_graph._engines) == 1
    for key in output:
        torch.testing.assert_close(output[key], saved[key], atol=0.0, rtol=0.0)


def test_rejected_capture_returns_its_reference_without_recomputing(pair, batch, monkeypatch):
    graph, regular = pair
    cache = graph._cuda_graph
    graph(batch, forces=True)
    cache.cache_memory_fraction = 0.0  # every capture is over budget
    forwards = []
    process_output = type(graph).process_output  # once per regular evaluation (no hook: hooks disable capture)

    def counted(self, data):
        forwards.append(1)
        return process_output(self, data)

    monkeypatch.setattr(graph, "process_output", counted.__get__(graph))
    with pytest.warns(RuntimeWarning, match="over the cache budget"):
        out = graph(batch, forces=True)
    _assert_same(out, regular(batch, forces=True))
    assert len(forwards) == 1  # the capture-time reference is the answer; the regular path is not run again
    assert not cache._engines and cache._gen is None and cache._answer is None
