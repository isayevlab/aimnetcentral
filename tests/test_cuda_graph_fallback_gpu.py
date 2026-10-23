"""GPU checks of graph fallback warnings, padded-graph reuse validation, release and capture side effects."""

import warnings

import pytest
import torch
from conftest import CAFFEINE_FILE, load_mol

from aimnet.calculators import AIMNet2Calculator
from aimnet.calculators.cuda_graph import _capture

pytestmark = [pytest.mark.gpu, pytest.mark.weights]

if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

WATER = [[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.239987, 0.927297, 0.0]]


def _water(batch=None):
    coord = torch.tensor(WATER, device="cuda")
    numbers = torch.tensor([8, 1, 1], device="cuda")
    if batch is None:
        return {"coord": coord, "numbers": numbers, "charge": torch.zeros((), device="cuda")}
    shift = 0.01 * torch.arange(batch, device="cuda").view(batch, 1, 1)
    return {
        "coord": coord.expand(batch, 3, 3) + shift,
        "numbers": numbers.expand(batch, 3).contiguous(),
        "charge": torch.zeros(batch, device="cuda"),
    }


def _graph_warnings(record):
    return [str(w.message) for w in record if "CUDA-graph replay" in str(w.message)]


@pytest.fixture(scope="module")
def regular():
    return AIMNet2Calculator("aimnet2", device="cuda")


@pytest.fixture
def graph():
    calc = AIMNet2Calculator("aimnet2", device="cuda", cuda_graph=True)
    assert calc._cuda_graph is not None
    yield calc
    calc.clear_cuda_graphs()


def _assert_close(result, reference):
    for key in reference:
        torch.testing.assert_close(result[key], reference[key], atol=2e-4, rtol=2e-5)


def test_mol_idx_input_warns_once(graph, regular):
    data = {**_water(), "mol_idx": torch.zeros(3, dtype=torch.long, device="cuda")}
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        for _ in range(3):
            result = graph(dict(data), forces=True)
    assert _graph_warnings(record) == [
        "AIMNet2Calculator: CUDA-graph replay is not used (flattened mol_idx batch input (e.g. TorchSim)); "
        "the regular code path runs instead."
    ]
    assert record[0].filename == __file__
    assert not graph._cuda_graph._engines
    _assert_close(result, regular(dict(data), forces=True))


def test_hessian_request_warns_once(graph):
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        for _ in range(2):
            graph(_water(), forces=True, hessian=True)
    assert len(_graph_warnings(record)) == 1
    assert "stress or Hessian" in _graph_warnings(record)[0]


def test_supported_calls_do_not_warn(graph):
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        for _ in range(3):
            graph(_water(), forces=True)
    assert not _graph_warnings(record)
    assert len(graph._cuda_graph._engines) == 1


def test_padded_graph_reuse_is_validated_once_per_row_count(graph, regular, monkeypatch):
    cache = graph._cuda_graph
    for _ in range(2):  # captured on the second sighting
        graph(_water(4), forces=True)
    ((key, entry),) = cache._engines.items()
    assert key[1] == 4 and entry.rows == {4}
    references = []
    original = cache._reference
    monkeypatch.setattr(cache, "_reference", lambda *a, **k: references.append(1) or original(*a, **k))
    first = graph(_water(3), forces=True)  # reuses the 4-row graph with one filler row: checked once
    assert entry.rows == {3, 4} and len(references) == 1
    second = graph(_water(3), forces=True)
    assert len(references) == 1  # later reuses replay without a regular reference
    reference = regular(_water(3), forces=True)
    _assert_close(first, reference)
    _assert_close(second, reference)


def test_failed_reuse_check_falls_back_to_the_regular_result(graph, regular, monkeypatch):
    cache = graph._cuda_graph
    for _ in range(2):
        graph(_water(4), forces=True)
    ((key, entry),) = cache._engines.items()

    def disagree(*args, **kwargs):
        raise RuntimeError("captured engine disagrees with the regular path: forces 1.00e+00 eV/A")

    with monkeypatch.context() as patch:
        patch.setattr(cache, "_compare", disagree)
        with pytest.warns(RuntimeWarning, match="disagrees") as record:
            result = graph(_water(3), forces=True)
    assert record[0].filename == __file__  # attributed to the caller, not to aimnet internals
    _assert_close(result, regular(_water(3), forces=True))
    # Only the reusing row count is affected: the validated 4-row graph keeps serving 4-row batches.
    assert cache._engines.get(key) is entry and 3 not in entry.rows
    launches = []
    original = entry.runner.launch
    monkeypatch.setattr(entry.runner, "launch", lambda *a: launches.append(1) or original(*a))
    _assert_close(graph(_water(4), forces=True), regular(_water(4), forces=True))
    assert launches == [1]
    # 3-row batches no longer reuse the 4-row graph; they stay regular until they earn their own capture.
    graph(_water(3), forces=True)
    assert launches == [1]


def test_clear_cuda_graphs_releases_the_pool(graph):
    for _ in range(2):
        graph(_water(), forces=True)
    stats = graph._cuda_graph.memory_stats()
    assert stats["engines"] == 1 and stats["pool_alive"] == 1
    graph.clear_cuda_graphs()
    stats = graph._cuda_graph.memory_stats()
    assert stats["engines"] == 0 and stats["pool_alive"] == 0 and stats["pool_reserved"] == 0
    for _ in range(2):  # captured again afterwards
        graph(_water(), forces=True)
    assert len(graph._cuda_graph._engines) == 1


def test_clear_cuda_graphs_is_a_no_op_without_graphs(regular):
    regular.clear_cuda_graphs()


def test_warm_up_returns_cached_blocks_but_does_not_synchronize(monkeypatch):
    """Blocks cached for the caller's stream cannot serve the capture stream: they are returned first (as the capture
    itself does), so warm-up memory does not stack on them. Ordering is stream-scoped, never a device sync."""
    calls = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: calls.append("synchronize"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda *a, **k: calls.append("empty_cache"))
    x = torch.ones(4, device="cuda")
    stream = _capture.warm_up(lambda t: t * 2, x, 2)
    assert calls == ["empty_cache"]
    assert stream is _capture.capture_stream(x.device)


def _caffeine_batch(rows, shift):
    molecule = load_mol(CAFFEINE_FILE)
    coord = torch.as_tensor(molecule["coord"], dtype=torch.float32, device="cuda")
    numbers = torch.as_tensor(molecule["numbers"], device="cuda")
    offsets = shift * torch.sin(torch.arange(rows * coord.numel(), device="cuda", dtype=torch.float32))
    return {
        "coord": coord.expand(rows, *coord.shape) + offsets.reshape(rows, *coord.shape),
        "numbers": numbers.expand(rows, -1).contiguous(),
        "charge": torch.zeros(rows, device="cuda"),
    }


def test_replays_from_two_streams_are_ordered():
    """Graphs share one memory pool and static input buffers: a replay launched on another stream must wait for the
    previous one. Both streams start after the same GPU delay, so without ordering the replays overlap."""
    graph = AIMNet2Calculator("aimnet2", device="cuda", cuda_graph=True)
    try:
        data_a, data_b = _caffeine_batch(64, 0.02), _caffeine_batch(64, -0.03)
        for _ in range(2):
            graph(data_a, forces=True, validate_species=False)
        assert len(graph._cuda_graph._engines) == 1
        expected_a = graph(data_a, forces=True, validate_species=False)
        expected_b = graph(data_b, forces=True, validate_species=False)
        torch.cuda.synchronize()
        streams = torch.cuda.Stream(), torch.cuda.Stream()
        delay = int(5e7)  # GPU cycles (~20 ms): longer than enqueueing both calls on the host
        for _ in range(4):
            outputs = []
            for stream, data in zip(streams, (data_a, data_b), strict=True):
                with torch.cuda.stream(stream):
                    torch.cuda._sleep(delay)
                    outputs.append(graph(data, forces=True, validate_species=False))
            torch.cuda.synchronize()  # also proves the cross-stream waits cannot deadlock
            for actual, expected in zip(outputs, (expected_a, expected_b), strict=True):
                for key in expected:
                    torch.testing.assert_close(actual[key], expected[key], atol=1e-5, rtol=1e-6)
    finally:
        graph.clear_cuda_graphs()
