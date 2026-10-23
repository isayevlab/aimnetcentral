"""Memory lifecycle of the CUDA-graph cache (``aimnet.calculators.cuda_graph.cache``).

The cached graphs of a calculator share one memory pool owned by the cache. The pool is created by the first
capture, accounted when a graph is captured or rebuilt, and released by ``clear``, a model change, an over-budget
capture or rebuild, and a failed capture. Pool memory is checked by pool id in the allocator snapshot, independently of
the cache's own accounting; replays and captures are counted with spies on ``torch.cuda.CUDAGraph``.

All tests in this module require CUDA and are marked with @pytest.mark.gpu.
"""

import gc
import warnings

import numpy as np
import pytest
import torch
from conftest import CAFFEINE_FILE, load_mol

from aimnet.calculators import AIMNet2Calculator
from aimnet.calculators.cuda_graph import cache as cache_module
from aimnet.calculators.cuda_graph import dense, sparse

pytestmark = [pytest.mark.gpu, pytest.mark.weights]

if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

MiB = 2**20
MODEL = "aimnet2-2025"
TIGHT_BUDGET = 0.005 * 6 * 2**30  # 30.72 MiB: cache_memory_fraction=0.005 on the 6 GiB GPU of the review
SEGMENT = 2 * MiB  # allocator granularity of small segments: tolerance for "no growth" between settled snapshots


def _settle() -> None:
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()


def _live_pools() -> set:
    """Ids of private (graph) memory pools that still own segments."""
    return {tuple(s["segment_pool_id"]) for s in torch.cuda.memory_snapshot()} - {(0, 0)}


def _no_new_pool(before: set) -> bool:
    """No pool other than those in ``before`` owns segments (pools of earlier calculators may be released meanwhile)."""
    return _live_pools() <= before


def _pool_bytes(pid) -> int:
    return sum(s["total_size"] for s in torch.cuda.memory_snapshot() if tuple(s["segment_pool_id"]) == tuple(pid))


def _set_budget(calc, nbytes: float) -> None:
    cache = calc._cuda_graph
    cache.cache_memory_fraction = nbytes / torch.cuda.get_device_properties(cache._device).total_memory


def _assert_close_all(out: dict, ref: dict) -> None:
    """Every key, at the regular path's run-to-run level on CUDA (as in test_cuda_graph.py)."""
    assert out.keys() == ref.keys()
    for k in ref:
        assert out[k].shape == ref[k].shape, k
        atol, rtol = {"energy": (1e-3, 0.0), "charges": (1e-5, 0.0)}.get(k, (1e-4, 1e-5))
        torch.testing.assert_close(out[k], ref[k], atol=atol, rtol=rtol)


def _jitter(data: dict, step: int) -> dict:
    coord = np.asarray(data["coord"])
    return {**data, "coord": coord + 0.02 * np.random.default_rng(step).normal(size=coord.shape)}


def _batch(mol: dict, b: int) -> dict:
    """[b, N, 3] dense batch of ``mol``; row 1 has two ``Z = 0`` padding atoms."""
    coord, numbers = np.stack([mol["coord"]] * b), np.stack([mol["numbers"]] * b)
    if b > 1:
        numbers[1, -2:], coord[1, -2:] = 0, 0.0
    return _jitter({"coord": coord, "numbers": numbers, "charge": [0.0] * b}, b)


def _water_cluster(n_side: int, spacing: float = 3.1, seed: int = 0) -> dict:
    """n_side**3 randomly oriented waters on a cubic grid (192 atoms for n_side=4: neighbor-list engine)."""
    rng = np.random.default_rng(seed)
    ang = np.deg2rad(104.52)
    water = np.array([[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [0.9572 * np.cos(ang), 0.9572 * np.sin(ang), 0.0]])
    coords = []
    for idx in np.ndindex(n_side, n_side, n_side):
        rot, r = np.linalg.qr(rng.normal(size=(3, 3)))
        coords.append(water @ (rot * np.sign(np.diag(r))).T + spacing * np.asarray(idx, dtype=float))
    return {"coord": np.concatenate(coords), "numbers": np.tile([8, 1, 1], n_side**3), "charge": 0.0}


@pytest.fixture(scope="module")
def caffeine():
    return load_mol(CAFFEINE_FILE)


@pytest.fixture
def make_pair():
    """Fresh (graph, regular) calculators; the graph calculator's cache is cleared at teardown."""
    made = []

    def make():
        graph = AIMNet2Calculator(MODEL, cuda_graph=True)
        regular = AIMNet2Calculator(MODEL, cuda_graph=False)
        assert graph._cuda_graph is not None
        made.append(graph)
        return graph, regular

    yield make
    for graph in made:
        graph._cuda_graph.clear()
    made.clear()
    _settle()


@pytest.fixture
def graph_calls(monkeypatch):
    """Numbers of CUDA-graph replays and captures during the test."""
    calls = {"replay": 0, "capture": 0}
    replay, capture_begin = torch.cuda.CUDAGraph.replay, torch.cuda.CUDAGraph.capture_begin

    def counted_replay(self):
        calls["replay"] += 1
        return replay(self)

    def counted_capture_begin(self, *args, **kwargs):
        calls["capture"] += 1
        return capture_begin(self, *args, **kwargs)

    monkeypatch.setattr(torch.cuda.CUDAGraph, "replay", counted_replay)
    monkeypatch.setattr(torch.cuda.CUDAGraph, "capture_begin", counted_capture_begin)
    return calls


class TestPoolLifetime:
    def test_construction_and_regular_calls_create_no_pool(self, make_pair, caffeine, graph_calls):
        before = _live_pools()
        graph, regular = make_pair()
        cache = graph._cuda_graph
        n = len(caffeine["numbers"])
        mol_idx = {
            "coord": np.concatenate([caffeine["coord"], caffeine["coord"] + 20.0]),
            "numbers": np.concatenate([caffeine["numbers"]] * 2),
            "charge": [0.0, 0.0],
            "mol_idx": np.repeat([0, 1], n),
        }
        for data in (caffeine, mol_idx, mol_idx):  # first sighting, then an input the cache does not handle
            _assert_close_all(graph(data, forces=True), regular(data, forces=True))
        _settle()
        assert cache._gen is None and cache.generations == 0 and graph_calls["capture"] == 0
        assert cache.memory_stats()["retained"] == 0
        assert _no_new_pool(before)

    def test_clear_releases_the_pool_of_a_live_calculator(self, make_pair, caffeine, graph_calls):
        graph, regular = make_pair()
        cache = graph._cuda_graph
        data = _batch(caffeine, 32)
        for step in range(3):
            d = _jitter(data, step)
            _assert_close_all(graph(d, forces=True), regular(d, forces=True))
        assert graph_calls["capture"] == 1 and graph_calls["replay"] == 2  # validated capture output + warm replay
        pid = cache._gen.pool.id
        _settle()
        held, reserved = _pool_bytes(pid), torch.cuda.memory_reserved()
        assert held > 0 and cache.memory_stats()["pool_reserved"] == held
        cache.clear()
        _settle()
        assert _pool_bytes(pid) == 0 and pid not in _live_pools()  # nothing pins any segment of the pool
        assert torch.cuda.memory_reserved() <= reserved - held
        assert cache._gen is None and cache.memory_stats()["retained"] == 0
        for step in range(3):  # the calculator stays usable
            d = _jitter(data, 10 + step)
            _assert_close_all(graph(d, forces=True), regular(d, forces=True))
        assert graph_calls["capture"] == 2

    def test_recapture_after_clear_uses_a_fresh_pool(self, make_pair, caffeine, graph_calls):
        graph, regular = make_pair()
        cache = graph._cuda_graph
        data = _batch(caffeine, 4)
        pids = []
        for cycle in range(3):
            replays = graph_calls["replay"]
            for step in range(4):  # regular, capture, replay, replay
                d = _jitter(data, 10 * cycle + step)
                _assert_close_all(graph(d, forces=True), regular(d, forces=True))
            assert graph_calls["replay"] - replays == 3
            pids.append(cache._gen.pool.id)
            cache.clear()
            _settle()
            assert pids[-1] not in _live_pools()
        assert len(set(pids)) == 3 and graph_calls["capture"] == 3

    def test_graphs_of_several_layouts_share_one_pool_and_retire_together(self, make_pair, caffeine, graph_calls):
        graph, regular = make_pair()
        cache = graph._cuda_graph
        layouts = [  # dense single E+F and E, padded B4 and B2 (no capacity reuse between them), sparse N=192
            (caffeine, True),
            (caffeine, False),
            (_batch(caffeine, 4), True),
            (_batch(caffeine, 2), True),
            (_batch(caffeine, 2), False),
            (_water_cluster(4), True),
        ]
        for data, forces in layouts:
            for _ in range(2):
                graph(data, forces=forces)
        gen = cache._gen
        assert len(cache._engines) == len(layouts) and all(e.gen is gen for e in cache._engines.values())
        assert cache.memory_stats()["pool_reserved"] == _pool_bytes(gen.pool.id) > 0
        replays = graph_calls["replay"]
        for step in range(3):  # alternate between the graphs
            for data, forces in layouts:
                d = _jitter(data, step)
                _assert_close_all(graph(d, forces=forces), regular(d, forces=forces))
        assert graph_calls["replay"] - replays == 3 * len(layouts)
        pid = gen.pool.id
        cache.clear()
        _settle()
        assert pid not in _live_pools()
        for data, forces in layouts[:2]:
            for _ in range(3):
                _assert_close_all(graph(data, forces=forces), regular(data, forces=forces))
        assert cache._gen is not None and cache._gen is not gen and cache._gen.pool.id != pid

    def test_model_change_retires_the_pool(self, make_pair, caffeine, graph_calls):
        graph, regular = make_pair()
        cache = graph._cuda_graph
        for _ in range(2):
            graph(caffeine, forces=True)
        pid = cache._gen.pool.id
        for calc in (graph, regular):
            with torch.no_grad():
                calc.model.mlps[0][0].weight.mul_(1.01)
        replays = graph_calls["replay"]
        _assert_close_all(graph(caffeine, forces=True), regular(caffeine, forces=True))
        assert graph_calls["replay"] == replays + 1  # launched before the contract check, discarded unread
        _settle()
        assert cache._gen is None and pid not in _live_pools()
        for step in range(3):  # regular, capture, replay: the changed model is recaptured
            d = _jitter(caffeine, step)
            _assert_close_all(graph(d, forces=True), regular(d, forces=True))
        assert cache._gen is not None and cache._gen.pool.id != pid

    def test_outputs_survive_replay_clear_and_recapture(self, make_pair, caffeine):
        graph, _ = make_pair()
        data = _batch(caffeine, 4)
        outs = [graph(_jitter(data, step), forces=True) for step in range(3)]  # regular, capture, replay
        kept = [{k: v.clone() for k, v in o.items()} for o in outs]
        graph(_jitter(data, 7), forces=True)
        graph._cuda_graph.clear()
        _settle()
        for step in range(3):
            graph(_jitter(data, 10 + step), forces=True)  # recaptured into a new pool
        for o, k in zip(outs, kept, strict=True):
            for key in k:
                assert torch.equal(o[key], k[key])

    def test_construct_capture_delete_cycles_reach_a_plateau(self, caffeine):
        before = _live_pools()
        allocated, reserved = [], []
        for _ in range(4):
            calc = AIMNet2Calculator(MODEL, cuda_graph=True)
            for data in (caffeine, _batch(caffeine, 4), _water_cluster(4)):
                for _ in range(3):
                    calc(data, forces=True)
            assert len(calc._cuda_graph._engines) == 3
            del calc
            _settle()
            allocated.append(torch.cuda.memory_allocated())
            reserved.append(torch.cuda.memory_reserved())
            assert _no_new_pool(before)
        # the first cycle may load process-wide state (kernels, cuBLAS workspaces); later cycles must not grow
        assert max(allocated[1:]) <= allocated[1] + MiB
        assert max(reserved[1:]) <= reserved[1] + SEGMENT

    def test_warm_replay_does_not_walk_the_allocator(self, make_pair, caffeine, monkeypatch, graph_calls):
        graph, _ = make_pair()
        layouts = (caffeine, _batch(caffeine, 4), _water_cluster(4))
        for data in layouts:
            for _ in range(2):
                graph(data, forces=True)
        walks = []
        snapshot = torch.cuda.memory_snapshot

        def counted_snapshot(*args, **kwargs):
            walks.append(args)
            return snapshot(*args, **kwargs)

        monkeypatch.setattr(torch.cuda, "memory_snapshot", counted_snapshot)
        replays = graph_calls["replay"]
        for step in range(5):
            for data in layouts:
                graph(_jitter(data, step), forces=True)
        assert graph_calls["replay"] - replays == 5 * len(layouts) and not walks

    def test_eviction_keeps_the_pool_for_later_captures(self, make_pair, caffeine, graph_calls):
        """The LRU limit drops engines, not the pool: an evicted layout is recaptured into the same pool."""
        graph, regular = make_pair()
        cache = graph._cuda_graph
        cache.cache_size = 2
        layouts = [_batch(caffeine, 4), _batch(caffeine, 2), caffeine]
        for data in layouts:
            for _ in range(2):
                graph(data, forces=True)
        pid = cache._gen.pool.id
        assert len(cache._engines) == 2 and graph_calls["capture"] == 3 and cache.generations == 1
        _assert_close_all(graph(layouts[0], forces=True), regular(layouts[0], forces=True))  # evicted: recaptured
        assert graph_calls["capture"] == 4 and cache._gen.pool.id == pid and cache.generations == 1
        assert cache.memory_stats()["pool_reserved"] == _pool_bytes(pid)


class TestBudget:
    def test_over_budget_capture_is_rejected_and_its_pool_released(self, make_pair, caffeine, graph_calls):
        graph, regular = make_pair()
        cache = graph._cuda_graph
        _set_budget(graph, TIGHT_BUDGET)
        before = _live_pools()
        data = _batch(caffeine, 64)
        with pytest.warns(RuntimeWarning, match="over the cache budget"):
            for step in range(3):
                d = _jitter(data, step)
                _assert_close_all(graph(d, forces=True), regular(d, forces=True))
        _settle()
        assert graph_calls["capture"] == 1 and not cache._engines and cache._gen is None
        assert cache.memory_stats()["retained"] == 0 and _no_new_pool(before)

    def test_rejected_layout_is_not_recaptured(self, make_pair, caffeine, graph_calls):
        graph, regular = make_pair()
        _set_budget(graph, TIGHT_BUDGET)
        data = _batch(caffeine, 64)
        reserved = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for step in range(10):
                d = _jitter(data, step)
                _assert_close_all(graph(d, forces=True), regular(d, forces=True))
                _settle()
                reserved.append(torch.cuda.memory_reserved())
        assert graph_calls["capture"] == 1 and graph_calls["replay"] == 1  # the capture-time check only
        assert max(reserved[2:]) <= reserved[2] + SEGMENT

    def test_smaller_layout_captures_after_a_rejection(self, make_pair, caffeine, graph_calls):
        graph, regular = make_pair()
        cache = graph._cuda_graph
        _set_budget(graph, TIGHT_BUDGET)
        big = _batch(caffeine, 64)
        with pytest.warns(RuntimeWarning, match="over the cache budget"):
            for _ in range(2):
                graph(big, forces=True)
        assert cache._gen is None
        replays = graph_calls["replay"]
        for step in range(4):  # regular, capture, replay, replay
            d = _jitter(caffeine, step)
            _assert_close_all(graph(d, forces=True), regular(d, forces=True))
        assert len(cache._engines) == 1 and graph_calls["replay"] - replays == 3
        assert 0 < cache.memory_stats()["retained"] <= cache._budget()
        _assert_close_all(graph(big, forces=True), regular(big, forces=True))  # still on the regular path
        assert graph_calls["capture"] == 2

    def test_admission_counts_the_shared_pool(self, make_pair, caffeine):
        graph, regular = make_pair()
        cache = graph._cuda_graph
        for data in (_batch(caffeine, 8), _batch(caffeine, 4)):
            for _ in range(2):
                graph(data, forces=True)
        stats = cache.memory_stats()
        pid = cache._gen.pool.id
        assert stats["engines"] == 2 and stats["pool_reserved"] == _pool_bytes(pid) > 0
        assert stats["retained"] == stats["pool_reserved"] + stats["engine_buffers"] and stats["engine_buffers"] > 0
        _set_budget(graph, stats["retained"] - 1)  # what the two graphs retain no longer fits: the next capture
        small = _batch(caffeine, 2)
        with pytest.warns(RuntimeWarning, match="over the cache budget"):
            for _ in range(2):
                _assert_close_all(graph(small, forces=True), regular(small, forces=True))
        _settle()
        assert cache._gen is None and not cache._engines and pid not in _live_pools()


class TestSparseRebuild:
    @pytest.mark.parametrize("budget", ["normal", "tight"])
    def test_rebuild_is_accounted(self, make_pair, graph_calls, budget):
        graph, regular = make_pair()
        cache = graph._cuda_graph
        data, compressed = _water_cluster(4), _water_cluster(4, spacing=2.6)  # ~1.7x the neighbors
        for _ in range(2):
            graph(data, forces=True)
        (entry,) = cache._engines.values()
        before, pid = cache.memory_stats(), cache._gen.pool.id
        if budget == "normal":
            _assert_close_all(graph(compressed, forces=True), regular(compressed, forces=True))
            after = cache.memory_stats()
            assert entry.runner.rebuilds == 1 and after["captures"] == before["captures"] + 1
            assert after["pool_reserved"] == _pool_bytes(pid) and after["retained"] > before["retained"]
            replays = graph_calls["replay"]
            _assert_close_all(graph(compressed, forces=True), regular(compressed, forces=True))
            assert graph_calls["replay"] == replays + 1
            return
        _set_budget(graph, before["retained"])  # the rebuilt engine retains more
        with pytest.warns(RuntimeWarning, match="rebuilt sparse engine exceeds the cache memory budget"):
            out = graph(compressed, forces=True)
        _assert_close_all(out, regular(compressed, forces=True))
        assert entry.runner.rebuilds == 1
        del entry
        _settle()
        assert cache._gen is None and not cache._engines and pid not in _live_pools()
        captures = graph_calls["capture"]
        for _ in range(2):  # the layout stays on the regular path
            _assert_close_all(graph(compressed, forces=True), regular(compressed, forces=True))
        assert graph_calls["capture"] == captures

    def test_failed_rebuild_releases_the_pool(self, make_pair, monkeypatch):
        graph, regular = make_pair()
        cache = graph._cuda_graph
        data, compressed = _water_cluster(4), _water_cluster(4, spacing=2.6)
        for _ in range(2):
            graph(data, forces=True)
        pid = cache._gen.pool.id

        def no_rebuild(*args, **kwargs):
            raise RuntimeError("injected rebuild failure")

        with monkeypatch.context() as m:
            m.setattr(sparse.GraphedSparseEF, "__init__", no_rebuild)
            with pytest.warns(RuntimeWarning, match="injected rebuild failure"):
                _assert_close_all(graph(compressed, forces=True), regular(compressed, forces=True))
        _settle()
        assert cache._gen is None and not cache._engines and pid not in _live_pools()
        for _ in range(2):
            _assert_close_all(graph(compressed, forces=True), regular(compressed, forces=True))
        assert not cache._engines  # the layout stays on the regular path


class TestFailedCaptures:
    def test_pool_creation_failure(self, make_pair, caffeine, monkeypatch, graph_calls):
        graph, regular = make_pair()
        cache = graph._cuda_graph

        def no_pool(self, device):
            raise RuntimeError("injected: no memory pool")

        with monkeypatch.context() as m:
            m.setattr(cache_module._Generation, "__init__", no_pool)
            with pytest.warns(RuntimeWarning, match="injected: no memory pool"):
                for _ in range(3):
                    _assert_close_all(graph(caffeine, forces=True), regular(caffeine, forces=True))
        assert cache._gen is None and not cache._engines and graph_calls["capture"] == 0
        data = _batch(caffeine, 4)
        for _ in range(3):  # regular, capture (validated output), warm replay
            _assert_close_all(graph(data, forces=True), regular(data, forces=True))
        assert len(cache._engines) == 1 and graph_calls["replay"] == 2

    @pytest.mark.parametrize("stage", ["warm-up", "capture"])
    def test_engine_failure_leaves_no_pool(self, make_pair, caffeine, monkeypatch, graph_calls, stage):
        graph, regular = make_pair()
        cache = graph._cuda_graph
        before = _live_pools()
        forward = dense.StaticDenseAIMNet2.forward

        def failing(self, coord):
            if stage == "capture" and not torch.cuda.is_current_stream_capturing():
                return forward(self, coord)
            raise RuntimeError(f"injected {stage} failure")

        with monkeypatch.context() as m:
            m.setattr(dense.StaticDenseAIMNet2, "forward", failing)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                for _ in range(3):
                    _assert_close_all(graph(caffeine, forces=True), regular(caffeine, forces=True))
        assert any(f"injected {stage} failure" in str(w.message) for w in caught)
        _settle()
        assert cache._gen is None and not cache._engines and _no_new_pool(before)
        data = _batch(caffeine, 4)
        for _ in range(3):
            _assert_close_all(graph(data, forces=True), regular(data, forces=True))
        assert len(cache._engines) == 1 and graph_calls["replay"] == 2

    def test_parity_failure_keeps_the_pool_of_other_graphs(self, make_pair, caffeine, monkeypatch, graph_calls):
        graph, regular = make_pair()
        cache = graph._cuda_graph
        data = _batch(caffeine, 4)
        for _ in range(2):
            graph(data, forces=True)
        pid = cache._gen.pool.id
        forward = dense.StaticDenseAIMNet2.forward

        def off_by_one_percent(self, coord):
            e, f, q = forward(self, coord)
            return e, f * 1.01, q

        with monkeypatch.context() as m:
            m.setattr(dense.StaticDenseAIMNet2, "forward", off_by_one_percent)
            with pytest.warns(RuntimeWarning, match="disagrees with the regular path"):
                for _ in range(3):
                    _assert_close_all(graph(caffeine, forces=True), regular(caffeine, forces=True))
        assert cache._gen is not None and cache._gen.pool.id == pid and len(cache._engines) == 1
        assert cache.memory_stats()["pool_reserved"] == _pool_bytes(pid)  # growth of the failed capture included
        replays = graph_calls["replay"]
        _assert_close_all(graph(data, forces=True), regular(data, forces=True))
        assert graph_calls["replay"] == replays + 1

    def test_fatal_cuda_error_is_raised(self, make_pair, caffeine, monkeypatch):
        graph, _ = make_pair()

        def fatal(self, coord):
            raise RuntimeError("CUDA error: an illegal memory access was encountered")

        with monkeypatch.context() as m:
            m.setattr(dense.StaticDenseAIMNet2, "forward", fatal)
            graph(caffeine, forces=True)  # first sighting: regular path
            with pytest.raises(RuntimeError, match="illegal memory access"):
                graph(caffeine, forces=True)


class TestAccountingHelpers:
    def test_failure_descriptions(self):
        assert cache_module._failure(ValueError("bad layout")) == "ValueError: bad layout"
        for text in ("CUDA error: an illegal memory access was encountered", "device-side assert triggered"):
            with pytest.raises(RuntimeError, match=text):
                cache_module._failure(RuntimeError(text))

    def test_pool_measure_without_a_per_pool_snapshot(self, make_pair, caffeine, monkeypatch):
        graph, _ = make_pair()
        for _ in range(2):
            graph(caffeine, forces=True)
        gen = graph._cuda_graph._gen
        expected = sum(int(s["total_size"]) for s in gen.measure())
        snapshot = torch.cuda.memory_snapshot

        def full_snapshot_only(*args, **kwargs):
            if args or kwargs:
                raise TypeError("memory_snapshot() takes no arguments")
            return snapshot()

        monkeypatch.setattr(torch.cuda, "memory_snapshot", full_snapshot_only)
        assert sum(int(s["total_size"]) for s in gen.measure()) == expected == gen.pool_bytes > 0
