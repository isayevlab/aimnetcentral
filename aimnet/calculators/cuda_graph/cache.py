"""CUDA-graph replay of repeated ``AIMNet2Calculator.eval`` calls.

``AIMNet2Calculator.eval`` asks ``CUDAGraphCache.eval`` first; ``None`` means "not handled" and the regular code path
runs unchanged.

Handled calls (all of):

* CUDA device; closed-shell AIMNet2 model (``d2features``, one charge channel) with the default long-range setup:
  full ``simple`` Coulomb over all pairs with embedded short range (or no external Coulomb) and optional external
  DFT-D3; not in training or deterministic mode;
* stock GELU modules/forward methods, without hooks or trainable parameters; module state changes invalidate graphs;
* input keys ``coord``, ``numbers``, ``charge`` and optionally ``mult`` (ignored by closed-shell models, as on the
  regular path); no ``cell``, ``mol_idx``, ``nbmat``, ...; no input tensor with ``requires_grad``; a single molecule
  ``[N, 3]`` without padding atoms, or a batch ``[B, N, 3]``; in dense mode (``N <= nb_threshold``) the batch may
  mix molecule sizes with ``Z = 0`` padding atoms;
* ``stress=False`` and ``hessian=False``.

Policy:

* single molecules and neighbor-list batches: a layout (species, charges, ``B``, ``N``, ``forces``, calculator
  settings) is left to the regular path the first time it is seen; from the second call on, a static-layout engine is
  built (~0.05-0.2 s) and replayed as one CUDA graph. A layout evaluated once costs what it costs on the regular path;
  repeated layouts (MD, optimization, scans) are replayed;
* dense batches (``B > 1``, padded or not): species and charges are inputs of the graph, so a graph is keyed by its
  shape (rows, ``N``, ``forces``, settings) and serves every composition; mixed-size batches and batches whose
  molecules change between calls replay without new captures. A batch of ``B`` uses a graph recorded for exactly
  ``B`` rows, else the smallest recorded one with at most ``max(1, B // 3)`` extra rows (copies of molecule 0,
  computed and discarded), else it records ``B`` rows once ``B`` has been seen twice. A batch that shrinks as
  molecules converge records a few graphs (64 -> 1: ten) instead of one per batch size;
* ``N <= nb_threshold`` uses the dense engine (``dense.py``), larger systems the neighbor-list engine (``sparse.py``);
* each new or capacity-rebuilt engine is checked against the regular path on the input that triggered the capture
  (a dense batch graph is then reused for other compositions); an engine that disagrees beyond float32 noise is not
  used;
* before a replay, captured allocation addresses and sizes are checked on the CPU. The full model contract
  (``contracts.py``) is checked on every call while the GPU replays; a changed model's result is discarded unread
  (the regular path answers). Detached views keep replaced parameter/buffer storage alive, while the address/size
  guard prevents replay after ``resize_`` reallocates a still-live storage;
* a failure while building, checking or running an engine sends that layout back to the regular path (warned once);
  CUDA errors that leave the context unusable (illegal address, device-side assert, ...) are raised.

Memory:

* the graphs of a calculator share one memory pool (a *generation*, ``_Generation``), created by the first capture:
  graphs are replayed one at a time and their outputs are copied before another replay, so their intermediates can
  share memory. The pool holds what its graphs use and keeps the blocks of dropped graphs reserved for later captures;
* the cache retains the pool's reserved segments plus the engines' own buffers outside it (static inputs, masks,
  neighbor buffers), both measured from the allocator when a graph is captured or rebuilt (never on replay). An
  engine is admitted only if this total stays within ``cache_memory_fraction`` of the device memory; otherwise its
  layout stays on the regular path and the generation is retired, which returns its pool to the device. Model
  weights, returned outputs and the temporary peak of a capture are not part of the budget;
* at most ``cache_size`` engines are kept (least recently used dropped first). Dropping an engine frees its buffers
  and returns its blocks to the pool for later captures; the pool reservation itself is returned only when the
  generation retires: ``clear``, a model change, an over-budget capture or rebuild, a capture that runs out of memory,
  a failed capture that leaves no engine, or deletion of the calculator.

Outputs have the regular path's keys, shapes and dtypes: ``energy`` ``[B]`` (``[1]`` for ``[N, 3]`` input),
``charges`` ``[B, N]`` (``[N]``), ``forces`` ``[B, N, 3]`` (``[N, 3]``, only when ``forces=True``).
"""

from __future__ import annotations

import collections
import contextlib
import math
import warnings
import weakref
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from aimnet.modules.lr import DFTD3, LRCoulomb

from .._graph_options import (
    cuda_graph_dependency_problem,
    cuda_graph_enabled,
    cuda_graph_platform_problem,
    external_stacklevel,
)
from .contracts import ModelContract
from .dense import GraphedEF, StaticDenseAIMNet2, check_dense_model
from .sparse import AutoGraphedSparseEF, check_sparse_model

_REQUIRED_KEYS = frozenset({"coord", "numbers", "charge"})
_ALLOWED_KEYS = _REQUIRED_KEYS | {"mult"}
_MAX_TRACKED_LAYOUTS = 1024  # layouts remembered for the "seen before" count
_REGULAR_ONLY = -1  # count of a layout that stays on the regular path
# Capture-time check against the regular path. FP32 noise measured on an L40S (Torch 2.13, Warp 1.16, nvalchemi
# 0.5.0, inference_kernels off and on; water, caffeine, taxol dense and sparse, padded dense batches up to 32x48,
# 960- and 4992-atom systems; at the capture geometry and after 0.02-0.2 A random displacements, |F|max <= 99 eV/A):
#   regular vs regular (repeat call or second calculator): energy <= 2.2e-7 eV/atom, forces <= 1.4e-4 eV/A,
#   charges <= 8.3e-7 e;
#   graph vs regular: energy <= 3.8e-7 eV/atom, charges <= 8.9e-7 e, forces <= 1.5e-5 eV/A up to 113 atoms,
#   2.3e-5 at 960 atoms and 2.2e-4 at 4992 atoms (2.2e-6 of |F|max): the force noise grows with the atom count
#   (long-range sums), hence the per-atom force term.
# The bounds keep a 5-10x-plus margin over these worst cases (old bounds: 1e-5 eV/atom, 1e-3 eV/A + 1e-5 |F|max,
# 1e-4 e), so only a genuinely wrong capture is rejected; a false rejection costs speed, never correctness.
_CHECK_E_PER_ATOM = 5e-6  # eV
_CHECK_F = 1e-4  # eV/A, plus _CHECK_F_REL of the largest force component and _CHECK_F_PER_ATOM per atom
_CHECK_F_REL = 1e-5
_CHECK_F_PER_ATOM = 1e-7  # eV/A
_CHECK_Q = 2e-5  # e
# CUDA errors after which the context is unusable: raised, never answered by the regular path
_FATAL_CUDA_ERRORS = (
    "illegal memory access",
    "device-side assert",
    "unspecified launch failure",
    "misaligned address",
    "illegal instruction",
)
_PACKAGE = __name__.rpartition(".")[0]
_UNSUPPORTED_MODEL = "the model is outside the graph contract (hooks, trainable parameters, custom modules or outputs)"


def _failure(exc: Exception) -> str:
    """Description of a recoverable failure; re-raises CUDA errors that leave the context unusable."""
    text = str(exc)
    if any(marker in text.lower() for marker in _FATAL_CUDA_ERRORS):
        raise exc
    return f"{type(exc).__name__}: {text}"


class _Generation:
    """Owner of the graph memory pool (``torch.cuda.MemPool``) the cached graphs are captured into.

    PyTorch keeps a pool while its owner or a graph captured into it is alive, so graphs can be dropped and
    recaptured into the same pool. ``retire`` drops the owner; once its graphs are gone too, the pool's segments are
    released (``empty_cache`` returns them to the device). A retired generation is never captured into again."""

    __slots__ = ("pool", "pool_bytes")

    def __init__(self, device: torch.device):
        with torch.cuda.device(device):
            self.pool: torch.cuda.MemPool | None = torch.cuda.MemPool()
        self.pool_bytes = 0  # reserved segments, as of the last ``measure``

    @property
    def id(self) -> tuple[int, int]:
        if self.pool is None:
            raise RuntimeError("the CUDA-graph memory pool was retired")
        return self.pool.id

    def measure(self) -> list:
        """The pool's segments from the allocator snapshot (capture / rebuild / diagnostic boundaries only)."""
        if self.pool is None:
            self.pool_bytes = 0
            return []
        pid = tuple(self.pool.id)
        try:
            segments = torch.cuda.memory_snapshot(pid, include_traces=False)
        except TypeError:  # no per-pool snapshot: filter the full one
            segments = torch.cuda.memory_snapshot()
        segments = [s for s in segments if tuple(s.get("segment_pool_id", ())) == pid]
        self.pool_bytes = sum(int(s["total_size"]) for s in segments)
        return segments

    def retire(self) -> None:
        self.pool = None
        self.pool_bytes = 0


def _owned_bytes(runner, segments: list, exclude: set) -> int:
    """Bytes of CUDA tensors an engine holds outside its graph pool (static inputs, masks, neighbor buffers).

    Follows the runner, the graph wrappers and engines of this package and their containers; storages inside the
    pool ``segments`` (graph outputs and capture allocations) and in ``exclude`` (model parameters and buffers) are
    not counted. Each storage counts once."""
    ranges = [(int(s["address"]), int(s["address"]) + int(s["total_size"])) for s in segments]
    seen, ptrs, total, stack = set(), set(exclude), 0, [runner]
    while stack:
        obj = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if isinstance(obj, torch.Tensor):
            if obj.is_cuda:
                storage = obj.untyped_storage()
                ptr = storage.data_ptr()
                if ptr not in ptrs and not any(lo <= ptr < hi for lo, hi in ranges):
                    ptrs.add(ptr)
                    total += storage.nbytes()
        elif isinstance(obj, dict):
            stack.extend(obj.values())
        elif isinstance(obj, (list, tuple)):
            stack.extend(obj)
        elif type(obj).__module__.startswith(_PACKAGE) or type(obj).__module__ == "aimnet.calculators.neighbor_buffers":
            stack.extend(getattr(obj, "__dict__", {}).values())  # (a pool owner has slots and no tensors)
    return total


def _packed_d3_cutoff(calc) -> float | None:
    """Cutoff of the neighbor list the regular path gives DFT-D3 for packed ``[N, 3]`` input (nb_mode 1).

    Dense ``[B, N, 3]`` input uses DFT-D3's own ``smoothing_off`` list (nb_mode 0), but packed input shares the
    calculator's long-range list (all pairs with full Coulomb, the D3 cutoff without Coulomb). Coordination numbers
    count every listed pair, so a dense replay of packed input must use that list."""
    for nblist in (getattr(calc, "_nblist_lr", None), getattr(calc, "_nblist_dftd3", None)):
        if nblist is not None:
            return float(nblist.cutoff)
    return None


def _max_filler_rows(b: int) -> int:
    """Filler rows a batch of ``b`` may carry when it reuses a graph recorded for a larger batch (a third of ``b``,
    at least one): a batch that shrinks as molecules converge reuses a few captures instead of one per batch size."""
    return max(1, b // 3)


class _Entry:
    __slots__ = ("gen", "keepalive", "no_reuse", "owned", "rebuilds", "rows", "runner", "storages", "validated")

    def __init__(self, runner, gen: _Generation, keepalive: tuple = ()):
        self.runner, self.gen = runner, gen
        self.owned = 0  # engine bytes outside the pool, measured at admission
        self.rebuilds = getattr(runner, "rebuilds", 0)
        self.validated = False  # species / charge validation passed for this layout (its content is in the key)
        self.rows: set[int] = set()  # dense batch sizes checked against the regular path on this padded graph
        self.no_reuse: set[int] = set()  # smaller batch sizes whose reuse check failed: never reuse this graph for them
        self.keepalive = keepalive  # storages of every parameter / buffer the graph reads
        seen, storages = set(), []
        for tensor in keepalive:
            storage = tensor.untyped_storage()
            if id(storage) not in seen:
                seen.add(id(storage))
                storages.append((storage, storage.data_ptr(), storage.nbytes()))
        self.storages = tuple(storages)

    def storage_valid(self) -> bool:
        """CPU-only prelaunch guard: detached aliases do not prevent storage reallocation.

        ``resize_`` can change the allocation *inside* a live StorageImpl, so
        keeping its tensor/storage objects alive is insufficient. CUDA graphs
        still hold the old raw address. Check before launching, not afterward.
        """
        return all(storage.data_ptr() == ptr and storage.nbytes() >= size for storage, ptr, size in self.storages)


class CUDAGraphCache:
    """Per-calculator layout cache and dispatcher (see the module docstring)."""

    def __init__(self, calc, capture_after: int = 2, cache_size: int = 16, cache_memory_fraction: float = 0.25):
        self._calc = weakref.proxy(calc)  # the calculator owns this cache
        self._model = calc.model
        self._device = torch.device(calc.device)
        self.capture_after = max(1, int(capture_after))
        self.cache_size = max(1, int(cache_size))
        self.cache_memory_fraction = float(cache_memory_fraction)
        self._engines: collections.OrderedDict[tuple, _Entry] = collections.OrderedDict()
        self._seen: collections.OrderedDict[tuple, int] = collections.OrderedDict()
        self._ident: tuple | None = None  # (identity key, owner weakrefs, layout key) of the last call
        self._warned = False
        self._checking = False  # the capture-time check is running the regular path
        self._contract = ModelContract(calc.model, calc.external_coulomb, calc.external_dftd3)
        self._contract_roots = (calc.model, calc.external_coulomb, calc.external_dftd3)
        self._model_state = self._contract.snapshot()
        self._answer: dict | None = None  # regular result of a capture that was rejected (see _regular_answer)
        self._gen: _Generation | None = None  # pool of the cached graphs; created by the first capture
        # Graphs share one memory pool and their static input buffers are overwritten at every launch, so replays must
        # not overlap, whatever stream they are launched on: each call waits for this event (recorded at the end of
        # the previous call) and records it again when its replay and output copies are enqueued.
        self._replayed = torch.cuda.Event()
        self.generations = 0  # pools created (diagnostic)
        self.captures = 0  # graphs captured, neighbor-capacity rebuilds included (diagnostic)

    @classmethod
    def create(cls, calc, cuda_graph: bool | None = False, **kwargs) -> CUDAGraphCache | None:
        """A cache for ``calc``, or None when it cannot apply (disabled, not CUDA, old dependencies, ROCm,
        deterministic or training mode, unsupported model). Requested replay that cannot apply is warned once."""
        if not cuda_graph_enabled(cuda_graph) or torch.device(calc.device).type != "cuda":
            return None
        reason = cuda_graph_dependency_problem() or cuda_graph_platform_problem() or cls._calculator_problem(calc)
        cache = None
        if reason is None:
            cache = cls(calc, **kwargs)
            if cache._model_state is None:
                reason = _UNSUPPORTED_MODEL
                cache = None
        if reason is not None:
            warn = getattr(calc, "_warn_cuda_graph_fallback", None)
            if warn is not None:
                warn(reason)
        return cache

    @staticmethod
    def _calculator_problem(calc) -> str | None:
        """Calculator-level reason the graph engines cannot reproduce the regular path, or None."""
        if getattr(calc, "_deterministic", False):
            return "deterministic=True"
        if getattr(calc, "_train", False):
            return "train=True"
        if isinstance(calc.model, torch.jit.ScriptModule):
            return "TorchScript (.jpt) models are not supported"
        try:
            check_dense_model(calc.model)
            check_sparse_model(calc.model, None)
        except (NotImplementedError, AttributeError) as exc:
            return f"unsupported model: {exc}"
        return None

    # --- calculator settings the engines reproduce ------------------------------------------------------------
    def _settings(self) -> tuple | None:
        """Hashable calculator settings an engine depends on, or None when the engines do not reproduce them."""
        return self._settings_reason()[0]

    def _settings_reason(self) -> tuple[tuple | None, str | None]:
        """(settings, None), or (None, why the engines do not reproduce the calculator's settings).

        Plain attribute reads; the model contract is checked separately by ``_check_model``."""
        calc = self._calc
        if calc.model is not self._model:
            return None, "the calculator's model was replaced after construction"
        if getattr(calc, "_train", False):
            return None, "train=True"
        if getattr(calc, "_deterministic", False):
            return None, "deterministic=True"
        c, d = calc.external_coulomb, calc.external_dftd3
        if (c is not None and type(c) is not LRCoulomb) or (d is not None and type(d) is not DFTD3):
            return None, "custom long-range modules"  # rejected before reading stock-only attributes
        if c is not None:
            if c.method != "simple":
                return None, f"{c.method} Coulomb (only full 'simple' Coulomb is captured)"
            if c.subtract_sr:
                return None, "long-range-only Coulomb (subtract_sr=True)"
            if calc._coulomb_cutoff != math.inf:
                return None, "truncated Coulomb cutoff (set_lr_cutoff)"
        elif d is not None and calc._dftd3_cutoff < d.smoothing_off:
            # the regular path would drop D3 pairs inside the smoothing range
            return None, "DFT-D3 cutoff below its smoothing range"
        return (
            None if c is None else id(c),
            None if d is None else (id(d), float(d.smoothing_on), float(d.smoothing_off), float(calc._dftd3_cutoff)),
            float(calc.cutoff),
            int(calc.nb_threshold),
            bool(calc.inference_kernels),
        ), None

    def _check_model(self) -> tuple[bool, bool]:
        """Check the model contract on the CPU: (supported, unchanged since the last check).

        A change drops every engine, so a supported changed model is recaptured (regular path first)."""
        calc = self._calc
        roots = (calc.model, calc.external_coulomb, calc.external_dftd3)
        if any(a is not b for a, b in zip(roots, self._contract_roots, strict=True)):
            self._contract = ModelContract(*roots)
            self._contract_roots = roots
        state = self._contract.snapshot()
        unchanged = state == self._model_state
        if not unchanged:
            # CPU-only tensor versions/configuration; no per-step device scalar reads.
            # Old graphs must not remain reachable after a changed model is restored.
            self.clear()  # waits for a discarded replay that may still be running
            self._model_state = state
        return state is not None, unchanged

    def _model_supported(self) -> bool:
        """``_check_model`` before a capture; an unsupported model state is warned once."""
        supported = self._check_model()[0]
        if not supported:
            self._skip(_UNSUPPORTED_MODEL)
        return supported

    def _captured_state(self) -> tuple:
        """Detached views of every parameter and buffer a graph reads: their storage outlives later replacement."""
        calc = self._calc
        roots = (calc.model, calc.external_coulomb, calc.external_dftd3)
        return tuple(t.detach() for m in roots if m is not None for t in (*m.parameters(), *m.buffers()))

    # --- layout keys (no device sync when the same numbers / charge tensors are passed again) ----------------
    @staticmethod
    def _host(value, dtype: torch.dtype) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            # Convert before numpy: e.g. bfloat16 has no NumPy representation.
            return value.detach().to(device="cpu", dtype=dtype).numpy()
        return np.asarray(value)

    @staticmethod
    def _identity(value) -> tuple[Any, Any] | None:
        """(identity key, owner) of a CUDA tensor or Python scalar; None when the content must be read.

        A view (e.g. ``numbers.unsqueeze(0)`` made fresh on every call by a wrapper) is identified through the tensor
        that owns its storage; ``_version`` is shared with that owner and guards in-place changes."""
        if isinstance(value, torch.Tensor):
            if not value.is_cuda:
                return None
            owner = value if value._base is None else value._base
            try:
                version = int(value._version)
            except RuntimeError:  # inference-mode tensors have no version counter
                return None
            key = ("t", id(owner), value.data_ptr(), version, tuple(value.shape), tuple(value.stride()))
            return key, owner
        if isinstance(value, (int, float)):
            return ("s", float(value)), None
        return None

    def _layout_key(self, numbers, charge, B: int, N: int, forces: bool, settings: tuple, packed: bool = False):
        """``packed``: ``[N, 3]`` input, which the regular path evaluates in nb_mode 1 (``[1, N, 3]`` is nb_mode 0 in
        dense mode); the two give DFT-D3 different neighbor lists, so they are separate layouts."""
        idn, idq = self._identity(numbers), self._identity(charge)
        ident = None
        if idn is not None and idq is not None:
            ident = (idn[0], idq[0], B, N, forces, settings, packed)
            hit = self._ident
            if (
                hit is not None
                and hit[0] == ident
                and all(r is None or r() is o for r, o in zip(hit[1], (idn[1], idq[1]), strict=True))
            ):
                return hit[2]
        z = self._host(numbers, torch.int64).astype(np.int64)
        if z.size != B * N or (z <= 0).any():
            return None  # padded or inconsistent input: regular path
        q = self._host(charge, torch.float64).astype(np.float64).reshape(-1)
        if q.size == 1:
            q = np.repeat(q, B)
        if q.size != B:
            return None
        key = (B, N, forces, z.reshape(B, N).tobytes(), q.tobytes(), settings, packed)
        if ident is not None and idn is not None and idq is not None:
            refs = tuple(None if o is None else weakref.ref(o) for o in (idn[1], idq[1]))
            self._ident = (ident, refs, key)
        return key

    def _count(self, key) -> int:
        """Number of times ``key`` was seen (``_REGULAR_ONLY`` for layouts kept on the regular path)."""
        n = self._seen.pop(key, 0)
        if n != _REGULAR_ONLY:
            n += 1
        self._seen[key] = n
        if len(self._seen) > _MAX_TRACKED_LAYOUTS:
            self._seen.popitem(last=False)
        return n

    # --- engine cache and its memory ------------------------------------------------------------------------------
    def _budget(self) -> float:
        return self.cache_memory_fraction * torch.cuda.get_device_properties(self._device).total_memory

    def _retained(self, entry: _Entry | None = None) -> int:
        """Bytes the cache retains, with ``entry`` (measured here) admitted: the pool's reserved segments plus engine
        buffers outside the pool. Walks the allocator: capture / rebuild / failure / diagnostic boundaries only."""
        gen = self._gen
        segments = [] if gen is None else gen.measure()
        owned = sum(e.owned for e in self._engines.values() if e is not entry)
        if entry is not None:
            entry.owned = _owned_bytes(entry.runner, segments, {ptr for _, ptr, _ in entry.storages})
            owned += entry.owned
        return (0 if gen is None else gen.pool_bytes) + owned

    @contextlib.contextmanager
    def _ordered(self):
        """Order this call's graph work after every earlier replay, on any stream (see ``_replayed``)."""
        stream = torch.cuda.current_stream(self._device)
        stream.wait_event(self._replayed)
        try:
            yield
        finally:
            self._replayed.record(stream)

    def _drain(self) -> None:
        """Wait on the host for every enqueued replay before engines (buffers, graphs) are dropped. Waits for this
        cache's work only (an event), not for the whole device. Rare boundaries: eviction, rejection, retirement."""
        stream = torch.cuda.current_stream(self._device)
        stream.wait_event(self._replayed)
        self._replayed.record(stream)
        self._replayed.synchronize()

    def _evict(self, keep: int) -> None:
        """Drop least recently used engines beyond ``keep``. Their blocks stay in the pool for later captures."""
        if len(self._engines) > keep:
            self._drain()
        while len(self._engines) > keep:
            self._engines.popitem(last=False)

    def _retire(self) -> None:
        """Drop every engine and the pool owner; the pool is released once its graphs are gone.

        A rare boundary (``clear``, model change, rejection): waits for replays that may still read the pool."""
        if self._gen is not None or self._engines:
            self._drain()
        self._engines.clear()
        if self._gen is not None:
            self._gen.retire()
            self._gen = None

    def _capture(
        self, key, B: int, N: int, x, data, forces: bool, padded: tuple | None = None, packed: bool = False
    ) -> tuple | None:
        """Capture, check and admit an engine, returning its validated E/F/q; None requests the regular path.

        ``padded``: the (species, charge) input buffers of a padded dense layout (``x`` holds its capacity rows).
        ``packed``: ``[N, 3]`` input (see ``_layout_key``)."""
        calc = self._calc
        self._evict(self.cache_size - 1)
        failure, oom, checked, reference = None, False, None, None
        engine: Any = None
        runner: Any = None
        entry: _Entry | None = None
        self._answer = None
        try:
            # Run the same-call reference before allocating a new private graph pool.
            # Only compact E/F/q outputs survive; warm-up can release its temporary cache.
            reference = self._reference(data, forces)
            if self._gen is None:
                self._gen = _Generation(self._device)
                self.generations += 1
            gen = self._gen
            if padded is not None:
                z, q = padded
                engine = StaticDenseAIMNet2(calc, z.to(torch.int64), q, padded=True)
                runner = GraphedEF(engine, x, forces=forces, pool=gen.id)
            else:
                # The key owns a value snapshot, so the runner must own the same immutable
                # snapshot, never caller storage (including already-CUDA same-dtype inputs).
                z = torch.as_tensor(data["numbers"], device=x.device).to(torch.int64).reshape(B, N).detach().clone()
                q = torch.as_tensor(data["charge"], dtype=torch.float32, device=x.device).reshape(-1).detach().clone()
                q = q.expand(B).contiguous() if q.numel() == 1 else q.reshape(B)
                if calc.nb_threshold >= N:
                    d3_cutoff = _packed_d3_cutoff(calc) if packed else None
                    engine = StaticDenseAIMNet2(calc, z, q, d3_cutoff=d3_cutoff)
                    runner = GraphedEF(engine, x, forces=forces, pool=gen.id)
                else:
                    runner = AutoGraphedSparseEF(calc, z, q, x, forces=forces, pool=gen)
            self.captures += 1
            entry = _Entry(runner, gen, self._captured_state())
            engine = runner = None
            checked = self._check(
                entry, x, data, forces, B, N, inputs=() if padded is None else padded, reference=reference
            )
            retained, budget = self._retained(entry), self._budget()
            if gen.pool_bytes == 0:
                failure = "RuntimeError: the allocator reports no segments for the graph pool (no pool accounting)"
            elif retained > budget:
                failure = (
                    f"RuntimeError: engine needs {retained / 2**20:.0f} MiB with the cached graphs, over the cache "
                    f"budget of {budget / 2**20:.0f} MiB"
                )
        except Exception as exc:  # never fail where the regular path works
            failure, oom = _failure(exc), isinstance(exc, torch.OutOfMemoryError)
        if failure is None:
            assert entry is not None  # set before any check could pass
            self._engines[key] = entry
            return checked
        engine = runner = entry = checked = None  # release rejected graph outputs before retiring its pool
        self._reject(key, failure, retire=oom)
        self._answer = reference  # the regular result of this very call, so the caller need not recompute it
        return None

    def _reference(self, data, forces: bool) -> dict[str, torch.Tensor]:
        """Same-call regular result (detached); its E/F/q are the capture-time reference."""
        self._checking = True
        try:
            result = self._calc(dict(data), forces=forces, validate_species=False)
            return {key: value.detach() if isinstance(value, torch.Tensor) else value for key, value in result.items()}
        finally:
            self._checking = False

    def _regular_answer(self):
        """The regular result computed by a rejected capture of this call, or None (the regular path runs)."""
        answer, self._answer = self._answer, None
        return answer

    def _check(self, entry: _Entry, coord, data, forces: bool, B: int, N: int, inputs=(), reference=None) -> tuple:
        """Raise unless the captured engine reproduces the regular path on this input (run once per layout).

        ``inputs``: species/charge buffers of a padded layout; its filler rows past ``B`` are not compared."""
        e, f, q = entry.runner(coord, *inputs)
        ref = self._reference(data, forces) if reference is None else reference
        return self._compare(e, f, q, ref, forces, B, N)

    @staticmethod
    def _compare(e, f, q, ref, forces: bool, B: int, N: int) -> tuple:
        """Raise unless the first ``B`` rows of a graph's E/F/q match the regular result ``ref`` within fp32 noise."""
        e, q = e[:B], q[:B]
        f = None if f is None else f[:B]
        bad = []
        de = float((e.detach().double() - ref["energy"].detach().double().reshape(B)).abs().max())
        if not de <= _CHECK_E_PER_ATOM * N:
            bad.append(f"energy {de:.2e} eV")
        dq = float((q.detach() - ref["charges"].detach().reshape(B, N)).abs().max())
        if not dq <= _CHECK_Q:
            bad.append(f"charges {dq:.2e} e")
        if forces:
            rf = ref["forces"].detach().reshape(B, N, 3)
            df = float((f - rf).abs().max())
            if not df <= _CHECK_F + _CHECK_F_REL * float(rf.abs().max()) + _CHECK_F_PER_ATOM * N:
                bad.append(f"forces {df:.2e} eV/A")
        if bad:
            raise RuntimeError("captured engine disagrees with the regular path: " + ", ".join(bad))
        return e, f, q

    def clear(self) -> None:
        """Drop every engine, release the graph memory pool and forget the layouts seen (including rejected ones).

        Waits for running replays; outputs returned earlier are copies and stay valid. The pool is returned with
        ``torch.cuda.empty_cache()``, which synchronizes the device and releases every unused cached block of the
        process. The next capture creates a new pool. Also required after deliberate .data/raw-storage mutation."""
        self._retire()
        self._seen.clear()
        self._ident = None
        torch.cuda.empty_cache()

    def memory_stats(self) -> dict[str, int]:
        """Bytes the cache retains: ``pool_reserved`` (graph pool segments) + ``engine_buffers`` (engine tensors
        outside the pool) = ``retained``, limited by ``budget`` at admission; counts of ``engines``, pools created
        (``generations``) and ``captures``. Walks the allocator (a diagnostic call)."""
        retained = self._retained()
        pool = 0 if self._gen is None else self._gen.pool_bytes
        return {
            "engines": len(self._engines),
            "pool_reserved": pool,
            "engine_buffers": retained - pool,
            "retained": retained,
            "budget": int(self._budget()),
            "pool_alive": int(self._gen is not None),
            "generations": self.generations,
            "captures": self.captures,
        }

    # --- dispatch -------------------------------------------------------------------------------------------------
    def _skip(self, reason: str) -> None:
        """Warn once per calculator and reason that this call uses the regular path (``reason`` is structural, not
        the normal first sighting of a layout)."""
        warn = getattr(self._calc, "_warn_cuda_graph_fallback", None)
        if warn is not None:
            warn(reason)

    @staticmethod
    def _input_problem(data: Mapping[str, Any]) -> str | None:
        """Why the input keys are outside the replayed contract, or None."""
        keys = data.keys()
        extra, missing = keys - _ALLOWED_KEYS, _REQUIRED_KEYS - keys
        if "mol_idx" in extra:
            return "flattened mol_idx batch input (e.g. TorchSim)"
        if "cell" in extra or "pbc" in extra:
            return "periodic cell input"
        if "nbmat" in extra:
            return "caller-supplied nbmat neighbor list"
        if extra:
            return f"unsupported input keys {sorted(extra)}"
        if missing:
            return f"input without {sorted(missing)}"
        return None

    def eval(self, data: Mapping[str, Any], forces: bool, validate_species: bool) -> dict[str, torch.Tensor] | None:
        """Evaluate ``data`` by CUDA-graph replay, or return None to let the regular path run."""
        if self._checking or not isinstance(data, Mapping):
            return None  # (the capture-time reference itself runs the regular path)
        problem = self._input_problem(data)
        if problem is not None:
            self._skip(problem)
            return None
        coord = data["coord"]
        if any(isinstance(value, torch.Tensor) and value.requires_grad for value in data.values()):
            self._skip("input tensors require grad")
            return None  # preserve differentiation through charge as well as coordinates
        shape = tuple(coord.shape) if hasattr(coord, "shape") else np.shape(coord)
        if len(shape) == 2 and shape[1] == 3:
            B, N, single = 1, int(shape[0]), True
        elif len(shape) == 3 and shape[2] == 3:
            B, N, single = int(shape[0]), int(shape[1]), False
        else:
            self._skip(f"coordinate shape {tuple(shape)} (expected [N, 3] or [B, N, 3])")
            return None
        if N < 2 or B < 1:
            self._skip("fewer than two atoms per structure")
            return None
        settings, problem = self._settings_reason()
        if settings is None:
            self._skip(problem or "unsupported calculator settings")
            return None
        forces = bool(forces)
        if B > 1 and self._calc.nb_threshold >= N:
            return self._eval_padded(data, coord, B, N, forces, settings, validate_species)
        key = self._layout_key(data["numbers"], data["charge"], B, N, forces, settings, packed=single)
        if key is None:
            self._skip(
                "zero-number padding atoms outside a dense batch (single molecule, or N > nb_threshold), "
                "or numbers/charge shapes that do not match coord"
            )
            return None
        entry = self._engines.get(key)
        if entry is None and (not self._model_supported() or self._count(key) < self.capture_after):
            return None  # unsupported model state, first sighting, or a layout kept on the regular path
        calc = self._calc
        if validate_species and (entry is None or not entry.validated):
            calc._validate_species_and_charge(data)
        calc._maybe_warn_mult_ignored(data)
        cached, entry = entry is not None, None  # hold no graph across the contract check: it may drop the cache
        dev = self._device
        with torch.inference_mode(False), torch.cuda.device(dev), self._ordered():
            x = torch.as_tensor(coord, dtype=torch.float32, device=dev).reshape(B, N, 3)
            checked = None
            if not cached:
                checked = self._capture(key, B, N, x, data, forces, packed=single)
                if checked is None:
                    return self._regular_answer()
            elif not self._launch(key, x):
                return None
            entry = self._engines[key]
            entry.validated = entry.validated or validate_species
            failure = None
            try:
                # The capture-time check already evaluated these exact inputs.
                e, f, q = entry.runner.result() if checked is None else checked
                rebuilds = getattr(entry.runner, "rebuilds", 0)
                if rebuilds != entry.rebuilds:
                    # Recapture changes the neighbor layout: repeat both the
                    # numerical acceptance gate and memory-budget accounting.
                    self.captures += rebuilds - entry.rebuilds
                    entry.rebuilds = rebuilds
                    self._check(entry, x, data, forces, B, N)
                    if self._retained(entry) > self._budget():
                        failure = "RuntimeError: rebuilt sparse engine exceeds the cache memory budget"
            except Exception as exc:
                failure = _failure(exc)
            if failure is not None:
                entry = checked = e = f = q = None
                self._reject(key, failure)
                return None
            out = {"energy": e.clone(), "charges": (q[0] if single else q).clone()}
            if forces:
                out["forces"] = (f[0] if single else f).clone()
        return out

    def _launch(self, key, x, *inputs) -> bool:
        """Launch the cached entry of ``key``, then check the model contract on the CPU while the GPU replays.

        False: the regular path answers (the replay, if any, is discarded unread)."""
        entry = self._engines[key]
        if not entry.storage_valid():
            # A resize can leave the captured raw pointers dangling.
            # Do not enqueue even a speculative replay in this case.
            del entry
            self._check_model()
            if self._engines:
                # Raw storage mutation need not bump tensor versions;
                # invalidate independently of the full model token.
                self.clear()
            return False
        failure = None
        try:
            entry.runner.launch(x, *inputs)
        except Exception as exc:
            failure = _failure(exc)
        del entry  # the contract check may drop the cache: hold no graph past it
        if failure is not None:
            self._reject(key, failure)
            return False
        supported, unchanged = self._check_model()
        if not (supported and unchanged):
            return False
        self._engines.move_to_end(key)
        return True

    def _eval_padded(self, data, coord, B: int, N: int, forces: bool, settings: tuple, validate_species: bool):
        """Dense batch, possibly with ``Z = 0`` padding: one graph per (batch capacity, ``N``), species and charges are
        per-call inputs, so mixed-size and shrinking batches replay without new captures."""
        calc = self._calc
        # A graph recorded for exactly B rows, else the smallest recorded one with few enough filler rows, else
        # record B rows exactly once this batch size has been seen ``capture_after`` times.
        cap, entry = B, self._engines.get(("padded", B, N, forces, settings))
        if entry is None:
            fits = [
                k[1]
                for k in self._engines
                if k[0] == "padded"
                and k[2:] == (N, forces, settings)
                and B < k[1] <= B + _max_filler_rows(B)
                and B not in self._engines[k].no_reuse
            ]
            if fits:
                cap = min(fits)
                entry = self._engines[("padded", cap, N, forces, settings)]
        key = ("padded", cap, N, forces, settings)
        if entry is None and (not self._model_supported() or self._count(key) < self.capture_after):
            return None
        if validate_species:  # the species can change on every call
            calc._validate_species_and_charge(data)
        calc._maybe_warn_mult_ignored(data)
        cached, entry = entry is not None, None  # hold no graph across the contract check: it may drop the cache
        dev = self._device
        with torch.inference_mode(False), torch.cuda.device(dev), self._ordered():
            x = torch.as_tensor(coord, dtype=torch.float32, device=dev).reshape(B, N, 3)
            z = torch.as_tensor(data["numbers"], device=dev).reshape(B, N)
            q = torch.as_tensor(data["charge"], dtype=torch.float32, device=dev).reshape(-1)
            q = q.expand(B) if q.numel() == 1 else q.reshape(B)
            if cap > B:  # filler rows: copies of molecule 0, computed and discarded
                x = torch.cat([x, x[:1].expand(cap - B, N, 3)])
                z = torch.cat([z, z[:1].expand(cap - B, N)])
                q = torch.cat([q, q[:1].expand(cap - B)])
            checked = None
            if not cached:
                checked = self._capture(key, B, N, x, data, forces, padded=(z, q))
                if checked is None:
                    return self._regular_answer()
                self._engines[key].rows.add(B)
            elif not self._launch(key, x, z, q):
                return None
            elif B not in self._engines[key].rows:
                # First reuse of this graph for another row count (filler rows): check it against the regular path
                # once, as a new capture is checked.
                checked = self._check_reuse(key, data, forces, B, N)
                if checked is None:
                    return self._regular_answer()
            e, f, qq = self._engines[key].runner.result() if checked is None else checked
            out = {"energy": e[:B].clone(), "charges": qq[:B].clone()}
            if forces:
                out["forces"] = f[:B].clone()
        return out

    def _check_reuse(self, key, data, forces: bool, B: int, N: int) -> tuple | None:
        """Compare the launched replay of ``key`` with the regular result of this call; None after a rejection (the
        regular result, when computed, is left for ``_regular_answer``). A failure only stops reuse of this graph for
        ``B`` rows: the graph stays validated for the row counts it already passed."""
        failure, reference = None, None
        entry: _Entry | None = None
        self._answer = None
        try:
            reference = self._reference(data, forces)
            entry = self._engines[key]
            e, f, q = entry.runner.result()
            checked = self._compare(e, f, q, reference, forces, B, N)
        except Exception as exc:
            failure = _failure(exc)
        else:
            entry.rows.add(B)
            return checked
        if entry is not None:
            entry.no_reuse.add(B)
        entry = None
        self._warn_disabled(f"{B} rows on the {key[1]}-row graph: {failure}")
        self._answer = reference
        return None

    def _reject(self, key, reason: str, retire: bool = False) -> None:
        """Keep ``key`` on the regular path (warned once) and drop its engine.

        The generation is retired as well when asked (out of memory), when no engine is left in it, or when it is over
        the budget, so a rejected or failed capture leaves no pool behind."""
        if key in self._engines:
            self._drain()
        self._engines.pop(key, None)
        self._seen.pop(key, None)
        self._seen[key] = _REGULAR_ONLY
        if retire or not self._engines or self._retained() > self._budget():
            self._retire()
            torch.cuda.empty_cache()
        self._warn_disabled(reason)

    def _warn_disabled(self, reason: str) -> None:
        """Warn once per cache that a layout fell back to the regular path, attributed to the user's call."""
        if not self._warned:
            warnings.warn(
                f"AIMNet2Calculator: CUDA-graph replay disabled for one input layout ({reason}); "
                "using the regular code path for it.",
                RuntimeWarning,
                stacklevel=external_stacklevel(),
            )
            self._warned = True
