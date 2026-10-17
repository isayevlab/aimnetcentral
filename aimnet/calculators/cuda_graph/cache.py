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
  ``[N, 3]`` or an equal-size batch ``[B, N, 3]`` without padding atoms;
* ``stress=False`` and ``hessian=False``.

Policy:

* a layout (species, charges, ``B``, ``N``, ``forces``, calculator settings) is left to the regular path the first time
  it is seen; from the second call on, a static-layout engine is built (~0.05-0.2 s) and replayed as one CUDA graph.
  A layout evaluated once costs what it costs on the regular path; repeated layouts (MD, optimization, scans) are
  replayed;
* ``N <= nb_threshold`` uses the dense engine (``dense.py``), larger systems the neighbor-list engine (``sparse.py``);
* each new or capacity-rebuilt engine is checked against the regular path on the same input; an engine that
  disagrees beyond float32 noise is not used;
* before a replay, captured allocation addresses and sizes are checked on the CPU. The full model contract
  (``contracts.py``) is checked on every call while the GPU runs; a changed model's result is discarded unread
  (the regular path answers). Detached views keep replaced parameter/buffer storage alive, while the address/size
  guard prevents replay after ``resize_`` reallocates a still-live storage;
* engines live in an LRU cache bounded by ``cache_size`` entries and ``cache_memory_fraction`` of GPU memory;
* a failure while building, checking or running an engine sends that layout back to the regular path (warned once).

Outputs have the regular path's keys, shapes and dtypes: ``energy`` ``[B]`` (``[1]`` for ``[N, 3]`` input),
``charges`` ``[B, N]`` (``[N]``), ``forces`` ``[B, N, 3]`` (``[N, 3]``, only when ``forces=True``).
"""

from __future__ import annotations

import collections
import math
import os
import warnings
import weakref
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from aimnet.modules.lr import DFTD3, LRCoulomb

from .contracts import ModelContract
from .dense import GraphedEF, StaticDenseAIMNet2, check_dense_model
from .sparse import AutoGraphedSparseEF, check_sparse_model

_REQUIRED_KEYS = frozenset({"coord", "numbers", "charge"})
_ALLOWED_KEYS = _REQUIRED_KEYS | {"mult"}
_MAX_TRACKED_LAYOUTS = 1024  # layouts remembered for the "seen before" count
_REGULAR_ONLY = -1  # count of a layout that stays on the regular path
# capture-time check against the regular path (measured engine-vs-regular differences: energy <= 3e-7 eV/atom,
# forces <= 1.6e-5 eV/A, charges <= 6e-7 e)
_CHECK_E_PER_ATOM = 1e-5  # eV
_CHECK_F = 1e-3  # eV/A, plus 1e-5 of the largest force component
_CHECK_Q = 1e-4  # e


def cuda_graph_enabled(cuda_graph: bool | None) -> bool:
    """Resolve the ``cuda_graph`` argument; ``None`` defers to ``AIMNET_CUDA_GRAPH`` (default on)."""
    if cuda_graph is not None:
        return bool(cuda_graph)
    return os.environ.get("AIMNET_CUDA_GRAPH", "1").strip().lower() not in ("0", "false", "no", "off")


class _Entry:
    __slots__ = ("keepalive", "mem", "rebuilds", "runner", "storages", "validated")

    def __init__(self, runner, mem: int, keepalive: tuple = ()):
        self.runner, self.mem = runner, mem
        self.rebuilds = getattr(runner, "rebuilds", 0)
        self.validated = False  # species / charge validation passed for this layout (its content is in the key)
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

    @classmethod
    def create(cls, calc, cuda_graph: bool | None = None, **kwargs) -> CUDAGraphCache | None:
        """A cache for ``calc``, or None when it cannot apply (disabled, not CUDA, unsupported model)."""
        if not cuda_graph_enabled(cuda_graph) or torch.device(calc.device).type != "cuda":
            return None
        if isinstance(calc.model, torch.jit.ScriptModule):
            return None
        try:
            check_dense_model(calc.model)
            check_sparse_model(calc.model, None)
        except (NotImplementedError, AttributeError):
            return None
        cache = cls(calc, **kwargs)
        return cache if cache._model_state is not None else None

    # --- calculator settings the engines reproduce ------------------------------------------------------------
    def _settings(self) -> tuple | None:
        """Hashable calculator settings an engine depends on, or None when the engines do not reproduce them.

        Plain attribute reads; the model contract is checked separately by ``_check_model``."""
        calc = self._calc
        if calc.model is not self._model or getattr(calc, "_train", False) or getattr(calc, "_deterministic", False):
            return None
        c, d = calc.external_coulomb, calc.external_dftd3
        if (c is not None and type(c) is not LRCoulomb) or (d is not None and type(d) is not DFTD3):
            return None  # reject custom modules before reading stock-only attributes
        if c is not None:
            if c.method != "simple" or c.subtract_sr or calc._coulomb_cutoff != math.inf:
                return None  # DSF/Ewald/PME, LR-only Coulomb, or set_lr_cutoff() truncated the full Coulomb
        elif d is not None and calc._dftd3_cutoff < d.smoothing_off:
            return None  # the regular path would drop D3 pairs inside the smoothing range
        return (
            None if c is None else id(c),
            None if d is None else (id(d), float(d.smoothing_on), float(d.smoothing_off)),
            float(calc.cutoff),
            int(calc.nb_threshold),
        )

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
            if self._engines:
                torch.cuda.synchronize(torch.device(calc.device))  # a discarded replay may still be running
            self.clear()
            self._model_state = state
        return state is not None, unchanged

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

    def _layout_key(self, numbers, charge, B: int, N: int, forces: bool, settings: tuple):
        idn, idq = self._identity(numbers), self._identity(charge)
        ident = None
        if idn is not None and idq is not None:
            ident = (idn[0], idq[0], B, N, forces, settings)
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
        key = (B, N, forces, z.reshape(B, N).tobytes(), q.tobytes(), settings)
        if ident is not None:
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

    # --- engine cache ---------------------------------------------------------------------------------------------
    def _evict(self, budget: float, keep: int) -> None:
        evicted = False
        while self._engines and (len(self._engines) > keep or sum(e.mem for e in self._engines.values()) > budget):
            self._engines.popitem(last=False)
            evicted = True
        if evicted:
            torch.cuda.empty_cache()

    def _build(self, key, B: int, N: int, coord, data, forces: bool) -> _Entry | None:
        calc = self._calc
        dev = coord.device
        budget = self.cache_memory_fraction * torch.cuda.get_device_properties(dev).total_memory
        self._evict(budget, self.cache_size - 1)
        # The key owns a value snapshot, so the runner must own the same immutable
        # snapshot, never caller storage (including already-CUDA same-dtype inputs).
        z = torch.as_tensor(data["numbers"], device=dev).to(torch.int64).reshape(B, N).detach().clone()
        q = torch.as_tensor(data["charge"], dtype=torch.float32, device=dev).reshape(-1).detach().clone()
        q = q.expand(B).contiguous() if q.numel() == 1 else q.reshape(B)
        torch.cuda.synchronize(dev)
        before = torch.cuda.memory_reserved(dev)
        if calc.nb_threshold >= N:
            runner = GraphedEF(StaticDenseAIMNet2(calc, z, q), coord, forces=forces)
        else:
            runner = AutoGraphedSparseEF(calc, z, q, coord, forces=forces)
        torch.cuda.synchronize(dev)
        entry = _Entry(runner, max(0, torch.cuda.memory_reserved(dev) - before), self._captured_state())
        self._check(entry, coord, data, forces, B, N)
        if entry.mem > budget:
            self._regular_only(key, RuntimeError(f"engine needs {entry.mem / 2**20:.0f} MiB, over the cache budget"))
            return None
        self._engines[key] = entry
        self._evict(budget, self.cache_size)
        return entry

    def _check(self, entry: _Entry, coord, data, forces: bool, B: int, N: int) -> None:
        """Raise unless the captured engine reproduces the regular path on this input (run once per layout)."""
        e, f, q = entry.runner(coord)
        self._checking = True
        try:
            ref = self._calc(dict(data), forces=forces, validate_species=False)
        finally:
            self._checking = False
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
            if not df <= _CHECK_F + 1e-5 * float(rf.abs().max()):
                bad.append(f"forces {df:.2e} eV/A")
        if bad:
            raise RuntimeError("captured engine disagrees with the regular path: " + ", ".join(bad))

    def clear(self) -> None:
        """Drop engines/graphs; also required after deliberate .data/raw-storage mutation."""
        self._engines.clear()
        self._seen.clear()
        self._ident = None
        torch.cuda.empty_cache()

    # --- dispatch -------------------------------------------------------------------------------------------------
    def eval(self, data: Mapping[str, Any], forces: bool, validate_species: bool) -> dict[str, torch.Tensor] | None:
        """Evaluate ``data`` by CUDA-graph replay, or return None to let the regular path run."""
        if self._checking or not isinstance(data, Mapping) or not _REQUIRED_KEYS <= data.keys() <= _ALLOWED_KEYS:
            return None
        coord = data["coord"]
        if any(isinstance(value, torch.Tensor) and value.requires_grad for value in data.values()):
            return None  # preserve differentiation through charge as well as coordinates
        shape = tuple(coord.shape) if hasattr(coord, "shape") else np.shape(coord)
        if len(shape) == 2 and shape[1] == 3:
            B, N, single = 1, int(shape[0]), True
        elif len(shape) == 3 and shape[2] == 3:
            B, N, single = int(shape[0]), int(shape[1]), False
        else:
            return None
        if N < 2 or B < 1:
            return None
        settings = self._settings()
        if settings is None:
            return None
        forces = bool(forces)
        key = self._layout_key(data["numbers"], data["charge"], B, N, forces, settings)
        if key is None:
            return None
        entry = self._engines.get(key)
        if entry is None and (not self._check_model()[0] or self._count(key) < self.capture_after):
            return None  # unsupported model state, first sighting, or a layout kept on the regular path
        calc = self._calc
        if validate_species and (entry is None or not entry.validated):
            calc._validate_species_and_charge(data)
        calc._maybe_warn_mult_ignored(data)
        dev = torch.device(calc.device)
        with torch.inference_mode(False), torch.cuda.device(dev):
            x = torch.as_tensor(coord, dtype=torch.float32, device=dev).reshape(B, N, 3)
            if entry is None:
                try:
                    entry = self._build(key, B, N, x, data, forces)
                except Exception as exc:  # never fail where the regular path works
                    self._regular_only(key, exc)
                    return None
                if entry is None:
                    return None
                entry.runner.launch(x)
            else:
                if not entry.storage_valid():
                    # A resize can leave the captured raw pointers dangling.
                    # Do not enqueue even a speculative replay in this case.
                    self._check_model()
                    if self._engines:
                        # Raw storage mutation need not bump tensor versions;
                        # invalidate independently of the full model token.
                        torch.cuda.synchronize(dev)
                        self.clear()
                    return None
                # Launch first; the contract check below runs on the CPU while the GPU replays.
                try:
                    entry.runner.launch(x)
                except Exception as exc:
                    self._regular_only(key, exc)
                    return None
                supported, unchanged = self._check_model()
                if not (supported and unchanged):
                    return None  # the replay is discarded unread; the regular path answers
                self._engines.move_to_end(key)
            entry.validated = entry.validated or validate_species
            try:
                e, f, q = entry.runner.result()
                rebuilds = getattr(entry.runner, "rebuilds", 0)
                if rebuilds != entry.rebuilds:
                    # Recapture changes the neighbor layout: repeat both the
                    # numerical acceptance gate and memory-budget accounting.
                    self._check(entry, x, data, forces, B, N)
                    entry.rebuilds = rebuilds
                    growth = entry.runner.unaccounted_memory_growth
                    entry.mem += growth
                    entry.runner.unaccounted_memory_growth = 0
                    budget = self.cache_memory_fraction * torch.cuda.get_device_properties(dev).total_memory
                    if entry.mem > budget:
                        self._regular_only(key, RuntimeError("rebuilt sparse engine exceeds the cache memory budget"))
                        return None
                    self._evict(budget, self.cache_size)
            except Exception as exc:
                self._regular_only(key, exc)
                return None
            out = {"energy": e.clone(), "charges": (q[0] if single else q).clone()}
            if forces:
                out["forces"] = (f[0] if single else f).clone()
        return out

    def _regular_only(self, key, exc: Exception) -> None:
        self._engines.pop(key, None)
        self._seen.pop(key, None)
        self._seen[key] = _REGULAR_ONLY
        if not self._warned:
            warnings.warn(
                f"AIMNet2Calculator: CUDA-graph replay disabled for one input layout ({type(exc).__name__}: {exc}); "
                "using the regular code path for it.",
                RuntimeWarning,
                stacklevel=4,
            )
            self._warned = True
