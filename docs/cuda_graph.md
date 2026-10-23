# CUDA-Graph Inference

CUDA-graph replay accelerates repeated energy, force, and charge evaluations of supported AIMNet2 models. It changes execution, not the trained potential, long-range terms, or units. This is an inference backend, not a training accelerator or a new model format.

## Defaults and Comparison Modes

`compile_model=False` and `cuda_graph=False` are the constructor defaults. Graph capture adds startup cost and retains GPU memory between calls, so enable it explicitly with `cuda_graph=True` for repeated CUDA evaluations. The optional graph backend is imported only after CUDA execution and graph opt-in have both been selected; CPU and graph-disabled calculators do not import it.

| Mode | Explicit constructor options | Execution |
| --- | --- | --- |
| Regular, stock primitives (default) | `compile_model=False, cuda_graph=False, inference_kernels=False` | Canonical calculator and model computation |
| Regular, shared optimized primitives | `compile_model=False, cuda_graph=False, inference_kernels=True` | Same model computation with eligible AEV/ConvSV and Coulomb kernels |
| Graph, stock primitives | `compile_model=False, cuda_graph=True, inference_kernels=False` | Capture and replay of the canonical computation with static input preparation |
| Graph, shared optimized primitives | `compile_model=False, cuda_graph=True, inference_kernels=True` | Same graph execution with the independently selected primitive kernels |
| Compile only | `compile_model=True, cuda_graph=False, inference_kernels=False` | Regular calculator with a compiled NN forward |
| Compile and graph selected | `compile_model=True, cuda_graph=True` | Eligible graph replay takes precedence and runs the eager model, not the compiled forward; only calls that fall back to the regular path use the compiled forward. A one-time `UserWarning` says so |

Here, “regular” does not mean “CPU” or “without Warp”; regular CUDA calculations already use supported Warp kernels. Dense/sparse layout, `torch.compile`, and CUDA-graph replay are separate choices.

```python
from aimnet.calculators import AIMNet2Calculator

regular = AIMNet2Calculator("aimnet2", compile_model=False, cuda_graph=False)
compiled = AIMNet2Calculator("aimnet2", compile_model=True, cuda_graph=False)
replay = AIMNet2Calculator("aimnet2", compile_model=False, cuda_graph=True)
replay_with_kernels = AIMNet2Calculator("aimnet2", cuda_graph=True, inference_kernels=True)
```

!!! important "Do not accidentally benchmark graph replay as torch.compile"

    Set `cuda_graph=False` for a compile-only comparison. With both options enabled, regular calls may incur compilation, while eligible replay uses the shared prepared-input model computation. This is not replay of the compiled forward.

**Precedence.** When options overlap, the calculator resolves them as follows and warns once at construction about every option that is overridden:

1. `cuda_graph=True` (CUDA, supported model and call): the call replays a CUDA graph of the **eager** model; `compile_model=True` does not affect replayed calls.
2. Otherwise, `compile_model=True`: the compiled forward runs. `inference_kernels=True` has **no effect** on compiled calls, which keep the stock primitives.
3. Otherwise, the eager model runs, with the optimized primitives when `inference_kernels=True`.

So `inference_kernels=True` together with `compile_model=True` is active only inside graph replays (`cuda_graph=True`).

Explicit `cuda_graph=True` or `False` overrides the environment. Omitting the argument is equivalent to `False`, even when `AIMNET_CUDA_GRAPH=1`. To let a deployment choose through the environment, explicitly pass `cuda_graph=None`: then `AIMNET_CUDA_GRAPH=1`, `true`, `yes`, or `on` enables replay (case-insensitive, whitespace ignored), and an unset or any other value disables it. CPU calculations always use the regular path regardless of this option; requesting replay on CPU warns once.

## Shared Computation and Independent Kernels

Regular and graph execution call the same `AIMNet2.forward_prepared` body for atomic features, charge updates, neural-network passes and output modules. The calculator's `_evaluate_prepared` implementation also owns external physics and force construction for both paths. Dense and sparse graph adapters prepare input buffers, masks and neighbors; they do not maintain separate versions of the model equations.

`inference_kernels=False` is a separate constructor default. Setting it to `True` selects eligible shared AEV/ConvSV layout operations and full-Coulomb kernels from the common modules, whether graphs are enabled or not. This makes kernel effects and replay effects independently measurable. Training, Hessians, stress, deterministic execution and caller-owned differentiation retain the corresponding regular derivative paths. The Triton and streaming Coulomb primitives support first-order inference, not force training or double backward.

For eligible dense inputs, the two independent feature convolutions within each later charge-update pass share one batched matrix multiplication. This reduces intermediate geometry-gradient accumulation without collecting gradients across passes. The operation uses ordinary PyTorch autograd, has no fixed pass count or backward traversal order, and is shared by eager and graph execution. Convolution hooks or custom call/prepare methods retain the individual module calls. This local contraction optimization does not expand the calculator's supported differentiation contracts.

For eligible multi-structure dense inputs, an explicit geometry continuation connects successive contractions. Its backward combines the incoming geometry gradient with the current matrix product. A lazily loaded FP32 Triton epilogue avoids materializing a separate addition and never overwrites an incoming gradient. CPU, autocast, higher-order backward and unsupported kernel shapes retain differentiable Torch operations. The public geometry tensor remains unchanged; there is no mutable backward collector, assumed traversal order or fixed number of passes. Fresh convolution outputs are masked in place on this optional path, retaining padding semantics without an extra clone. These execution changes do not alter calculator defaults or enable force training through the first-order AEV kernel.

A second optional ConvSV primitive combines vector projection, squared norm, scalar/vector packing and padding masking, and writes its input gradient in the contraction's native layout. It is limited to multi-structure CUDA FP32 inputs, frozen projection weights and bounded basis sizes. Trainable weights, autocast and unsupported inputs retain Torch expressions; higher-order backward of this primitive recomputes from its original inputs with differentiable Torch operations. The enclosing calculator still enforces the stricter first-order inference-kernel contract described above.

Graph acceleration remains restricted to the model and input contracts below. Sharing computation removes duplicated equations; it does not by itself qualify every model, hook, device or derivative request for replay.

## Supported Calls

All of these conditions must hold:

- CUDA execution, inference mode (`train=False`), and `deterministic=False`.
- A stock, closed-shell AIMNet2 module with one charge channel, the supported `d2features` architecture, stock GELU layers and forward methods, canonical output heads, and no hooks or trainable parameters.
- Full `simple` external Coulomb with the supported short-range treatment, or no external Coulomb; external DFT-D3 is optional. DSF, Ewald, PME, and truncated full Coulomb use the regular path.
- Input keys are exactly `coord`, `numbers`, `charge`, with optional `mult`. The latter retains the regular closed-shell behavior and warning semantics; it does not add open-shell support.
- Coordinates are `[N, 3]` or `[B, N, 3]`, with `N >= 2`. No input tensor requires gradients. A single molecule `[N, 3]` and a neighbor-list batch (`N > nb_threshold`) have no zero-number padding. A dense batch (`B > 1` and `N <= nb_threshold`, default 120) may mix molecule sizes padded with `numbers == 0` atoms, the regular calculator's padded `[B, N, 3]` format. Flattened `mol_idx` batches use the regular path.
- No stress or Hessian request.

`B` is the number of independent structures and `N` is atoms per structure (the padded size of a dense batch), not a training batch-size setting. Changing coordinates is supported; fixed layout does **not** mean fixed geometry. A single molecule or neighbor-list batch is captured per layout: atomic species, total charges, batch size, atom count, and force-request mode. A dense batch is captured per shape (rows, `N`, force request): species and total charges, including the number and position of padding atoms, are graph inputs that may change on every call. A batch of `B` rows replays a graph recorded for exactly `B` rows, else the smallest recorded one with at most `max(1, B // 3)` extra rows (copies of the first molecule, computed and discarded), else it records `B` rows on its second sighting. The first reuse of a graph for a smaller row count is checked against the regular path once, as a new capture is. A batch that shrinks from 64 molecules to 1 records about ten graphs.

Outputs retain regular keys, shapes, dtypes, and device: energy is in **eV**, forces are in **eV/Å**, and charges are in **e**. Forces are the energy-derived `-∂E/∂R` values, not a separately fitted force model. FP32 operation ordering can change values at rounding level; bitwise equality is not promised.

### Artifacts and Models

Registry models and v2 `.pt` artifacts are eligible only if the loaded module satisfies the contract above. A `.pt` suffix alone does not establish compatibility. Legacy `.jpt` models, custom output heads, SR-repulsion heads, SiLU substitutions, and other custom forward implementations are **not** accelerated by this backend. They use the regular path if the existing loader and calculator support them; this change does not add compatibility to previously unsupported artifacts. In particular, `compile_model=True` remains unsupported for legacy `.jpt` files.

## Capture, Memory, and Fallback

The first call for a layout uses the regular path. On the second call, a fresh regular reference is evaluated before the engine is captured, retaining only E/F/charges. One complete warm-up runs on the capture stream; capture and numerical comparison follow. The comparison admits a graph only if it matches the regular result within FP32 noise: energy within 5e-6 eV per atom, charges within 2e-5 e, and forces within 1e-4 eV/Å + 1e-5 × max|F| + 1e-7 eV/Å per atom (5–10× or more above the graph-versus-regular differences measured on an L40S for 3- to 4,992-atom systems; the force noise grows with system size through long-range sums). A rejected graph only costs speed: that layout stays on the regular path. The validated output is copied to the caller without an extra replay. The reference always uses the current call's coordinates, species and charges, not the first call's geometry. Later calls replay the graph. `forces=False` and `forces=True` have separate engines. `N <= nb_threshold` uses a dense engine; larger systems use a sparse neighbor-list engine. The existing `nb_threshold=120` default is unchanged.

| Policy | Current implementation |
| --- | --- |
| New-layout setup | Regular reference, warm-up, capture and numerical check; measure separately from steady replay |
| Dense batch graphs | Keyed by shape; species, charges, and padding are per-call inputs; bounded batch-capacity reuse |
| Graph memory pool | One per calculator at a time, created by its first capture (none for construction or regular-only calls), shared by its graphs |
| Cache | Per calculator, LRU, at most 16 engines |
| Cache budget | 25% of device total memory, applied to the memory the cache retains (below), measured at every capture and rebuild |
| Over-budget capture or rebuild | Reject that layout; release the shared generation if no admitted engine remains or its remaining retained bytes exceed the budget. Existing valid graphs may keep a pool that still fits |
| Sparse neighbor overflow | Rebuild capacity, repeat the numerical check, and recheck the cache budget |
| `calc.clear_cuda_graphs()` | Drops every engine, releases the pool, and forgets the layouts seen; the next capture creates a new pool |
| Unsupported call | Regular path; one `UserWarning` per calculator and reason (for example `mol_idx` input, `cell`, caller `nbmat`, inputs requiring grad, stress/Hessian, `deterministic=True`, unsupported model, old dependencies) |
| Capture/check/replay failure | Layout falls back to regular; one `RuntimeWarning` per calculator. CUDA errors that leave the context unusable (illegal address, device-side assert) are raised |

**Retained memory.** The cache retains the reserved segments of its graph pool and the engines' own buffers outside it (static inputs, masks, neighbor buffers). Both are measured from the allocator when a graph is captured or rebuilt, never on replay, and `calc._cuda_graph.memory_stats()` reports them. A capture or rebuild that would exceed the budget is rejected: that layout stays on the regular path (it is not captured again until `clear_cuda_graphs()`, a model change, or until it drops out of the 1,024 most recently seen layouts). After removing the rejected entry, the generation is retired if no admitted engine remains or if the remaining pool plus engine buffers still exceeds the budget. Otherwise existing valid graphs may retain their shared pool. OOM always retires the generation. Retired layouts can be captured again when eligible on later calls; a smaller layout can still be captured after a rejection.

**Not in the budget.** Model weights, outputs held by the caller (always copies), the CUDA context and library state, other calculators and applications, and the **temporary peak of a capture**: regular reference evaluation, warm-up, capture, and numerical comparison all run before admission, so admission does not prevent every out-of-memory error. A capture that runs out of memory is answered by the regular path and releases the pool. The budget is a cache admission policy, not a hard cap on process VRAM. Measure these quantities again for the shared-computation implementation: timings and memory figures from the earlier specialized engines are historical results, not measurements of this revision.

**Eviction versus release.** The LRU limit drops lookup entries and their engine buffers; their graph blocks stay in the pool and are reused by later captures, so eviction does not lower the pool reservation. The pool is returned to the device when the generation is retired: by `clear_cuda_graphs()`, a model or contract change, a capture that runs out of memory, deletion of the calculator, or rejection/failure when no admitted engine remains or the remaining retained memory exceeds the budget. Existing valid graphs may keep their shared pool after another layout is rejected if that pool and their buffers still fit the budget. `clear_cuda_graphs()` first waits for this calculator's enqueued replays, then releases the pool with `torch.cuda.empty_cache()`. That call synchronizes the device and also releases other unused cached blocks of the process; it happens only at these rare retirement boundaries. Outputs returned earlier stay valid. Each capture also empties the allocator cache once before its warm-up (blocks cached for the caller's stream cannot serve the capture stream; on a 96-atom batch shrinking from 64 rows to 1, peak reserved memory was 2578 MiB with this and 3330 MiB without), and PyTorch's `torch.cuda.graph` synchronizes the device and empties the cache again for every capture. Expect a short process-wide pause at most once per new layout.

**Allocator API.** The pool is owned through `torch.cuda.MemPool`; accounting reads the per-pool allocator snapshot (`torch.cuda.memory_snapshot(pool_id)`, or the full snapshot filtered by `segment_pool_id`). A just-captured graph whose pool reports no segments is not cached, so no entry is recorded at zero cost. These lifecycle and accounting paths were validated with Torch 2.11 (earlier revision) and Torch 2.13 (this revision).

The first capture in a process may additionally compile/load Warp or Triton kernels; measure this cold-start cost separately.

`clear_cuda_graphs()` returns the pool but is not free: the next repeated layout needs regular evaluation, warm-up, allocation, capture and validation again. Keep a calculator alive for repeated inference when its model and input contract are unchanged. The one-warm-up capture schedule and pool lifecycle have been tested on the GPU/Torch combination above; validate cold capture, recapture, numerical agreement and memory release on other targets before relying on the measured costs.

PBC/cell input, user neighbor lists or `mol_idx`, padded neighbor-list batches (`N > nb_threshold`), deterministic mode, and higher-derivative requests retain regular behavior. `coord.requires_grad`, `charge.requires_grad`, or trainable parameters also prevent replay, preserving the regular autograd contract. Graph outputs are ordinary inference results; do not use this backend as a differentiable training layer. Use `train=True` for parameter training, and explicitly set `cuda_graph=False` when developing custom differentiation workflows.

## Mutation and Concurrency

The graph owns species/charge snapshots rather than aliases of user buffers. Ordinary PyTorch in-place updates, `load_state_dict`, parameter/buffer replacement, and changes to supported AEV/D3 constants invalidate cached engines. Captured storage addresses and capacity are checked **before** launch. The complete model contract is checked while the GPU replays; changed-model output is discarded and regular evaluation answers the call.

Mutations through `.data`, raw pointers, or external kernels can bypass tensor version counters and are outside this contract. Construct a new calculator after such changes. Replays launched on different CUDA streams are ordered: each call waits on the GPU for the previous replay (an event, not a host synchronization), because a calculator's graphs share one memory pool and static input buffers. Do not mutate a model concurrently with its evaluation, or call one calculator from several threads at once. Use separate calculators with serialized capture for independent workers. Runtime monkeypatching of global helper functions is not a supported deployment pattern; use `cuda_graph=False` while developing such changes.

## Dependencies and Validation Scope

This backend uses Python/PyTorch, Warp, and optional Triton dense AEV/ConvSV primitives; there is no separate C/C++ or assembly extension to build. If Triton cannot be imported, or the GPU has compute capability below 8.0, the dense engine uses the corresponding PyTorch expressions; `inference_kernels=True` then warns once at construction. Use the Triton version selected by your PyTorch installation rather than independently forcing an unrelated version.

The optional `pip install "aimnet[cuda-graph]"` extra declares the Linux Triton dependency; it does not change calculator defaults. CPU/other-platform core installations do not require this extra.

Graph capture requires `nvalchemi-toolkit-ops>=0.5.0` and `warp-lang>=1.16`. The package itself still accepts `nvalchemi-toolkit-ops>=0.4.1` for the regular path; with older versions installed, `cuda_graph=True` warns once, names the versions and the upgrade command, and uses the regular path without attempting a capture (0.4.1 does not order its Warp launches on PyTorch's stream, so its captures return wrong results). ROCm builds are refused the same way, since the Warp/nvalchemi kernels are CUDA-only. Reusable neighbor buffers use the public nvalchemi Torch API, which resets buffers and orders Torch/Warp work internally. The adapters do not import a private neighbor launcher or maintain a separate Warp stream context around neighbor and D3 calls.

Long-range neighbor storage belongs to the modules that consume it. Streaming Coulomb alone does not need an all-pairs matrix. When D3 shares the regular full-Coulomb list, the graph adapter retains those pairs, including pairs outside D3's energy smoothing range: coordination-number inputs must remain equivalent to the regular calculation. D3-only lists preserve the calculator's D3 cutoff policy. The dense adapter follows the regular path of the input shape: `[B, N, 3]` input keeps D3's own `smoothing_off` list (nb_mode 0), while packed `[N, 3]` input receives the regular packed path's long-range list (all pairs with full Coulomb), so the two shapes are separate cached layouts. No shorter physical cutoff is introduced to obtain a memory or timing improvement.

The package's declared PyTorch minimum and CI version matrix do not by themselves establish graph-backend coverage. Earlier specialized-engine validation, including padded batches and the graph-pool lifecycle, used RTX 3060 Laptop / WSL2 / Torch 2.11 / Warp 1.16 / Triton 3.6. That history does not establish the speed or peak memory of the shared-computation revision. This revision (dependency gate, admission tolerances, fallback warnings, cross-stream ordering) was validated on an NVIDIA L40S (compute capability 8.9) with Torch 2.13, Warp 1.16, nvalchemi-toolkit-ops 0.5.0 and Triton 3.7: the GPU test set, parity against the regular path from 3 to 4,992 atoms, and the FP32 noise measurements behind the tolerances. The release lock (Torch 2.14.1, Warp 1.18, nvalchemi-toolkit-ops 0.5.0) is qualified separately. Other GPU generations, operating systems, and supported Torch versions need their own release qualification. No V100, H100, or CPU-native-engine speedup is implied.

Historical validation of the preceding revision included the minimum Torch 2.10 CPU/core lane and an installed-wheel CPU smoke test without Triton. Repeat these checks for the shared-computation revision before release; historical checks do not qualify its CUDA graphs on Torch 2.10 or other GPU hardware.

## Fair Benchmarking

- Use the **same checkpoint**, exact inputs, long-range settings, precision, hardware, and thread counts. A different trained architecture or a legacy ensemble is a separate model comparison, not a backend speedup.
- First compare the four combinations of `cuda_graph=False/True` and `inference_kernels=False/True`, keeping `compile_model=False`. This separates kernel improvements from replay improvements. Then report a separate compile-only baseline (`compile_model=True, cuda_graph=False, inference_kernels=False`). Record dense/sparse layout and the environment override.
- Warm each path independently, synchronize CUDA around wall-clock timing, and report cold startup separately from steady state. Use multiple timing rounds and vary construction/measurement order.
- Measure energy-only and E+F separately. Include both fixed and moving geometries, and state whether host-to-device transfers are timed.
- Report allocated and reserved VRAM, capture peak, and retained cache memory separately. Do not infer memory savings from runtime.
- For CPU comparisons, fix and report `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and PyTorch thread counts. This backend adds no CPU replay accelerator; CPU remains a useful compatibility/control measurement.
- Test E/F/charge parity and relevant derivative/long-range checks before interpreting a timing ratio as usable acceleration.

See also [Performance Tuning](tutorials/performance.md) and [Calculator API](calculator.md).
