# CUDA-Graph Inference

CUDA-graph replay accelerates repeated energy, force, and charge evaluations of supported AIMNet2 models. It changes execution, not the trained potential, long-range terms, or units. This is an inference backend, not a training accelerator or a new model format.

## Defaults and Comparison Modes

`compile_model=False` remains the constructor default. It controls **`torch.compile` only**. The separate `cuda_graph=None` default enables eligible CUDA-graph replay unless `AIMNET_CUDA_GRAPH` disables it.

| Mode | Explicit constructor options in this fork | Execution |
| --- | --- | --- |
| Upstream-style regular | `compile_model=False, cuda_graph=False` | Unchanged regular calculator path |
| Upstream-style compiled | `compile_model=True, cuda_graph=False` | Regular calculator with a compiled NN forward |
| CUDA-graph replay | `compile_model=False, cuda_graph=True` | Regular first call; capture and replay on eligible repeated layouts |
| Fork default | `compile_model=False, cuda_graph=None` | Same graph eligibility, controlled by the environment |
| Both enabled | `compile_model=True, cuda_graph=True` | Eligible graph replay takes precedence; other calls use the regular/compiled path |

The upstream reference does not have the `cuda_graph` keyword: omit it when running unmodified upstream code. Here, “regular” does not mean “CPU” or “without Warp”; regular CUDA calculations already use supported Warp kernels. Dense/sparse layout, `torch.compile`, and CUDA-graph replay are separate choices.

```python
from aimnet.calculators import AIMNet2Calculator

regular = AIMNet2Calculator("aimnet2", compile_model=False, cuda_graph=False)
compiled = AIMNet2Calculator("aimnet2", compile_model=True, cuda_graph=False)
replay = AIMNet2Calculator("aimnet2", compile_model=False, cuda_graph=True)
```

!!! important "Do not accidentally benchmark graph replay as torch.compile"

    Set `cuda_graph=False` for a compile-only comparison. With both options enabled, the first call may incur compilation and later eligible calls use the specialized graph engine instead. There is generally no reason to pay both startup costs for a known graph-compatible workload.

Explicit `cuda_graph=True` or `False` overrides the environment. With `None`, `AIMNET_CUDA_GRAPH=0`, `false`, `no`, or `off` disables replay (case-insensitive). CPU calculations always use the regular path regardless of this option.

## Supported Calls

All of these conditions must hold:

- CUDA execution, inference mode (`train=False`), and `deterministic=False`.
- A stock, closed-shell AIMNet2 module with one charge channel, the supported `d2features` architecture, stock GELU layers and forward methods, canonical output heads, and no hooks or trainable parameters.
- Full `simple` external Coulomb with the supported short-range treatment, or no external Coulomb; external DFT-D3 is optional. DSF, Ewald, PME, and truncated full Coulomb use the regular path.
- Input keys are exactly `coord`, `numbers`, `charge`, with optional `mult`. The latter retains the regular closed-shell behavior and warning semantics; it does not add open-shell support.
- Coordinates are `[N, 3]` or `[B, N, 3]`, with `N >= 2`. A batch has equal atom counts and no zero-number padding. No input tensor requires gradients.
- No stress or Hessian request.

`B` is the number of independent structures and `N` is atoms per structure, not a training batch-size setting. Changing coordinates is supported; fixed layout does **not** mean fixed geometry. Within a layout, atomic species, total charges, batch size, atom count, and force-request mode stay fixed.

Outputs retain regular keys, shapes, dtypes, and device: energy is in **eV**, forces are in **eV/Å**, and charges are in **e**. Forces are the energy-derived `-∂E/∂R` values, not a separately fitted force model. FP32 operation ordering can change values at rounding level; bitwise equality is not promised.

### Artifacts and Models

Registry models and v2 `.pt` artifacts are eligible only if the loaded module satisfies the contract above. A `.pt` suffix alone does not establish compatibility. Legacy `.jpt` models, custom CPCM/QEq/Flow heads, SR-repulsion heads, SiLU substitutions, and other custom forward implementations are **not** accelerated by this backend. They use the regular path if the existing loader and calculator support them; this change does not add compatibility to previously unsupported artifacts. In particular, `compile_model=True` remains unsupported for legacy `.jpt` files.

## Capture, Memory, and Fallback

The first call for a layout uses the regular path. On the second call, the engine captures its operations and compares E/F/charges with regular evaluation at the same geometry. Later calls replay the graph. `forces=False` and `forces=True` have separate engines. `N <= nb_threshold` uses a dense engine; larger systems use a sparse neighbor-list engine. The existing `nb_threshold=120` default is unchanged.

| Policy | Current implementation |
| --- | --- |
| New-layout setup | Capture plus numerical check; historical measurements were approximately 0.07–0.48 s after warmup for tested 20–4,487-atom inputs |
| Cache | Per calculator, LRU, at most 16 engines |
| Cache budget | 25% of device total memory, using tracked reserved-memory growth; over-budget engines are not retained |
| Sparse neighbor overflow | Rebuild capacity, repeat the numerical check, and recheck the cache budget |
| Unsupported call | Regular path; no graph result returned |
| Capture/check/replay failure | Layout falls back to regular; one `RuntimeWarning` per calculator |

The memory budget is a **cache admission policy**, not a hard cap on process VRAM. Model weights, temporary capture allocations, CUDA context, allocator pools, other calculators, and other applications also consume memory. Capture can temporarily need more memory than warm replay. The first capture in a process may additionally compile/load Warp or Triton kernels; measure this cold-start cost separately.

PBC/cell input, user neighbor lists or `mol_idx`, padded batches, deterministic mode, and higher-derivative requests retain regular behavior. `coord.requires_grad`, `charge.requires_grad`, or trainable parameters also prevent replay, preserving the regular autograd contract. Graph outputs are ordinary inference results; do not use this backend as a differentiable training layer. Use `train=True` for parameter training, and explicitly set `cuda_graph=False` when developing custom differentiation workflows.

## Mutation and Concurrency

The graph owns species/charge snapshots rather than aliases of user buffers. Ordinary PyTorch in-place updates, `load_state_dict`, parameter/buffer replacement, and changes to supported AEV/D3 constants invalidate cached engines. Captured storage addresses and capacity are checked **before** launch. The complete model contract is checked while the GPU replays; changed-model output is discarded and regular evaluation answers the call.

Mutations through `.data`, raw pointers, or external kernels can bypass tensor version counters and are outside this contract. Construct a new calculator after such changes. Do not mutate a model concurrently with its evaluation, or share one mutable calculator across concurrent threads/streams. Use separate calculators with serialized capture for independent workers. Runtime monkeypatching of global helper functions is not a supported deployment pattern; use `cuda_graph=False` while developing such changes.

## Dependencies and Validation Scope

This backend uses Python/PyTorch, Warp, and an optional Triton dense AEV kernel; there is no separate C/C++ or assembly extension to build. If Triton cannot be imported, the dense engine uses the PyTorch AEV expression. Use the Triton version selected by your PyTorch installation rather than independently forcing an unrelated version.

The optional `pip install "aimnet[cuda-graph]"` extra declares the Linux Triton dependency; it does not change calculator defaults. CPU/other-platform core installations do not require this extra.

The sparse engine currently depends on the private nvalchemiops fixed-capacity neighbor-list helper. An incompatible helper can prevent capture, in which case the regular path is used. This dependency and fallback require regression checks when upgrading nvalchemiops.

The package's declared PyTorch minimum and upstream version matrix do not by themselves establish graph-backend coverage. Initial graph correctness, safety, and performance validation was on RTX 3060 Laptop / WSL2 / Torch 2.11 / Warp 1.16 / Triton 3.6. Other GPU generations, operating systems, and supported Torch versions need their own release qualification. No V100, H100, or CPU-native-engine speedup is implied by these GPU results.

The minimum Torch 2.10 CPU/core test lane and a fresh installed-wheel CPU smoke test without Triton have also passed. These checks establish CPU/package compatibility within their tested scope, not CUDA-graph qualification on Torch 2.10 or other GPU hardware.

## Fair Benchmarking

- Use the **same checkpoint**, exact inputs, long-range settings, precision, hardware, and thread counts. A different trained architecture or a legacy ensemble is a separate model comparison, not a backend speedup.
- Report regular (`compile_model=False, cuda_graph=False`), compile-only (`True, False`), and replay (`False, True`) separately; record dense/sparse layout and the environment override.
- Warm each path independently, synchronize CUDA around wall-clock timing, and report cold startup separately from steady state. Use multiple timing rounds and vary construction/measurement order.
- Measure energy-only and E+F separately. Include both fixed and moving geometries, and state whether host-to-device transfers are timed.
- Report allocated and reserved VRAM, capture peak, and retained cache memory separately. Do not infer memory savings from runtime.
- For CPU comparisons, fix and report `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and PyTorch thread counts. This fork adds no CPU replay accelerator; CPU remains a useful compatibility/control measurement.
- Test E/F/charge parity and relevant derivative/long-range checks before interpreting a timing ratio as usable acceleration.

See also [Performance Tuning](tutorials/performance.md) and [Calculator API](calculator.md).
