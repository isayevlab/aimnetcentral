# Performance Tuning: Maximum Speed

## What You'll Learn

- What happens during Warp kernel compilation and how to manage warmup costs
- How `compile_model` works and what it does (and does not) compile
- How to choose between dense and sparse execution modes via `nb_threshold`
- How to benchmark correctly and avoid common performance pitfalls

## Prerequisites

- AIMNet2 installed with GPU support (`pip install aimnet` + CUDA-enabled PyTorch)
- Familiarity with AIMNet2 calculator usage (see [First Calculation](single_point.md))
- A CUDA-capable GPU (performance tuning is primarily relevant for GPU execution)

## Step 1: Understanding Startup Costs

AIMNet2 uses NVIDIA Warp for GPU-accelerated kernels. There are two one-time costs you will encounter:

### Warp Initialization (2-5 seconds)

The first time any AIMNet2 module imports the Warp kernels, `wp.init()` runs automatically. This initializes the Warp runtime and detects available GPUs:

```python
import aimnet  # Triggers wp.init() on first kernel import: ~2-5 seconds
```

This cost is paid once per Python process.

### Kernel JIT Compilation (10-30 seconds)

The first GPU calculation triggers just-in-time (JIT) compilation of Warp CUDA kernels. Compiled kernels are cached on disk, so subsequent runs of the same Python environment start faster:

```python
from aimnet.calculators import AIMNet2Calculator
import torch

calc = AIMNet2Calculator("aimnet2")

# First call: triggers kernel JIT compilation (~10-30s)
result = calc({
    "coord": torch.randn(10, 3),
    "numbers": torch.tensor([6, 1, 1, 1, 1, 7, 1, 1, 1, 8]),
    "charge": 0.0,
}, forces=True)

# Second call: fast (~milliseconds)
result = calc({
    "coord": torch.randn(10, 3),
    "numbers": torch.tensor([6, 1, 1, 1, 1, 7, 1, 1, 1, 8]),
    "charge": 0.0,
}, forces=True)
```

!!! note Warp caches compiled kernels in `~/.cache/warp/` (or the directory set by `WARP_CACHE_PATH`). If you delete this cache, the next run will recompile.

## Step 2: Using torch.compile (compile_model)

The `compile_model=True` parameter applies `torch.compile()` to the neural network forward pass:

```python
# Without compilation
calc = AIMNet2Calculator("aimnet2")

# With compilation
calc_compiled = AIMNet2Calculator("aimnet2", compile_model=True)
```

### What Gets Compiled

`torch.compile()` optimizes **only the neural network forward pass** (the model itself). It does **not** compile:

- Neighbor list construction (adaptive neighbor lists)
- External Coulomb calculations (LRCoulomb, DSF, Ewald)
- External DFTD3 dispersion corrections
- Input preprocessing and output postprocessing

This means the speedup from compilation depends on what fraction of total compute time is spent in the NN forward pass vs. these other components.

### When to Use compile_model

**Good candidates for compilation:**

- Long MD trajectories (hundreds to thousands of steps)
- Geometry optimization (many force evaluations on similar-sized systems)
- Repeated evaluation on the same molecular system

**Poor candidates for compilation:**

- Single-point energy calculations (compilation overhead exceeds savings)
- Processing datasets with varying molecule sizes (may trigger recompilation)
- Quick exploratory calculations

### Compile Options

You can pass additional arguments to `torch.compile()` via `compile_kwargs`:

```python
# Default compilation (recommended starting point)
calc = AIMNet2Calculator("aimnet2", compile_model=True)

# With max-autotune for maximum optimization (longer compile time)
calc = AIMNet2Calculator(
    "aimnet2",
    compile_model=True,
    compile_kwargs={"mode": "max-autotune"},
)
```

!!! warning "Do not use reduce-overhead with varying input sizes" The `reduce-overhead` mode uses CUDA graphs, which record a fixed execution pattern. If your input sizes vary between calls (different numbers of atoms or neighbors), this causes **graph breaks** and can actually slow things down or produce errors. Only use `reduce-overhead` when the system size is truly constant.

    ```python
    # AVOID for varying sizes
    calc = AIMNet2Calculator(
        "aimnet2",
        compile_model=True,
        compile_kwargs={"mode": "reduce-overhead"},  # Bad for varying sizes
    )

    # SAFE default
    calc = AIMNet2Calculator("aimnet2", compile_model=True)
    ```

## Step 3: Dense vs Sparse Mode

The calculator automatically chooses between two execution modes based on system size, device, and PBC. Understanding this choice is key to performance tuning.

### Mode Selection Rules

| Condition | Mode | Complexity | Neighbor Lists |
| --- | --- | --- | --- |
| N <= `nb_threshold` AND CUDA AND non-PBC | **Dense** | O(N^2) | No (all-pairs) |
| N > `nb_threshold` | **Sparse** | O(N) | Yes |
| CPU (any size) | **Sparse** | O(N) | Yes |
| PBC (any size) | **Sparse** | O(N) | Yes |

**Dense mode** treats every atom pair as interacting (fully connected graph). This is fast on GPU for small molecules because it avoids the overhead of neighbor list construction, and GPU parallelism handles the O(N^2) interactions efficiently.

**Sparse mode** uses adaptive neighbor lists to limit interactions to atoms within a cutoff distance. This scales linearly with system size and is required for large systems, CPU execution, and periodic boundary conditions.

### Tuning nb_threshold

The default `nb_threshold=120` works well for most use cases. Adjust it based on your hardware:

```python
# Memory-constrained GPU (e.g., 8 GB) -- switch to sparse earlier
calc = AIMNet2Calculator("aimnet2", nb_threshold=80)

# High-memory GPU (e.g., 40+ GB) -- stay in dense mode longer
calc = AIMNet2Calculator("aimnet2", nb_threshold=200)
```

!!! warning "Avoid crossing the dense/sparse boundary with torch.compile" If you use `compile_model=True`, make sure all your inputs consistently land in the same mode (all dense or all sparse). Crossing the boundary (e.g., some calls with 100 atoms in dense mode, others with 150 atoms in sparse mode) causes recompilation each time, negating the benefits. Set `nb_threshold` so that your typical workload stays in one mode.

## Step 4: GPU Memory Considerations

### Adaptive Neighbor List Growth

The `AdaptiveNeighborList` starts with a buffer sized from an initial density estimate. If the actual number of neighbors exceeds the buffer, it **grows by 1.5x** and retries. This means:

- First calculations on dense systems may trigger one or two buffer growths
- Once the buffer stabilizes, subsequent calculations reuse the same allocation
- Buffer shrinks automatically when utilization drops below 50% of target

### Hessian Memory

Hessian calculation requires O(N^2) memory because the full second-derivative matrix has shape `(N, 3, N, 3)`. For a 100-atom molecule, this is 100 x 3 x 100 x 3 = 90,000 float values (~360 KB in float32), and for 1000 atoms it becomes 1000 x 3 x 1000 x 3 = 9,000,000 values (~36 MB in float32).

!!! warning Hessian computation is limited to **single molecules** and scales quadratically with atom count. For molecules larger than ~200 atoms, you may run out of GPU memory. Consider computing the Hessian on a CPU for large systems, or using finite-difference approaches for specific vibrational modes.

### Synchronization Points

The adaptive neighbor list contains one GPU-CPU synchronization point: `num_neighbors.max().item()` is called to determine the actual maximum neighbor count for trimming. This is a single `.item()` call per neighbor list computation (not per atom). For most workloads, this overhead is negligible compared to the NN forward pass.

## Step 5: Benchmarking Correctly

Incorrect benchmarking is the most common performance pitfall. Follow these rules:

### Rule 1: Warmup Before Timing

The first 1-2 calls are always slower due to kernel compilation, memory allocation, and `torch.compile` tracing. Always discard warmup calls:

```python
import time
import torch
from aimnet.calculators import AIMNet2Calculator

calc = AIMNet2Calculator("aimnet2", compile_model=True)
data = {
    "coord": torch.randn(50, 3, device="cuda"),
    "numbers": torch.randint(1, 9, (50,), device="cuda"),
    "charge": torch.tensor([0.0], device="cuda"),
}

# Warmup (discard timing)
for _ in range(2):
    calc(data, forces=True)

# Synchronize GPU before timing
torch.cuda.synchronize()
start = time.perf_counter()

n_iterations = 100
for _ in range(n_iterations):
    result = calc(data, forces=True)

# Synchronize GPU after timing
torch.cuda.synchronize()
elapsed = time.perf_counter() - start

print(f"Average time: {elapsed / n_iterations * 1000:.2f} ms/call")
```

### Rule 2: Synchronize the GPU

GPU operations are asynchronous. Without `torch.cuda.synchronize()`, you measure only the time to **launch** operations, not the time to **complete** them:

```python
# WRONG -- measures launch time only
start = time.perf_counter()
result = calc(data, forces=True)
elapsed = time.perf_counter() - start  # Misleadingly fast

# CORRECT -- measures actual compute time
torch.cuda.synchronize()
start = time.perf_counter()
result = calc(data, forces=True)
torch.cuda.synchronize()
elapsed = time.perf_counter() - start  # True wall time
```

### Rule 3: Use Realistic Inputs

Benchmark with inputs representative of your actual workload. System size, atom types, and whether PBC is active all affect performance characteristics.

## Step 6: Things to Avoid

### Do Not Use torch.autocast

AIMNet2 models are trained in float32 and expect float32 inputs. Mixed-precision via `torch.autocast` can produce incorrect results:

```python
# WRONG -- may produce incorrect results
with torch.autocast("cuda"):
    result = calc(data, forces=True)

# CORRECT -- use default float32
result = calc(data, forces=True)
```

### Do Not Set train=True for Inference

The `train` parameter controls whether gradients are tracked on model parameters. Setting `train=True` for inference wastes memory and slows computation:

```python
# WRONG for inference
calc = AIMNet2Calculator("aimnet2", train=True)

# CORRECT for inference (default)
calc = AIMNet2Calculator("aimnet2")  # train=False by default
```

!!! note `train=False` (the default) disables `requires_grad` on all model parameters. This reduces memory usage and improves `torch.compile` compatibility. The calculator still computes forces and Hessians correctly via autograd on the input coordinates.

### Do Not Use Multiple GPUs

AIMNet2 does not support multi-GPU execution (DataParallel or DistributedDataParallel). Use a single GPU per calculator instance. If you have multiple GPUs, run independent processes on each:

```python
# Process 1
calc = AIMNet2Calculator("aimnet2", device="cuda:0")

# Process 2 (separate Python process)
calc = AIMNet2Calculator("aimnet2", device="cuda:1")
```

## Performance Checklist

| Setting | Recommended | Notes |
| --- | --- | --- |
| Device | `"cuda"` (auto-detected) | 10-50x speedup over CPU |
| `compile_model` | `True` for repeated evals | Skip for single-point calculations |
| `compile_kwargs` | `{}` (default) | Avoid `reduce-overhead` with varying sizes |
| `nb_threshold` | `120` (default) | Lower for memory-constrained GPUs |
| `train` | `False` (default) | Only `True` when actually training |
| Warmup | 2 calls before timing | First calls include compilation overhead |
| Sync | `torch.cuda.synchronize()` | Required for accurate GPU timing |

## What's Next

- [Batch Processing](batch_processing.md) -- Combine batching with performance tuning
- [Periodic Systems](periodic_systems.md) -- PBC-specific performance considerations
- [Calculator API](../calculator.md) -- Full reference for all constructor parameters
