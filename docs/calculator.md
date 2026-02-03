# AIMNet2Calculator

This document provides detailed documentation of the `AIMNet2Calculator` class behavior.

## Overview

`AIMNet2Calculator` is a helper class for loading AIMNet2 models and performing inference. It handles:

- Model loading (from registry, file path, or `nn.Module`)
- External long-range (LR) module setup (Coulomb, DFTD3)
- Neighbor list computation and management
- Input preprocessing and output postprocessing
- Batching and periodic boundary conditions (PBC)

For LR module math and behavior, see [long_range.md](long_range.md).

## Quick Start

### Basic Inference

```python
from aimnet.calculators import AIMNet2Calculator

calc = AIMNet2Calculator("aimnet2")
result = calc({
    "coord": coords,    # (N, 3)
    "numbers": numbers, # (N,)
    "charge": 0.0,
})
energy = result["energy"]
```

### With Forces

```python
result = calc(data, forces=True)
forces = result["forces"]  # (N, 3)
```

### Periodic Systems

```python
calc.set_lrcoulomb_method("dsf", cutoff=15.0)
result = calc({
    "coord": coords,
    "numbers": numbers,
    "charge": 0.0,
    "cell": cell,  # (3, 3)
}, forces=True, stress=True)
```

### Changing Coulomb Methods

```python
# DSF (recommended for PBC)
calc.set_lrcoulomb_method("dsf", cutoff=15.0, dsf_alpha=0.2)

# Ewald (high accuracy, currently limited to non-batched case)
calc.set_lrcoulomb_method("ewald", ewald_accuracy=1e-8)

# Simple (all pairs)
calc.set_lrcoulomb_method("simple")
```

## Constructor

```python
AIMNet2Calculator(
    model: str | nn.Module = "aimnet2",
    nb_threshold: int = 120,
    needs_coulomb: bool | None = None,
    needs_dispersion: bool | None = None,
    device: str | None = None,
    compile_model: bool = False,
    compile_kwargs: dict | None = None,
)
```

### Parameters

#### `model`

Model to use for inference.

| Type                  | Behavior                                                             |
| --------------------- | -------------------------------------------------------------------- |
| `str` (registry name) | Loads from model registry (e.g., `"aimnet2"`), downloading if needed |
| `str` (file path)     | Loads from `.pt` (v2) or `.jpt` (v1 legacy) file if the path exists  |
| `torch.nn.Module`     | Uses provided module directly                                        |

For `torch.nn.Module`, metadata is read from `model.metadata` attribute if available (v2 models).

#### `nb_threshold`

Threshold for batching/flattening decisions. Default: `120`.

The calculator uses two input modes:

1. **Fully connected (mode 0)**: 3D batched input for coordinates `(num_mols, num_atoms, 3)` with all-pairs
   interactions (dense O(N²)). Fast on GPU for small systems.
2. **Flattened + neighbor lists (mode 1)**: 2D input for coordinates `(num_atoms, 3)` with `mol_idx` and neighbor
   lists (sparse O(N)). Used for large systems, CPU execution, and periodic systems.

| Condition                   | Behavior                                                       | Complexity            |
| --------------------------- | -------------------------------------------------------------- | --------------------- |
| `N > nb_threshold`          | If input is mode 0 (3D), flatten to mode 1 (2D with `mol_idx`) | O(N) linear           |
| `device == "cpu"`           | If input is mode 0 (3D), always flatten to mode 1              | O(N) linear           |
| `N < nb_threshold` and CUDA | Keep mode 0 (3D)                                               | O(N²) fully connected |

This affects memory usage and performance for batched inference. The mode=0 path uses a fully connected
graph (all-pairs interactions), which scales as O(N²) but is fast for GPU. The mode=1
path uses neighbor lists, which scale linearly with system size. Fully connected mode is not used
for periodic systems; PBC inputs always go through neighbor lists.

#### `needs_coulomb`

Whether to attach external Coulomb module.

| Value            | Behavior                                    |
| ---------------- | ------------------------------------------- |
| `None` (default) | Determined from model metadata              |
| `True`           | Force external Coulomb (overrides metadata) |
| `False`          | No external Coulomb (overrides metadata)    |

Only affects v2 format models. Legacy JIT models have embedded Coulomb.
If you override this flag on a model without Coulomb metadata, ensure it is
compatible with the expected subtraction for short range Coulomb contribution (see `coulomb_mode` in model metadata).

#### `needs_dispersion`

Whether to attach external DFTD3 module.

| Value            | Behavior                                  |
| ---------------- | ----------------------------------------- |
| `None` (default) | Determined from model metadata            |
| `True`           | Force external DFTD3 (overrides metadata) |
| `False`          | No external DFTD3 (overrides metadata)    |

Only affects new-format models. Raises `ValueError` if `needs_dispersion=True` but `d3_params` are missing in metadata.

#### `device`

Device to run the model on.

| Value            | Behavior                                      |
| ---------------- | --------------------------------------------- |
| `None` (default) | Auto-detect: uses CUDA if available, else CPU |
| `"cuda"`         | Force CUDA device                             |
| `"cpu"`          | Force CPU device                              |
| `"cuda:N"`       | Specific CUDA device (e.g., `"cuda:1"`)       |

#### `compile_model`

Whether to compile the model with `torch.compile()` for faster inference.

| Value             | Behavior                             |
| ----------------- | ------------------------------------ |
| `False` (default) | No compilation                       |
| `True`            | Compile model with `torch.compile()` |

Compilation adds overhead on first call but speeds up subsequent calls. Useful for
MD trajectories, geometry optimizations, or repeated evaluations.

#### `compile_kwargs`

Additional keyword arguments to pass to `torch.compile()`. Default is `None`.

```python
# Example: use reduce-overhead mode for lower latency
calc = AIMNet2Calculator("aimnet2", compile_model=True, compile_kwargs={"mode": "reduce-overhead"})
```

See [torch.compile documentation](https://pytorch.org/docs/stable/generated/torch.compile.html)
for available options.

### Metadata Resolution

```
Priority: explicit flags > model metadata > no external modules
```

| Model Source             | Metadata Source            |
| ------------------------ | -------------------------- |
| File path (`.pt`/`.jpt`) | Loaded from file           |
| `nn.Module`              | `model.metadata` attribute |
| No metadata + no flags   | No external LR modules     |

## Properties

### `device`

Device string (e.g., `"cuda"`, `"cpu"`, `"cuda:1"`). Set via constructor parameter or auto-detected.

### `cutoff`

Short-range model cutoff in Ångströms. Typically 5.0 Å.

### `cutoff_lr`

Primary long-range cutoff reference. Used for backward compatibility with legacy models.

### `coulomb_cutoff`

Coulomb-specific cutoff distance. Tracked separately from the DFTD3 cutoff.

| Method     | Value                                                       |
| ---------- | ----------------------------------------------------------- |
| `"simple"` | `inf` (all pairs)                                           |
| `"dsf"`    | Configured cutoff (default 15.0 Å)                          |
| `"ewald"`  | `None` (Ewald manages its own real-space cutoff internally) |

### `dftd3_cutoff`

DFTD3-specific cutoff distance. Default: 15.0 Å.

**Neighbor list behavior:**

The calculator keeps Coulomb and DFTD3 cutoffs independent. Long-range neighbor lists are:

- **Shared** when both cutoffs are finite and within 20% of each other
- **Separate** when both cutoffs are finite and differ by more than 20%
- **All pairs** for `"simple"` Coulomb (effectively no cutoff)
- **Ignored by Ewald**, which builds its own real-space/reciprocal sums

**Data dictionary keys:**

LR modules prefer their specific suffix, falling back to `_lr`:

- **LRCoulomb**: Tries `nbmat_coulomb` first, falls back to `nbmat_lr`
- **DFTD3/D3TS**: Tries `nbmat_dftd3` first, falls back to `nbmat_lr`

When neighbor lists are shared, all keys point to the same array.

**Modifying cutoffs:**

- `set_lr_cutoff(cutoff)`: Updates both Coulomb and DFTD3 cutoffs
- `set_lrcoulomb_method(method, cutoff)`: Updates Coulomb cutoff only
- `set_dftd3_cutoff(cutoff)`: Updates DFTD3 cutoff only

### `has_external_coulomb`

`True` if external `LRCoulomb` module is attached. `False` for legacy models with embedded Coulomb.

### `has_external_dftd3`

`True` if external `DFTD3` module is attached. `False` for legacy models or D3TS models.

### `coulomb_method`

Current Coulomb method: `"simple"`, `"dsf"`, `"ewald"`, or `None`.

Returns `None` for:

- Legacy models with embedded Coulomb
- Models without Coulomb

**Note on Ewald:**

Ewald summation uses its own internal real-space cutoff based on accuracy requirements.
When Ewald is selected, `coulomb_cutoff` is `None` and does not contribute to neighbor list computation.

## Methods

### `eval(data, forces=False, stress=False, hessian=False)`

Main inference method. Also callable via `calculator(data, ...)`.

**Parameters:**

| Parameter | Type   | Default  | Description                             |
| --------- | ------ | -------- | --------------------------------------- |
| `data`    | `dict` | required | Input data dictionary                   |
| `forces`  | `bool` | `False`  | Compute atomic forces                   |
| `stress`  | `bool` | `False`  | Compute stress tensor (requires `cell`) |
| `hessian` | `bool` | `False`  | Compute Hessian matrix                  |

**Returns:** Dictionary with computed outputs.

**Example:**

```python
calc = AIMNet2Calculator("aimnet2")
result = calc.eval({
    "coord": coords,      # (N, 3) or (B, N, 3)
    "numbers": numbers,   # (N,) or (B, N)
    "charge": charge,     # (1,) or (B,)
}, forces=True)

energy = result["energy"]
forces = result["forces"]
charges = result["charges"]
```

### `set_lrcoulomb_method(method, cutoff=15.0, dsf_alpha=0.2, ewald_accuracy=1e-8)`

Set the long-range Coulomb method.

**Parameters:**

| Parameter        | Type    | Default  | Description                                    |
| ---------------- | ------- | -------- | ---------------------------------------------- |
| `method`         | `str`   | required | `"simple"`, `"dsf"`, or `"ewald"`              |
| `cutoff`         | `float` | `15.0`   | Cutoff for DSF method (Å). Not used for Ewald. |
| `dsf_alpha`      | `float` | `0.2`    | Alpha parameter for DSF method                 |
| `ewald_accuracy` | `float` | `1e-8`   | Target accuracy for Ewald summation            |

**Behavior:**

| Method     | Description                    | Coulomb cutoff         |
| ---------- | ------------------------------ | ---------------------- |
| `"simple"` | Direct Coulomb sum (all pairs) | `inf`                  |
| `"dsf"`    | Damped shifted force           | Configured cutoff      |
| `"ewald"`  | Ewald summation                | Computed from accuracy |

**Ewald Accuracy Parameter:**

For Ewald summation, the `ewald_accuracy` parameter controls the real-space and
reciprocal-space cutoffs. The cutoffs are computed automatically based on system
geometry:

\[
\eta = \frac{(V^2 / N)^{1/6}}{\sqrt{2\pi}}
\]

\[
r\_{\text{cutoff}} = \sqrt{-2 \ln \varepsilon} \cdot \eta
\]

\[
k\_{\text{cutoff}} = \frac{\sqrt{-2 \ln \varepsilon}}{\eta}
\]

Where \(\varepsilon\) is the accuracy parameter, \(V\) is the cell volume, and
\(N\) is the number of atoms. Lower accuracy values (e.g., `1e-10`) give higher
precision but require more computation.

**Notes:**

- Updates external `LRCoulomb` module if present
- Automatically updates neighbor lists
- Issues warning for legacy models (no effect)
- Auto-switches to `"dsf"` when PBC is detected with `"simple"` method (see PBC notes below)

**Example:**

```python
calc = AIMNet2Calculator("aimnet2")
calc.set_lrcoulomb_method("dsf", cutoff=12.0, dsf_alpha=0.20)
calc.set_lrcoulomb_method("ewald", ewald_accuracy=1e-6)
```

### `set_lr_cutoff(cutoff)`

Set the unified long-range cutoff for all LR modules.

**Parameters:**

| Parameter | Type    | Description                               |
| --------- | ------- | ----------------------------------------- |
| `cutoff`  | `float` | Cutoff distance (Å) for LR neighbor lists |

**Notes:**

- Updates both Coulomb and DFTD3 cutoffs
- Ewald method ignores this cutoff (uses its own internal cutoff)
- Automatically rebuilds neighbor lists

**Example:**

```python
calc = AIMNet2Calculator("aimnet2")
calc.set_lr_cutoff(20.0)  # Updates both Coulomb and DFTD3 cutoffs
```

### `set_dftd3_cutoff(cutoff=None, smoothing_fraction=None)`

Set DFTD3 cutoff and smoothing.

**Parameters:**

| Parameter            | Type   | Default | Description |
| -------------------- | ------ | ------- | ----------- | ------------------------------------------ |
| `cutoff`             | `float | None`   | `15.0`      | Cutoff distance (Å)                        |
| `smoothing_fraction` | `float | None`   | `0.2`       | Fraction of cutoff used as smoothing width |

**Notes:**

- Only updates smoothing parameters for external DFTD3 modules
- Always updates neighbor list cutoffs used by the calculator
- For legacy models with embedded DFTD3, the embedded module’s smoothing parameters do not change,
  but the neighbor list cutoff provided by the calculator can still change dispersion behavior

**Example:**

```python
calc = AIMNet2Calculator("aimnet2")
calc.set_dftd3_cutoff(cutoff=20.0, smoothing_fraction=0.25)  # smoothing from 15A to 20A
```

## Input Format

### Required Keys

| Key       | Type      | Shape                   | Description            |
| --------- | --------- | ----------------------- | ---------------------- |
| `coord`   | `float32` | `(N, 3)` or `(B, N, 3)` | Atomic coordinates (Å) |
| `numbers` | `int64`   | `(N,)` or `(B, N)`      | Atomic numbers         |
| `charge`  | `float32` | `(1,)` or `(B,)`        | Molecular charge(s)    |

### Optional Keys

| Key              | Type      | Shape                   | Description                          |
| ---------------- | --------- | ----------------------- | ------------------------------------ |
| `mult`           | `float32` | `(B,)`                  | Multiplicity                         |
| `mol_idx`        | `int64`   | `(N,)`                  | Molecule index per atom              |
| `cell`           | `float32` | `(3, 3)` or `(B, 3, 3)` | Unit cell vectors                    |
| `nbmat`          | `int64`   | `(N, max_nb)`           | Pre-computed neighbor matrix         |
| `nbmat_lr`       | `int64`   | `(N, max_nb)`           | Long-range neighbor matrix           |
| `nb_pad_mask`    | `bool`    | `(N, max_nb)`           | Optional padding mask for `nbmat`    |
| `nb_pad_mask_lr` | `bool`    | `(N, max_nb)`           | Optional padding mask for `nbmat_lr` |
| `shifts`         | `float32` | `(N, max_nb, 3)`        | PBC shifts for neighbors             |
| `shifts_lr`      | `float32` | `(N, max_nb, 3)`        | PBC shifts for LR neighbors          |

### Input Conversion

The calculator automatically converts:

- NumPy arrays → PyTorch tensors
- Python lists → PyTorch tensors
- Scalar tensors → Shape `(1,)`
- All tensors → Correct dtype and device

Any keys not listed in required/optional tables are ignored during input conversion.

## Output Format

| Key       | Shape                   | Description                   |
| --------- | ----------------------- | ----------------------------- |
| `energy`  | `(1,)` or `(B,)`        | Total energy per molecule     |
| `charges` | `(N,)` or `(B, N)`      | Atomic partial charges        |
| `forces`  | `(N, 3)` or `(B, N, 3)` | Atomic forces (if requested)  |
| `stress`  | `(3, 3)` or `(B, 3, 3)` | Stress tensor (if requested)  |
| `hessian` | `(N, 3, N, 3)`          | Hessian matrix (if requested) |

**Notes:**

- `forces` requires `forces=True` in `eval()`
- `stress` requires `stress=True` and `cell` in input
- `hessian` requires `hessian=True`, only for single molecules

## Batching and Neighbor Modes

The calculator chooses between dense and sparse execution based on system size and device. The goal
is to keep small GPU workloads fast while keeping large or CPU workloads linear in memory.

### Dense Mode (O(N²))

- **When**: `N < nb_threshold` **and** CUDA is available
- **Input**: 3D batched `(B, N, 3)`
- **Behavior**: No neighbor list; the model uses a fully connected graph
- **Tradeoff**: Fast on GPU for small molecules, but quadratic memory

### Sparse Mode (O(N))

- **When**: `N > nb_threshold` **or** CPU execution
- **Input**: Flattened 2D `(N_total, 3)` with `mol_idx`
- **Behavior**: Adaptive neighbor lists limit interactions to within `cutoff`
- **Tradeoff**: Linear memory with a small overhead for neighbor list construction

### Mode 2: Batched Sparse (manual)

- **Input**: 3D batched `(B, N, 3)` plus 3D neighbor matrix `(B, N, max_nb)`
- **Note**: Supported by the model, but not selected automatically. Use this mode by
  supplying a 3D `nbmat` explicitly.

### Choosing the Right Mode

The calculator automatically selects between two execution modes:

**Mode 0 (Dense, O(N²) fully connected):** Every atom interacts with every other atom in an all-to-all manner. No neighbor list is constructed. This mode is only used for **batches of small molecules with the same number of atoms on GPU**. Specifically:

- Input must be 3D batched coordinates (B, N, 3)
- N ≤ nb_threshold (default 120 atoms per molecule)
- CUDA device available
- No periodic boundary conditions

In this case, the O(N²) fully connected approach is more efficient than constructing neighbor lists.

**Mode 1 (Sparse, O(N) with neighbor lists):** Uses adaptive neighbor lists to limit interactions to atoms within cutoff distance. This mode is used in all other cases:

- Periodic boundary conditions (required for periodic images)
- CPU execution
- Large systems (N > nb_threshold)
- Variable-sized molecules

**Key Takeaways:**

- PBC always requires neighbor lists (Mode 1)
- CPU always uses neighbor lists (Mode 1)
- Small batched molecules on GPU use fully connected (Mode 0)
- Large systems use neighbor lists (Mode 1)

### Flattening Logic

For 3D batched inputs, `AIMNet2Calculator` decides whether to flatten based on `nb_threshold`:

```python
# nb_threshold default is 120
if device == "cpu" or max_atoms > nb_threshold:
    # FLATTEN -> Mode 1 (Sparse)
    # Computes neighbor list
else:
    # KEEP 3D -> Mode 0 (Dense)
    # Implicit all-pairs
```

## Periodic Boundary Conditions (PBC)

### Input Requirements

```python
data = {
    "coord": coords,
    "numbers": numbers,
    "charge": charge,
    "cell": cell,  # (3, 3) or (num_systems, 3, 3)
}
```

### Behavior

1. Coordinates wrapped into unit cell via `move_coord_to_cell()`
2. Neighbor lists include periodic image shifts
3. Coulomb method auto-switches to `"dsf"` if `"simple"` (with warning).
   For legacy JIT models the embedded Coulomb method cannot be changed at runtime; the warning
   indicates that only the calculator’s external setting was updated.
4. Multiple molecules with PBC: raises `NotImplementedError`

### Coulomb Method for PBC

| Initial Method      | Action                              |
| ------------------- | ----------------------------------- |
| `"simple"`          | Auto-switch to `"dsf"` with warning |
| `"dsf"` / `"ewald"` | No change                           |

## Neighbor List Management

### Adaptive Neighbor Lists

The calculator uses `AdaptiveNeighborList` for automatic buffer management in **Mode 1**:

- **Initial sizing**: Based on density estimate and cutoff
- **Overflow handling**: Increases buffer by 1.5x and retries
- **Underutilization**: Shrinks if utilization < 2/3 of target (hysteresis)
- **Minimum buffer**: 16 neighbors
- **Memory alignment**: Rounded to multiples of 16

### Neighbor List Format and Padding

- Neighbor lists are stored as integer matrices `nbmat` with shape `(N_total, max_neighbors)`.
- Each row contains neighbor indices for a single atom.
- Rows are padded with a dummy index (typically `N_total`) when an atom has fewer neighbors than
  `max_neighbors`.
- The buffer grows on overflow (×1.5) and shrinks when utilization drops well below target,
  which helps performance remain stable as density changes.

### Module Suffix Fallback

LR modules prefer their specific neighbor list key, with fallback to `_lr`:

- **LRCoulomb (simple/dsf)**: Tries `nbmat_coulomb`, falls back to `nbmat_lr`
- **DFTD3/D3TS**: Tries `nbmat_dftd3`, falls back to `nbmat_lr`

**Note:** Ewald uses its own internal neighbor list and ignores calculator cutoffs.

## Device Handling

### Automatic Selection

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

### Placement

| Component          | Device                     |
| ------------------ | -------------------------- |
| Model              | `self.device`              |
| External LRCoulomb | `self.device`              |
| External DFTD3     | `self.device`              |
| Input tensors      | Converted to `self.device` |
| Output tensors     | Remain on `self.device`    |

## External LR Module Configuration

### LRCoulomb Setup

When `needs_coulomb=True`:

```python
LRCoulomb(
    key_in="charges",
    key_out="energy",
    method="simple",  # Default, changeable via set_lrcoulomb_method()
    rc=metadata.get("coulomb_sr_rc", 4.6),
    envelope=metadata.get("coulomb_sr_envelope", "exp"),
    subtract_sr=not sr_embedded,  # Based on coulomb_mode
)
```

### DFTD3 Setup

When `needs_dispersion=True` and `d3_params` available:

```python
DFTD3(
    s8=d3_params["s8"],
    a1=d3_params["a1"],
    a2=d3_params["a2"],
    s6=d3_params.get("s6", 1.0),
)
```

### How LR Modules Are Attached

External LR modules are attached based on model metadata unless overridden by constructor flags:

- If `needs_coulomb=True`, an external `LRCoulomb` is created.
  If `coulomb_mode="sr_embedded"`, the model already subtracts SR Coulomb internally and the
  external module adds full Coulomb on top.
- If `needs_dispersion=True` and `d3_params` are present, an external `DFTD3` is created.
  If `d3_params` are missing, initialization raises `ValueError`.

Explicit `needs_coulomb` / `needs_dispersion` flags override metadata.

### Cutoff Handling for LR Modules

- **Coulomb**: `set_lrcoulomb_method()` selects the method and updates the Coulomb cutoff
  (`inf` for `"simple"`, finite for `"dsf"`, `None` for `"ewald"`).
- **DFTD3**: `set_dftd3_cutoff()` updates the DFTD3 cutoff and smoothing window.
- **Unified control**: `set_lr_cutoff()` sets both Coulomb and DFTD3 cutoffs to the same value.
- **Ewald**: Uses its own internal neighbor list; calculator cutoffs do not apply.

## Default Values

### LR Module Defaults

| Parameter                                        | Default | Description                                 |
| ------------------------------------------------ | ------- | ------------------------------------------- |
| `set_lrcoulomb_method(..., cutoff=15.0)`         | `15.0`  | Default LR cutoff for DSF (Å)               |
| `set_lrcoulomb_method(..., ewald_accuracy=1e-8)` | `1e-8`  | Default accuracy for Ewald summation        |
| `set_dftd3_cutoff(..., smoothing_fraction=0.2)`  | `0.2`   | DFTD3 smoothing width as fraction of cutoff |

### Coulomb Defaults

| Parameter             | Default | Description                               |
| --------------------- | ------- | ----------------------------------------- |
| `coulomb_sr_rc`       | `4.6` Å | Short-range Coulomb cutoff                |
| `coulomb_sr_envelope` | `"exp"` | Envelope function (`"exp"` or `"cosine"`) |

#### SR Coulomb Cutoff Constraint

**`coulomb_sr_rc` must be ≤ model `cutoff`**

The short-range Coulomb cutoff defines the distance within which SR Coulomb interactions are computed by the embedded `SRCoulomb` module. This cutoff must be less than or equal to the model's short-range cutoff because:

- SRCoulomb uses the same neighbor list as the neural network
- Atom pairs beyond the model cutoff are not visible to SRCoulomb
- Typical configuration: `coulomb_sr_rc=4.6` Å with model `cutoff=5.0` Å

The envelope function (`"exp"` or `"cosine"`) determines how the SR interaction smoothly decays to zero at the cutoff boundary.

## Legacy Model Compatibility

Legacy JIT models (`.jpt`) have different behavior:

| Feature                  | Legacy                        | New Format                             |
| ------------------------ | ----------------------------- | -------------------------------------- |
| Coulomb                  | Embedded in model             | External module                        |
| DFTD3/D3BJ               | Embedded in model             | External module                        |
| `set_lrcoulomb_method()` | Warning, no effect            | Updates method                         |
| `set_lr_cutoff()`        | No effect on embedded modules | Updates `cutoff_lr` for all LR modules |
| `set_dftd3_cutoff()`     | No effect on embedded modules | Updates smoothing for external DFTD3   |
| `has_external_coulomb`   | `False`                       | `True` (if applicable)                 |

## Error Handling

### Common Errors

| Condition                                   | Error                 |
| ------------------------------------------- | --------------------- |
| Invalid model type                          | `TypeError`           |
| Missing required input key                  | `KeyError`            |
| Hessian with multiple molecules             | `NotImplementedError` |
| PBC with multiple molecules                 | `NotImplementedError` |
| Invalid Coulomb method                      | `ValueError`          |
| `needs_dispersion=True` without `d3_params` | `ValueError`          |

### Warnings

| Condition                                | Warning                  |
| ---------------------------------------- | ------------------------ |
| `set_lrcoulomb_method()` on legacy model | Warns, no effect         |
| PBC with `"simple"` Coulomb              | Auto-switches to `"dsf"` |

## Complete Example

```python
import torch
from aimnet.calculators import AIMNet2Calculator

# Create calculator
calc = AIMNet2Calculator("aimnet2")

# Check configuration
print(f"Device: {calc.device}")
print(f"Cutoff: {calc.cutoff}")
print(f"Has external Coulomb: {calc.has_external_coulomb}")
print(f"Coulomb method: {calc.coulomb_method}")

# Configure for PBC
calc.set_lrcoulomb_method("dsf", cutoff=12.0)

# Prepare input
data = {
    "coord": torch.randn(20, 3),
    "numbers": torch.randint(1, 10, (20,)),
    "charge": torch.tensor([0.0]),
}

# Run inference
result = calc.eval(data, forces=True)

print(f"Energy: {result['energy']}")
print(f"Forces shape: {result['forces'].shape}")
print(f"Charges shape: {result['charges'].shape}")
```

## Performance Tips

### Hardware Acceleration

**Use GPU for best performance:**

```python
# Automatically uses CUDA if available
calc = AIMNet2Calculator("aimnet2")
print(calc.device)  # "cuda" or "cpu"
```

GPU provides 10-50x speedup over CPU for typical workloads.

### Compile Mode

**Use `compile_model=True` for additional speedup:**

```python
# Basic compilation
calc = AIMNet2Calculator("aimnet2", compile_model=True)

# With custom compile options
calc = AIMNet2Calculator(
    "aimnet2",
    compile_model=True,
    compile_kwargs={"mode": "reduce-overhead"}
)
```

**Characteristics:**

- First call will be slow (compilation overhead)
- Subsequent calls are faster
- Works with both periodic and non-periodic systems

**When to use:**

- Long MD trajectories
- Geometry optimization with many steps
- Repeated evaluation on same system size

**When not to use:**

- Single evaluations (compilation overhead outweighs benefit)
- Varying system sizes (may trigger recompilation)

### Memory Management

**Tune `nb_threshold` for your workload:**

```python
# Conservative (less memory, earlier sparse mode)
calc = AIMNet2Calculator("aimnet2", nb_threshold=80)

# Aggressive (more memory, faster on GPU)
calc = AIMNet2Calculator("aimnet2", nb_threshold=150)
```

**For large systems on GPU:**

- Lower `nb_threshold` to use sparse mode
- Reduces memory footprint
- Enables processing larger molecules

**For many small molecules:**

- Higher `nb_threshold` to use dense mode longer
- Maximizes GPU parallelism
- Faster overall throughput

### Pre-compute Neighbor Lists

**Avoid recomputation by caching:**

```python
# First call: computes neighbor lists
result1 = calc(data, forces=True)

# If geometry unchanged, reuse same data dict
# Calculator caches neighbor lists internally
result2 = calc(data, forces=False)  # Reuses cached lists
```

**For custom workflows, provide nbmat explicitly:**

```python
# Compute once
nbmat, _, shifts = compute_neighbor_list(coords, cutoff=5.0)

# Use many times
for _ in range(1000):
    result = calc({
        "coord": coords,
        "numbers": numbers,
        "charge": 0.0,
        "nbmat": nbmat,
        "shifts": shifts,
    }, forces=True)
```

### Coulomb Method Selection

**Choose method for your system:**

| System Type       | Method     | Parameter      | Notes            |
| ----------------- | ---------- | -------------- | ---------------- |
| Small non-PBC     | `"simple"` | N/A            | All pairs, exact |
| Large non-PBC     | `"dsf"`    | cutoff=12-15 Å | O(N) scaling     |
| PBC systems       | `"dsf"`    | cutoff=12-15 Å | Recommended      |
| High-accuracy PBC | `"ewald"`  | accuracy=1e-8  | Research-grade   |

```python
# Fast for non-PBC
calc.set_lrcoulomb_method("simple")

# Efficient for PBC
calc.set_lrcoulomb_method("dsf", cutoff=15.0)

# High-accuracy for research
calc.set_lrcoulomb_method("ewald", ewald_accuracy=1e-8)
```

### Multi-threading (CPU)

**Set thread count for CPU execution:**

```python
import torch
torch.set_num_threads(4)  # Use 4 CPU cores

calc = AIMNet2Calculator("aimnet2")  # Will use CPU
```
