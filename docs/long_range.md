# Long-Range Modules

This page documents the long-range (LR) modules implemented in `aimnet/modules/lr.py`. All modules operate on
the shared data dictionary and add their contributions to `data[key_out]` (usually `energy`).

## Choosing a Coulomb Method

Select the appropriate method based on your system and accuracy requirements.

### Simple (All-Pairs Pairwise Coulomb)

**Only for non-periodic systems.** The calculator automatically switches to DSF if PBC is detected.

**When to use:**

- Small non-periodic systems (< 100 atoms)
- Quick calculations where exact Coulomb is acceptable
- Non-production exploratory work

**When NOT to use:**

- Periodic systems (NOT SUPPORTED - auto-switches to DSF)
- Large systems (O(N²) fully connected becomes prohibitive)
- Production MD (inefficient for repeated evaluation)

**Configuration:**

```python
calc.set_lrcoulomb_method("simple")
# No cutoff - all pairs evaluated
# WARNING: Will auto-switch to DSF if PBC (cell) is provided
```

**Characteristics:**

- Exact pairwise 1/r Coulomb sum for non-periodic systems
- All atom pairs evaluated (fully connected)
- O(N²) complexity
- Not applicable to periodic boundary conditions

### DSF (Damped Shifted Force)

**When to use:**

- Periodic boundary conditions (recommended)
- Large non-periodic systems (> 200 atoms)
- Production molecular dynamics
- When computational efficiency matters

**When NOT to use:**

- When highest accuracy is required (use Ewald instead)
- Very small systems where Simple is faster

**Configuration:**

```python
calc.set_lrcoulomb_method("dsf", cutoff=15.0, dsf_alpha=0.2)
```

**Parameters:**

| Parameter   | Typical Range | Default | Notes                                             |
| ----------- | ------------- | ------- | ------------------------------------------------- |
| `cutoff`    | 12-20 Å       | 15.0 Å  | Larger = more accurate, more expensive            |
| `dsf_alpha` | 0.1-0.3       | 0.2     | Damping strength; 0.2 works well for most systems |

**Tuning guidelines:**

- Start with cutoff=15.0 Å, alpha=0.2 (default)
- Dense systems: reduce cutoff to 12 Å
- Dilute/surface systems: increase cutoff to 18-20 Å
- Validate: energies should converge within ~0.1 kcal/mol as cutoff increases

**Characteristics:**

- Smooth truncation at cutoff with shifted force
- Maintains charge neutrality
- O(N) scaling with neighbor lists
- Energy and forces continuous at cutoff
- Based on Wolf summation method

### Ewald Summation

**When to use:**

- Research-grade accuracy required
- Benchmarking and validation studies
- Systems where electrostatics dominate behavior
- When computational cost is acceptable

**When NOT to use:**

- Long MD trajectories (slower than DSF)
- When DSF accuracy is sufficient
- Very large systems (reciprocal space cost increases)

**Configuration:**

```python
# Default accuracy (1e-8)
calc.set_lrcoulomb_method("ewald")

# Custom accuracy (higher precision, more computation)
calc.set_lrcoulomb_method("ewald", ewald_accuracy=1e-10)

# Lower accuracy (faster, but less precise)
calc.set_lrcoulomb_method("ewald", ewald_accuracy=1e-6)
```

**Accuracy Parameter:**

The `ewald_accuracy` parameter controls the real-space and reciprocal-space cutoffs.
Lower values give higher precision but require more computation. The cutoffs are
computed automatically based on system geometry:

\[
\eta = \frac{(V^2 / N)^{1/6}}{\sqrt{2\pi}}
\]

\[
r*{\text{cutoff}} = \sqrt{-2 \ln \varepsilon} \cdot \eta, \quad
k*{\text{cutoff}} = \frac{\sqrt{-2 \ln \varepsilon}}{\eta}
\]

Where \(\varepsilon\) is the accuracy parameter, \(V\) is the cell volume, and
\(N\) is the number of atoms.

**Characteristics:**

- Splits Coulomb into real-space + reciprocal-space + self-energy terms
- Configurable accuracy target (default 1e-8)
- Automatically determines k-space vectors based on accuracy
- O(N log N) to O(N^1.5) complexity depending on implementation
- Most accurate method for periodic systems

**When Ewald matters:**

- Computing precise thermodynamic properties
- Benchmark comparisons against QM calculations
- Validation of other Coulomb methods
- Systems with significant long-range charge ordering

## Method Comparison

| Method | Complexity               | PBC Support | Typical Use Case             | Notes                    |
| ------ | ------------------------ | ----------- | ---------------------------- | ------------------------ |
| Simple | O(N²) fully connected    | No          | Small molecules, quick tests | Auto-switches for PBC    |
| DSF    | O(N) with neighbor lists | Yes         | Production MD, large systems | Recommended for PBC      |
| Ewald  | O(N log N) to O(N^1.5)   | Yes         | High-accuracy benchmarks     | Research-grade precision |

**Accuracy hierarchy:** Ewald > DSF > Simple (for PBC)

**Speed hierarchy:** Simple (small N) > DSF > Ewald

**Recommendation:** Use DSF for periodic systems unless you need Ewald's precision.

## Common Data and Neighbor-List Handling

**Neighbor-list keys and suffix fallback**

- Coulomb modules prefer `nbmat_coulomb` and fall back to `nbmat_lr`.
- Dispersion modules (`DFTD3`, `D3TS`) prefer `nbmat_dftd3` and fall back to `nbmat_lr`.

This fallback behavior is implemented via `nbops.resolve_suffix`, and distance calculation is lazily computed
with `ops.lazy_calc_dij`.

**Distance and masking**

Distances use `d_ij{suffix}` derived from `coord` (and `shifts{suffix}` when PBC is present). Padding/diagonal
pairs are masked via `mask_ij{suffix}`. For DSF, pairs beyond `Rc` are also masked.

## LRCoulomb

`LRCoulomb` computes a long-range Coulomb contribution, optionally subtracting the short-range (SR) part to
avoid double counting.

**Inputs**

- `charges` (default `key_in`)
- `coord`, `cell` (for Ewald)
- Neighbor lists as described above

**Methods**

1. **Simple (full pairwise Coulomb)**

- Pair energy: `e_ij = q_i q_j / r_ij`
- Total energy: `E = factor * sum_i sum_j e_ij` (accumulated in float64)
- If `subtract_sr=True`, subtracts the SR energy described below.

2. **DSF (damped shifted force)**

Uses `ops.coulomb_matrix_dsf`:

```
J(r) = erfc(alpha r)/r
     - erfc(alpha Rc)/Rc
     + (r - Rc) * (erfc(alpha Rc)/Rc^2 + 2 alpha exp(-(alpha Rc)^2) / (Rc sqrt(pi)))
```

Pairs with `r > Rc` or masked entries are zeroed. Energy is:

`E = factor * sum_i sum_j q_i q_j J(r_ij)`

If `subtract_sr=True`, the SR term is subtracted.

3. **Ewald**

Uses `ops.coulomb_matrix_ewald`, which implements real-space + reciprocal-space + self terms with
fixed accuracy (`1e-8`). This path requires a single molecule with `coord.ndim == 2` and `cell.ndim == 2`.

If `subtract_sr=True`, the SR term is subtracted.

**SR subtraction (shared by all methods)**

The SR term uses an envelope cutoff:

- `exp`: `fc(d) = exp(-1 / (1 - (d/rc)^2)) / 0.36787944117144233`
- `cosine`: `fc(d) = 0.5 * (cos(pi * d / rc) + 1)`

SR pair energy: `e_ij = fc(r_ij) * q_i q_j / r_ij`

The SR contribution is summed per molecule and subtracted from `key_out` when `subtract_sr=True`.

## SRCoulomb

`SRCoulomb` subtracts the SR Coulomb contribution from `key_out`. It uses the same SR formula as `LRCoulomb`
(envelope + cutoff) and is typically embedded in models trained with SR Coulomb so that external LR methods
can add the full Coulomb energy without double counting.

## DispParam

`DispParam` produces per-atom dispersion parameters (`c6`, `alpha`) from a reference table.

- Reference is loaded from a `.pt` file or provided as `ref_c6` / `ref_alpha`.
- Per-atom scaling uses `disp_param_mult = exp(clamp(disp_param, -4, 4))`.
- Output `disp_param` has shape `(N, 2)` with columns `(c6, alpha)`.

## D3TS

`D3TS` implements a DFT-D3-like pairwise dispersion with the TS combination rule.

**Inputs**

- `disp_param` (from `DispParam`)
- `coord`, `numbers`
- Neighbor lists with `_dftd3` or `_lr` suffix

**Key equations**

TS combination rule:

`c6_ij = 2 c6_i c6_j / (c6_i * alpha_j / alpha_i + c6_j * alpha_i / alpha_j)`

Let `rr = r4r2[numbers]` and `rr_ij = 3 * rr_i * rr_j`. The BJ-like radius term is:

`r0_ij = a1 * sqrt(rr_ij) + a2`

Distances are converted to Bohr: `r = d_ij * Bohr_inv`.

Energy per pair:

`e_ij = c6_ij * (s6 / (r^6 + r0_ij^6) + s8 * rr_ij / (r^8 + r0_ij^8))`

Total energy is summed per molecule and multiplied by `-half_Hartree`.

## DFTD3

`DFTD3` computes DFT-D3 dispersion correction using the BJ damping function with C6/C8 terms. Uses the `dftd3_energy` custom autograd op from `aimnet.modules.ops`. No 3-body term (Axilrod-Teller-Muto) is included.

### Smoothing Window

Dispersion interactions are smoothly damped to zero near the cutoff to ensure continuous energy and forces.

The smoothing window is defined by:

```python
smoothing_on = cutoff * (1 - smoothing_fraction)
smoothing_off = cutoff
```

**Parameters:**

| Parameter            | Typical Value | Default | Effect                                          |
| -------------------- | ------------- | ------- | ----------------------------------------------- |
| `cutoff`             | 15-20 Å       | 15.0 Å  | Interaction cutoff distance                     |
| `smoothing_fraction` | 0.1-0.3       | 0.2     | Width of smoothing region as fraction of cutoff |

**Example:** With cutoff=15.0 Å and smoothing_fraction=0.2:

- Full dispersion energy for r < 12.0 Å (smoothing_on)
- Smoothly interpolated for 12.0 Å < r < 15.0 Å
- Zero contribution for r > 15.0 Å

### Tuning DFTD3 Parameters

**Adjusting cutoff:**

```python
# Default: 15.0 Å
calc.set_dftd3_cutoff(cutoff=15.0, smoothing_fraction=0.2)

# For dense systems: shorter cutoff
calc.set_dftd3_cutoff(cutoff=12.0, smoothing_fraction=0.2)

# For surface/interface: longer cutoff
calc.set_dftd3_cutoff(cutoff=20.0, smoothing_fraction=0.15)
```

**Smoothing fraction guidelines:**

- Smaller (0.1): Sharper transition, may need tighter convergence
- Default (0.2): Good balance for most systems
- Larger (0.3): Smoother transition, more conservative

**Validation:**
Test energy convergence by varying cutoff. Dispersion energy should change by < 0.05 kcal/mol when cutoff increases by 2 Å.

### Forces

If `compute_forces=True` and coordinates require grad, forces are automatically computed as `-grad(energy)` and added to `data["forces"]`.

### D3 Parameters

DFTD3 requires functional-specific parameters stored in model metadata:

```python
d3_params = {
    "s6": 1.0,      # C6 scaling
    "s8": 0.3908,   # C8 scaling (functional-dependent)
    "a1": 0.5660,   # BJ damping parameter
    "a2": 3.1280,   # BJ damping parameter
}
```

These are set during model training/export and should not be modified for inference.

## Complete Examples

### DSF for Periodic System

```python
from aimnet.calculators import AIMNet2Calculator
import torch

# Create calculator
calc = AIMNet2Calculator("aimnet2")

# Configure DSF method
calc.set_lrcoulomb_method("dsf", cutoff=15.0, dsf_alpha=0.2)

# Periodic system
cell = torch.tensor([
    [10.0, 0.0, 0.0],
    [0.0, 10.0, 0.0],
    [0.0, 0.0, 10.0],
])

result = calc({
    "coord": coords,  # (N, 3)
    "numbers": numbers,  # (N,)
    "charge": 0.0,
    "cell": cell,
}, forces=True, stress=True)

print(f"Energy: {result['energy'].item():.4f} eV")
print(f"Stress: {result['stress']}")  # (3, 3)
```

### Ewald for High Accuracy

```python
# Configure Ewald method with default accuracy (1e-8)
calc.set_lrcoulomb_method("ewald")

# Same periodic system as above
result_ewald = calc({
    "coord": coords,
    "numbers": numbers,
    "charge": 0.0,
    "cell": cell,
}, forces=True)

# Compare with DSF
calc.set_lrcoulomb_method("dsf", cutoff=15.0)
result_dsf = calc({
    "coord": coords,
    "numbers": numbers,
    "charge": 0.0,
    "cell": cell,
}, forces=True)

energy_diff = abs(result_ewald["energy"] - result_dsf["energy"])
print(f"DSF vs Ewald energy difference: {energy_diff.item():.4f} eV")
# Typically < 0.01 eV for well-converged cutoffs
```

### Changing Methods at Runtime

```python
calc = AIMNet2Calculator("aimnet2")

# Start with simple
calc.set_lrcoulomb_method("simple")
result_simple = calc(data)

# Switch to DSF
calc.set_lrcoulomb_method("dsf", cutoff=15.0)
result_dsf = calc(data)

# Switch to Ewald
calc.set_lrcoulomb_method("ewald", ewald_accuracy=1e-8)
result_ewald = calc(data)

# Compare energies
print(f"Simple: {result_simple['energy'].item():.4f} eV")
print(f"DSF:    {result_dsf['energy'].item():.4f} eV")
print(f"Ewald:  {result_ewald['energy'].item():.4f} eV")
```

### Separate Coulomb and DFTD3 Cutoffs

```python
# Set different cutoffs for Coulomb and DFTD3
calc.set_lrcoulomb_method("dsf", cutoff=12.0)  # Coulomb cutoff
calc.set_dftd3_cutoff(cutoff=15.0, smoothing_fraction=0.2)  # DFTD3 cutoff

# Calculator will use separate neighbor lists
result = calc(data, forces=True)

# Check cutoffs
print(f"Coulomb cutoff: {calc.coulomb_cutoff} Å")
print(f"DFTD3 cutoff: {calc.dftd3_cutoff} Å")
```

## Calculator Integration (External Modules)

`AIMNet2Calculator` attaches external LR modules based on model metadata:

- External `LRCoulomb` is created when `needs_coulomb=True`. If `coulomb_mode="sr_embedded"`,
  the model already subtracts SR Coulomb, so the external module adds the full Coulomb energy.
- External `DFTD3` is created when `needs_dispersion=True` and `d3_params` are present.

See [calculator.md](calculator.md) for attachment logic and runtime configuration methods.
