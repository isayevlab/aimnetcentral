# Model Format and Conversion

This document describes the AIMNet2 model format, metadata structure, and conversion between legacy and new formats.
It reflects the current implementation in `aimnet.models.base`, `aimnet.models.utils`, and `aimnet.calculators.calculator`.

## Model Formats

AIMNet2 supports two model formats:

| Format | Extension | Version | Description |
|--------|-----------|---------|-------------|
| Legacy | `.jpt` | 1 | TorchScript JIT-compiled model with embedded LR modules |
| New | `.pt` | 2 | State dict with embedded YAML config and metadata |

### Format Detection

When loading a model via `load_model()`, the format is automatically detected:

- **New format**: Dictionary containing `"model_yaml"` key
- **Legacy format**: `torch.jit.ScriptModule` instance

## Metadata Structure

Model metadata is returned by `load_model()` as a `ModelMetadata` TypedDict.
For early v2 bundles that predate `format_version`, `load_model()` defaults to `format_version=2`.

### Core Fields

| Field | Type | Description |
|-------|------|-------------|
| `format_version` | `int` | `1` = legacy JIT, `2` = new format (default for early v2 bundles) |
| `cutoff` | `float` | Model short-range cutoff radius (Å) |
| `implemented_species` | `list[int]` | Supported atomic numbers |

### Coulomb Configuration

| Field | Type | Description |
|-------|------|-------------|
| `needs_coulomb` | `bool` | If `True`, calculator should add external Coulomb |
| `coulomb_mode` | `str` | What's embedded: `"sr_embedded"`, `"full_embedded"`, or `"none"` |
| `coulomb_sr_rc` | `float | None` | SR Coulomb cutoff (only if `coulomb_mode="sr_embedded"`) |
| `coulomb_sr_envelope` | `str | None` | Envelope function: `"exp"` (mollifier) or `"cosine"` |

### Dispersion Configuration

| Field | Type | Description |
|-------|------|-------------|
| `needs_dispersion` | `bool` | If `True`, calculator should add external DFTD3 |
| `d3_params` | `dict | None` | D3 parameters: `{s6, s8, a1, a2}` |

## Which Format Should I Use?

### Decision Matrix

| Scenario | Format | Model Type | Notes |
|----------|--------|------------|-------|
| Training new model | v2 (.pt) | Export after training | Flexible, modern |
| Need runtime Coulomb control | v2 (.pt) | Convert from v1 if needed | Switch simple/DSF/Ewald |
| Production inference | v2 (.pt) | Preferred | Smaller, more flexible |
| Legacy deployment | v1 (.jpt) | Keep as-is | If compatibility required |
| Experimenting with methods | v2 (.pt) | Required | Runtime reconfiguration |
| Fixed pipeline | Either | Use what works | No strong preference |

### Quick Selection Guide

**Use v2 (.pt) if:**
- Training new models
- Need to try different Coulomb methods
- Want runtime flexibility
- Prefer modern PyTorch features

**Keep v1 (.jpt) if:**
- Existing deployment works
- Don't need to change methods
- Legacy compatibility required
- No issues with current setup

## Coulomb Modes

The `coulomb_mode` field describes what Coulomb treatment is embedded in the model.

### Coulomb Mode Comparison

```
┌─────────────────────────────────────────────────────────────────┐
│ sr_embedded (v2 format - RECOMMENDED)                           │
├─────────────────────────────────────────────────────────────────┤
│ Model:      E_NN - E_SR (SR Coulomb subtracted)                 │
│ Calculator: + E_full (adds full Coulomb externally)             │
│ Total:      E_NN + E_LR (SR cancels out)                        │
│                                                                  │
│ Runtime control: ✓ Can switch simple/DSF/Ewald                  │
│ File size:       Smaller (no LR modules embedded)               │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ full_embedded (v1 legacy format)                                │
├─────────────────────────────────────────────────────────────────┤
│ Model:      E_NN + E_Coulomb (full Coulomb embedded in JIT)     │
│ Calculator: (nothing)                                           │
│ Total:      E_NN + E_Coulomb                                    │
│                                                                  │
│ Runtime control: ✗ Fixed method, warning only                   │
│ File size:       Larger (modules in JIT)                        │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ none (no Coulomb)                                               │
├─────────────────────────────────────────────────────────────────┤
│ Model:      E_NN only                                           │
│ Calculator: (nothing)                                           │
│ Total:      E_NN                                                │
│                                                                  │
│ Runtime control: N/A                                            │
│ Use case:       Models without electrostatics                   │
└─────────────────────────────────────────────────────────────────┘
```

### `"sr_embedded"`

- Model has **SRCoulomb** (short-range) embedded
- Model outputs: `E_NN - E_SR`
- Calculator adds **full** Coulomb externally: `E_total = (E_NN - E_SR) + E_full = E_NN + E_LR`
- Uses `coulomb_sr_rc` and `coulomb_sr_envelope` from metadata
- User can switch Coulomb method (simple/DSF/Ewald) at runtime via `set_lrcoulomb_method()`

#### SR Coulomb Cutoff (`coulomb_sr_rc`)

The short-range Coulomb cutoff defines the distance within which SR Coulomb interactions are computed by the embedded `SRCoulomb` module.

**Constraint:** `coulomb_sr_rc <= model cutoff`

The SR cutoff must be less than or equal to the model's short-range cutoff (`cutoff`) because:
- SRCoulomb uses the same neighbor list as the neural network
- Atom pairs beyond the model cutoff are not visible to SRCoulomb
- Typical value: 4.6 Å (with model cutoff of 5.0 Å)

#### SR Envelope (`coulomb_sr_envelope`)

The envelope function defines how the SR interaction decays at the cutoff:
- `"exp"`: Smooth mollifier-based decay (default)
- `"cosine"`: Cosine-based decay

### `"full_embedded"`

- Legacy JIT model with **full Coulomb** embedded
- Model outputs: `E_NN + E_Coulomb` directly
- No external Coulomb needed (`needs_coulomb=False`)
- Coulomb method cannot be changed at runtime

### `"none"`

- No Coulomb treatment in model
- `needs_coulomb=False`
- Model outputs: `E_NN` only

## Dispersion Modes

### External DFTD3 (`needs_dispersion=True`)

- DFTD3/D3BJ module removed from model during export
- D3 parameters (`s6`, `s8`, `a1`, `a2`) stored in `d3_params` metadata
- Calculator creates external DFTD3 module
- Cutoff can be configured via `set_dftd3_cutoff()` for external DFTD3 only

**Note:** DFTD3 cutoff/smoothing values are not currently stored in metadata. External DFTD3
defaults to 15.0 Å cutoff and 0.8 smoothing fraction unless overridden at runtime.

### Embedded D3TS

- D3TS (learned parameters) remains embedded in model
- `needs_dispersion=False` for D3TS models
- Cannot be modified at runtime

### No Dispersion

- `needs_dispersion=False` and no D3TS
- Model outputs energy without dispersion correction

## File Structure

### New Format (.pt)

```python
{
    "format_version": 2,        # Default for early v2 bundles may be omitted
    "model_yaml": str,           # Core model YAML config (no LR modules)
    "cutoff": float,
    "needs_coulomb": bool,
    "needs_dispersion": bool,
    "coulomb_mode": str,
    "coulomb_sr_rc": float | None,
    "coulomb_sr_envelope": str | None,
    "d3_params": dict | None,    # DFTD3 params for external use
    "implemented_species": list[int],
    "state_dict": dict,          # Model weights (SAE baked in)
}
```

### Legacy Format (.jpt)

TorchScript module with attributes:
- `cutoff`: Model cutoff
- `cutoff_lr`: Long-range cutoff (if applicable)
- LRCoulomb and DFTD3/D3BJ modules embedded

## Exporting Models

Use the `aimnet export` CLI command:

```bash
aimnet export weights.pt model_v2.pt --model config.yaml --sae sae.yaml
```

### Export Process

1. Load model YAML config, SAE (self-atomic energies), and weights
2. Strip LRCoulomb/DFTD3 modules from config
3. Add SRCoulomb if LRCoulomb was present (requires determinable `rc`)
4. Build core model from modified config
5. Load weights (with `strict=False` for module changes)
6. Bake SAE into `atomic_shift.shifts.weight` as float64
7. Mask unimplemented species (set NaN in `afv.weight`)
8. Save with metadata

### Export Options

```bash
aimnet export weights.pt model.pt \
    --model config.yaml \
    --sae sae.yaml \
    --needs-coulomb    # Override: force external Coulomb
    --needs-dispersion # Override: force external DFTD3
```

Explicit flags override auto-detection from config.

## Converting Legacy Models

Use the `aimnet convert` CLI command:

```bash
aimnet convert model.jpt config.yaml model_v2.pt
```

### Conversion Process

1. Load legacy JIT model and YAML config
2. Extract `cutoff` from model attribute
3. Extract `implemented_species` from `afv.weight` (non-NaN entries)
4. Strip LR modules from config, add SRCoulomb if needed (requires determinable `rc`)
5. Build core model from modified config
6. Load weights from JIT state dict
7. Validate keys (filter expected missing/unexpected)
8. Convert `atomic_shift` to float64
9. Save with metadata

### Key Changes During Conversion

| Legacy | New |
|--------|-----|
| `outputs.lrcoulomb.*` | Removed |
| `outputs.dftd3.*` | Removed |
| `outputs.d3bj.*` | Removed |
| (none) | `outputs.srcoulomb.*` added |

## Loading Models

```python
from aimnet.models.base import load_model

model, metadata = load_model("model.pt", device="cuda")

# Access metadata
print(metadata["cutoff"])
print(metadata["needs_coulomb"])
print(metadata["coulomb_mode"])
```

### Loading Behavior

- New format: Parses YAML, builds model, loads state dict
- Legacy format: Returns JIT model directly
- Metadata always returned as `ModelMetadata` dict
- `atomic_shift` converted to float64 after loading (precision)

## Metadata Behavior Summary

| Model Type | `needs_coulomb` | `coulomb_mode` | Calculator Behavior |
|------------|-----------------|----------------|---------------------|
| New with Coulomb | `True` | `"sr_embedded"` | Adds external LRCoulomb |
| New without Coulomb | `False` | `"none"` | No external Coulomb |
| Legacy JIT | `False` | `"full_embedded"` | Coulomb embedded in JIT |

| Model Type | `needs_dispersion` | Calculator Behavior |
|------------|-------------------|---------------------|
| New with DFTD3/D3BJ | `True` | Adds external DFTD3 |
| New with D3TS | `False` | D3TS embedded |
| New without dispersion | `False` | No dispersion |
| Legacy with DFTD3 | `False` | Embedded in JIT (d3_params extracted for diagnostics) |

## API Reference

### `load_model(path, device="cpu")`

Load model from file with automatic format detection.

**Parameters:**
- `path` (`str`): Path to model file (`.pt` or `.jpt`)
- `device` (`str`): Device to load model on

**Returns:**
- `model` (`nn.Module`): Loaded model
- `metadata` (`ModelMetadata`): Metadata dictionary

### `ModelMetadata` (TypedDict)

See [Metadata Structure](#metadata-structure) for field definitions.

### Migration Guide

### Why Migrate to v2 Format?

**Benefits of v2 (.pt) over v1 (.jpt):**

- **Runtime flexibility**: Change Coulomb method (simple/DSF/Ewald) without retraining
- **Smaller files**: Separate external modules reduce file size
- **Better debugging**: Access model structure and weights directly
- **Modern workflow**: Compatible with latest PyTorch features
- **Metadata**: Rich metadata for validation and documentation

**When to convert:**
- You have legacy `.jpt` models and want runtime Coulomb control
- You're training new models (use v2 from the start)
- You need to modify model architecture post-training

**When to keep v1:**
- Legacy compatibility required
- Model works fine and no new features needed
- Deployment pipeline expects JIT models

### Step-by-Step Migration

#### 1. Prepare Required Files

You'll need:
- `model.jpt` - Your legacy JIT model
- `config.yaml` - Original model configuration

```bash
# If you don't have config.yaml, you may need to reconstruct it
# from training logs or model inspection
```

#### 2. Run Conversion

```bash
aimnet convert model.jpt config.yaml model_v2.pt
```

**What happens during conversion:**
- Extracts model weights from JIT state dict
- Strips embedded LRCoulomb/DFTD3 modules
- Adds SRCoulomb if LRCoulomb was present
- Preserves atomic shifts (SAE) as float64
- Detects implemented species from weights
- Generates metadata dictionary

#### 3. Validate Conversion

```python
from aimnet.calculators import AIMNet2Calculator
import torch

# Load both models
calc_v1 = AIMNet2Calculator("model.jpt")
calc_v2 = AIMNet2Calculator("model_v2.pt")

# Test data
data = {
    "coord": torch.randn(10, 3),
    "numbers": torch.randint(1, 9, (10,)),
    "charge": 0.0,
}

# Compare energies (should match within tolerance)
result_v1 = calc_v1(data, forces=True)
result_v2 = calc_v2(data, forces=True)

energy_diff = (result_v1["energy"] - result_v2["energy"]).abs()
force_diff = (result_v1["forces"] - result_v2["forces"]).abs().max()

print(f"Energy difference: {energy_diff:.2e} eV")
print(f"Max force difference: {force_diff:.2e} eV/Å")

assert energy_diff < 1e-5, "Energy mismatch!"
assert force_diff < 1e-4, "Force mismatch!"
```

**Expected differences:**
- Energies: < 1e-5 eV (numerical precision)
- Forces: < 1e-4 eV/Å (gradient precision)

#### 4. Test Runtime Flexibility

```python
# v2 models support runtime method changes
calc_v2.set_lrcoulomb_method("dsf", cutoff=15.0)
result_dsf = calc_v2(data)

calc_v2.set_lrcoulomb_method("ewald", ewald_accuracy=1e-8)
result_ewald = calc_v2(data)

# v1 models show warning but don't change
calc_v1.set_lrcoulomb_method("dsf", cutoff=15.0)
# Warning: Cannot change method for legacy models
```

### Common Conversion Issues

#### Issue: Missing config.yaml

**Problem:** You have a `.jpt` model but no configuration file.

**Solution:** Inspect the model to reconstruct config:

```python
import torch
model = torch.jit.load("model.jpt")

# Inspect attributes
print(f"Cutoff: {model.cutoff}")
print(f"Cutoff LR: {model.cutoff_lr}")

# May need to manually create config based on model structure
```

#### Issue: Weight Mismatch

**Problem:** Conversion completes but validation shows large differences.

**Solution:** Check for module name mismatches:

```bash
# Use verbose mode to see what's happening
aimnet convert model.jpt config.yaml model_v2.pt --verbose

# Check for unexpected missing keys
# Some modules may have been renamed
```

#### Issue: Implemented Species Mismatch

**Problem:** Converted model has wrong `implemented_species`.

**Solution:** Species are auto-detected from non-NaN entries in `afv.weight`. Verify:

```python
from aimnet.models.base import load_model
model, metadata = load_model("model_v2.pt")
print(metadata["implemented_species"])

# If wrong, may need to fix config before conversion
```

### Exporting New Models

For newly trained models, export directly to v2:

```bash
aimnet export weights.pt model_v2.pt \
    --model config.yaml \
    --sae sae.yaml
```

**Optional flags:**

```bash
# Override auto-detection
--needs-coulomb       # Force external Coulomb
--needs-dispersion    # Force external DFTD3
```

## CLI Commands

```bash
# Export trained model
aimnet export weights.pt output.pt --model config.yaml --sae sae.yaml

# Convert legacy JIT model
aimnet convert model.jpt config.yaml output.pt

# Calculate SAE from dataset
aimnet calc_sae dataset.h5 sae.yaml
```