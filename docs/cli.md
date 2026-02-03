# CLI Reference

AIMNet2 provides command-line tools for training, model export, conversion, and data preprocessing.

## Installation

CLI tools are available with the `train` extras:

```bash
pip install "aimnet[train] @ git+https://github.com/isayevlab/aimnetcentral.git"
```

## Commands Overview

| Command | Purpose | Typical Use |
|---------|---------|-------------|
| `aimnet train` | Train a model | Training from scratch |
| `aimnet export` | Export trained weights to inference format | After training |
| `aimnet convert` | Convert legacy .jpt to new .pt format | Migrating old models |
| `aimnet calc_sae` | Calculate self-atomic energies | Before training |

## aimnet train

Train an AIMNet2 model from a configuration file.

### Basic Usage

```bash
aimnet train --config config.yaml --model model.yaml
```

### Options

```bash
aimnet train [OPTIONS]
```

| Option | Type | Description |
|--------|------|-------------|
| `--config PATH` | Required | Training configuration YAML file |
| `--model PATH` | Required | Model architecture YAML file |
| `--resume PATH` | Optional | Resume from checkpoint |
| `--device DEVICE` | Optional | Device to use (cuda/cpu) |

### Example Configuration

Training requires two YAML files:

**config.yaml** (training parameters):

```yaml
data:
  train: data/train.h5
  val: data/val.h5
  batch_size: 32

optimizer:
  lr: 0.001
  weight_decay: 0.0

training:
  epochs: 100
  checkpoint_dir: checkpoints/
  log_interval: 10
```

**model.yaml** (architecture):

```yaml
class: aimnet.models.AIMNet2
kwargs:
  nfeature: 16
  hidden: [[512, 380], [512, 380]]
  aev:
    rc_s: 5.0
    nshifts_s: 16
  outputs:
    energy_mlp:
      class: aimnet.modules.Output
      # ... output configuration
```

### Complete Example

```bash
# Train with W&B logging
aimnet train \
  --config experiments/train_config.yaml \
  --model models/aimnet2.yaml \
  --device cuda

# Resume training
aimnet train \
  --config experiments/train_config.yaml \
  --model models/aimnet2.yaml \
  --resume checkpoints/last.pt
```

See [train.md](train.md) for detailed training documentation.

## aimnet export

Export trained model weights to inference format (.pt).

### Basic Usage

```bash
aimnet export weights.pt output.pt --model model.yaml --sae sae.yaml
```

### Options

```bash
aimnet export INPUT OUTPUT [OPTIONS]
```

**Positional Arguments:**

| Argument | Description |
|----------|-------------|
| `INPUT` | Path to trained weights (.pt checkpoint) |
| `OUTPUT` | Path for exported model (.pt file) |

**Options:**

| Option | Description |
|--------|-------------|
| `--model PATH` | Model architecture YAML (required) |
| `--sae PATH` | Self-atomic energies YAML (required) |
| `--needs-coulomb` | Force external Coulomb module |
| `--needs-dispersion` | Force external DFTD3 module |

### Export Process

The export process:

1. Loads model architecture from YAML
2. Strips LRCoulomb/DFTD3 modules
3. Adds SRCoulomb if LRCoulomb was present
4. Loads trained weights
5. Bakes SAE into atomic_shift as float64
6. Masks unimplemented species
7. Saves with metadata

### Examples

**Basic export:**

```bash
aimnet export \
  checkpoints/best.pt \
  models/aimnet2_production.pt \
  --model configs/aimnet2.yaml \
  --sae configs/sae.yaml
```

**Override auto-detection:**

```bash
# Force external Coulomb and DFTD3
aimnet export weights.pt model.pt \
  --model config.yaml \
  --sae sae.yaml \
  --needs-coulomb \
  --needs-dispersion
```

**Export without dispersion:**

```bash
# Model without DFTD3 correction
aimnet export weights.pt model_no_d3.pt \
  --model config_no_d3.yaml \
  --sae sae.yaml
```

## aimnet convert

Convert legacy .jpt (JIT) models to new .pt format.

### Basic Usage

```bash
aimnet convert model.jpt config.yaml output.pt
```

### Options

```bash
aimnet convert INPUT CONFIG OUTPUT
```

**Positional Arguments:**

| Argument | Description |
|----------|-------------|
| `INPUT` | Legacy .jpt model file |
| `CONFIG` | Original model configuration YAML |
| `OUTPUT` | Output .pt model file |

### Conversion Process

1. Loads legacy JIT model
2. Extracts weights from state dict
3. Strips embedded LR modules
4. Adds SRCoulomb if needed
5. Converts atomic_shift to float64
6. Generates metadata
7. Saves as v2 format

### Example

```bash
# Convert legacy model
aimnet convert \
  legacy/aimnet2_old.jpt \
  configs/aimnet2.yaml \
  models/aimnet2_new.pt

# Verify conversion
python -c "
from aimnet.calculators import AIMNet2Calculator
calc = AIMNet2Calculator('models/aimnet2_new.pt')
print(f'Cutoff: {calc.cutoff}')
print(f'Has external Coulomb: {calc.has_external_coulomb}')
"
```

See [model_format.md](model_format.md#migration-guide) for detailed migration guide.

## aimnet calc_sae

Calculate self-atomic energies (SAE) from a dataset.

### Basic Usage

```bash
aimnet calc_sae dataset.h5 output_sae.yaml
```

### Options

```bash
aimnet calc_sae INPUT OUTPUT [OPTIONS]
```

**Positional Arguments:**

| Argument | Description |
|----------|-------------|
| `INPUT` | HDF5 dataset with single-atom energies |
| `OUTPUT` | Output YAML file for SAE values |

**Options:**

| Option | Description |
|--------|-------------|
| `--elements LIST` | Comma-separated atomic numbers (e.g., "1,6,7,8") |

### SAE Format

Output YAML format:

```yaml
# Self-atomic energies in eV
1: -13.587  # Hydrogen
6: -1027.592  # Carbon
7: -1483.525  # Nitrogen
8: -2039.734  # Oxygen
```

### Example

```bash
# Calculate SAE for H, C, N, O
aimnet calc_sae \
  data/single_atoms.h5 \
  configs/sae.yaml \
  --elements "1,6,7,8"

# Use in training
aimnet train \
  --config train_config.yaml \
  --model model.yaml
  # SAE loaded from train_config
```

### Creating Single-Atom Dataset

SAE calculation requires single-atom reference calculations:

```python
import h5py
import numpy as np

# Single-atom energies from reference calculations (e.g., DFT)
sae_data = {
    1: -13.587,   # H atom energy
    6: -1027.592, # C atom energy
    # ... more elements
}

# Create HDF5 dataset
with h5py.File("single_atoms.h5", "w") as f:
    for z, energy in sae_data.items():
        grp = f.create_group(f"atom_{z}")
        grp["energy"] = energy
        grp["numbers"] = [z]
```

## Common Workflows

### Complete Training Workflow

```bash
# 1. Calculate SAE
aimnet calc_sae single_atoms.h5 sae.yaml

# 2. Train model
aimnet train \
  --config config.yaml \
  --model aimnet2.yaml

# 3. Export best checkpoint
aimnet export \
  checkpoints/best.pt \
  production_model.pt \
  --model aimnet2.yaml \
  --sae sae.yaml

# 4. Validate
python validate_model.py production_model.pt
```

### Legacy Migration Workflow

```bash
# 1. Convert .jpt to .pt
aimnet convert \
  old_model.jpt \
  original_config.yaml \
  new_model.pt

# 2. Test equivalence
python test_equivalence.py old_model.jpt new_model.pt

# 3. Update inference code
# Replace: calc = AIMNet2Calculator("old_model.jpt")
# With:    calc = AIMNet2Calculator("new_model.pt")
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AIMNET_CACHE_DIR` | Model cache directory | `~/.cache/aimnet/` |
| `CUDA_VISIBLE_DEVICES` | GPU device selection | All available |

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Invalid arguments |
| 3 | File not found |
| 4 | Configuration error |

## Getting Help

```bash
# General help
aimnet --help

# Command-specific help
aimnet train --help
aimnet export --help
aimnet convert --help
aimnet calc_sae --help
```

## See Also

- [Training Guide](train.md) - Detailed training documentation
- [Model Format](model_format.md) - Model file specifications
- [Calculator API](calculator.md) - Python inference API
