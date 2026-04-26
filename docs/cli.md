# CLI Reference

AIMNet2 provides command-line tools for training, model export, conversion, and data preprocessing.

## Installation

CLI tools are available with the `train` extras:

```bash
pip install "aimnet[train] @ git+https://github.com/isayevlab/aimnetcentral.git"
```

## Commands Overview

| Command | Purpose | Typical Use |
| --- | --- | --- |
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
aimnet train [OPTIONS] [ARGS]...
```

| Option | Type | Description |
| --- | --- | --- |
| `--config PATH` | Optional | Extra training configuration YAML (may be passed multiple times; merged over the default) |
| `--model PATH` | Optional | Model architecture YAML (defaults to bundled `aimnet/models/aimnet2.yaml`) |
| `--load PATH` | Optional | Path to existing model weights to load before training |
| `--save PATH` | Optional | Path to save model weights |
| `--no-default-config` | Flag | Skip loading `aimnet/train/default_train.yaml` |
| `ARGS` | Variadic | Dot-separated overrides applied last (e.g., `data.train=mydata.h5 run_name=firstrun`) |

Trailing tokens after the named options (`ARGS`) are positional dot-form overrides into the merged training config — they are not flags.

Device selection is controlled via `CUDA_VISIBLE_DEVICES`; training uses all visible GPUs in DDP mode.

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
# Train with W&B logging (GPU controlled via CUDA_VISIBLE_DEVICES)
aimnet train \
  --config experiments/train_config.yaml \
  --model models/aimnet2.yaml

# Continue from existing weights
aimnet train \
  --config experiments/train_config.yaml \
  --model models/aimnet2.yaml \
  --load checkpoints/last.pt \
  --save checkpoints/continued.pt
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

| Argument | Description                              |
| -------- | ---------------------------------------- |
| `INPUT`  | Path to trained weights (.pt checkpoint) |
| `OUTPUT` | Path for exported model (.pt file)       |

**Options:**

| Option               | Description                          |
| -------------------- | ------------------------------------ |
| `--model PATH`       | Model architecture YAML (required)   |
| `--sae PATH`         | Self-atomic energies YAML (required) |
| `--needs-coulomb`    | Force external Coulomb module        |
| `--needs-dispersion` | Force external DFTD3 module          |

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

| Argument | Description                       |
| -------- | --------------------------------- |
| `INPUT`  | Legacy .jpt model file            |
| `CONFIG` | Original model configuration YAML |
| `OUTPUT` | Output .pt model file             |

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

Calculate self-atomic energies (SAE) from a dataset by fitting a per-element energy shift to the training data.

### Basic Usage

```bash
aimnet calc_sae dataset.h5 output_sae.yaml
```

### Options

```bash
aimnet calc_sae [OPTIONS] DS OUTPUT
```

**Positional Arguments:**

| Argument | Description |
| --- | --- |
| `DS` | HDF5 dataset (`SizeGroupedDataset` layout) containing `numbers` and `energy` |
| `OUTPUT` | Output YAML file for fitted SAE values |

**Options:**

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `--samples` | int | 100000 | Maximum number of dataset samples used for the fit |

### SAE Format

Output YAML format:

```yaml
# Self-atomic energies in eV
1: -13.587 # Hydrogen
6: -1027.592 # Carbon
7: -1483.525 # Nitrogen
8: -2039.734 # Oxygen
```

### Example

```bash
# Fit SAE from the training dataset (uses up to 100k samples by default)
aimnet calc_sae \
  data/dataset.h5 \
  configs/sae.yaml

# Limit the number of samples used for the fit
aimnet calc_sae \
  data/dataset.h5 \
  configs/sae.yaml \
  --samples 50000

# Use in training
aimnet train \
  --config train_config.yaml \
  --model model.yaml \
  data.sae.energy.file=configs/sae.yaml
```

### How SAE Is Computed

`aimnet calc_sae` does not require a separate single-atom dataset. It operates on the same `SizeGroupedDataset` HDF5 file used for training, fits a per-element energy shift via a least-squares regression on `(numbers, energy)`, trims outliers (2nd/98th percentiles of the residual), and refits. The element list is inferred automatically from the dataset.

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

| Variable               | Description           | Default            |
| ---------------------- | --------------------- | ------------------ |
| `AIMNET_CACHE_DIR`     | Model cache directory | `~/.cache/aimnet/` |
| `CUDA_VISIBLE_DEVICES` | GPU device selection  | All available      |

## Exit Codes

| Code | Meaning             |
| ---- | ------------------- |
| 0    | Success             |
| 1    | General error       |
| 2    | Invalid arguments   |
| 3    | File not found      |
| 4    | Configuration error |

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
