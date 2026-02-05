# AIMNet2 training examples.

## General workflow

### Dataset preparation

The training dataset must be formatted as an HDF5 file, with groups containing molecules of uniform size. For example, the dataset below contains 25,768 molecules with 28 atoms and 19,404 molecules with 29 atoms.

```
$ h5ls -r dataset.h5
/028                     Group
/028/charge              Dataset {25768}
/028/charges             Dataset {25768, 28}
/028/coord               Dataset {25768, 28, 3}
/028/energy              Dataset {25768}
/028/forces              Dataset {25768, 28, 3}
/028/numbers             Dataset {25768, 28}
/029                     Group
/029/charge              Dataset {19404}
/029/charges             Dataset {19404, 29}
/029/coord               Dataset {19404, 29, 3}
/029/energy              Dataset {19404}
/029/forces              Dataset {19404, 29, 3}
/029/numbers             Dataset {19404, 29}
```

Units should be based on Angstrom, electron-volt, and electron charge.

### Training Configuration

To access available options for the training script execute the following command:

```
$ aimnet train --help
```

Key components for initiating training include:

- **Training Configuration:** The base configuration file `aimnet/train/default_train.yaml` can be customized using command-line options or a separate YAML configuration file, which will override or extend default values. It is crucial to, at minimum, define the `run_name` and `data.train`.

- **Model Definition:** By default, the model defined in `aimnet/models/aimnet2.yaml` is used.

- **Self-Atomic Energies File:** This file can be generated using the following command:

```
$ aimnet calc_sae dataset.h5 dataset_sae.yaml
```

### Weights & Biases (W&B) Logging

The training script integrates with Weights & Biases (W&B), a platform for experiment tracking (free for personal and academic use). To monitor training progress, either a W&B account or a local Docker-based W&B server is necessary. By default, W&B operates in offline mode.

**Setting Up W&B**

- **Online Account:**

```
$ wandb login
```

- **Project and Entity Configuration:** Create a configuration file (e.g., `extra_conf.yaml`) with your W&B project and entity details:

```
wandb:
  init:
    mode: online
    entity: your_username
    project: your_project_name
```

Pass this configuration to the `aimnet train` command using the `--config` parameter.

### Launching Training

For optimal data loader performance, it is recommended to disable numpy multithreading:

```
$ export OMP_NUM_THREADS=1
```

By default, training will utilize all available GPUs in a single-node, distributed data-parallel mode. To restrict training to a specific GPU (e.g., GPU 0):

```
$ export CUDA_VISIBLE_DEVICES=0
```

Finally, initiate the training script with default parameters and the specified `run_name`:

```
$ aimnet train data.train=dataset.h5 data.sae.energy.file=dataset_sae.yaml run_name=firstrun
```

### Model Export for Distribution

To export a trained model for distribution and use with AIMNet calculators:

```
$ aimnet export weights.pt model.pt --model config.yaml --sae model.sae
```

Arguments:

- `weights.pt`: Raw PyTorch weights file from training
- `model.pt`: Output model file
- `--model`: Path to model YAML configuration file
- `--sae`: Path to self-atomic energies file

The export command creates a self-contained `.pt` file with:

- Model architecture configuration
- Trained weights with SAE baked into atomic shifts
- Metadata for Coulomb and dispersion handling

**Model Format**

The new model format separates the core neural network from long-range corrections:

- Core model computes NN energy minus short-range Coulomb
- Long-range Coulomb (LRCoulomb) is applied externally by the calculator
- DFTD3 dispersion is applied externally by the calculator

This allows switching Coulomb methods (simple/DSF/Ewald) at inference time without re-exporting.

**Metadata Schema**

The v2 model format includes the following metadata fields:

| Field                 | Type      | Description                                             |
| --------------------- | --------- | ------------------------------------------------------- |
| `format_version`      | int       | 2 for new format                                        |
| `cutoff`              | float     | Model cutoff radius                                     |
| `needs_coulomb`       | bool      | True if calculator should add external Coulomb          |
| `needs_dispersion`    | bool      | True if calculator should add external DFTD3            |
| `coulomb_mode`        | str       | "sr_embedded" (has SRCoulomb) or "none"                 |
| `coulomb_sr_rc`       | float?    | SR Coulomb cutoff (if coulomb_mode="sr_embedded")       |
| `coulomb_sr_envelope` | str?      | "exp" or "cosine" (if coulomb_mode="sr_embedded")       |
| `d3_params`           | dict?     | D3 parameters {s6, s8, a1, a2} if needs_dispersion=True |
| `implemented_species` | list[int] | Supported atomic numbers                                |

**Runtime Configuration**

For new-format models, the calculator provides methods to configure external modules:

```python
from aimnet.calculators import AIMNet2Calculator

calc = AIMNet2Calculator("model.pt")

# Switch Coulomb method (only if model uses external Coulomb)
if calc.has_external_coulomb:
    calc.set_lrcoulomb_method("dsf", cutoff=15.0)  # For PBC systems

# Adjust DFTD3 cutoff (only if model uses external dispersion)
if calc.has_external_dftd3:
    calc.set_dftd3_cutoff(20.0, smoothing_fraction=0.8)
```

**Backward Compatibility**

The calculator automatically detects and loads both formats:

- New `.pt` files with embedded metadata (format version 2)
- Legacy JIT-compiled `.jpt` files (format version 1)

For legacy models, Coulomb and dispersion modules remain embedded in the model.

### Converting Legacy JIT Models

To convert existing `.jpt` models to the new format:

```
$ aimnet convert model.jpt config.yaml model_new.pt
```

Arguments:

- `model.jpt`: Legacy JIT-compiled model file
- `config.yaml`: Model YAML configuration file
- `model_new.pt`: Output file in new format

The conversion:

- Extracts model cutoff and D3 parameters from the JIT model
- Rebuilds the architecture from YAML with SRCoulomb embedded
- Loads weights from the JIT model
- Creates a new-format bundle with all necessary metadata

### Registry Migration Workflow

For maintainers migrating the model registry from legacy `.jpt` to new `.pt` format:

1. **Convert each model:**

   ```bash
   aimnet convert aimnet2_wb97m_d3_0.jpt aimnet/models/aimnet2_dftd3_wb97m.yaml aimnet2_wb97m_d3_0.pt
   ```

2. **Validate conversion:**

   ```bash
   python scripts/validate_conversion.py aimnet2_wb97m_d3_0.pt aimnet2_wb97m_d3_0.jpt \
       --structure tests/data/caffeine.xyz
   ```

   The validation script compares energies and forces between formats and reports any discrepancies.

3. **Upload to GCS:**

   ```bash
   gsutil cp aimnet2_wb97m_d3_0.pt gs://aimnetcentral/AIMNet2/aimnet2_wb97m_d3_0.pt
   ```

4. **Update model_registry.yaml:**
   ```yaml
   models:
     aimnet2_wb97m_d3_0:
       file: aimnet2_wb97m_d3_0.pt
       url: https://storage.googleapis.com/aimnetcentral/AIMNet2/aimnet2_wb97m_d3_0.pt
   ```
