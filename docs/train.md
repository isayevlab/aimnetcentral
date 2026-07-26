# AIMNet2 training examples.

## General workflow

### Dataset preparation

The training dataset must be formatted as an HDF5 file, with groups containing molecules of uniform size. For example, the dataset below contains 25,768 molecules with 28 atoms and 19,404 molecules with 29 atoms.

```bash
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

```bash
$ aimnet train --help
```

Key components for initiating training include:

- **Training Configuration:** The base configuration file `aimnet/train/default_train.yaml` can be customized using command-line options or a separate YAML configuration file, which will override or extend default values. It is crucial to, at minimum, define the `run_name` and `data.train`.

- **Model Definition:** By default, the model defined in `aimnet/models/aimnet2.yaml` is used.

- **Self-Atomic Energies File:** This file can be generated using the following command:

```bash
$ aimnet calc_sae dataset.h5 dataset_sae.yaml
```

### Weights & Biases (W&B) Logging

The training script integrates with Weights & Biases (W&B), a platform for experiment tracking (free for personal and academic use). To monitor training progress, either a W&B account or a local Docker-based W&B server is necessary. By default, W&B operates in offline mode.

**Setting Up W&B**

- **Online Account:**

```bash
$ wandb login
```

- **Project and Entity Configuration:** Create a configuration file (e.g., `extra_conf.yaml`) with your W&B project and entity details:

```yaml
wandb:
  init:
    mode: online
    entity: your_username
    project: your_project_name
```

Pass this configuration to the `aimnet train` command using the `--config` parameter.

### Launching Training

For optimal data loader performance, it is recommended to disable numpy multithreading:

```bash
$ export OMP_NUM_THREADS=1
```

By default, training will utilize all available GPUs in a single-node, distributed data-parallel mode. To restrict training to a specific GPU (e.g., GPU 0):

```bash
$ export CUDA_VISIBLE_DEVICES=0
```

Finally, initiate the training script with default parameters and the specified `run_name`:

```bash
$ aimnet train data.train=dataset.h5 data.sae.energy.file=dataset_sae.yaml run_name=firstrun
```

### Model Export for Distribution

To export a trained model for distribution and use with AIMNet calculators:

```bash
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

**Output format:** `aimnet export` writes a v2 `.pt` artifact containing model YAML, weights, and runtime metadata. See [Model Format](model_format.md) for the artifact structure, metadata fields, suffix routing, and import policy.

For inference-time Coulomb and dispersion settings, see the [Calculator](calculator.md) guide.

### Converting Legacy JIT Models

To convert an existing `.jpt` model:

```bash
$ aimnet convert model.jpt config.yaml model_new.pt
```

See [Converting Legacy Models](model_format.md#converting-legacy-models) for the complete conversion and validation workflow.

### Publishing Registry Artifacts

Maintainers publishing a registry artifact must:

1. **Convert the model:**

   ```bash
   aimnet convert aimnet2_wb97m_d3_0.jpt aimnet/models/aimnet2_dftd3_wb97m.yaml aimnet2_wb97m_d3_0.pt
   ```

2. **Validate conversion:**

   ```bash
   python scripts/validate_conversion.py aimnet2_wb97m_d3_0.pt aimnet2_wb97m_d3_0.jpt \
       --structure tests/data/caffeine.xyz
   ```

   The validation script compares energies and forces between formats and reports any discrepancies.

3. **Publish the final artifact** to approved registry storage and record its public HTTPS download URL.

4. **Update `model_registry.yaml`:**

   ```yaml
   models:
     aimnet2-wb97m-d3_0:
       family: wb97m-d3
       file: aimnet2_wb97m_d3_0.pt
       url: <public-https-download-url>
       sha256: <sha256-of-published-bytes>
   ```

   Registry keys follow the convention `aimnet2-<family>_<member>` (dash separates `aimnet2` from the family tag, trailing `_<int>` is the ensemble member index). The `file:` field keeps the original filename so cached `.pt` downloads remain valid.

   The digest is the immutable identity of the uploaded bytes. Calculate it by streaming the final public download URL twice from fresh temporary directories, reject non-HTTPS or cross-origin redirects, and require both passes to agree. Obtain maintainer approval against the authoritative upload/source record before committing the registry change. If bytes change later, publish a new filename and registry entry rather than silently replacing the digest.
