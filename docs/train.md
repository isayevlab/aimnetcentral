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

### Model Compilation for Distribution

To compile a trained model for distribution and subsequent use with AIMNet calculators:

```
$ aimnet jitcompile my_model.pt my_model.jpt
```
