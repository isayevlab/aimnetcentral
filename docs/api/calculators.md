# Calculators

Calculator interfaces for molecular simulations using AIMNet2.

## AIMNet2Calculator

The core calculator for running AIMNet2 inference. It handles model loading, device management, and application of long-range interactions (Coulomb and Dispersion).

### Key Features

- **Format Support**: Loads both legacy `.jpt` models and new `.pt` format.
- **Long-Range Interactions**: Automatically attaches `LRCoulomb` and `DFTD3` modules based on model metadata.
- **Overrides**: You can force specific long-range behavior using `needs_coulomb` and `needs_dispersion` arguments.
- **Batching**: Automatically batches large molecules/systems based on `nb_threshold`.

::: aimnet.calculators.AIMNet2Calculator
    options:
      show_root_heading: true
      show_source: true

## AIMNet2ASE

[ASE (Atomic Simulation Environment)](https://wiki.fysik.dtu.dk/ase/) calculator interface.

!!! note "Installation"

    Requires the `ase` extra: `pip install aimnet[ase]`

This calculator integrates with ASE's `Atoms` object, supporting energy, forces, stress, and dipole moment calculations. It operates in **eV** and **Angstrom**.

### Usage Example

```python
from ase.io import read
from aimnet.calculators import AIMNet2ASE

atoms = read("molecule.xyz")
atoms.calc = AIMNet2ASE("aimnet2")

print(atoms.get_potential_energy())
print(atoms.get_forces())
```

::: aimnet.calculators.aimnet2ase.AIMNet2ASE
    options:
      show_root_heading: true
      show_source: true

## AIMNet2Pysis

[PySisyphus](https://pysisyphus.readthedocs.io/) calculator interface.

!!! note "Installation"

    Requires the `pysis` extra: `pip install aimnet[pysis]`

This interface adapts AIMNet2 for use with PySisyphus optimizers. It handles unit conversion automatically:

- **Input**: Converts Angstrom (PySisyphus) to Angstrom (AIMNet2).
- **Output**: Converts eV/Angstrom (AIMNet2) to **Hartree/Bohr** (PySisyphus).

::: aimnet.calculators.aimnet2pysis.AIMNet2Pysis
    options:
      show_root_heading: true
      show_source: true

## AIMNet2TorchSim

[TorchSim](https://torchsim.github.io/torch-sim/) `ModelInterface` wrapper.

!!! note "Installation"

    Requires the `torchsim` extra and Python 3.12+: `pip install "aimnet[torchsim]"`. Add the `ase` extra for ASE-based input/output examples: `pip install "aimnet[torchsim,ase]"`.

`AIMNet2TorchSim` wraps an `AIMNet2Calculator` as a `torch-sim-atomistic` model for static evaluation, geometry optimization, molecular dynamics, and autobatched workloads.

### Usage Example

```python
import ase.io
import torch_sim as ts

from aimnet.calculators import AIMNet2Calculator, AIMNet2TorchSim

atoms = ase.io.read("molecule.xyz")

base_calc = AIMNet2Calculator("aimnet2")
calc = AIMNet2TorchSim(base_calc)

results = ts.static(system=atoms, model=calc)
print(results[0]["potential_energy"], results[0]["forces"])
```

!!! note "Stress"

    By default `compute_stress=False`. Pass `compute_stress=True` when constructing `AIMNet2TorchSim` for NPT integrators and PBC cell relaxation.

!!! note "TorchSim extras"

    AIMNet partial charges are returned as both `charges` and `partial_charges`. Set per-system `charge` and NSE `mult` through TorchSim system extras.

::: aimnet.calculators.aimnet2torchsim.AIMNet2TorchSim
    options:
      show_root_heading: true
      show_source: true

## Model Registry

Utilities for loading pre-trained models. Models are automatically downloaded from the remote repository to the local model cache (`AIMNET_CACHE_DIR` when set, otherwise `~/.cache/aimnet/`) upon first use.

### CLI Command

You can clear the local model cache using the CLI:

```bash
aimnet clear_model_cache
```

::: aimnet.calculators.model_registry
    options:
      show_root_heading: true
      show_source: true
