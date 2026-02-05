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

## Model Registry

Utilities for loading pre-trained models. Models are automatically downloaded from the remote repository to a local cache (`aimnet/calculators/assets/`) upon first use.

### CLI Command

You can clear the local model cache using the CLI:

```bash
aimnet clear_model_cache
```

::: aimnet.calculators.model_registry
options:
show_root_heading: true
show_source: true
