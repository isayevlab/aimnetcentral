---
title: AIMNet Machine-Learned Interatomic Potential
---

# AIMNet2: Machine-Learned Interatomic Potential

[![Release](https://img.shields.io/github/v/release/isayevlab/aimnetcentral)](https://img.shields.io/github/v/release/isayevlab/aimnetcentral)
[![Build status](https://img.shields.io/github/actions/workflow/status/isayevlab/aimnetcentral/main.yml?branch=main)](https://github.com/isayevlab/aimnetcentral/actions/workflows/main.yml?query=branch%3Amain)
[![License](https://img.shields.io/github/license/isayevlab/aimnetcentral)](https://img.shields.io/github/license/isayevlab/aimnetcentral)

## What is AIMNet2?

AIMNet2 is a neural network potential for fast and accurate atomistic simulations. Built on PyTorch, it provides:

- Accurate predictions for neutral, charged, organic, and elemental-organic systems
- Fast inference on both CPU and GPU
- Integration with popular simulation packages (ASE, PySisyphus)
- Configurable long-range electrostatics (DSF, Ewald) for periodic systems

AIMNet2 combines a graph neural network architecture with flexible long-range interactions, making it suitable for molecular dynamics, geometry optimization, and property prediction across diverse chemical systems.

## Key Features

- **Accurate and Versatile**: Handles neutral, charged, organic, and elemental-organic systems with consistent accuracy
- **Flexible Interfaces**: Calculator API for direct inference, plus ASE and PySisyphus integration for simulation workflows
- **Configurable Long-Range**: Choose between Simple, DSF, or Ewald methods for Coulomb interactions
- **GPU Accelerated**: CUDA support with optional `compile_mode` for ~5x MD speedup
- **Periodic Boundary Conditions**: Full support for bulk and surface systems

## Quick Start

```python
from aimnet.calculators import AIMNet2Calculator

# Load model
calc = AIMNet2Calculator("aimnet2")

# Run inference
result = calc({
    "coord": coords,    # (N, 3) array in Angstrom
    "numbers": numbers, # (N,) atomic numbers
    "charge": 0.0,      # molecular charge
}, forces=True)

# Access results
energy = result["energy"]   # eV
forces = result["forces"]   # eV/Angstrom
charges = result["charges"] # partial charges
```

## Available Models

| Model           | Elements                                      | Description                      |
| --------------- | --------------------------------------------- | -------------------------------- |
| `aimnet2`       | H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br, I | wB97M-D3 (default)               |
| `aimnet2_b973c` | H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br, I | B97-3c functional                |
| `aimnet2_2025`  | H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br, I | B97-3c + improved intermolecular |
| `aimnet2nse`    | H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br, I | Open-shell chemistry             |
| `aimnet2pd`     | H, B, C, N, O, F, Si, P, S, Cl, Se, Br, Pd, I | Palladium-containing systems     |

_Each model has ensemble members (append \_0 to \_3). Ensemble averaging recommended for production._

## Installation

Basic installation:

```bash
pip install aimnet
```

With optional features:

```bash
# ASE integration
pip install "aimnet[ase]"

# PySisyphus integration
pip install "aimnet[pysis]"

# Training tools
pip install "aimnet[train]"
```

**Requirements:** Python 3.11 or 3.12. GPU support optional (PyTorch with CUDA).

## Documentation Guide

### Core Documentation

- **[Getting Started](getting_started.md)** - Installation and first calculations
- **[Calculator API](calculator.md)** - Comprehensive reference for `AIMNet2Calculator`
- **[Model Format](model_format.md)** - Understanding model files and metadata
- **[Long Range](long_range.md)** - Coulomb and dispersion methods

### Workflows

- **[Training](train.md)** - Training custom models
- **[CLI Reference](cli.md)** - Command-line tools

### API Reference

- **[Calculators](api/calculators.md)** - `AIMNet2Calculator`, `AIMNet2ASE`, `AIMNet2Pysis`
- **[Modules](api/modules.md)** - Core neural network modules
- **[Data](api/data.md)** - Dataset and sampling utilities
- **[Config](api/config.md)** - Configuration utilities

## Common Use Cases

### Periodic Systems

```python
result = calc({
    "coord": coords,
    "numbers": numbers,
    "charge": 0.0,
    "cell": cell_vectors,  # (3, 3) in Angstrom
}, forces=True, stress=True)
```

### Configuring Long-Range Methods

```python
# DSF for periodic systems
calc.set_lrcoulomb_method("dsf", cutoff=15.0)

# Ewald for high accuracy (accuracy parameter controls precision)
calc.set_lrcoulomb_method("ewald", ewald_accuracy=1e-8)
```

### ASE Integration

```python
from ase.io import read
from aimnet.calculators import AIMNet2ASE

atoms = read("structure.xyz")
atoms.calc = AIMNet2ASE("aimnet2")
energy = atoms.get_potential_energy()
```

## Support and Contributing

- **Issues**: [GitHub Issues](https://github.com/isayevlab/aimnetcentral/issues)
- **Discussions**: [GitHub Discussions](https://github.com/isayevlab/aimnetcentral/discussions)
- **Repository**: [github.com/isayevlab/aimnetcentral](https://github.com/isayevlab/aimnetcentral)

## Citation

If you use AIMNet2, please cite:

```bibtex
@article{aimnet2,
  title={AIMNet2: A Neural Network Potential to Meet Your Neutral, Charged, Organic, and Elemental-Organic Needs},
  author={Anstine, Dylan M and Zubatyuk, Roman and Isayev, Olexandr},
  journal={Chemical Science},
  volume={16},
  pages={10228--10244},
  year={2025},
  doi={10.1039/D4SC08572H}
}
```

## License

See [LICENSE](https://github.com/isayevlab/aimnetcentral/blob/main/LICENSE) for details.
