---
title: AIMNet Machine-Learned Interatomic Potential
---

# AIMNet2: Machine-Learned Interatomic Potential

[![Release](https://img.shields.io/github/v/release/isayevlab/aimnetcentral)](https://img.shields.io/github/v/release/isayevlab/aimnetcentral) [![Build status](https://img.shields.io/github/actions/workflow/status/isayevlab/aimnetcentral/main.yml?branch=main)](https://github.com/isayevlab/aimnetcentral/actions/workflows/main.yml?query=branch%3Amain) [![License](https://img.shields.io/github/license/isayevlab/aimnetcentral)](https://img.shields.io/github/license/isayevlab/aimnetcentral)

## What is AIMNet2?

AIMNet2 is a neural network potential for fast and accurate atomistic simulations. Built on PyTorch, it provides:

- Accurate predictions for neutral, charged, organic, and elemental-organic systems
- Fast inference on both CPU and GPU
- Integration with popular simulation packages (ASE, PySisyphus)
- Configurable long-range electrostatics (DSF, Ewald) for periodic systems

AIMNet2 combines a graph neural network architecture with flexible long-range interactions, making it suitable for molecular dynamics, geometry optimization, and property prediction across diverse chemical systems.

## Explore AIMNet2

### Choose a Model

Find the right model for your chemistry -- general organic, open-shell radicals, Pd catalysis, non-covalent interactions, or high-throughput screening.

[Model Selection Guide](models/guide.md)

### Learn the Basics

Step-by-step tutorials from your first single-point calculation through geometry optimization, molecular dynamics, and batch processing.

[Start with Single-Point Calculations](tutorials/single_point.md)

### Advanced Workflows

Conformer search, reaction paths and transition states, transition metal catalysis, non-covalent interactions, and charged systems.

[Advanced Use Cases](advanced/conformer_search.md)

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

AIMNet2 provides five model families covering a wide range of chemistry -- from general organic molecules to open-shell radicals and palladium catalysis. Each model has ensemble members (append `_0` to `_3`) for uncertainty estimation.

See the **[Model Selection Guide](models/guide.md)** for a detailed comparison and decision flowchart.

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

### First Steps

- **[Getting Started](getting_started.md)** - Installation, setup, and verifying your environment

### Models

- **[Model Selection Guide](models/guide.md)** - Decision flowchart for choosing the right model
- **[Architecture Overview](models/architecture.md)** - AEV descriptors, ConvSV, charge equilibration internals
- **[AIMNet2 (wB97M-D3)](models/aimnet2.md)** - General-purpose default model
- **[AIMNet2-2025](models/aimnet2_2025.md)** - Recommended B97-3c model (supersedes B97-3c)
- **[AIMNet2-B97-3c](models/aimnet2_b973c.md)** - Legacy B97-3c (for reproducibility)
- **[AIMNet2-NSE](models/aimnet2nse.md)** - Open-shell and radical chemistry
- **[AIMNet2-Pd](models/aimnet2pd.md)** - Palladium-containing systems

### Tutorials

- **[Single-Point Calculations](tutorials/single_point.md)** - Your first calculation
- **[Geometry Optimization](tutorials/geometry_optimization.md)** - Structure relaxation with ASE
- **[Molecular Dynamics](tutorials/molecular_dynamics.md)** - NVT/NPT simulations
- **[Periodic Systems](tutorials/periodic_systems.md)** - Crystals, surfaces, and PBC
- **[Batch Processing](tutorials/batch_processing.md)** - Processing molecular datasets
- **[Performance Tuning](tutorials/performance.md)** - `compile_model=True`, GPU optimization

### Advanced Use Cases

- **[Open-Shell Chemistry](advanced/open_shell.md)** - Radicals and spin states
- **[Pd Catalysis](advanced/transition_metal_catalysis.md)** - Transition metal reactions
- **[Non-Covalent Interactions](advanced/intermolecular_interactions.md)** - H-bonding, pi-stacking
- **[Conformer Search](advanced/conformer_search.md)** - Conformational sampling
- **[Reaction Paths & TS](advanced/reaction_paths.md)** - Transition states with PySisyphus
- **[Charged Systems](advanced/charged_systems.md)** - Ions and zwitterions

### Reference

- **[Calculator API](calculator.md)** - Comprehensive reference for `AIMNet2Calculator`
- **[Long Range Methods](long_range.md)** - Coulomb and dispersion methods
- **[Model Format](model_format.md)** - Understanding model files and metadata
- **[Training](train.md)** - Training custom models
- **[CLI Reference](cli.md)** - Command-line tools

### API Reference

- **[Calculators](api/calculators.md)** - `AIMNet2Calculator`, `AIMNet2ASE`, `AIMNet2Pysis`
- **[Modules](api/modules.md)** - Core neural network modules
- **[Data](api/data.md)** - Dataset and sampling utilities
- **[Config](api/config.md)** - Configuration utilities

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
