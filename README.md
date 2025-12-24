[![Release](https://img.shields.io/github/v/release/isayevlab/aimnetcentral)](https://github.com/isayevlab/aimnetcentral/releases)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://www.python.org/)
[![Build status](https://img.shields.io/github/actions/workflow/status/isayevlab/aimnetcentral/main.yml?branch=main)](https://github.com/isayevlab/aimnetcentral/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/isayevlab/aimnetcentral/branch/main/graph/badge.svg)](https://codecov.io/gh/isayevlab/aimnetcentral)
[![License](https://img.shields.io/github/license/isayevlab/aimnetcentral)](https://github.com/isayevlab/aimnetcentral/blob/main/LICENSE)

- **Github repository**: <https://github.com/isayevlab/aimnetcentral/>
- **Documentation** <https://isayevlab.github.io/aimnetcentral/>

# AIMNet2 : ML interatomic potential for fast and accurate atomistic simulations

## Key Features

- Accurate and Versatile: AIMNet2 excels at modeling neutral, charged, organic, and elemental-organic systems.
- Flexible Interfaces: Use AIMNet2 through convenient calculators for popular simulation packages like ASE and PySisyphus.
- Flexible Long-Range Interactions: Optionally employ the Damped-Shifted Force (DSF) or Ewald summation Coulomb models for accurate calculations in large or periodic systems.

## Requirements

### Python Version

AIMNet2 requires **Python 3.11 or 3.12**.

### GPU Support (Optional)

AIMNet2 works on CPU out of the box. For GPU acceleration:

- **CUDA GPU**: Install PyTorch with CUDA support from [pytorch.org](https://pytorch.org/get-started/locally/)
- **compile_mode**: Requires CUDA for ~5x MD speedup (see Performance Optimization)

Example PyTorch installation with CUDA 12.4:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

## Available Models

| Model                | Alias           | Elements                                      | Description                  |
| -------------------- | --------------- | --------------------------------------------- | ---------------------------- |
| `aimnet2_wb97m_d3_X` | `aimnet2`       | H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br, I | wB97M-D3 (default)           |
| `aimnet2_b973c_d3_X` | `aimnet2_b973c` | H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br, I | B97-3c functional            |
| `aimnet2_2025_b973c_d3_X` | `aimnet2_b973c` | H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br, I | B97-3c + improved intermolecular interactions            |
| `aimnet2nse_X`       | `aimnet2nse`    | H, C, N, O, F, S, Cl                          | Open-shell chemistry         |
| `aimnet2-pd_X`       | `aimnet2pd`     | H, C, N, O, F, P, S, Cl, Pd                   | Palladium-containing systems |

_X = 0-3 for ensemble members. Ensemble averaging recommended for production use._

## Installation

### Basic Installation

Install from GitHub:

```bash
pip install git+https://github.com/isayevlab/aimnetcentral.git
```

### Optional Features

AIMNet2 provides optional extras for different use cases:

**ASE Calculator** (for atomistic simulations with ASE):

```bash
pip install "aimnet[ase] @ git+https://github.com/isayevlab/aimnetcentral.git"
```

**PySisyphus Calculator** (for reaction path calculations):

```bash
pip install "aimnet[pysis] @ git+https://github.com/isayevlab/aimnetcentral.git"
```

**Training** (for model training and development):

```bash
pip install "aimnet[train] @ git+https://github.com/isayevlab/aimnetcentral.git"
```

**All Features**:

```bash
pip install "aimnet[ase,pysis,train] @ git+https://github.com/isayevlab/aimnetcentral.git"
```

### Development Installation

For contributors, use [uv](https://docs.astral.sh/uv/) for fast dependency management:

```bash
git clone https://github.com/isayevlab/aimnetcentral.git
cd aimnetcentral
make install
source .venv/bin/activate
```

## Quick Start

### Basic Usage (Core)

```python
from aimnet.calculators import AIMNet2Calculator

# Load a pre-trained model
calc = AIMNet2Calculator("aimnet2")

# Prepare input
data = {
    "coord": coordinates,  # Nx3 array
    "numbers": atomic_numbers,  # N array
    "charge": 0.0,
}

# Run inference
results = calc(data, forces=True)
print(results["energy"], results["forces"])
```

### Output Data

The calculator returns a dictionary with the following keys:

| Key       | Shape                   | Description                          |
| --------- | ----------------------- | ------------------------------------ |
| `energy`  | `(,)` or `(B,)`         | Total energy in eV                   |
| `charges` | `(N,)` or `(B, N)`      | Atomic partial charges in e          |
| `forces`  | `(N, 3)` or `(B, N, 3)` | Atomic forces in eV/A (if requested) |
| `hessian` | `(N, 3, N, 3)`          | Second derivatives (if requested)    |
| `stress`  | `(3, 3)`                | Stress tensor for PBC (if requested) |

_B = batch size, N = number of atoms_

### ASE Integration

With `aimnet[ase]` installed:

```python
from ase.io import read
from aimnet.calculators import AIMNet2ASE

atoms = read("molecule.xyz")
atoms.calc = AIMNet2ASE("aimnet2")

energy = atoms.get_potential_energy()
forces = atoms.get_forces()
```

### Periodic Boundary Conditions

For periodic systems, provide a unit cell:

```python
data = {
    "coord": coordinates,
    "numbers": atomic_numbers,
    "charge": 0.0,
    "cell": cell_vectors,  # 3x3 array in Angstrom
}
results = calc(data, forces=True, stress=True)
```

### Long-Range Coulomb Methods

Configure electrostatic interactions for large or periodic systems:

```python
# Damped-Shifted Force (DSF) - recommended for periodic systems
calc.set_lrcoulomb_method("dsf", cutoff=15.0, dsf_alpha=0.2)

# Ewald summation - for accurate periodic electrostatics
calc.set_lrcoulomb_method("ewald", cutoff=15.0)
```

### Performance Optimization

For molecular dynamics simulations, use `compile_mode` for ~5x speedup:

```python
calc = AIMNet2Calculator("aimnet2", compile_mode=True)
```

Requirements:

- CUDA GPU required
- Not compatible with periodic boundary conditions
- Best for repeated inference on similar-sized systems

### Training

With `aimnet[train]` installed:

```bash
aimnet train --config my_config.yaml --model aimnet2.yaml
```

## Development

Common development tasks using `make`:

```bash
make check       # Run linters and code quality checks
make test        # Run tests with coverage
make docs        # Build and serve documentation
make build       # Build distribution packages
```

## Citation

If you use AIMNet2 in your research, please cite the appropriate paper:

**AIMNet2 (main model):**

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

**AIMNet2-NSE:** [ChemRxiv preprint](https://chemrxiv.org/engage/chemrxiv/article-details/692d304c65a54c2d4a7ab3c7)

**AIMNet2-Pd:** [ChemRxiv preprint](https://chemrxiv.org/engage/chemrxiv/article-details/67d7b7f7fa469535b97c021a)

## License

See [LICENSE](LICENSE) file for details.
