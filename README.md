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
- Flexible Long-Range Interactions: Optionally employ the Dumped-Shifted Force (DSF) or Ewald summation Coulomb models for accurate calculations in large or periodic systems.

## Installation

### Basic Installation

Install from GitHub:

```bash
pip install git+https://github.com/isayevlab/aimnetcentral.git
```

For non-default pytorch installation (e.g. CUDA on Windows), install pytorch first, see [pytorch.org](https://pytorch.org/).

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

If you use AIMNet2 in your research, please cite:

```bibtex
@article{aimnet2,
  title={AIMNet2: A Neural Network Potential to Meet Your Neutral, Charged, Organic, and Elemental-Organic Needs},
  author={Zubatyuk, Roman and Smith, Justin S and Nebgen, Benjamin and Tretiak, Sergei and Isayev, Olexandr},
  journal={},
  year={2024}
}
```

## License

See [LICENSE](LICENSE) file for details.
