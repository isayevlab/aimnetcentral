[![Release](https://img.shields.io/github/v/release/isayevlab/aimnetcentral)](https://github.com/isayevlab/aimnetcentral/releases) [![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/) [![Build status](https://img.shields.io/github/actions/workflow/status/isayevlab/aimnetcentral/main.yml?branch=main)](https://github.com/isayevlab/aimnetcentral/actions/workflows/main.yml?query=branch%3Amain) [![codecov](https://codecov.io/gh/isayevlab/aimnetcentral/branch/main/graph/badge.svg)](https://codecov.io/gh/isayevlab/aimnetcentral) [![License](https://img.shields.io/github/license/isayevlab/aimnetcentral)](https://github.com/isayevlab/aimnetcentral/blob/main/LICENSE)

- **Documentation**: <https://isayevlab.github.io/aimnetcentral/>
- **Repository**: <https://github.com/isayevlab/aimnetcentral/>

# AIMNet2 : ML Interatomic Potential for Fast and Accurate Atomistic Simulations

AIMNet2 is a neural network interatomic potential that predicts energies, forces, atomic charges, stress tensors, and Hessians for organic and elemental-organic molecules. It supports 14 elements (H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br, I) with specialized models for open-shell chemistry and palladium catalysis.

## Key Features

- Accurate for neutral, charged, organic, and elemental-organic systems
- GPU-accelerated with NVIDIA Warp CUDA kernels and `torch.compile` support
- ASE, PySisyphus, and TorchSim calculator interfaces
- Periodic boundary conditions with DSF and Ewald Coulomb methods
- DFT-D3 dispersion corrections (BJ damping, GPU-accelerated)
- Adaptive neighbor lists with automatic dense/sparse mode selection

## Installation

### Requirements

- **Python** 3.11+
- **PyTorch** 2.8+ ([pytorch.org](https://pytorch.org/get-started/locally/))

### Using pip

```bash
# Install PyTorch first (with CUDA if you have a GPU)
pip install torch --index-url https://download.pytorch.org/whl/cu126

# Install AIMNet2
pip install aimnet
```

### Using uv (recommended for fast installs)

```bash
# Install PyTorch + AIMNet2
uv pip install torch --index-url https://download.pytorch.org/whl/cu126
uv pip install aimnet
```

### Using conda/mamba

```bash
# Create environment with PyTorch from conda-forge
mamba create -n aimnet python=3.12 pytorch pytorch-cuda=12.6 -c pytorch -c nvidia -c conda-forge
mamba activate aimnet

# Install AIMNet2 via pip (not yet on conda-forge)
pip install aimnet
```

### Optional Extras

```bash
pip install "aimnet[ase]"             # ASE calculator interface
pip install "aimnet[pysis]"           # PySisyphus reaction path calculator
pip install "aimnet[sella]"           # Sella TS optimizer (includes ASE)
pip install "aimnet[hf]"              # Hugging Face Hub model loading
pip install "aimnet[torchsim,ase]"    # TorchSim integration (Python 3.12+)
pip install "aimnet[train]"           # Training pipeline (W&B, ignite)
pip install "aimnet[ase,pysis,sella,hf,torchsim,train]" # All extras available on this Python
```

### Development Setup

```bash
git clone https://github.com/isayevlab/aimnetcentral.git
cd aimnetcentral
make install        # Creates venv, installs all extras + dev tools
source .venv/bin/activate
```

## Available Models

| Model | Elements | Description |
| --- | --- | --- |
| `aimnet2` | H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br, I | wB97M-D3 (default) |
| `aimnet2-2025` | H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br, I | B97-3c + improved intermolecular (recommended) |
| `aimnet2-b973c` | H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br, I | B97-3c (superseded by aimnet2-2025) |
| `aimnet2-nse` | H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br, I | Open-shell / radical chemistry |
| `aimnet2-pd` | H, B, C, N, O, F, Si, P, S, Cl, Se, Br, Pd, I | Pd systems with CPCM solvation (THF) |
| `aimnet2-rxn` | H, C, N, O | Reactive chemistry (TS, NEB, IRC) |

Each model has 4 ensemble members (0-3). Models are auto-downloaded on first use. Previously published aliases (`aimnet2_2025`, `aimnet2nse`, `aimnet2pd`, etc.) continue to resolve.

## Quick Start

### Core Calculator

```python
from aimnet.calculators import AIMNet2Calculator

calc = AIMNet2Calculator("aimnet2")

results = calc(
    {"coord": coordinates, "numbers": atomic_numbers, "charge": 0.0},
    forces=True,
)
print(results["energy"], results["forces"])
```

### ASE Integration

```python
from ase.io import read
from aimnet.calculators import AIMNet2ASE

atoms = read("molecule.xyz")
atoms.calc = AIMNet2ASE("aimnet2", charge=0)

energy = atoms.get_potential_energy()
forces = atoms.get_forces()
```

### Periodic Systems

```python
data = {
    "coord": coordinates,
    "numbers": atomic_numbers,
    "charge": 0.0,
    "cell": cell_vectors,  # 3x3 array in Angstrom
}
results = calc(data, forces=True, stress=True)

# Configure Coulomb method for periodic systems
calc.set_lrcoulomb_method("dsf", cutoff=15.0, dsf_alpha=0.2)
# or Ewald summation with the default nvalchemiops accuracy
calc.set_lrcoulomb_method("ewald", ewald_accuracy=1e-6)
```

### Performance: torch.compile

For molecular dynamics, `compile_model=True` gives ~5x speedup (requires CUDA):

```python
calc = AIMNet2Calculator("aimnet2", compile_model=True)
```

### Output Reference

| Key       | Shape                   | Description                          |
| --------- | ----------------------- | ------------------------------------ |
| `energy`  | `(,)` or `(B,)`         | Total energy in eV                   |
| `charges` | `(N,)` or `(B, N)`      | Atomic partial charges in e          |
| `forces`  | `(N, 3)` or `(B, N, 3)` | Atomic forces in eV/A (if requested) |
| `hessian` | `(N, 3, N, 3)`          | Second derivatives (if requested)    |
| `stress`  | `(3, 3)`                | Stress tensor for PBC (if requested) |

### Loading from Hugging Face

AIMNet2 models are available on [Hugging Face](https://huggingface.co/isayevlab). Install the optional HF extras:

```bash
pip install "aimnet[hf]"
```

```python
from aimnet.calculators import AIMNet2Calculator

# Load from Hugging Face — downloads and caches automatically
calc = AIMNet2Calculator("isayevlab/aimnet2-wb97m-d3")

# All available HF repos:
# isayevlab/aimnet2-wb97m-d3   general purpose (wB97M-D3)
# isayevlab/aimnet2-2025       improved intermolecular (B97-3c)
# isayevlab/aimnet2-nse        open-shell / radicals
# isayevlab/aimnet2-pd         palladium chemistry
# isayevlab/aimnet2-rxn        reactive chemistry / TS / IRC

# Load a specific ensemble member (0–3) or a pinned revision:
calc = AIMNet2Calculator("isayevlab/aimnet2-wb97m-d3", ensemble_member=2)
calc = AIMNet2Calculator("isayevlab/aimnet2-wb97m-d3", revision="v1.0")

# Private repos — pass a HF token:
calc = AIMNet2Calculator("myorg/private-model", token="hf_...")

# Local directory in HF repo layout (config.json + ensemble_N.safetensors):
calc = AIMNet2Calculator("/path/to/local/repo")

# Existing registry aliases still work without any HF deps:
calc = AIMNet2Calculator("aimnet2")
```

Try the [interactive demo](https://huggingface.co/spaces/isayevlab/aimnet2-demo)!

## How It Works

### Architecture

AIMNet2 uses a message-passing neural network with iterative charge equilibration:

1. **AEVSV** - Gaussian basis expansion of pairwise distances and displacement vectors
2. **ConvSV** - Sparse indexed convolution combining atomic features with local geometry (GPU-accelerated via NVIDIA Warp kernels)
3. **MLP passes** - Iterative refinement with charge prediction and Coulomb-aware features
4. **Output modules** - Energy, forces (via autograd), charges, stress, Hessian

### Dense vs Sparse Mode

The calculator automatically selects the optimal strategy:

- **Dense mode (O(N^2))** - Small molecules on GPU. Fully connected graph, maximum parallelism.
- **Sparse mode (O(N))** - Large systems or CPU. Adaptive neighbor lists with ~75% buffer utilization, 16-byte aligned allocations, automatic overflow handling.

The threshold is configurable via `nb_threshold` (default: 120 atoms).

### Long-Range Corrections

- **DFT-D3** dispersion with BJ damping (GPU-accelerated via nvalchemiops)
- **Coulomb**: Simple (all-pairs), DSF (damped-shifted force), or Ewald summation
- Long-range modules support inference forces and stress where documented; DSF and DFT-D3 use specialized backend paths, with Hessian support limited to the documented pure-torch DFT-D3 path

## Training

```bash
pip install "aimnet[train]"
aimnet train --config my_config.yaml --model aimnet2.yaml
```

See the [training documentation](https://isayevlab.github.io/aimnetcentral/train/) for dataset preparation, configuration, and W&B integration.

## Development

```bash
make check       # Linters and code quality (ruff, markdownlint, prettier)
make test        # Tests with coverage (pytest, parallel)
make docs        # Build and serve documentation (mkdocs)
make docs-test   # Validate docs build
```

## Citation

If you use AIMNet2 in your research, please cite:

**AIMNet2:**

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
