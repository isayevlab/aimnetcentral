[![PyPI](https://img.shields.io/pypi/v/aimnet)](https://pypi.org/project/aimnet/) [![Release](https://img.shields.io/github/v/release/isayevlab/aimnetcentral)](https://github.com/isayevlab/aimnetcentral/releases) [![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/) [![Build status](https://img.shields.io/github/actions/workflow/status/isayevlab/aimnetcentral/main.yml?branch=main)](https://github.com/isayevlab/aimnetcentral/actions/workflows/main.yml?query=branch%3Amain) [![codecov](https://codecov.io/gh/isayevlab/aimnetcentral/branch/main/graph/badge.svg)](https://codecov.io/gh/isayevlab/aimnetcentral) [![License](https://img.shields.io/github/license/isayevlab/aimnetcentral)](https://github.com/isayevlab/aimnetcentral/blob/main/LICENSE)

# AIMNet2

Fast neural-network interatomic potentials for organic, elemental-organic, open-shell, reactive, and palladium chemistry.

- **Documentation:** <https://isayevlab.github.io/aimnetcentral/>
- **Repository:** <https://github.com/isayevlab/aimnetcentral>
- **PyPI:** <https://pypi.org/project/aimnet/>
- **Demo:** <https://huggingface.co/spaces/isayevlab/aimnet2-demo>

## What It Does

AIMNet2 predicts energies, forces, atomic charges, stress tensors, and Hessians for molecular and periodic atomistic simulations. It combines AIMNet2 neural potentials with adaptive dense/sparse neighbor handling, optional long-range electrostatics, DFT-D3 dispersion, and integrations for simulation workflows.

**Highlights**

- Pretrained model families for general organic chemistry, B97-3c, open-shell chemistry, palladium catalysis, and reactive paths
- Core inference through `AIMNet2Calculator`
- ASE, PySisyphus, Sella, TorchSim, Hugging Face Hub, and CLI workflows
- Periodic systems with DSF, Ewald, and PME Coulomb methods
- DFT-D3(BJ) dispersion through GPU-accelerated nvalchemiops kernels
- Python 3.11, 3.12, and 3.13 support; TorchSim extra is available on Python 3.12+

## Install

AIMNet2 requires Python 3.11+ and PyTorch 2.8+.

```bash
# CPU or PyTorch-default install
pip install aimnet

# CUDA example: install PyTorch first, then AIMNet2
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install aimnet
```

With `uv`:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu126
uv pip install aimnet
```

Optional integrations:

```bash
pip install "aimnet[ase]"              # ASE calculator
pip install "aimnet[pysis]"            # PySisyphus reaction paths
pip install "aimnet[sella]"            # Sella TS optimization, includes ASE
pip install "aimnet[hf]"               # Hugging Face Hub model loading
pip install "aimnet[torchsim,ase]"     # TorchSim, Python 3.12+
pip install "aimnet[train]"            # training and export commands
```

For a full development environment:

```bash
git clone https://github.com/isayevlab/aimnetcentral.git
cd aimnetcentral
make install
source .venv/bin/activate
```

## Quick Start

```python
from aimnet.calculators import AIMNet2Calculator

calc = AIMNet2Calculator("aimnet2")

result = calc(
    {"coord": coordinates, "numbers": atomic_numbers, "charge": 0.0},
    forces=True,
)

print(result["energy"])   # eV
print(result["forces"])   # eV/Angstrom
print(result["charges"])  # e
```

### ASE

```python
from ase.io import read
from aimnet.calculators import AIMNet2ASE

atoms = read("molecule.xyz")
atoms.calc = AIMNet2ASE("aimnet2", charge=0)

energy = atoms.get_potential_energy()
forces = atoms.get_forces()
```

### TorchSim

```python
import ase.io
import torch_sim as ts

from aimnet.calculators import AIMNet2Calculator, AIMNet2TorchSim

atoms = ase.io.read("molecule.xyz")
model = AIMNet2TorchSim(AIMNet2Calculator("aimnet2"))

results = ts.static(system=atoms, model=model)
print(results[0]["potential_energy"], results[0]["forces"])
```

### Periodic Systems

```python
calc = AIMNet2Calculator("aimnet2")
calc.set_lrcoulomb_method("ewald", ewald_accuracy=1e-6)

result = calc(
    {
        "coord": coordinates,
        "numbers": atomic_numbers,
        "charge": 0.0,
        "cell": cell_vectors,
        "pbc": True,
    },
    forces=True,
    stress=True,
)
```

## Model Families

| Model | Elements | Best for |
| --- | --- | --- |
| `aimnet2` | H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br, I | General organic and elemental-organic chemistry, wB97M-D3 |
| `aimnet2-2025` | H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br, I | B97-3c with improved intermolecular interactions |
| `aimnet2-b973c` | H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br, I | Legacy B97-3c reproducibility |
| `aimnet2-nse` | H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br, I | Open-shell systems and radicals |
| `aimnet2-pd` | H, B, C, N, O, F, Si, P, S, Cl, Se, Br, Pd, I | Pd catalysis with B97-3c/CPCM(THF) reference data |
| `aimnet2-rxn` | H, C, N, O | Reactive chemistry, transition states, NEB, IRC; net-neutral systems only |

Each family has four ensemble members indexed `0` through `3`. Short aliases load member `0` by default. Previously published aliases such as `aimnet2_2025`, `aimnet2nse`, and `aimnet2pd` continue to resolve.

See the [model selection guide](https://isayevlab.github.io/aimnetcentral/models/guide/) for limitations, aliases, citations, and download links.

## Hugging Face Models

Install the HF extra and pass a repo ID directly:

```bash
pip install "aimnet[hf]"
```

```python
from aimnet.calculators import AIMNet2Calculator

calc = AIMNet2Calculator("isayevlab/aimnet2-wb97m-d3")
calc = AIMNet2Calculator("isayevlab/aimnet2-wb97m-d3", ensemble_member=2)
calc = AIMNet2Calculator("isayevlab/aimnet2-wb97m-d3", revision="v1.0")
```

Published repositories:

| HF repo | Alias | Use |
| --- | --- | --- |
| `isayevlab/aimnet2-wb97m-d3` | `aimnet2` | General organic chemistry |
| `isayevlab/aimnet2-2025` | `aimnet2-2025` | Improved intermolecular interactions |
| `isayevlab/aimnet2-nse` | `aimnet2-nse` | Open-shell chemistry |
| `isayevlab/aimnet2-pd` | `aimnet2-pd` | Palladium chemistry |
| `isayevlab/aimnet2-rxn` | `aimnet2-rxn` | Reactive paths and transition states |

Private repos can be loaded with `token=` or the `HF_TOKEN` environment variable. Local HF-style directories with `config.json` and `ensemble_N.safetensors` are also supported.

## Outputs

| Key       | Shape                   | Units           |
| --------- | ----------------------- | --------------- |
| `energy`  | scalar or `(B,)`        | eV              |
| `forces`  | `(N, 3)` or `(B, N, 3)` | eV/Angstrom     |
| `charges` | `(N,)` or `(B, N)`      | electron charge |
| `stress`  | `(3, 3)`                | eV/Angstrom^3   |
| `hessian` | `(N, 3, N, 3)`          | eV/Angstrom^2   |

Hessians are single-molecule only and are incompatible with `compile_model=True`. Long-range backends differ in derivative support; see the [calculator](https://isayevlab.github.io/aimnetcentral/calculator/) and [long-range](https://isayevlab.github.io/aimnetcentral/long_range/) docs for the exact contracts.

## Training and CLI

```bash
pip install "aimnet[train]"
aimnet train --config my_config.yaml --model aimnet2.yaml
```

The `aimnet` entry point is installed with the core package. Training, export, and self-atomic-energy commands require the `train` extra.

## Development

```bash
make check       # formatting, linting, dependency checks
make test        # pytest suite
make docs        # serve docs locally
make docs-test   # strict docs build
```

CI runs the core test suite across Python 3.11-3.13 and separate optional-extra lanes for ASE, PySisyphus, Sella, Hugging Face, TorchSim, docs, and security checks.

## Citation

If AIMNet2 is useful in your work, please cite the relevant model papers:

**AIMNet2**

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

**AIMNet2-NSE:** Kalita, B.; Zubatyuk, R.; Anstine, D. M.; Bergeler, M.; Settels, V.; Stork, C.; Spicher, S.; Isayev, O. AIMNet2-NSE: A Transferable Reactive Neural Network Potential for Open-Shell Chemistry. _Angew. Chem. Int. Ed._ **2026**. DOI: [10.1002/anie.202516763](https://doi.org/10.1002/anie.202516763)

**AIMNet2-Pd:** Anstine, D. M.; Zubatyuk, R.; Gallegos, L.; Paton, R.; Wiest, O.; Nebgen, B.; Jones, T.; Gomes, G.; Tretiak, S.; Isayev, O. Transferable Machine Learning Interatomic Potential for Pd-Catalyzed Cross-Coupling Reactions. _ChemRxiv_ **2025**. DOI: [10.26434/chemrxiv-2025-n36r6](https://doi.org/10.26434/chemrxiv-2025-n36r6)

## License

See [LICENSE](LICENSE).
