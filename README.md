[![PyPI](https://img.shields.io/pypi/v/aimnet)](https://pypi.org/project/aimnet/) [![Release](https://img.shields.io/github/v/release/isayevlab/aimnetcentral)](https://github.com/isayevlab/aimnetcentral/releases) [![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)](https://www.python.org/) [![Build status](https://img.shields.io/github/actions/workflow/status/isayevlab/aimnetcentral/main.yml?branch=main)](https://github.com/isayevlab/aimnetcentral/actions/workflows/main.yml?query=branch%3Amain) [![License](https://img.shields.io/github/license/isayevlab/aimnetcentral)](https://github.com/isayevlab/aimnetcentral/blob/main/LICENSE)

# AIMNet2

Fast neural-network interatomic potentials for organic, elemental-organic, open-shell, reactive, and palladium chemistry.

- **Documentation:** <https://isayevlab.github.io/aimnetcentral/>
- **Repository:** <https://github.com/isayevlab/aimnetcentral>
- **PyPI:** <https://pypi.org/project/aimnet/>
- **Demo:** <https://huggingface.co/spaces/isayevlab/aimnet2-demo>

## What It Does

AIMNet2 predicts energies, forces, atomic charges, stress tensors, and Hessians for molecules and periodic systems. Pretrained models cover general organic chemistry, open-shell and reactive chemistry, and palladium catalysis.

- One calculator, `AIMNet2Calculator`, for single molecules, batches, and periodic cells
- ASE, PySisyphus, Sella, TorchSim, Hugging Face Hub, and command-line workflows
- Long-range Coulomb (DSF, Ewald, PME) and DFT-D3(BJ) dispersion on GPU
- Optional CUDA-graph replay and `torch.compile` for repeated evaluations
- Python 3.11 to 3.14 and PyTorch 2.10 or newer

## Install

```bash
pip install aimnet
```

On Linux the default PyTorch wheel from PyPI includes CUDA. To pick a specific CUDA build, install PyTorch first, then AIMNet2:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install aimnet
```

Optional integrations:

```bash
pip install "aimnet[ase]"              # ASE calculator
pip install "aimnet[pysis]"            # PySisyphus reaction paths
pip install "aimnet[sella]"            # Sella transition-state search, includes ASE
pip install "aimnet[hf]"               # Hugging Face Hub model loading
pip install "aimnet[torchsim,ase]"     # TorchSim, Python 3.12+
pip install "aimnet[train]"            # training and export commands
```

For offline or HPC clusters, see [Offline and HPC installation](https://isayevlab.github.io/aimnetcentral/offline_hpc/).

## Quick Start

```python
from aimnet.calculators import AIMNet2Calculator

calc = AIMNet2Calculator("aimnet2")  # downloads the model on first use

water = {
    "coord": [[0.000, 0.000, 0.119], [0.000, 0.763, -0.477], [0.000, -0.763, -0.477]],
    "numbers": [8, 1, 1],
    "charge": 0.0,
}
result = calc(water, forces=True)

print(result["energy"])   # eV
print(result["forces"])   # eV/Å
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
        "coord": coordinates,       # (N, 3), Å
        "numbers": atomic_numbers,  # (N,)
        "charge": 0.0,
        "cell": cell_vectors,       # (3, 3), Å; fully periodic
    },
    forces=True,
    stress=True,
)
```

See the [periodic systems tutorial](https://isayevlab.github.io/aimnetcentral/tutorials/periodic_systems/) and [long-range methods](https://isayevlab.github.io/aimnetcentral/long_range/) for choosing a Coulomb method.

### Performance

`compile_model=False` and `cuda_graph=False` are the defaults. For many repeated evaluations of the same system size on a GPU, such as MD or geometry optimization, `cuda_graph=True` replays captured CUDA graphs and is often several times faster for small and medium molecules. See [CUDA-graph inference](https://isayevlab.github.io/aimnetcentral/cuda_graph/) for supported models and memory costs, and [performance tuning](https://isayevlab.github.io/aimnetcentral/tutorials/performance/) for batching and `torch.compile`.

## Model Families

| Model | Elements | Best for |
| --- | --- | --- |
| `aimnet2` | H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br, I | General organic and elemental-organic chemistry, wB97M-D3 |
| `aimnet2-2025` | H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br, I | B97-3c with improved intermolecular interactions |
| `aimnet2-b973c` | H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br, I | Legacy B97-3c reproducibility |
| `aimnet2-nse` | H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br, I | Open-shell systems and radicals |
| `aimnet2-pd` | H, B, C, N, O, F, Si, P, S, Cl, Se, Br, Pd, I | Pd catalysis with B97-3c/CPCM(THF) reference data |
| `aimnet2-rxn` | H, C, N, O | Reactive chemistry, transition states, NEB, IRC; net-neutral systems only |

Each family has four ensemble members, `0` to `3`; short aliases load member `0`. Previously published names such as `aimnet2_2025`, `aimnet2nse`, and `aimnet2pd` still resolve.

The [model selection guide](https://isayevlab.github.io/aimnetcentral/models/guide/) covers known limitations, aliases, citations, and download links.

## Hugging Face Models

With the `hf` extra installed, pass a repository ID directly:

```python
from aimnet.calculators import AIMNet2Calculator

calc = AIMNet2Calculator("isayevlab/aimnet2-wb97m-d3")
calc = AIMNet2Calculator("isayevlab/aimnet2-wb97m-d3", ensemble_member=2)
calc = AIMNet2Calculator("isayevlab/aimnet2-wb97m-d3", revision="v1.0")
```

| HF repo | Alias | Use |
| --- | --- | --- |
| `isayevlab/aimnet2-wb97m-d3` | `aimnet2` | General organic chemistry |
| `isayevlab/aimnet2-2025` | `aimnet2-2025` | Improved intermolecular interactions |
| `isayevlab/aimnet2-nse` | `aimnet2-nse` | Open-shell chemistry |
| `isayevlab/aimnet2-pd` | `aimnet2-pd` | Palladium chemistry |
| `isayevlab/aimnet2-rxn` | `aimnet2-rxn` | Reactive paths and transition states |

Private repositories load with `token=` or the `HF_TOKEN` environment variable. Local directories with `config.json` and `ensemble_N.safetensors` also work.

## Outputs

| Key       | Shape                               | Units  |
| --------- | ----------------------------------- | ------ |
| `energy`  | `(B,)`                              | eV     |
| `forces`  | `(N, 3)` or `(B, N, 3)`             | eV/Å   |
| `charges` | `(N,)` or `(B, N)`                  | e      |
| `stress`  | `(3, 3)` or `(B, 3, 3)`             | eV/Å^3 |
| `hessian` | `(N, 3, N, 3)` or `(B, N, 3, N, 3)` | eV/Å^2 |

`B` is the number of systems (1 for a single molecule) and `N` the number of atoms. Stress uses the ASE sign convention. Batched Hessians are evaluated per system and returned as a list when the systems differ in size. The [calculator reference](https://isayevlab.github.io/aimnetcentral/calculator/) documents input layouts, batching, and the derivative support of each long-range method.

## Training

```bash
pip install "aimnet[train]"
aimnet train data.train=dataset.h5 data.sae.energy.file=dataset_sae.yaml run_name=firstrun
```

The `aimnet` command is installed with the core package; training, export, and self-atomic-energy commands need the `train` extra. The [training guide](https://isayevlab.github.io/aimnetcentral/train/) covers datasets, configuration, compiled training, stress training, and exporting models.

## Development

```bash
git clone https://github.com/isayevlab/aimnetcentral.git
cd aimnetcentral
make install     # uv environment with all extras and pre-commit hooks
make check       # formatting, linting, dependency checks
make test        # pytest suite
make docs        # serve the docs locally
```

CI tests Python 3.11, 3.13, and 3.14, PyTorch 2.10 to 2.14, and each optional integration. Contributions are welcome; see [CONTRIBUTING.md](https://github.com/isayevlab/aimnetcentral/blob/main/CONTRIBUTING.md).

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

**AIMNet2-2025**

If you use AIMNet2-2025 (`aimnet2-2025`), please also cite [the AIMNet2-2025 paper](https://doi.org/10.1088/2632-2153/aea39f):

```bibtex
@article{nayal2026critical,
  title={Critical benchmarking of machine-learned interatomic potentials for intermolecular and noncovalent interactions},
  author={Nayal, Kamal and Cho, IlKwon and Isayev, Olexandr},
  journal={Machine Learning: Science and Technology},
  year={2026},
  doi={10.1088/2632-2153/aea39f}
}
```

**AIMNet2-NSE:** Kalita, B.; Zubatyuk, R.; Anstine, D. M.; Bergeler, M.; Settels, V.; Stork, C.; Spicher, S.; Isayev, O. AIMNet2-NSE: A Transferable Reactive Neural Network Potential for Open-Shell Chemistry. _Angew. Chem. Int. Ed._ **2026**. DOI: [10.1002/anie.202516763](https://doi.org/10.1002/anie.202516763)

**AIMNet2-Pd:** Anstine, D. M.; Zubatyuk, R.; Gallegos, L.; Paton, R.; Wiest, O.; Nebgen, B.; Jones, T.; Gomes, G.; Tretiak, S.; Isayev, O. Transferable Machine Learning Interatomic Potential for Pd-Catalyzed Cross-Coupling Reactions. _ChemRxiv_ **2025**. DOI: [10.26434/chemrxiv-2025-n36r6](https://doi.org/10.26434/chemrxiv-2025-n36r6)

## License

MIT; see [LICENSE](https://github.com/isayevlab/aimnetcentral/blob/main/LICENSE).
