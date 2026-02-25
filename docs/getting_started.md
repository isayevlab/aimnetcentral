# Getting Started with AIMNet2

This guide will help you install AIMNet2 and verify your setup.

## Prerequisites

- **Python**: Version 3.11 or 3.12
- **pip**: Package installer (usually included with Python)
- **Optional**: CUDA-capable GPU for faster inference

## Installation

### Basic Installation

Install AIMNet2 from PyPI:

```bash
pip install aimnet
```

### With ASE Integration

For molecular dynamics and geometry optimization with ASE:

```bash
pip install "aimnet[ase]"
```

### GPU Support

AIMNet2 works on CPU by default. For GPU acceleration, install PyTorch with CUDA:

```bash
# For CUDA 12.4
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Check [pytorch.org](https://pytorch.org/get-started/locally/) for other CUDA versions.

## Verify Your Installation

After installing, confirm that AIMNet2 loads correctly:

```python
from aimnet.calculators import AIMNet2Calculator

calc = AIMNet2Calculator("aimnet2")
print("AIMNet2 loaded successfully")
```

The first time you load a model, it will be downloaded automatically.

## Loading Your Molecule

Most workflows start from a molecular structure file. Here are the most common approaches:

### From an XYZ file (recommended)

```python
from ase.io import read
from aimnet.calculators import AIMNet2ASE, AIMNet2Calculator

atoms = read("molecule.xyz")
base_calc = AIMNet2Calculator("aimnet2")
atoms.calc = AIMNet2ASE(base_calc, charge=0)
energy = atoms.get_potential_energy()
```

### From a SMILES string (requires RDKit)

```python
from rdkit import Chem
from rdkit.Chem import AllChem
from ase import Atoms

mol = Chem.MolFromSmiles("CCO")  # ethanol
mol = Chem.AddHs(mol)
AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())

atoms = Atoms(
    numbers=[atom.GetAtomicNum() for atom in mol.GetAtoms()],
    positions=mol.GetConformer().GetPositions(),
)
```

See the [Single-Point tutorial](tutorials/single_point.md) for what to do next with your molecule.

## What's Next

Now that AIMNet2 is installed, continue with:

- **[Your First Calculation](tutorials/single_point.md)** - Compute energies, forces, and charges step by step
- **[Choosing the Right Model](models/guide.md)** - Find the best model for your chemistry
- **[Optimizing Performance](tutorials/performance.md)** - GPU acceleration and `compile_model=True` for speed

## Getting Help

- **Documentation**: You're reading it! Check the sidebar for more topics
- **Issues**: [GitHub Issues](https://github.com/isayevlab/aimnetcentral/issues)
- **Discussions**: [GitHub Discussions](https://github.com/isayevlab/aimnetcentral/discussions)
