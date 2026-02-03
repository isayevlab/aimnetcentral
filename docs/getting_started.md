# Getting Started with AIMNet2

This guide will help you get up and running with AIMNet2 for molecular simulations.

## Prerequisites

- **Python**: Version 3.11 or 3.12
- **pip**: Package installer (usually included with Python)
- **Optional**: CUDA-capable GPU for faster inference

## Installation

### Basic Installation

Install AIMNet2 from GitHub:

```bash
pip install git+https://github.com/isayevlab/aimnetcentral.git
```

### With ASE Integration

For molecular dynamics and geometry optimization with ASE:

```bash
pip install "aimnet[ase] @ git+https://github.com/isayevlab/aimnetcentral.git"
```

### GPU Support

AIMNet2 works on CPU by default. For GPU acceleration, install PyTorch with CUDA:

```bash
# For CUDA 12.4
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Check [pytorch.org](https://pytorch.org/get-started/locally/) for other CUDA versions.

## Your First Calculation

Let's compute the energy and forces for a water molecule:

```python
import torch
from aimnet.calculators import AIMNet2Calculator

# Create calculator with default model
calc = AIMNet2Calculator("aimnet2")

# Water molecule coordinates (Angstrom)
coords = torch.tensor([
    [0.0000,  0.0000,  0.1173],  # O
    [0.0000,  0.7572, -0.4692],  # H
    [0.0000, -0.7572, -0.4692],  # H
])

# Run calculation
result = calc({
    "coord": coords,
    "numbers": torch.tensor([8, 1, 1]),  # O, H, H
    "charge": 0.0,
}, forces=True)

print(f"Energy: {result['energy'].item():.4f} eV")
print(f"Forces shape: {result['forces'].shape}")
```

## Understanding the Output

The calculator returns a dictionary with:

| Key       | Shape                   | Description                                    | Units |
| --------- | ----------------------- | ---------------------------------------------- | ----- |
| `energy`  | `()` or `(B,)`          | Total energy                                   | eV    |
| `charges` | `(N,)` or `(B, N)`      | Atomic partial charges                         | e     |
| `forces`  | `(N, 3)` or `(B, N, 3)` | Atomic forces (if requested)                   | eV/Å  |
| `stress`  | `(3, 3)` or `(B, 3, 3)` | Stress tensor (if requested with PBC)          | eV/Å³ |
| `hessian` | `(N, 3, N, 3)`          | Hessian matrix (if requested, single molecule) | eV/Å² |

## Common Tasks

### Batch Processing

Process multiple molecules at once. For molecules with different atom counts, use
flat coordinates with `mol_idx` to indicate which molecule each atom belongs to:

```python
# Water (3 atoms) and ammonia cation (5 atoms)
# Use flat coordinates with mol_idx for different-sized molecules
coords = torch.tensor([
    # Water: O at origin, two H atoms
    [0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0],
    # Ammonia cation: N at origin, four H atoms (tetrahedral)
    [0.0, 0.0, 0.0], [0.59, 0.59, 0.59], [0.59, -0.59, -0.59],
    [-0.59, 0.59, -0.59], [-0.59, -0.59, 0.59],
])

result = calc({
    "coord": coords,                                      # (8, 3) flat
    "numbers": torch.tensor([8, 1, 1, 7, 1, 1, 1, 1]),    # O, H, H, N, H, H, H, H
    "charge": torch.tensor([0.0, 1.0]),                   # Water: 0, NH4+: +1
    "mol_idx": torch.tensor([0, 0, 0, 1, 1, 1, 1, 1]),    # Which molecule each atom belongs to
}, forces=True)

# result["energy"] has shape (2,) - one energy per molecule
# result["forces"] has shape (8, 3) - forces for all atoms
```

### Periodic Systems

For systems with periodic boundary conditions:

```python
# Bulk silicon cell
cell = torch.tensor([
    [5.43, 0.00, 0.00],
    [0.00, 5.43, 0.00],
    [0.00, 0.00, 5.43],
])

result = calc({
    "coord": coords,
    "numbers": numbers,
    "charge": 0.0,
    "cell": cell,
}, forces=True, stress=True)

# Access stress tensor
stress = result["stress"]  # (3, 3)
```

For periodic systems, configure the Coulomb method:

```python
# Use DSF for periodic systems (recommended)
calc.set_lrcoulomb_method("dsf", cutoff=15.0)
```

### Charged Systems

AIMNet2 handles charged molecules:

```python
# Hydroxide ion (OH-)
result = calc({
    "coord": coords,
    "numbers": torch.tensor([8, 1]),  # O, H
    "charge": -1.0,  # Net charge
}, forces=True)
```

### Using Different Models

Choose a specific model for your chemistry:

```python
# For palladium-containing systems
calc_pd = AIMNet2Calculator("aimnet2pd")

# For open-shell systems
calc_nse = AIMNet2Calculator("aimnet2nse")

# Ensemble averaging (recommended for production)
calcs = [AIMNet2Calculator(f"aimnet2_{i}") for i in range(4)]
energies = [calc(data)["energy"] for calc in calcs]
avg_energy = sum(energies) / len(energies)
```

## ASE Integration

Use AIMNet2 with ASE for optimization and molecular dynamics:

```python
from ase.io import read
from ase.optimize import BFGS
from aimnet.calculators import AIMNet2ASE

# Load structure
atoms = read("molecule.xyz")

# Attach calculator
atoms.calc = AIMNet2ASE("aimnet2")

# Optimize geometry
opt = BFGS(atoms)
opt.run(fmax=0.01)  # Converge to 0.01 eV/Angstrom

# Get final energy
final_energy = atoms.get_potential_energy()
```

## Next Steps

Now that you're familiar with the basics, explore:

- **[Calculator API](calculator.md)** - Complete reference for all calculator features
- **[Long Range Methods](long_range.md)** - Choosing Coulomb methods for your system
- **[Model Format](model_format.md)** - Understanding model files and metadata
- **[Training Guide](train.md)** - Training custom models on your data

## Performance Tips

1. **Use GPU**: Install CUDA-enabled PyTorch for 10-50x speedup
2. **Batch molecules**: Process similar-sized molecules together
3. **Compile mode**: Use `compile_model=True` for MD
4. **Adjust nb_threshold**: Lower values use less memory, higher values are faster on GPU

```python
# For molecular dynamics with compilation
calc = AIMNet2Calculator("aimnet2", compile_model=True)

# Force specific device
calc = AIMNet2Calculator("aimnet2", device="cuda:0")
```

## Getting Help

- **Documentation**: You're reading it! Check the sidebar for more topics
- **Issues**: [GitHub Issues](https://github.com/isayevlab/aimnetcentral/issues)
- **Discussions**: [GitHub Discussions](https://github.com/isayevlab/aimnetcentral/discussions)
