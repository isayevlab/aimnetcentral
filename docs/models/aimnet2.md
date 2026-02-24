# AIMNet2 (wB97M-D3)

## Overview

AIMNet2 is the **default general-purpose model** for organic and main-group molecular chemistry. It is trained against wB97M-D3/def2-TZVPP reference data, a modern range-separated hybrid functional with DFT-D3 dispersion correction widely regarded as one of the most accurate for thermochemistry and non-covalent interactions.

**Supported elements:** H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br, I (14 elements)

**Registry alias:** `aimnet2` (loads ensemble member `aimnet2_wb97m_d3_0`)

**Ensemble members:** `aimnet2_wb97m_d3_0` through `aimnet2_wb97m_d3_3` (4 models)

If you are unsure which model to use, start here. AIMNet2 provides the best balance of accuracy and element coverage for most molecular chemistry applications.

## Strengths and Limitations

### Strengths

- Broad element coverage spanning organic, bioorganic, and main-group chemistry
- High-quality wB97M-D3 reference level with reliable thermochemistry
- Good accuracy for conformational energies, reaction energies, and molecular geometries
- Built-in DFT-D3 dispersion and long-range Coulomb corrections
- Validated on drug-like molecules, amino acids, and small organics

### Limitations

!!! warning "No transition metals" AIMNet2 does not support any transition metal elements. For palladium chemistry, use [AIMNet2-Pd](aimnet2pd.md).

!!! warning "Molecular systems only" The training data consists of molecular (gas-phase) structures. While periodic boundary conditions are supported computationally, the model has not been validated for bulk materials, surfaces, or extended solids.

!!! warning "System size" Validated for systems up to approximately 100 heavy atoms. Larger systems may work but accuracy has not been systematically benchmarked beyond this range.

!!! warning "Closed-shell only" This model assumes closed-shell electronic structure. For radicals, triplet states, or any system with unpaired electrons, use [AIMNet2-NSE](aimnet2nse.md).

!!! warning "Single-reference DFT" Trained on single-determinant DFT data. Systems with strong multi-reference character (e.g., biradicals, stretched bonds near dissociation, certain transition states) may not be reliable.

## Typical Use Cases

- **Drug-like molecule energetics** -- conformational energies, tautomer ranking, protonation states
- **Reaction thermochemistry** -- reaction energies and enthalpies for organic transformations
- **Geometry optimization** -- equilibrium structures of organic and bioorganic molecules
- **Molecular dynamics** -- sampling conformational space at near-DFT accuracy
- **High-throughput screening** -- rapid energy evaluation across molecular libraries (for B97-3c-level screening, consider [AIMNet2-2025](aimnet2_2025.md))

## Quick Example

Single-point energy and forces for aspirin (acetylsalicylic acid):

```python
import torch
from aimnet.calculators import AIMNet2Calculator

calc = AIMNet2Calculator("aimnet2", compile_model=True)

# Aspirin: C9H8O4 (21 atoms)
coords = torch.tensor([
    [-3.2005,  0.5364, -0.0398],  # C
    [-1.8893,  1.0051, -0.0282],  # C
    [-0.8360,  0.0875, -0.0114],  # C
    [-1.1294, -1.2783, -0.0067],  # C
    [-2.4406, -1.7333, -0.0183],  # C
    [-3.4786, -0.8310, -0.0345],  # C
    [ 0.5290,  0.5850,  0.0021],  # C
    [ 1.5835, -0.2987,  0.0163],  # O
    [ 0.7609,  1.7867, -0.0008],  # O
    [ 2.9063,  0.2310,  0.0278],  # C
    [ 3.0266,  1.5268,  0.0206],  # O
    [ 3.8795, -0.7152,  0.0451],  # C
    [-1.6548,  2.0646, -0.0325],  # H
    [-0.3177, -1.9947,  0.0065],  # H
    [-2.6587, -2.7970, -0.0147],  # H
    [-4.4941, -1.2099, -0.0435],  # H
    [-4.0011,  1.2644, -0.0527],  # O
    [-5.3078,  0.7771, -0.0637],  # H
    [ 3.5437, -1.7496,  0.0486],  # H
    [ 4.9141, -0.3821,  0.0545],  # H
    [ 4.7786, -0.5819, -0.8637],  # H
])

numbers = torch.tensor([6, 6, 6, 6, 6, 6, 6, 8, 8, 6, 8, 6, 1, 1, 1, 1, 8, 1, 1, 1, 1])

result = calc({
    "coord": coords,
    "numbers": numbers,
    "charge": torch.tensor(0.0),
}, forces=True)

print(f"Energy: {result['energy'].item():.6f} eV")
print(f"Forces (max component): {result['forces'].abs().max().item():.6f} eV/A")
print(f"Partial charges: {result['charges']}")
```

### ASE Integration

```python
from ase.io import read
from ase.optimize import BFGS
from aimnet.calculators import AIMNet2ASE, AIMNet2Calculator

base_calc = AIMNet2Calculator("aimnet2", compile_model=True)
atoms = read("aspirin.xyz")
atoms.calc = AIMNet2ASE(base_calc, charge=0)

opt = BFGS(atoms)
opt.run(fmax=0.01)

print(f"Optimized energy: {atoms.get_potential_energy():.6f} eV")
```

## Computational Details

### Training Data

The model is trained on a diverse dataset of organic and main-group molecules computed at the wB97M-D3/def2-TZVPP level of theory. The training set covers neutral and charged species, various conformations, and bond-breaking geometries.

### Long-Range Corrections

AIMNet2 includes external DFT-D3 dispersion and long-range Coulomb modules. For non-periodic (gas-phase) calculations, these are handled automatically. For periodic systems, configure the Coulomb method explicitly:

```python
calc.set_lrcoulomb_method("dsf", cutoff=15.0)
```

See [Long-Range Methods](../long_range.md) for details on DSF vs Ewald.

### Ensemble Averaging

Four independently trained ensemble members are available (`aimnet2_wb97m_d3_0` through `aimnet2_wb97m_d3_3`). Use ensemble averaging for production calculations to improve accuracy and estimate prediction uncertainty:

```python
models = [AIMNet2Calculator(f"aimnet2_wb97m_d3_{i}") for i in range(4)]
results = [m(data, forces=True) for m in models]

energies = torch.stack([r["energy"] for r in results])
avg_energy = energies.mean(dim=0)
std_energy = energies.std(dim=0)  # Uncertainty estimate
```

### Performance

- Use `compile_model=True` for molecular dynamics or repeated evaluations on similarly-sized systems
- First call with compilation incurs a warmup cost; subsequent calls are significantly faster
- GPU inference is recommended for systems larger than about 10 atoms

## References

Anstine, D. M.; Zubatyuk, R.; Isayev, O. AIMNet2: A Neural Network Potential to Meet your Neutral, Charged, Organic, and Elemental-Organic Needs. _Chemical Science_ **2025**, _16_, 10228--10244. DOI: [10.1039/D4SC08572H](https://doi.org/10.1039/D4SC08572H)
