# AIMNet2-B97-3c

!!! warning "Superseded by AIMNet2-2025" This model has been superseded by [AIMNet2-2025](aimnet2_2025.md), which provides improved intermolecular interaction accuracy with no regression for intramolecular chemistry. **Use `aimnet2_2025` for all new work.** The original `aimnet2_b973c` is retained only for reproducing previously published results.

## Overview

AIMNet2-B97-3c is the **original B97-3c screening model** trained against B97-3c reference data. B97-3c is a composite density functional designed for speed, combining the B97 GGA functional with a small basis set (def2-mTZVP) and geometric counterpoise corrections. While the reference level is less accurate than wB97M-D3, the resulting ML potential inherits different error characteristics that can be advantageous for rapid ranking and filtering tasks.

**Supported elements:** H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br, I (14 elements)

**Registry alias:** `aimnet2_b973c` (loads ensemble member `aimnet2_b973c_d3_0`)

**Ensemble members:** `aimnet2_b973c_d3_0` through `aimnet2_b973c_d3_3` (4 models)

## Strengths and Limitations

### Strengths

- Same element coverage as the default AIMNet2 model
- Trained on a cost-effective but well-established composite functional
- Useful as a complementary model for cross-checking results from [AIMNet2 (wB97M-D3)](aimnet2.md)
- Good for relative energy rankings in conformer searches and high-throughput workflows

### Limitations

!!! warning "Lower reference accuracy" B97-3c is a GGA-level composite method. It is systematically less accurate than wB97M-D3 for barrier heights, reaction energies involving significant electron correlation changes, and non-covalent interaction energies. For higher accuracy, use [AIMNet2 (wB97M-D3)](aimnet2.md).

!!! warning "Barrier heights" GGA functionals tend to underestimate reaction barriers. If accurate transition state energies are important, prefer the wB97M-D3 model or use [AIMNet2-NSE](aimnet2nse.md) for open-shell transition states.

!!! warning "Same scope restrictions as AIMNet2" No transition metals, closed-shell only, molecular (gas-phase) training data. See the [AIMNet2 page](aimnet2.md) for the full list of limitations.

## Typical Use Cases

- **Reproducing published results** -- use when replicating calculations from papers that used `aimnet2_b973c`
- **Cross-validation** -- compare B97-3c and wB97M-D3 predictions to gauge sensitivity to the reference method

!!! tip "For new projects, use AIMNet2-2025" For high-throughput screening, conformer ranking, and any new B97-3c-level work, switch to [`aimnet2_2025`](aimnet2_2025.md) which provides strictly better accuracy for the same computational cost.

## Quick Example

Ranking conformer energies for a flexible molecule:

```python
import torch
from aimnet.calculators import AIMNet2Calculator

# Use B97-3c model for fast screening
calc = AIMNet2Calculator("aimnet2_b973c", compile_model=True)

# Load multiple conformers (example: 3 conformers of butane, 14 atoms each)
# In practice, read from a multi-frame XYZ file
conformer_coords = [...]  # list of (N, 3) tensors
numbers = torch.tensor([6, 1, 1, 1, 6, 1, 1, 6, 1, 1, 6, 1, 1, 1])  # C4H10

energies = []
for coords in conformer_coords:
    result = calc({
        "coord": coords,
        "numbers": numbers,
        "charge": torch.tensor(0.0),
    })
    energies.append(result["energy"].item())

# Rank by relative energy
e_min = min(energies)
for i, e in enumerate(energies):
    print(f"Conformer {i}: {(e - e_min) * 1000:.1f} meV relative")
```

### ASE Integration

```python
from ase.io import read
from ase.optimize import BFGS
from aimnet.calculators import AIMNet2ASE, AIMNet2Calculator

base_calc = AIMNet2Calculator("aimnet2_b973c", compile_model=True)
atoms = read("molecule.xyz")
atoms.calc = AIMNet2ASE(base_calc, charge=0)

opt = BFGS(atoms)
opt.run(fmax=0.05)  # Looser threshold for screening

print(f"Energy: {atoms.get_potential_energy():.6f} eV")
```

## Computational Details

### Training Data

The model is trained on molecular structures computed at the B97-3c level of theory. B97-3c uses the B97 GGA functional with a modified triple-zeta basis set (def2-mTZVP) and includes geometric counterpoise (gCP), DFT-D3, and short-range basis set incompleteness (SRB) corrections.

### Long-Range Corrections

Like all AIMNet2 models, external DFT-D3 dispersion and long-range Coulomb modules are included. Configuration is identical to the default model:

```python
# For periodic systems
calc.set_lrcoulomb_method("dsf", cutoff=15.0)
```

### Ensemble Averaging

Four ensemble members are available for uncertainty estimation:

```python
models = [AIMNet2Calculator(f"aimnet2_b973c_d3_{i}") for i in range(4)]
results = [m(data) for m in models]

energies = torch.stack([r["energy"] for r in results])
spread = energies.std(dim=0)  # Large spread signals unreliable prediction
```

### When to Upgrade to wB97M-D3

Consider switching to the default [AIMNet2 (wB97M-D3)](aimnet2.md) model when:

- Absolute energy accuracy matters (not just relative rankings)
- Computing reaction barrier heights
- Studying non-covalent interactions where dispersion accuracy is critical
- Publishing results that require a higher reference level

## References

Anstine, D. M.; Zubatyuk, R.; Isayev, O. AIMNet2: A Neural Network Potential to Meet your Neutral, Charged, Organic, and Elemental-Organic Needs. _Chemical Science_ **2025**, _16_, 10228--10244. DOI: [10.1039/D4SC08572H](https://doi.org/10.1039/D4SC08572H)
