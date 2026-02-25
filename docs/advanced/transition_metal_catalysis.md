# Pd-Catalyzed Reactions

## What You'll Learn

- How to use the `aimnet2pd` model for palladium organometallic chemistry
- Computing relative stabilities of Pd coordination geometries
- Modeling a step of a Suzuki cross-coupling catalytic cycle
- Understanding the limitations of transition metal ML potentials

## Prerequisites

- Familiarity with [AIMNet2Calculator](../calculator.md) and the [ASE interface](../tutorials/geometry_optimization.md)
- Basic understanding of organometallic coordination chemistry
- ASE installation for geometry optimization

## Spotlight Model: AIMNet2-Pd

The `aimnet2pd` model extends AIMNet2 to palladium-containing organometallic systems. It is trained on wB97M-D3/CPCM reference data with **THF implicit solvation**, so predicted energetics include continuum solvent effects relevant to homogeneous catalysis. This enables rapid exploration of catalytic reaction profiles at near-DFT accuracy with solvent stabilization built in.

**Supported elements:** H, B, C, N, O, F, Si, P, S, Cl, Se, Br, Pd, I (14 elements)

!!! warning "No Arsenic"

    The Pd model replaces As with Pd in the element set. You **cannot** use AsPh3 ligands or any arsenic-containing species with `aimnet2pd`. Common phosphine ligands (PPh3, PMe3, etc.) are fully supported.

!!! warning "Palladium Only"

    `aimnet2pd` supports Pd as the only transition metal. It does **not** cover Ni, Cu, Fe, Ru, Rh, Ir, or any other transition metals. For reactions catalyzed by other metals, DFT remains necessary.

## Loading the Pd Model

```python
from aimnet.calculators import AIMNet2Calculator, AIMNet2ASE

# Direct calculator
calc = AIMNet2Calculator("aimnet2pd")

# ASE interface
from ase import Atoms
base_calc = AIMNet2Calculator("aimnet2pd", compile_model=True)
ase_calc = AIMNet2ASE(base_calc, charge=0)
```

## Pd Coordination Geometries

Pd(II) complexes typically adopt square planar geometry, while Pd(0) prefers linear or trigonal coordination. Comparing the relative stability of different coordination arrangements is a useful validation test.

```python
import torch
from aimnet.calculators import AIMNet2Calculator

calc = AIMNet2Calculator("aimnet2pd")

# PdCl4(2-): ideal square planar
# Pd at origin, 4 Cl in xy-plane
d_pd_cl = 2.30  # approximate Pd-Cl bond length in Angstrom
sq_planar = torch.tensor([
    [0.0, 0.0, 0.0],           # Pd
    [d_pd_cl, 0.0, 0.0],       # Cl
    [0.0, d_pd_cl, 0.0],       # Cl
    [-d_pd_cl, 0.0, 0.0],      # Cl
    [0.0, -d_pd_cl, 0.0],      # Cl
])
numbers_pdcl4 = torch.tensor([46, 17, 17, 17, 17])

result_sq = calc({
    "coord": sq_planar,
    "numbers": numbers_pdcl4,
    "charge": torch.tensor(-2.0),
}, forces=True)

print(f"Square planar PdCl4(2-) energy: {result_sq['energy'].item():.4f} eV")
print(f"Max force: {result_sq['forces'].abs().max().item():.4f} eV/A")
```

!!! tip "Optimize Before Comparing"

    Starting geometries from textbook bond lengths and angles are approximate. Always optimize structures before comparing energies:

    ```python
    from ase import Atoms
    from ase.optimize import BFGS
    from aimnet.calculators import AIMNet2ASE, AIMNet2Calculator

    base_calc = AIMNet2Calculator("aimnet2pd", compile_model=True)
    atoms = Atoms("PdCl4", positions=sq_planar.numpy())
    atoms.calc = AIMNet2ASE(base_calc, charge=-2)

    opt = BFGS(atoms)
    opt.run(fmax=0.01)
    print(f"Optimized energy: {atoms.get_potential_energy():.4f} eV")
    ```

## Suzuki Coupling: Oxidative Addition

The Suzuki cross-coupling reaction is one of the most important Pd-catalyzed reactions in synthetic chemistry. The catalytic cycle involves:

1. **Oxidative addition** of aryl halide to Pd(0)
2. Transmetalation with organoboron
3. Reductive elimination to form C-C bond

Here we model the oxidative addition step: the reaction of bromobenzene with a Pd(0)-phosphine complex.

### Step 1: Build and Optimize the Pd(0) Complex

A simplified Pd(0) complex with PH3 ligands (PH3 as a model for PPh3):

```python
import torch
import numpy as np
from ase import Atoms
from ase.optimize import BFGS
from aimnet.calculators import AIMNet2ASE, AIMNet2Calculator

base_calc = AIMNet2Calculator("aimnet2pd", compile_model=True)

# Pd(PH3)2: linear Pd(0) complex
# Pd at origin, two PH3 along x-axis
pd_ph3_2 = Atoms(
    symbols="PdPH3PH3",
    positions=[
        [0.0, 0.0, 0.0],       # Pd
        [-2.30, 0.0, 0.0],     # P (left)
        [-2.90, 1.20, 0.50],   # H
        [-2.90, -0.60, 1.10],  # H
        [-2.90, -0.60, -1.10], # H
        [2.30, 0.0, 0.0],      # P (right)
        [2.90, 1.20, 0.50],    # H
        [2.90, -0.60, 1.10],   # H
        [2.90, -0.60, -1.10],  # H
    ],
)

pd_ph3_2.calc = AIMNet2ASE(base_calc, charge=0)
opt = BFGS(pd_ph3_2, logfile=None)
opt.run(fmax=0.01)

e_pd0 = pd_ph3_2.get_potential_energy()
print(f"Pd(PH3)2 optimized energy: {e_pd0:.4f} eV")
```

### Step 2: Build and Optimize Bromobenzene

```python
# Bromobenzene (PhBr)
phbr = Atoms(
    symbols="C6H5Br",
    positions=[
        [ 0.000,  1.398, 0.0],   # C (ipso, bonded to Br)
        [ 1.211,  0.699, 0.0],   # C
        [ 1.211, -0.699, 0.0],   # C
        [ 0.000, -1.398, 0.0],   # C
        [-1.211, -0.699, 0.0],   # C
        [-1.211,  0.699, 0.0],   # C
        [ 2.155,  1.244, 0.0],   # H
        [ 2.155, -1.244, 0.0],   # H
        [ 0.000, -2.488, 0.0],   # H
        [-2.155, -1.244, 0.0],   # H
        [-2.155,  1.244, 0.0],   # H
        [ 0.000,  3.283, 0.0],   # Br
    ],
)

phbr.calc = AIMNet2ASE(base_calc, charge=0)
opt = BFGS(phbr, logfile=None)
opt.run(fmax=0.01)

e_phbr = phbr.get_potential_energy()
print(f"PhBr optimized energy: {e_phbr:.4f} eV")
```

### Step 3: Build and Optimize the Oxidative Addition Product

The product is a Pd(II) complex: trans-Pd(Ph)(Br)(PH3)2.

```python
# Oxidative addition product: Pd(Ph)(Br)(PH3)2
# Square planar Pd(II) with Ph and Br trans to each other
product = Atoms(
    symbols="PdBrPH3PH3C6H5",
    positions=[
        # Pd center
        [0.0, 0.0, 0.0],
        # Br (trans to Ph, along -y)
        [0.0, -2.50, 0.0],
        # PH3 (along +x)
        [2.30, 0.0, 0.0],
        [2.90, 1.20, 0.50],
        [2.90, -0.60, 1.10],
        [2.90, -0.60, -1.10],
        # PH3 (along -x)
        [-2.30, 0.0, 0.0],
        [-2.90, 1.20, 0.50],
        [-2.90, -0.60, 1.10],
        [-2.90, -0.60, -1.10],
        # Phenyl ring (along +y, bonded to Pd via ipso C)
        [0.0, 2.05, 0.0],        # C (ipso)
        [1.21, 2.75, 0.0],       # C
        [1.21, 4.15, 0.0],       # C
        [0.0, 4.85, 0.0],        # C
        [-1.21, 4.15, 0.0],      # C
        [-1.21, 2.75, 0.0],      # C
        [2.16, 2.20, 0.0],       # H
        [2.16, 4.70, 0.0],       # H
        [0.0, 5.94, 0.0],        # H
        [-2.16, 4.70, 0.0],      # H
        [-2.16, 2.20, 0.0],      # H
    ],
)

product.calc = AIMNet2ASE(base_calc, charge=0)
opt = BFGS(product, logfile=None)
opt.run(fmax=0.01)

e_product = product.get_potential_energy()
print(f"Pd(Ph)(Br)(PH3)2 optimized energy: {e_product:.4f} eV")
```

### Step 4: Compute Reaction Energetics

```python
# Oxidative addition reaction energy:
# Pd(PH3)2 + PhBr -> Pd(Ph)(Br)(PH3)2
e_rxn = e_product - e_pd0 - e_phbr
e_rxn_kcal = e_rxn * 23.0609  # eV to kcal/mol

print(f"Oxidative addition energy: {e_rxn_kcal:.1f} kcal/mol")
# Typically exothermic: approximately -20 to -30 kcal/mol
# depending on ligands and level of theory
```

## Important Limitations

### Implicit Solvation (THF)

!!! tip "CPCM solvation is built in"

    Unlike other AIMNet2 models, `aimnet2pd` is trained on wB97M-D3/CPCM reference data with **THF as the implicit solvent**. All predicted energetics include continuum solvent stabilization effects appropriate for homogeneous catalysis in THF or similar non-polar aprotic solvents.

    For reactions in very different solvent environments (e.g., water, DMSO, DMF), the THF solvation model may not capture the correct solvent effects. In those cases, additional solvation corrections or explicit solvent modeling may be needed.

### Limited Oxidation States

!!! note "Training Data Coverage"

    The training data covers common Pd oxidation states (Pd(0) and Pd(II)) in typical organometallic coordination environments. Less common oxidation states (Pd(I), Pd(IV)) or unusual coordination geometries may fall outside the training domain.

    Use ensemble uncertainty to identify potentially unreliable predictions:

    ```python
    from aimnet.calculators import AIMNet2Calculator

    calcs = [AIMNet2Calculator(f"aimnet2-pd_{i}") for i in range(4)]
    results = [c(data, forces=True) for c in calcs]

    import torch
    energies = torch.stack([r["energy"] for r in results])
    std = energies.std(dim=0)
    print(f"Ensemble std: {std.item():.4f} eV")
    # High std (> 0.1 eV) suggests out-of-distribution prediction
    ```

### Ligand Considerations

- **Phosphines** (PR3): Well represented in training data. PH3, PMe3, PPh3-like ligands are expected to be reliable.
- **N-heterocyclic carbenes** (NHC): May be less well represented. Validate with ensemble uncertainty.
- **Halides** (F, Cl, Br, I): Supported as ligands and substrates.
- **Boron compounds**: Supported (B is in the element set), relevant for Suzuki coupling transmetalation steps.

!!! warning "No AsPh3 or Arsenic Ligands"

    Arsenic is **not** in the `aimnet2pd` element set. Triphenylarsine (AsPh3) and other As-containing ligands cannot be used. Use the standard `aimnet2` model for As-containing systems (but without Pd).

## Tips for Catalytic Cycle Modeling

1. **Start simple.** Use small model ligands (PH3 instead of PPh3) to test the reaction profile before scaling up to realistic ligands.

2. **Optimize each species.** Always perform geometry optimization (BFGS to fmax < 0.01 eV/A) before comparing energies.

3. **Check spin states.** Pd(0) is typically d10 singlet, Pd(II) is d8 singlet (square planar). Set `charge` appropriately for charged intermediates.

4. **Validate with DFT.** For key stationary points (reactants, products, transition states), compare AIMNet2-Pd energies with DFT to gauge accuracy.

5. **Use ensemble uncertainty.** Large ensemble variance flags potentially unreliable structures.

## What's Next

- [Model Selection Guide](../models/guide.md) -- overview of all AIMNet2 model variants
- [Charged Systems](charged_systems.md) -- handling charged catalytic intermediates
- [Long-Range Methods](../long_range.md) -- electrostatics for charged metal complexes
