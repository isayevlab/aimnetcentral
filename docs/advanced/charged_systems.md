# Charged Systems: Ions, Zwitterions & Electrochemistry

## What You'll Learn

- How to set molecular charge correctly for single molecules and batches
- Working with anions, cations, and zwitterions
- Dipole moment caveats for charged species
- Long-range electrostatics for periodic charged systems
- Understanding charge prediction behavior in molecular dynamics

## Prerequisites

- Familiarity with [AIMNet2Calculator](../calculator.md) and the [ASE interface](../tutorials/geometry_optimization.md)
- Understanding of [long-range methods](../long_range.md) (DSF, Ewald)

## Relevant Models

Charge handling is universal across all AIMNet2 models. The examples in this guide work with any model variant (`aimnet2`, `aimnet2_2025`, `aimnet2nse`, `aimnet2pd`, `aimnet2_b973c`). Choose the model based on your chemistry, not your charge state.

## Setting the Charge

### Direct Calculator

With `AIMNet2Calculator`, the charge is passed in the input dictionary as a scalar or tensor:

```python
import torch
from aimnet.calculators import AIMNet2Calculator

calc = AIMNet2Calculator("aimnet2")

# Neutral molecule
result = calc({
    "coord": coords,
    "numbers": numbers,
    "charge": torch.tensor(0.0),
})

# Anion (e.g., hydroxide OH-)
result = calc({
    "coord": coords_oh,
    "numbers": numbers_oh,
    "charge": torch.tensor(-1.0),
})

# Cation (e.g., ammonium NH4+)
result = calc({
    "coord": coords_nh4,
    "numbers": numbers_nh4,
    "charge": torch.tensor(1.0),
})
```

### Batched Calculations

When computing multiple molecules in a single batch using `mol_idx`, provide the charge as a 1D tensor with one value per molecule:

```python
import torch
from aimnet.calculators import AIMNet2Calculator

calc = AIMNet2Calculator("aimnet2")

# Two molecules: first neutral, second with charge -1
result = calc({
    "coord": combined_coords,    # (N_total, 3)
    "numbers": combined_numbers,  # (N_total,)
    "charge": torch.tensor([0.0, -1.0]),
    "mol_idx": mol_idx,           # (N_total,) mapping atoms to molecules
})
```

### ASE Interface

With `AIMNet2ASE`, set the charge at construction or update it later:

```python
from aimnet.calculators import AIMNet2ASE, AIMNet2Calculator

# Set charge at construction
base_calc = AIMNet2Calculator("aimnet2")
ase_calc = AIMNet2ASE(base_calc, charge=-1)

# Update charge later
ase_calc.set_charge(-1)

# Or set charge for a cation
ase_calc = AIMNet2ASE("aimnet2", charge=1)
```

!!! warning "Always Set the Correct Charge"

    The charge parameter directly affects predicted partial atomic charges and total energy. An incorrect charge will produce incorrect results without raising an error. The model constrains partial charges to sum to the specified total charge.

## Worked Examples

### Hydroxide Anion (OH-)

```python
import torch
from aimnet.calculators import AIMNet2Calculator

calc = AIMNet2Calculator("aimnet2")

# OH- ion
oh_coords = torch.tensor([
    [0.0, 0.0, 0.0],     # O
    [0.0, 0.0, 0.9572],  # H
])
oh_numbers = torch.tensor([8, 1])

result = calc({
    "coord": oh_coords,
    "numbers": oh_numbers,
    "charge": torch.tensor(-1.0),
}, forces=True)

print(f"OH- energy: {result['energy'].item():.4f} eV")
print(f"Partial charges: {result['charges']}")
# Expect: negative charges summing to -1.0
print(f"Charge sum: {result['charges'].sum().item():.4f}")
```

### Protonated Methylamine (CH3NH3+)

```python
import torch
from aimnet.calculators import AIMNet2Calculator

calc = AIMNet2Calculator("aimnet2")

# Methylammonium: CH3-NH3+
coords = torch.tensor([
    [ 0.000,  0.000,  0.000],   # C
    [ 0.000,  0.000,  1.520],   # N
    [ 1.030,  0.000, -0.380],   # H (on C)
    [-0.515,  0.893, -0.380],   # H (on C)
    [-0.515, -0.893, -0.380],   # H (on C)
    [ 0.943,  0.000,  1.920],   # H (on N)
    [-0.471,  0.817,  1.920],   # H (on N)
    [-0.471, -0.817,  1.920],   # H (on N)
])
numbers = torch.tensor([6, 7, 1, 1, 1, 1, 1, 1])

result = calc({
    "coord": coords,
    "numbers": numbers,
    "charge": torch.tensor(1.0),
}, forces=True)

print(f"CH3NH3+ energy: {result['energy'].item():.4f} eV")
print(f"Partial charges: {result['charges']}")
print(f"Charge sum: {result['charges'].sum().item():.4f}")
# Should sum to +1.0
```

### Amino Acid Zwitterion (Glycine)

Amino acids exist as zwitterions at physiological pH, with a protonated amine (NH3+) and deprotonated carboxylate (COO-). The overall charge is zero, but the internal charge separation is large.

```python
import torch
from aimnet.calculators import AIMNet2Calculator

calc = AIMNet2Calculator("aimnet2")

# Glycine zwitterion: +H3N-CH2-COO-
# Overall neutral, but with internal charge separation
coords = torch.tensor([
    [ 0.000,  0.000,  0.000],   # N (NH3+)
    [ 1.475,  0.000,  0.000],   # C (alpha)
    [ 2.011,  1.420,  0.000],   # C (carboxyl)
    [ 1.349,  2.399,  0.337],   # O
    [ 3.191,  1.557, -0.336],   # O
    [-0.380,  1.000,  0.200],   # H (on N)
    [-0.380, -0.500,  0.850],   # H (on N)
    [-0.380, -0.500, -0.850],   # H (on N)
    [ 1.830, -0.500,  0.920],   # H (on C-alpha)
    [ 1.830, -0.500, -0.920],   # H (on C-alpha)
])
numbers = torch.tensor([7, 6, 6, 8, 8, 1, 1, 1, 1, 1])

result = calc({
    "coord": coords,
    "numbers": numbers,
    "charge": torch.tensor(0.0),  # Overall neutral
}, forces=True)

print(f"Glycine zwitterion energy: {result['energy'].item():.4f} eV")
print(f"Partial charges:")
for i, (z, q) in enumerate(zip(numbers, result['charges'].flatten())):
    elem = {1: "H", 6: "C", 7: "N", 8: "O"}[z.item()]
    print(f"  {elem}({i}): {q.item():+.4f}")
# Expect: N has positive charge, O atoms have negative charges
```

## Dipole Moment Caveats

AIMNet2 computes the dipole moment classically from the predicted partial atomic charges:

```
mu = sum_i(q_i * r_i)
```

where `q_i` is the partial charge on atom `i` and `r_i` is its position vector.

!!! warning "Origin Dependence for Charged Species"

    For a **neutral** molecule (total charge = 0), the dipole moment is independent of the coordinate origin. This is the standard physical definition.

    For a **charged** species (total charge != 0), the dipole moment depends on the choice of coordinate origin. Shifting all coordinates by a constant vector `d` changes the dipole by `Q_total * d`, where `Q_total` is the total charge. This is a fundamental property of multipole expansions, not a limitation of AIMNet2.

    **Practical consequence:** Dipole moments are only physically meaningful for neutral molecules. For charged species, compare dipoles only if all geometries use the same coordinate convention (e.g., center of mass at the origin).

```python
from ase import Atoms
from aimnet.calculators import AIMNet2ASE, AIMNet2Calculator
import numpy as np

# Example: dipole of neutral water (origin-independent)
base_calc = AIMNet2Calculator("aimnet2")
water = Atoms("OH2", positions=[
    [0.0, 0.0, 0.1173],
    [0.0, 0.7572, -0.4692],
    [0.0, -0.7572, -0.4692],
])
water.calc = AIMNet2ASE(base_calc, charge=0)

dipole = water.calc.get_dipole_moment(water)
print(f"Water dipole: {np.linalg.norm(dipole):.3f} e*A")
# Approximately 0.38 e*A = 1.85 D
```

## Charge Behavior in Molecular Dynamics

!!! warning "Atomic Charges Are Not Conserved Across MD Frames"

    In AIMNet2, atomic partial charges are **predicted independently for each frame** during molecular dynamics. The model predicts charges that sum to the specified total charge, but the distribution among atoms can fluctuate from frame to frame.

    This means:

    - Charge on any individual atom may change between MD steps
    - Only the total charge is constrained to be constant
    - There is no continuity constraint on per-atom charges between frames
    - This is physically reasonable (charge redistribution happens in real
      molecules) but differs from fixed-charge force fields

    For applications requiring smooth charge evolution (e.g., computing current from charge fluxes), consider post-processing to smooth the charge trajectory.

## Ewald for Periodic Charged Systems

### Single Molecule in a Periodic Box

For a single charged molecule in a periodic box, Ewald summation handles the long-range electrostatics exactly. However, note that the current Ewald implementation uses a full N x N interaction matrix, not a neighbor-list approach:

!!! warning "Ewald Scaling"

    The Ewald implementation builds a full N x N Coulomb matrix, making it suitable for **single molecules** in periodic boxes but expensive for large systems. For large periodic systems, prefer DSF.

```python
import torch
from aimnet.calculators import AIMNet2Calculator

calc = AIMNet2Calculator("aimnet2")
calc.set_lrcoulomb_method("ewald", ewald_accuracy=1e-8)

# Charged molecule in periodic box
cell = torch.tensor([
    [15.0, 0.0, 0.0],
    [0.0, 15.0, 0.0],
    [0.0, 0.0, 15.0],
])

result = calc({
    "coord": coords_ion,
    "numbers": numbers_ion,
    "charge": torch.tensor(-1.0),
    "cell": cell,
}, forces=True)
```

### Automatic Method Switching for PBC

When periodic boundary conditions are detected (a `cell` is provided), the calculator automatically switches from the "simple" Coulomb method to DSF:

```python
import torch
from aimnet.calculators import AIMNet2Calculator

calc = AIMNet2Calculator("aimnet2")
# Default method is "simple"

# Providing a cell triggers automatic switch to DSF
cell = torch.tensor([
    [10.0, 0.0, 0.0],
    [0.0, 10.0, 0.0],
    [0.0, 0.0, 10.0],
])

result = calc({
    "coord": coords,
    "numbers": numbers,
    "charge": torch.tensor(-1.0),
    "cell": cell,
}, forces=True)
# A warning is issued: "Switching to DSF Coulomb for PBC"
```

To avoid the warning, set the method explicitly before the first periodic calculation:

```python
calc.set_lrcoulomb_method("dsf", cutoff=15.0)
```

## Protonation and Deprotonation Energetics

A common application is computing protonation/deprotonation energies. Since AIMNet2 is gas-phase only, these are **gas-phase proton affinities**:

```python
import torch
from aimnet.calculators import AIMNet2Calculator

calc = AIMNet2Calculator("aimnet2")
ev_to_kcal = 23.0609

# Example: gas-phase proton affinity of acetate
# CH3COO- + H+ -> CH3COOH

# Acetate anion (CH3COO-)
acetate_coords = torch.tensor([
    [ 0.000,  0.000,  0.000],   # C (methyl)
    [ 1.520,  0.000,  0.000],   # C (carboxyl)
    [ 2.100,  1.090,  0.000],   # O
    [ 2.100, -1.090,  0.000],   # O
    [-0.400,  1.000,  0.200],   # H
    [-0.400, -0.500,  0.900],   # H
    [-0.400, -0.500, -0.900],   # H
])
acetate_numbers = torch.tensor([6, 6, 8, 8, 1, 1, 1])

# Acetic acid (CH3COOH) - add proton to one oxygen
acetic_coords = torch.tensor([
    [ 0.000,  0.000,  0.000],   # C (methyl)
    [ 1.520,  0.000,  0.000],   # C (carboxyl)
    [ 2.100,  1.090,  0.000],   # O (C=O)
    [ 2.100, -1.090,  0.000],   # O (C-OH)
    [-0.400,  1.000,  0.200],   # H
    [-0.400, -0.500,  0.900],   # H
    [-0.400, -0.500, -0.900],   # H
    [ 3.060, -1.090,  0.000],   # H (on O-H)
])
acetic_numbers = torch.tensor([6, 6, 8, 8, 1, 1, 1, 1])

e_acetate = calc({
    "coord": acetate_coords,
    "numbers": acetate_numbers,
    "charge": torch.tensor(-1.0),
})["energy"].item()

e_acetic = calc({
    "coord": acetic_coords,
    "numbers": acetic_numbers,
    "charge": torch.tensor(0.0),
})["energy"].item()

# Gas-phase proton affinity (no proton energy needed for relative comparison)
# For absolute PA, the proton energy is 0 by convention in gas phase
delta_e = (e_acetic - e_acetate) * ev_to_kcal
print(f"Deprotonation energy (gas phase): {-delta_e:.1f} kcal/mol")
```

!!! note "Gas-Phase vs Solution"

    Gas-phase proton affinities differ substantially from solution-phase pKa values. Solvation stabilization of charged species is typically 50-80 kcal/mol for ions in water. Do not directly compare AIMNet2 gas-phase energetics with experimental solution-phase data without solvation corrections.

## What's Next

- [Non-Covalent Interactions](intermolecular_interactions.md) -- binding energies of charged and neutral complexes
- [Long-Range Methods](../long_range.md) -- detailed DSF and Ewald configuration
- [Model Selection Guide](../models/guide.md) -- choosing the right model for your chemistry
