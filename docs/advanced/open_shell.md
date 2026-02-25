# Radical and Open-Shell Chemistry

## What You Will Learn

- Why closed-shell models produce incorrect results for radical species
- How the NSE (Nuclear Spin-resolved Electrons) architecture handles unpaired electrons
- Setting the `mult` parameter for doublets, triplets, and higher spin states
- Computing C-H bond dissociation energies
- Comparing radical stability across isomers
- Recognizing when AIMNet2-NSE is insufficient and DFT is needed

## Prerequisites

- Familiarity with the [AIMNet2Calculator API](../calculator.md) and [ASE integration](../tutorials/geometry_optimization.md)
- Understanding of spin multiplicity (mult = 2S+1, where S is total spin)
- Installation: `pip install "aimnet[ase]"`

## Why Closed-Shell Models Fail for Radicals

Standard AIMNet2 models (`aimnet2`, `aimnet2_b973c`, `aimnet2_2025`) use a single charge channel internally (`num_charge_channels=1`). This means the model predicts one set of atomic partial charges and implicitly assumes all electrons are paired. When applied to a radical -- a species with one or more unpaired electrons -- the model has no mechanism to represent the spin density distribution and produces unreliable energies and forces.

!!! warning "Do not use closed-shell models for open-shell systems" Applying `aimnet2` to a doublet radical or triplet state will silently produce results that look reasonable but have large systematic errors. Always use `aimnet2nse` when unpaired electrons are present.

Common symptoms of using the wrong model:

- Bond dissociation energies that are off by 10-30 kcal/mol
- Incorrect radical stability ordering
- Geometry optimizations that converge to unphysical structures
- Vibrational frequencies with unexpected imaginary modes

## How NSE Works

AIMNet2-NSE extends the standard architecture with **two charge channels** (`num_charge_channels=2`). These represent alpha and beta electron populations separately:

1. **Preprocessing**: The molecular charge and multiplicity are decomposed into alpha and beta channel inputs:

   - Alpha channel charge: `(charge/2) + (mult-1)/2`
   - Beta channel charge: `(charge/2) - (mult-1)/2`

2. **Inference**: The neural network processes both channels through the same architecture, predicting separate alpha and beta atomic charges at each iteration.

3. **Postprocessing**: The two channels are recombined:
   - Total charges: `alpha_charges + beta_charges`
   - Spin charges: `alpha_charges - beta_charges` (spin density per atom)

This design allows the model to capture the spatial distribution of unpaired electrons without requiring an explicit wavefunction.

## Setting the Multiplicity

The `mult` parameter specifies the spin multiplicity (2S+1):

| System                      | Unpaired electrons | S   | mult        |
| --------------------------- | ------------------ | --- | ----------- |
| Closed-shell molecule       | 0                  | 0   | 1 (singlet) |
| Organic radical (e.g., CH3) | 1                  | 1/2 | 2 (doublet) |
| Triplet carbene             | 2                  | 1   | 3 (triplet) |
| Quartet nitrogen atom       | 3                  | 3/2 | 4 (quartet) |

### With AIMNet2Calculator (direct)

```python
import torch
from aimnet.calculators import AIMNet2Calculator

calc = AIMNet2Calculator("aimnet2nse")

result = calc({
    "coord": coords,
    "numbers": numbers,
    "charge": torch.tensor(0.0),
    "mult": torch.tensor(2.0),   # Doublet radical
}, forces=True)
```

### With AIMNet2ASE (ASE integration)

```python
from aimnet.calculators import AIMNet2ASE

# Doublet radical
calc = AIMNet2ASE("aimnet2nse", charge=0, mult=2)

# Change multiplicity later
calc.set_mult(3)  # Switch to triplet
```

!!! note "Default multiplicity" If `mult` is not provided, the calculator defaults to `mult=1` (singlet). For closed-shell molecules with `aimnet2nse`, this is correct and the model reduces to standard behavior. You do not need to switch models for closed-shell species.

## Worked Example: C-H Bond Dissociation Energy

Bond dissociation energy (BDE) measures the energy required to break a bond homolytically, producing two radical fragments. This is a fundamental test for open-shell models because it requires accurate energies for both the parent molecule and the radical products.

For toluene (C6H5-CH3), breaking a benzylic C-H bond:

C6H5-CH3 -> C6H5-CH2 (radical) + H (radical)

BDE = E(radical) + E(H) - E(toluene)

```python
import torch
from ase import Atoms
from ase.optimize import BFGS
from aimnet.calculators import AIMNet2ASE, AIMNet2Calculator

base_calc = AIMNet2Calculator("aimnet2nse", compile_model=True)

# --- Step 1: Optimize toluene (closed-shell singlet) ---
toluene = Atoms(
    symbols="C7H8",
    positions=[
        # Ring carbons
        [ 0.000,  1.402,  0.000],  # C1
        [ 1.214,  0.701,  0.000],  # C2
        [ 1.214, -0.701,  0.000],  # C3
        [ 0.000, -1.402,  0.000],  # C4
        [-1.214, -0.701,  0.000],  # C5
        [-1.214,  0.701,  0.000],  # C6
        [ 0.000,  2.902,  0.000],  # C-methyl
        # H atoms (approximate positions)
        [ 2.156,  1.244,  0.000],
        [ 2.156, -1.244,  0.000],
        [ 0.000, -2.490,  0.000],
        [-2.156, -1.244,  0.000],
        [-2.156,  1.244,  0.000],
        [ 0.000,  3.302,  1.030],  # Methyl H
        [ 0.891,  3.302, -0.515],  # Methyl H
        [-0.891,  3.302, -0.515],  # Methyl H
    ],
)

toluene.calc = AIMNet2ASE(base_calc, charge=0, mult=1)
opt = BFGS(toluene, logfile=None)
opt.run(fmax=0.01)
E_toluene = toluene.get_potential_energy()

# --- Step 2: Optimize benzyl radical (doublet) ---
# Remove one methyl hydrogen
benzyl = Atoms(
    symbols="C7H7",
    positions=toluene.positions[:-1],  # Remove last H
)

benzyl.calc = AIMNet2ASE(base_calc, charge=0, mult=2)
opt = BFGS(benzyl, logfile=None)
opt.run(fmax=0.01)
E_benzyl = benzyl.get_potential_energy()

# --- Step 3: Hydrogen atom energy (doublet) ---
H_atom = Atoms("H", positions=[[0.0, 0.0, 0.0]])
H_atom.calc = AIMNet2ASE(base_calc, charge=0, mult=2)
E_H = H_atom.get_potential_energy()

# --- Step 4: Compute BDE ---
BDE_eV = E_benzyl + E_H - E_toluene
BDE_kcal = BDE_eV * 23.0609  # eV to kcal/mol

print(f"Toluene energy:       {E_toluene:.4f} eV")
print(f"Benzyl radical energy: {E_benzyl:.4f} eV")
print(f"H atom energy:        {E_H:.4f} eV")
print(f"Benzylic C-H BDE:     {BDE_kcal:.1f} kcal/mol")
# Expected: ~90 kcal/mol (experimental: 89.8 kcal/mol)
```

!!! warning "Zero-point energy corrections" The BDE values computed above are purely electronic (Delta E_e). For thermochemically accurate results, include zero-point energy (ZPE) corrections by computing the Hessian at each optimized geometry and extracting harmonic frequencies. The ZPE-corrected BDE is:

    BDE_0 = E(radical1) + E(radical2) - E(molecule) + ZPE(radical1) + ZPE(radical2) - ZPE(molecule)

    See the [Geometry Optimization tutorial](../tutorials/geometry_optimization.md) for thermochemistry workflows using `ase.thermochemistry.IdealGasThermo`.

## Radical Stability Ordering

A key validation for any open-shell method is reproducing the correct stability ordering of radicals. More stable radicals have lower BDEs. For carbon radicals, the expected order is:

methyl < primary < secondary < tertiary < allyl < benzyl

```python
import torch
from ase import Atoms
from ase.optimize import BFGS
from aimnet.calculators import AIMNet2ASE, AIMNet2Calculator

base_calc = AIMNet2Calculator("aimnet2nse", compile_model=True)


def compute_bde(parent_atoms, radical_atoms, h_energy):
    """Compute BDE given parent molecule and radical fragment."""
    # Optimize parent (singlet)
    parent_atoms.calc = AIMNet2ASE(base_calc, charge=0, mult=1)
    opt = BFGS(parent_atoms, logfile=None)
    opt.run(fmax=0.01)
    e_parent = parent_atoms.get_potential_energy()

    # Optimize radical (doublet)
    radical_atoms.calc = AIMNet2ASE(base_calc, charge=0, mult=2)
    opt = BFGS(radical_atoms, logfile=None)
    opt.run(fmax=0.01)
    e_radical = radical_atoms.get_potential_energy()

    bde_kcal = (e_radical + h_energy - e_parent) * 23.0609
    return bde_kcal


# Get H atom energy once
H_atom = Atoms("H", positions=[[0.0, 0.0, 0.0]])
H_atom.calc = AIMNet2ASE(base_calc, charge=0, mult=2)
E_H = H_atom.get_potential_energy()

# Define parent/radical pairs and compute BDEs
# (Use your own geometry definitions for each species)
# Results should show: methyl > ethyl > isopropyl > tert-butyl
```

## Side-by-Side: aimnet2 vs aimnet2nse

This comparison demonstrates the systematic error of using a closed-shell model for radical species.

```python
import torch
from ase import Atoms
from ase.optimize import BFGS
from aimnet.calculators import AIMNet2ASE, AIMNet2Calculator

# Methyl radical (CH3, doublet)
ch3 = Atoms(
    symbols="CH3",
    positions=[
        [0.000,  0.000,  0.000],
        [1.079,  0.000,  0.000],
        [-0.540,  0.935,  0.000],
        [-0.540, -0.935,  0.000],
    ],
)

# --- Closed-shell model (INCORRECT for radicals) ---
calc_closed = AIMNet2Calculator("aimnet2", compile_model=True)
ch3_closed = ch3.copy()
ch3_closed.calc = AIMNet2ASE(calc_closed, charge=0, mult=1)
# Note: mult is ignored by aimnet2 (it only has 1 charge channel)
opt = BFGS(ch3_closed, logfile=None)
opt.run(fmax=0.01)
E_closed = ch3_closed.get_potential_energy()
print(f"aimnet2 (closed-shell): {E_closed:.4f} eV")

# --- Open-shell model (CORRECT) ---
calc_nse = AIMNet2Calculator("aimnet2nse", compile_model=True)
ch3_nse = ch3.copy()
ch3_nse.calc = AIMNet2ASE(calc_nse, charge=0, mult=2)
opt = BFGS(ch3_nse, logfile=None)
opt.run(fmax=0.01)
E_nse = ch3_nse.get_potential_energy()
print(f"aimnet2nse (open-shell): {E_nse:.4f} eV")

print(f"Energy difference: {abs(E_closed - E_nse) * 23.0609:.1f} kcal/mol")
```

!!! tip "When in doubt, use aimnet2nse" The NSE model handles closed-shell species correctly (mult=1 reduces to standard behavior). If your workflow involves a mix of closed-shell and open-shell species -- for example, computing BDEs -- use `aimnet2nse` throughout for consistent energetics.

## When to Fall Back to Multi-Reference Methods

AIMNet2-NSE handles most radical chemistry well, but certain electronic structure situations are beyond the reach of any single-reference method (including both AIMNet2 and standard DFT):

**Multi-reference cases (use multi-configurational methods such as CASSCF, CASPT2, MRCI, or NEVPT2):**

- **Diradicals with singlet ground states** -- e.g., ortho-benzyne, trimethylenemethane. These require describing two singly occupied orbitals that are antiferromagnetically coupled.
- **Stretched symmetric bonds** -- e.g., H2 at 3x equilibrium distance. The wavefunction requires equal contributions from bonding and antibonding configurations.
- **Conical intersections** -- Points where potential energy surfaces cross, important in photochemistry.
- **Near-degenerate spin states** -- When singlet-triplet gaps are below ~2 kcal/mol, the accuracy of any single-reference method becomes questionable.

**Practical indicators that multi-reference methods may be needed:**

- Optimization oscillates without converging
- Spin density is delocalized in an unphysical way
- Energy differences between spin states are unexpectedly small
- The system has low-lying excited states (check literature)

!!! info "Recommendation" For routine radical chemistry -- organic radical stability, BDEs, H-atom abstractions, radical additions -- AIMNet2-NSE is reliable and orders of magnitude faster than DFT. Reserve multi-reference methods (CASSCF, CASPT2, MRCI, NEVPT2) for the specific pathological cases listed above. Note that standard DFT is also a single-reference method and will fail for the same multi-reference cases.

## What's Next

- [AIMNet2-NSE model details](../models/aimnet2nse.md) -- training data, accuracy benchmarks, element coverage
- [Reaction Paths and Transition States](reaction_paths.md) -- using NSE for bond-breaking transition states with PySisyphus
- [Model Selection Guide](../models/guide.md) -- choosing the right model for your chemistry
- [Geometry Optimization tutorial](../tutorials/geometry_optimization.md) -- structure relaxation fundamentals
