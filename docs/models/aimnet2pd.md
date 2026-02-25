# AIMNet2-Pd

## Overview

AIMNet2-Pd extends AIMNet2 to **palladium-catalyzed organometallic chemistry**. It is the first AIMNet2 variant to include a transition metal, enabling fast and accurate simulations of Pd-containing complexes, catalytic intermediates, and reaction profiles at near-DFT accuracy.

**Supported elements:** H, B, C, N, O, F, Si, P, S, Cl, Se, Br, Pd, I (14 elements)

**Registry alias:** `aimnet2pd` (loads ensemble member `aimnet2-pd_0`)

**Ensemble members:** `aimnet2-pd_0` through `aimnet2-pd_3` (4 models)

**DFT reference:** wB97M-D3 with CPCM implicit solvation (THF solvent)

!!! warning "Element coverage difference" AIMNet2-Pd replaces **As** (arsenic) with **Pd** (palladium) compared to the standard AIMNet2 element set. Arsenic is **not supported** by this model.

## Strengths and Limitations

### Strengths

- First neural network potential for Pd organometallic chemistry in the AIMNet2 family
- Handles common Pd oxidation states (0 and +2) and coordination geometries
- Trained on diverse Pd-ligand environments including phosphines, amines, halides, and carbenes
- Suitable for studying catalytic cycles, ligand exchange, and oxidative addition/reductive elimination
- Same high-quality wB97M-D3 reference level as the default AIMNet2
- **Implicit CPCM solvation for THF** baked into the model -- energetics include continuum solvent effects relevant to homogeneous catalysis without additional setup

### Limitations

!!! warning "Palladium only" This model supports only palladium among transition metals. It cannot be used for other catalytic metals such as Ni, Cu, Fe, Ru, Rh, or Ir. For systems without transition metals, use the standard [AIMNet2](aimnet2.md) model instead.

!!! warning "No arsenic" Arsenic (As) is excluded from the element set. If your Pd complex contains arsine ligands (AsR3), this model cannot be used.

!!! tip "Implicit solvation included" Unlike other AIMNet2 models, AIMNet2-Pd is trained on wB97M-D3/CPCM reference data with **THF as the implicit solvent**. This means solvent stabilization effects relevant to homogeneous catalysis in THF or similar non-polar aprotic solvents are captured directly. For reactions in very different solvent environments (e.g., water, DMSO), additional solvation corrections may still be needed.

!!! warning "Closed-shell only" AIMNet2-Pd assumes closed-shell electronic structure. Open-shell Pd intermediates (e.g., Pd(I) species, radical mechanisms) require DFT treatment. For organic radical chemistry without Pd, use [AIMNet2-NSE](aimnet2nse.md).

!!! warning "Coordination environment" The model has been trained on common coordination environments. Unusual geometries (e.g., Pd clusters, Pd nanoparticles, bulk Pd metal) are outside the training domain and should not be trusted.

## Typical Use Cases

- **Suzuki-Miyaura cross-coupling** -- reaction profiles for the full catalytic cycle including oxidative addition, transmetalation, and reductive elimination
- **Oxidative addition** -- barriers and thermodynamics for Pd(0) insertion into C-X bonds (X = Cl, Br, I)
- **Pd coordination chemistry** -- ligand binding energies, conformational preferences of Pd complexes
- **Catalyst screening** -- compare different phosphine or NHC ligands on Pd centers
- **Solution-phase energetics** -- CPCM/THF solvation is baked in, making energetics directly comparable to solution-phase experimental data in THF or similar solvents

## Quick Example

Single-point calculation on a Pd complex (simplified Pd(PH3)2Cl2):

```python
import torch
from aimnet.calculators import AIMNet2Calculator

calc = AIMNet2Calculator("aimnet2pd", compile_model=True)

# trans-Pd(PH3)2Cl2 (simplified model complex)
coords = torch.tensor([
    [ 0.0000,  0.0000,  0.0000],  # Pd
    [ 2.3500,  0.0000,  0.0000],  # Cl
    [-2.3500,  0.0000,  0.0000],  # Cl
    [ 0.0000,  2.3200,  0.0000],  # P
    [ 0.0000, -2.3200,  0.0000],  # P
    [ 0.0000,  2.9200,  1.2100],  # H (on P)
    [ 1.0500,  2.9200, -0.6050],  # H (on P)
    [-1.0500,  2.9200, -0.6050],  # H (on P)
    [ 0.0000, -2.9200, -1.2100],  # H (on P)
    [ 1.0500, -2.9200,  0.6050],  # H (on P)
    [-1.0500, -2.9200,  0.6050],  # H (on P)
])

# Atomic numbers: Pd=46, Cl=17, P=15, H=1
numbers = torch.tensor([46, 17, 17, 15, 15, 1, 1, 1, 1, 1, 1])

result = calc({
    "coord": coords,
    "numbers": numbers,
    "charge": torch.tensor(0.0),
}, forces=True)

print(f"Energy: {result['energy'].item():.6f} eV")
print(f"Max force: {result['forces'].abs().max().item():.6f} eV/A")
```

### ASE Integration

```python
from ase.io import read
from ase.optimize import BFGS
from aimnet.calculators import AIMNet2ASE, AIMNet2Calculator

base_calc = AIMNet2Calculator("aimnet2pd", compile_model=True)
atoms = read("pd_complex.xyz")
atoms.calc = AIMNet2ASE(base_calc, charge=0)

opt = BFGS(atoms)
opt.run(fmax=0.01)

print(f"Optimized energy: {atoms.get_potential_energy():.6f} eV")
```

### Comparing Ligand Binding Energies

```python
import torch
from aimnet.calculators import AIMNet2Calculator

calc = AIMNet2Calculator("aimnet2pd", compile_model=True)

def get_energy(coords, numbers, charge=0.0):
    return calc({
        "coord": coords,
        "numbers": numbers,
        "charge": torch.tensor(charge),
    })["energy"].item()

# Compute: Pd-L complex energy - Pd fragment energy - free ligand energy
e_complex = get_energy(complex_coords, complex_numbers)
e_pd_frag = get_energy(pd_frag_coords, pd_frag_numbers)
e_ligand = get_energy(ligand_coords, ligand_numbers)

binding_energy = e_complex - e_pd_frag - e_ligand
print(f"Ligand binding energy: {binding_energy * 23.0609:.1f} kcal/mol")
```

## Computational Details

### Training Data

AIMNet2-Pd is trained on wB97M-D3/CPCM (THF solvent) reference data for diverse Pd-containing molecular complexes. The implicit CPCM solvation model for THF is baked into the training data, so all predicted energetics include continuum solvent effects appropriate for THF and similar non-polar aprotic solvents. The training set covers:

- Pd(0) and Pd(II) oxidation states
- Common ligand types: phosphines, amines, halides, N-heterocyclic carbenes
- Catalytic intermediates from cross-coupling reactions
- Various coordination numbers (2-coordinate linear to 4-coordinate square planar)

### Long-Range Corrections

External DFT-D3 dispersion and long-range Coulomb modules are included. The dispersion correction is particularly important for Pd complexes with bulky ligands where London dispersion contributes significantly to ligand binding.

### Ensemble Averaging

```python
models = [AIMNet2Calculator(f"aimnet2-pd_{i}") for i in range(4)]
results = [m(data, forces=True) for m in models]

energies = torch.stack([r["energy"] for r in results])
print(f"Energy: {energies.mean().item():.6f} +/- {energies.std().item():.6f} eV")
```

!!! note "Ensemble member naming" The Pd model ensemble members use a hyphen: `aimnet2-pd_0` through `aimnet2-pd_3` (note the hyphen between "aimnet2" and "pd").

### Performance

Pd complexes are typically moderate in size (20-80 atoms). Using `compile_model=True` is recommended for reaction profile scans or molecular dynamics where many sequential evaluations are performed on similarly-sized systems.

## References

Anstine, D. M.; Zubatyuk, R.; Isayev, O. AIMNet2: A Neural Network Potential to Meet your Neutral, Charged, Organic, and Elemental-Organic Needs. _Chemical Science_ **2025**, _16_, 10228--10244. DOI: [10.1039/D4SC08572H](https://doi.org/10.1039/D4SC08572H)

Anstine, D. M.; Zubatyuk, R.; Gallegos, L.; Paton, R.; Wiest, O.; Nebgen, B.; Jones, T.; Gomes, G.; Tretiak, S.; Isayev, O. Transferable Machine Learning Interatomic Potential for Pd-Catalyzed Cross-Coupling Reactions. _ChemRxiv_ **2025**. DOI: [10.26434/chemrxiv-2025-n36r6](https://doi.org/10.26434/chemrxiv-2025-n36r6)
