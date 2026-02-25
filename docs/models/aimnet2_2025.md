# AIMNet2-2025

!!! tip "Recommended B97-3c Model" AIMNet2-2025 is the **recommended B97-3c-level model** and supersedes the original [AIMNet2-B97-3c](aimnet2_b973c.md). It provides improved intermolecular interaction accuracy with no regression for intramolecular chemistry. Use this model for all new work requiring a B97-3c reference level.

## Overview

AIMNet2-2025 is the **current-generation B97-3c model**, combining the cost-effective B97-3c reference level with improved training for non-covalent chemistry. It supersedes the original AIMNet2-B97-3c (`aimnet2_b973c`) and is recommended for all applications: high-throughput screening, conformer ranking, binding energy calculations, and any workflow where B97-3c-level accuracy is appropriate.

**Supported elements:** H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br, I (14 elements)

**Registry alias:** `aimnet2_2025` (loads ensemble member `aimnet2_b973c_2025_d3_0`)

**Ensemble members:** `aimnet2_b973c_2025_d3_0` through `aimnet2_b973c_2025_d3_3` (4 models)

## Strengths and Limitations

### Strengths

- Improved accuracy for hydrogen bonding geometries and energies
- Better description of pi-stacking and CH-pi interactions
- More reliable binding energy predictions for molecular complexes
- Same broad element coverage as other AIMNet2 models
- Recommended for any study where non-covalent interactions drive the chemistry

### Limitations

!!! warning "B97-3c base level" The underlying reference data is at the B97-3c level. While intermolecular interactions are improved through enhanced training, the base functional limitations for intramolecular thermochemistry (e.g., barrier heights) remain. For general thermochemistry, prefer [AIMNet2 (wB97M-D3)](aimnet2.md).

!!! warning "Non-covalent improvements may not transfer to all motifs" The improved intermolecular accuracy has been validated on common non-covalent interaction types. Unusual or exotic interaction motifs (e.g., halogen bonding with heavy halogens, aerogen bonding) may not benefit equally.

!!! warning "Same scope restrictions" No transition metals, closed-shell only, molecular training data. See the [AIMNet2 page](aimnet2.md) for the full list of limitations.

## Typical Use Cases

- **General-purpose B97-3c calculations** -- recommended replacement for `aimnet2_b973c` in all workflows
- **High-throughput conformer screening** -- rapid ranking of large conformer sets with B97-3c accuracy
- **Binding energy calculations** -- compute interaction energies for molecular dimers and complexes using the supramolecular approach (complex minus monomers)
- **Hydrogen-bonded systems** -- study water clusters, nucleobase pairs, protein-ligand hydrogen bonds
- **Pi-stacking interactions** -- aromatic stacking in drug-receptor binding, crystal packing, DNA base stacking
- **Conformational analysis driven by non-covalent contacts** -- systems where intramolecular hydrogen bonds or dispersion contacts determine the preferred conformer

## Quick Example

Computing the binding energy of a water dimer:

```python
import torch
from aimnet.calculators import AIMNet2Calculator

calc = AIMNet2Calculator("aimnet2_2025", compile_model=True)

# Water dimer coordinates (Angstrom)
dimer_coords = torch.tensor([
    # Monomer A (donor)
    [ 0.0000,  0.0000,  0.1173],  # O
    [ 0.0000,  0.7572, -0.4692],  # H
    [ 0.0000, -0.7572, -0.4692],  # H
    # Monomer B (acceptor)
    [ 2.9000,  0.0000,  0.0000],  # O
    [ 3.4000,  0.7600,  0.0000],  # H
    [ 3.4000, -0.7600,  0.0000],  # H
])
dimer_numbers = torch.tensor([8, 1, 1, 8, 1, 1])

# Compute dimer energy
e_dimer = calc({
    "coord": dimer_coords,
    "numbers": dimer_numbers,
    "charge": torch.tensor(0.0),
})["energy"].item()

# Compute monomer energies separately
mono_a_coords = dimer_coords[:3]
mono_b_coords = dimer_coords[3:]
mono_numbers = torch.tensor([8, 1, 1])

e_mono_a = calc({
    "coord": mono_a_coords,
    "numbers": mono_numbers,
    "charge": torch.tensor(0.0),
})["energy"].item()

e_mono_b = calc({
    "coord": mono_b_coords,
    "numbers": mono_numbers,
    "charge": torch.tensor(0.0),
})["energy"].item()

binding_energy = e_dimer - e_mono_a - e_mono_b
print(f"Binding energy: {binding_energy * 1000:.1f} meV")
print(f"Binding energy: {binding_energy * 23.0609:.2f} kcal/mol")
```

!!! note "No BSSE correction needed" Unlike ab initio methods with finite basis sets, neural network potentials do not suffer from basis set superposition error (BSSE). The supramolecular approach (dimer minus monomers) directly gives the interaction energy without counterpoise correction.

### ASE Integration

```python
from ase.build import molecule
from aimnet.calculators import AIMNet2ASE, AIMNet2Calculator

base_calc = AIMNet2Calculator("aimnet2_2025", compile_model=True)
atoms = molecule("H2O")
atoms.calc = AIMNet2ASE(base_calc, charge=0)

energy = atoms.get_potential_energy()
print(f"Water energy: {energy:.6f} eV")
```

## Computational Details

### Training Data

AIMNet2-2025 builds on the B97-3c training set with additional emphasis on intermolecular interaction geometries and energies. The enhanced training improves the description of non-covalent contacts without sacrificing intramolecular accuracy.

### Long-Range Corrections

Non-covalent interactions at longer range are captured by the external DFT-D3 dispersion and long-range Coulomb modules. For molecular complexes in vacuum, these activate automatically. For periodic systems with significant intermolecular interactions (e.g., molecular crystals), proper long-range electrostatics configuration is important:

```python
# Ewald summation for high-accuracy periodic electrostatics
calc.set_lrcoulomb_method("ewald", ewald_accuracy=1e-8)

# Or DSF for faster periodic calculations
calc.set_lrcoulomb_method("dsf", cutoff=15.0)
```

### Ensemble Averaging

For binding energy calculations, ensemble averaging across all four members is recommended to assess reliability:

```python
models = [AIMNet2Calculator(f"aimnet2_b973c_2025_d3_{i}") for i in range(4)]

binding_energies = []
for m in models:
    e_dim = m(dimer_data)["energy"].item()
    e_a = m(mono_a_data)["energy"].item()
    e_b = m(mono_b_data)["energy"].item()
    binding_energies.append(e_dim - e_a - e_b)

import statistics
print(f"Binding energy: {statistics.mean(binding_energies)*23.0609:.2f} "
      f"+/- {statistics.stdev(binding_energies)*23.0609:.2f} kcal/mol")
```

## References

Anstine, D. M.; Zubatyuk, R.; Isayev, O. AIMNet2: A Neural Network Potential to Meet your Neutral, Charged, Organic, and Elemental-Organic Needs. _Chemical Science_ **2025**, _16_, 10228--10244. DOI: [10.1039/D4SC08572H](https://doi.org/10.1039/D4SC08572H)
