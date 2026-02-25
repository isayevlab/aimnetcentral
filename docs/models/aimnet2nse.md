# AIMNet2-NSE

## Overview

AIMNet2-NSE (Neutral Spin Equilibrated) is the model for **open-shell molecular systems**. It extends the AIMNet2 architecture with spin-multiplicity awareness, using `num_charge_channels=2` to handle both charge and spin state information. This allows it to describe radicals, triplet states, and other systems with unpaired electrons that closed-shell models cannot treat correctly.

**Supported elements:** H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br, I (14 elements)

**Registry alias:** `aimnet2nse` (loads ensemble member `aimnet2nse_0`)

**Ensemble members:** `aimnet2nse_0` through `aimnet2nse_3` (4 models)

**DFT reference:** wB97M-D3/def2-TZVPP (same functional as the default AIMNet2)

## Strengths and Limitations

### Strengths

- Handles open-shell systems with proper spin-state dependence
- Uses the `mult` parameter to specify spin multiplicity (2S+1)
- High-quality wB97M-D3 reference data including open-shell species
- Can describe bond dissociation curves where standard models fail
- Essential for radical stability comparisons and spin-state energetics

### Limitations

!!! warning "Single-determinant DFT reference" AIMNet2-NSE is trained on unrestricted Kohn-Sham DFT data, which uses a single Slater determinant. It is **not reliable** for systems with genuine multi-reference character, including:

    - Biradicals with significant open-shell singlet character
    - Near-degenerate spin states in transition-metal-free systems
    - Strongly stretched bonds in singlet states (e.g., homolytic dissociation on a singlet surface)

    If you suspect multi-reference character, validate against CASSCF or MRCI calculations.

!!! warning "Spin contamination" The underlying DFT reference data may contain spin contamination from unrestricted KS calculations. The model learns from this data as-is, so predictions for heavily spin-contaminated states should be treated with caution.

!!! warning "Same element and system scope" No transition metals, molecular (gas-phase) training data only. For Pd-containing open-shell systems, neither this model nor [AIMNet2-Pd](aimnet2pd.md) is currently suitable -- use DFT directly.

## How Spin Multiplicity Works

AIMNet2-NSE accepts a `mult` parameter specifying the spin multiplicity (2S+1) of the system:

| Multiplicity (`mult`) | Unpaired electrons | Example systems |
| --- | --- | --- |
| 1 (singlet) | 0 | Closed-shell molecules, singlet carbenes |
| 2 (doublet) | 1 | Organic radicals, radical anions/cations |
| 3 (triplet) | 2 | Triplet oxygen, triplet carbenes, diradicals |

The multiplicity is passed as a floating-point tensor alongside the molecular charge:

```python
result = calc({
    "coord": coords,
    "numbers": numbers,
    "charge": torch.tensor(0.0),
    "mult": torch.tensor(2.0),   # Doublet radical
}, forces=True)
```

For closed-shell calculations with AIMNet2-NSE, set `mult=1.0`. This allows direct comparison of open-shell and closed-shell energies on the same potential energy surface.

## Typical Use Cases

- **Radical stability** -- compare bond dissociation energies across different C-H, N-H, or O-H bonds
- **Spin-state energetics** -- singlet-triplet gaps for carbenes, nitrenes, and other reactive intermediates
- **Bond dissociation** -- homolytic bond cleavage where the products are radicals
- **Radical reaction paths** -- hydrogen atom transfer, radical addition, beta-scission

## Quick Example

Computing the C-H bond dissociation energy of methane:

```python
import torch
from aimnet.calculators import AIMNet2Calculator

calc = AIMNet2Calculator("aimnet2nse", compile_model=True)

# Methane (CH4) - singlet, closed shell
ch4_coords = torch.tensor([
    [ 0.0000,  0.0000,  0.0000],  # C
    [ 0.6276,  0.6276,  0.6276],  # H
    [-0.6276, -0.6276,  0.6276],  # H
    [-0.6276,  0.6276, -0.6276],  # H
    [ 0.6276, -0.6276, -0.6276],  # H
])
ch4_numbers = torch.tensor([6, 1, 1, 1, 1])

e_ch4 = calc({
    "coord": ch4_coords,
    "numbers": ch4_numbers,
    "charge": torch.tensor(0.0),
    "mult": torch.tensor(1.0),  # Singlet
})["energy"].item()

# Methyl radical (CH3) - doublet
ch3_coords = torch.tensor([
    [ 0.0000,  0.0000,  0.0000],  # C
    [ 1.0770,  0.0000,  0.0000],  # H
    [-0.5385,  0.9326,  0.0000],  # H
    [-0.5385, -0.9326,  0.0000],  # H
])
ch3_numbers = torch.tensor([6, 1, 1, 1])

e_ch3 = calc({
    "coord": ch3_coords,
    "numbers": ch3_numbers,
    "charge": torch.tensor(0.0),
    "mult": torch.tensor(2.0),  # Doublet radical
})["energy"].item()

# Hydrogen atom - doublet
h_coords = torch.tensor([[0.0, 0.0, 0.0]])
h_numbers = torch.tensor([1])

e_h = calc({
    "coord": h_coords,
    "numbers": h_numbers,
    "charge": torch.tensor(0.0),
    "mult": torch.tensor(2.0),  # Doublet
})["energy"].item()

bde = e_ch3 + e_h - e_ch4
print(f"C-H BDE (methane): {bde:.4f} eV")
print(f"C-H BDE (methane): {bde * 23.0609:.1f} kcal/mol")
```

### ASE Integration

When using AIMNet2-NSE through the ASE interface, set the spin multiplicity via the `mult` parameter:

```python
from ase.io import read
from ase.optimize import BFGS
from aimnet.calculators import AIMNet2ASE, AIMNet2Calculator

base_calc = AIMNet2Calculator("aimnet2nse", compile_model=True)

# Doublet radical
atoms = read("radical.xyz")
atoms.calc = AIMNet2ASE(base_calc, charge=0, mult=2)

opt = BFGS(atoms)
opt.run(fmax=0.01)

print(f"Optimized energy: {atoms.get_potential_energy():.6f} eV")
```

To change the spin state during a calculation:

```python
atoms.calc.set_mult(3)  # Switch to triplet
e_triplet = atoms.get_potential_energy()

atoms.calc.set_mult(1)  # Switch to singlet
e_singlet = atoms.get_potential_energy()

gap = e_triplet - e_singlet
print(f"Singlet-triplet gap: {gap:.4f} eV ({gap * 23.0609:.1f} kcal/mol)")
```

## Computational Details

### Training Data

AIMNet2-NSE is trained on unrestricted wB97M-D3/def2-TZVPP calculations spanning singlet, doublet, and triplet states. The training set includes closed-shell molecules, organic radicals, and excited-state-like geometries to provide broad coverage of open-shell chemistry.

### Architecture Difference

The key architectural difference from the standard AIMNet2 model is `num_charge_channels=2`. The first channel handles molecular charge (as in all AIMNet2 models), and the second channel encodes spin multiplicity. This allows the model to learn spin-dependent energy contributions without separate models for each spin state.

### Long-Range Corrections

External DFT-D3 and long-range Coulomb modules are included, configured identically to the default AIMNet2 model.

### Ensemble Averaging

```python
models = [AIMNet2Calculator(f"aimnet2nse_{i}") for i in range(4)]
data = {
    "coord": coords,
    "numbers": numbers,
    "charge": torch.tensor(0.0),
    "mult": torch.tensor(2.0),
}
results = [m(data, forces=True) for m in models]

energies = torch.stack([r["energy"] for r in results])
print(f"Energy: {energies.mean().item():.6f} +/- {energies.std().item():.6f} eV")
```

## References

Anstine, D. M.; Zubatyuk, R.; Isayev, O. AIMNet2: A Neural Network Potential to Meet your Neutral, Charged, Organic, and Elemental-Organic Needs. _Chemical Science_ **2025**, _16_, 10228--10244. DOI: [10.1039/D4SC08572H](https://doi.org/10.1039/D4SC08572H)

Anstine, D. M.; Zubatyuk, R.; Isayev, O. AIMNet2-NSE: A Neural Network Potential for Organic Radical Chemistry. _Angew. Chem. Int. Ed._ **2025**. DOI: [10.1002/anie.202516763](https://doi.org/10.1002/anie.202516763)
