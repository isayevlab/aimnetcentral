# Conformational Sampling

## What You Will Learn

- Building a conformer search pipeline with RDKit and AIMNet2
- Choosing between `aimnet2` (accuracy) and `aimnet2_2025` (B97-3c speed) for screening
- Batch geometry optimization of conformer ensembles
- Boltzmann-weighted population analysis
- Computing dihedral angles for Ramachandran-style analysis
- Managing GPU memory when processing many conformers

## Prerequisites

- Familiarity with the [AIMNet2Calculator API](../calculator.md) and [ASE integration](../getting_started.md)
- Installation: `pip install "aimnet[ase]" rdkit`
- Basic understanding of conformational analysis in computational chemistry

## The Conformer Search Pipeline

Conformational sampling follows a generate-optimize-rank workflow:

1. **Generate** candidate conformers with RDKit's ETKDG algorithm (fast, approximate)
2. **Optimize** each conformer with AIMNet2 (accurate gradients at ML speed)
3. **Rank** by energy and compute Boltzmann populations

This combines the strengths of each method: RDKit provides broad sampling of conformational space, while AIMNet2 provides near-DFT accuracy for energy ranking at a fraction of the cost.

### Model Choice: Speed vs Accuracy

| Model | Functional | Relative Speed | Best For |
| --- | --- | --- | --- |
| `aimnet2_2025` | B97-3c (improved) | Faster | Initial screening, large libraries, many conformers |
| `aimnet2` | wB97M-D3 | Baseline | Final ranking, publication-quality results |

!!! tip "Two-stage workflow" For large molecules with many conformers, use `aimnet2_2025` to screen and discard high-energy conformers (e.g., > 5 kcal/mol above minimum), then re-optimize the survivors with `aimnet2` for final ranking.

## Step 1: Generate Conformers with RDKit

RDKit's ETKDG (Experimental Torsion Knowledge Distance Geometry) algorithm generates diverse 3D conformers using torsional preferences from experimental crystal structures.

```python
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

# Parse SMILES and add hydrogens
smiles = "CC(=O)NC(C)C(=O)NC"  # Alanine dipeptide (Ace-Ala-NMe)
mol = Chem.MolFromSmiles(smiles)
mol = Chem.AddHs(mol)

# Count rotatable bonds to estimate conformational complexity
n_rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
print(f"Rotatable bonds: {n_rot}")

# Generate conformers with ETKDG
params = AllChem.ETKDGv3()
params.randomSeed = 42
params.numThreads = 0        # Use all available CPU cores
params.pruneRmsThresh = 0.5  # Remove near-duplicates (Angstrom RMSD)

n_confs = min(50 * (n_rot + 1), 500)  # Scale with flexibility, cap at 500
conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)
print(f"Generated {len(conf_ids)} conformers")

# Optional: quick MMFF pre-optimization to remove clashes
results = AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=0, maxIters=200)
```

## Step 2: Optimize Conformers with AIMNet2

Convert RDKit conformers to ASE Atoms objects and optimize each with BFGS.

```python
import numpy as np
from ase import Atoms
from ase.optimize import BFGS
from aimnet.calculators import AIMNet2ASE, AIMNet2Calculator

# Create calculator (shared across all conformers)
base_calc = AIMNet2Calculator("aimnet2", compile_model=True)

# Extract atomic numbers once (same for all conformers)
atomic_nums = [atom.GetAtomicNum() for atom in mol.GetAtoms()]


def rdkit_conf_to_ase(mol, conf_id):
    """Convert an RDKit conformer to an ASE Atoms object."""
    conf = mol.GetConformer(conf_id)
    positions = conf.GetPositions()
    return Atoms(numbers=atomic_nums, positions=positions)


# Optimize all conformers
optimized = []
for i, conf_id in enumerate(conf_ids):
    atoms = rdkit_conf_to_ase(mol, conf_id)
    atoms.calc = AIMNet2ASE(base_calc, charge=0)

    opt = BFGS(atoms, logfile=None)
    converged = opt.run(fmax=0.01, steps=500)

    energy = atoms.get_potential_energy()
    optimized.append({
        "conf_id": conf_id,
        "atoms": atoms,
        "energy": energy,
        "converged": converged,
    })

    if (i + 1) % 10 == 0:
        print(f"Optimized {i + 1}/{len(conf_ids)} conformers")

# Filter unconverged
optimized = [c for c in optimized if c["converged"]]
print(f"{len(optimized)} conformers converged")
```

## Step 3: Energy Ranking and Boltzmann Populations

Rank conformers by relative energy and compute thermodynamic populations.

```python
import numpy as np

# Sort by energy
optimized.sort(key=lambda c: c["energy"])

# Relative energies in kcal/mol
E_min = optimized[0]["energy"]
for c in optimized:
    c["rel_energy_eV"] = c["energy"] - E_min
    c["rel_energy_kcal"] = c["rel_energy_eV"] * 23.0609

# Boltzmann weighting at 298.15 K
kT_kcal = 0.5922  # kcal/mol at 298.15 K (R * T)
rel_energies = np.array([c["rel_energy_kcal"] for c in optimized])
boltzmann_factors = np.exp(-rel_energies / kT_kcal)
partition_function = boltzmann_factors.sum()
populations = boltzmann_factors / partition_function

for i, c in enumerate(optimized):
    c["population"] = populations[i]

# Report top conformers
print(f"\n{'Rank':<6}{'Rel. E (kcal/mol)':<20}{'Population (%)':<16}")
print("-" * 42)
for i, c in enumerate(optimized[:10]):
    print(f"{i+1:<6}{c['rel_energy_kcal']:<20.2f}{c['population']*100:<16.1f}")

# Cumulative population of top N
cumulative = np.cumsum(populations[:10])
print(f"\nTop 10 conformers cover {cumulative[-1]*100:.1f}% of population")
```

!!! note "Temperature matters" At room temperature (298 K), kT is about 0.6 kcal/mol. Conformers more than 3 kcal/mol above the minimum typically have negligible population (< 1%). At higher temperatures or for entropy-driven processes, more conformers become relevant.

## Worked Example: Alanine Dipeptide Ramachandran Plot

The alanine dipeptide (Ace-Ala-NMe) is a classic benchmark in biomolecular simulation. Its conformational landscape is described by two backbone dihedral angles, phi and psi, which can be plotted as a Ramachandran diagram.

### Define the Dihedral Angles

```python
import numpy as np


def get_dihedral(positions, i, j, k, l):
    """Compute dihedral angle (degrees) for atoms i-j-k-l."""
    b1 = positions[j] - positions[i]
    b2 = positions[k] - positions[j]
    b3 = positions[l] - positions[k]

    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    n1 /= np.linalg.norm(n1)
    n2 /= np.linalg.norm(n2)
    b2_norm = b2 / np.linalg.norm(b2)

    m1 = np.cross(n1, b2_norm)
    x = np.dot(n1, n2)
    y = np.dot(m1, n2)

    return np.degrees(np.arctan2(-y, x))
```

### Extract Phi/Psi from Optimized Conformers

```python
# Identify the phi/psi atom indices from the alanine dipeptide
# phi: C(acetyl)-N-CA-C(carbonyl)
# psi: N-CA-C(carbonyl)-N(methyl)
#
# These indices depend on your specific atom ordering.
# For the SMILES "CC(=O)NC(C)C(=O)NC" with hydrogens:
# Inspect your molecule to find the correct backbone atom indices.

# Example (verify indices for your molecule):
phi_atoms = (1, 3, 4, 6)   # C-N-CA-C
psi_atoms = (3, 4, 6, 8)   # N-CA-C-N

phi_angles = []
psi_angles = []
energies = []

for c in optimized:
    pos = c["atoms"].get_positions()
    phi = get_dihedral(pos, *phi_atoms)
    psi = get_dihedral(pos, *psi_atoms)
    phi_angles.append(phi)
    psi_angles.append(psi)
    energies.append(c["rel_energy_kcal"])
```

### Plot the Ramachandran Diagram

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 7))

scatter = ax.scatter(
    phi_angles, psi_angles,
    c=energies,
    cmap="RdYlGn_r",
    s=60,
    edgecolors="black",
    linewidths=0.5,
    vmin=0,
    vmax=max(5, max(energies)),
)

cbar = plt.colorbar(scatter, ax=ax, label="Relative energy (kcal/mol)")
ax.set_xlabel("Phi (degrees)")
ax.set_ylabel("Psi (degrees)")
ax.set_title("Alanine Dipeptide Ramachandran Plot (AIMNet2)")
ax.set_xlim(-180, 180)
ax.set_ylim(-180, 180)
ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")

# Mark known regions
ax.annotate("C7eq", xy=(-80, 80), fontsize=10, ha="center")
ax.annotate("C5", xy=(-160, 160), fontsize=10, ha="center")
ax.annotate("alphaR", xy=(-60, -40), fontsize=10, ha="center")

plt.tight_layout()
plt.savefig("ramachandran.png", dpi=150)
plt.show()
```

## Memory Management for Large Ensembles

When processing hundreds of conformers, GPU memory can become a bottleneck. Process conformers in batches to avoid out-of-memory errors.

```python
import gc
import torch
from ase import Atoms
from ase.optimize import BFGS
from aimnet.calculators import AIMNet2ASE, AIMNet2Calculator

base_calc = AIMNet2Calculator("aimnet2", compile_model=True)

BATCH_SIZE = 50  # Process this many conformers at a time


def optimize_batch(conformer_list, base_calc):
    """Optimize a batch of conformers, clearing GPU memory between batches."""
    results = []
    for atoms in conformer_list:
        atoms.calc = AIMNet2ASE(base_calc, charge=0)
        opt = BFGS(atoms, logfile=None)
        converged = opt.run(fmax=0.01, steps=500)
        results.append({
            "atoms": atoms.copy(),  # Copy to detach from calculator
            "energy": atoms.get_potential_energy(),
            "converged": converged,
        })
    return results


# Process in batches
all_atoms = [rdkit_conf_to_ase(mol, cid) for cid in conf_ids]
all_results = []

for i in range(0, len(all_atoms), BATCH_SIZE):
    batch = all_atoms[i : i + BATCH_SIZE]
    batch_results = optimize_batch(batch, base_calc)
    all_results.extend(batch_results)

    # Clear GPU cache between batches
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    print(f"Processed {min(i + BATCH_SIZE, len(all_atoms))}/{len(all_atoms)}")
```

### Duplicate Removal

After optimization, different initial conformers may converge to the same minimum. Remove duplicates by comparing energies and RMSD.

```python
import numpy as np


def rmsd(pos1, pos2):
    """Compute RMSD between two coordinate sets (same atom ordering)."""
    return np.sqrt(np.mean(np.sum((pos1 - pos2) ** 2, axis=1)))


def remove_duplicates(conformers, energy_tol=0.001, rmsd_tol=0.3):
    """Remove duplicate conformers based on energy and RMSD thresholds.

    Parameters
    ----------
    conformers : list of dict
        Each dict has 'atoms' (ASE Atoms) and 'energy' (float, eV).
    energy_tol : float
        Energy tolerance in eV. Conformers within this energy are
        candidates for RMSD comparison.
    rmsd_tol : float
        RMSD tolerance in Angstrom. Conformers with RMSD below this
        are considered duplicates.

    Returns
    -------
    list of dict
        Unique conformers.
    """
    unique = []
    for conf in conformers:
        is_duplicate = False
        for u in unique:
            if abs(conf["energy"] - u["energy"]) < energy_tol:
                r = rmsd(
                    conf["atoms"].get_positions(),
                    u["atoms"].get_positions(),
                )
                if r < rmsd_tol:
                    is_duplicate = True
                    break
        if not is_duplicate:
            unique.append(conf)
    return unique


unique_conformers = remove_duplicates(all_results)
print(f"Unique conformers: {len(unique_conformers)} (from {len(all_results)})")
```

## What's Next

- [Model Selection Guide](../models/guide.md) -- choosing `aimnet2` vs `aimnet2_2025` for your workflow
- [Batch Processing tutorial](../tutorials/batch_processing.md) -- efficient processing of molecular datasets
- [Performance Tuning tutorial](../tutorials/performance.md) -- `compile_model=True` and GPU optimization
- [Reaction Paths and Transition States](reaction_paths.md) -- finding transition states between conformers
