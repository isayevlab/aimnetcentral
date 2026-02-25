# Batch Processing: Processing Molecular Datasets

## What You'll Learn

- How to construct `mol_idx` tensors for batching molecules of different sizes
- How to process multi-frame XYZ files efficiently with AIMNet2
- How to manage GPU memory when processing large datasets
- How to batch molecules with different atom counts using the flat coordinate format

## Prerequisites

- AIMNet2 installed (`pip install aimnet`)
- Familiarity with PyTorch tensors and basic AIMNet2 usage (see [First Calculation](single_point.md))
- A multi-structure dataset (XYZ file, SDF, or similar)

## Step 1: Understanding the Flat Coordinate Format

When processing multiple molecules together, AIMNet2 uses a **flat coordinate format**. Instead of padding all molecules to the same size, you concatenate all atomic coordinates into a single `(N_total, 3)` tensor and use `mol_idx` to indicate which molecule each atom belongs to:

```python
import torch
from aimnet.calculators import AIMNet2Calculator

calc = AIMNet2Calculator("aimnet2")

# Water (3 atoms) + Methane (5 atoms) + Ammonia (4 atoms)
coords = torch.tensor([
    # Water (molecule 0)
    [0.000,  0.000,  0.117],  # O
    [0.000,  0.757, -0.469],  # H
    [0.000, -0.757, -0.469],  # H
    # Methane (molecule 1)
    [0.000,  0.000,  0.000],  # C
    [0.629,  0.629,  0.629],  # H
    [0.629, -0.629, -0.629],  # H
    [-0.629,  0.629, -0.629], # H
    [-0.629, -0.629,  0.629], # H
    # Ammonia (molecule 2)
    [0.000,  0.000,  0.116],  # N
    [0.000,  0.939, -0.271],  # H
    [0.813, -0.470, -0.271],  # H
    [-0.813, -0.470, -0.271], # H
])

numbers = torch.tensor([8, 1, 1, 6, 1, 1, 1, 1, 7, 1, 1, 1])
mol_idx = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2])
charge = torch.tensor([0.0, 0.0, 0.0])  # One charge per molecule

result = calc(
    {"coord": coords, "numbers": numbers, "charge": charge, "mol_idx": mol_idx},
    forces=True,
)

# Energy has shape (3,) -- one value per molecule
print(f"Energies: {result['energy']}")
# Forces has shape (12, 3) -- one force vector per atom
print(f"Forces shape: {result['forces'].shape}")
```

!!! warning "mol_idx must be sorted in non-decreasing order" The `mol_idx` tensor **must** be sorted: all atoms for molecule 0 come first, then all atoms for molecule 1, and so on. Values like `[0, 1, 0, 1]` will produce incorrect results. Always use `[0, 0, 1, 1]`.

!!! warning "charge shape must match number of molecules" When using `mol_idx`, the `charge` tensor must have shape `(num_molecules,)` with one charge value per molecule. A scalar charge only works for single-molecule calculations.

## Step 2: Processing a Multi-Frame XYZ File

A common workflow is reading conformers or trajectory frames from an XYZ file and processing them in batches. Here is how to do it with ASE for reading and the low-level calculator for batched inference:

```python
import torch
from ase.io import read
from aimnet.calculators import AIMNet2Calculator

calc = AIMNet2Calculator("aimnet2")

# Read all frames from a multi-frame XYZ file
frames = read("conformers.xyz", index=":")
print(f"Loaded {len(frames)} conformers")

# All conformers of the same molecule have identical atom count
n_atoms = len(frames[0])
n_frames = len(frames)
```

### Same-Size Molecules (3D Batched Input)

When all molecules have the same number of atoms (e.g., conformers of one molecule), you can use the simpler 3D batched format:

```python
# Stack into (B, N, 3) tensor
coords = torch.tensor([f.positions for f in frames], dtype=torch.float32)
numbers = torch.tensor(frames[0].numbers, dtype=torch.long)  # Same for all conformers
numbers = numbers.unsqueeze(0).expand(n_frames, -1)           # (B, N)
charge = torch.zeros(n_frames)                                # (B,)

result = calc(
    {"coord": coords, "numbers": numbers, "charge": charge},
    forces=True,
)

# result["energy"] has shape (B,)
# result["forces"] has shape (B, N, 3)
energies = result["energy"]
relative = energies - energies.min()
print(f"Relative energies (eV): {relative}")
```

!!! tip For conformers of the same molecule, the 3D batched format `(B, N, 3)` is more convenient than flat coordinates with `mol_idx`. The calculator automatically decides between dense mode (small molecules on GPU) and sparse mode (large molecules or CPU).

### Different-Size Molecules (Flat Format with mol_idx)

When molecules have different atom counts, use the flat format:

```python
# Example: processing a dataset with varied molecules
molecules = [
    {"symbols": [8, 1, 1],       "coords": [[0,0,0.12], [0,0.76,-0.47], [0,-0.76,-0.47]], "charge": 0.0},
    {"symbols": [6, 1, 1, 1, 1], "coords": [[0,0,0], [.63,.63,.63], [.63,-.63,-.63], [-.63,.63,-.63], [-.63,-.63,.63]], "charge": 0.0},
]

all_coords = []
all_numbers = []
all_mol_idx = []
all_charges = []

for i, mol in enumerate(molecules):
    n = len(mol["symbols"])
    all_coords.extend(mol["coords"])
    all_numbers.extend(mol["symbols"])
    all_mol_idx.extend([i] * n)
    all_charges.append(mol["charge"])

result = calc({
    "coord": torch.tensor(all_coords, dtype=torch.float32),
    "numbers": torch.tensor(all_numbers, dtype=torch.long),
    "mol_idx": torch.tensor(all_mol_idx, dtype=torch.long),
    "charge": torch.tensor(all_charges, dtype=torch.float32),
}, forces=True)
```

## Step 3: Chunked Processing for Large Datasets

For datasets with thousands of structures, processing everything at once may exceed GPU memory. Break the dataset into chunks:

```python
import torch
from ase.io import read
from aimnet.calculators import AIMNet2Calculator

calc = AIMNet2Calculator("aimnet2")

frames = read("large_dataset.xyz", index=":")
n_atoms = len(frames[0])
batch_size = 64  # Adjust based on available GPU memory

all_energies = []
all_forces = []

for start in range(0, len(frames), batch_size):
    batch = frames[start:start + batch_size]
    B = len(batch)

    coords = torch.tensor([f.positions for f in batch], dtype=torch.float32)
    numbers = torch.tensor(batch[0].numbers).unsqueeze(0).expand(B, -1)
    charge = torch.zeros(B)

    result = calc(
        {"coord": coords, "numbers": numbers, "charge": charge},
        forces=True,
    )

    # Move results to CPU immediately to free GPU memory
    all_energies.append(result["energy"].cpu())
    all_forces.append(result["forces"].cpu())

    # Free GPU cache periodically
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

energies = torch.cat(all_energies)
forces = torch.cat(all_forces)
print(f"Processed {len(energies)} structures")
```

!!! tip "Memory management" Calling `.cpu()` on result tensors moves them off the GPU immediately. Combined with `torch.cuda.empty_cache()`, this prevents GPU memory from growing unboundedly across batches. This is especially important when processing thousands of structures.

## Step 4: Worked Example -- Conformer Ranking

Here is a complete example that reads conformers, computes energies in batches, and ranks them by relative energy:

```python
import torch
from ase.io import read
from aimnet.calculators import AIMNet2Calculator

# --- 1. Load conformers ---
frames = read("taxol_conformers.xyz", index=":")
n_conformers = len(frames)
n_atoms = len(frames[0])
print(f"Loaded {n_conformers} conformers of {n_atoms}-atom molecule")

# --- 2. Set up calculator ---
calc = AIMNet2Calculator("aimnet2")

# --- 3. Batch-process conformers ---
batch_size = 32
all_energies = []

for start in range(0, n_conformers, batch_size):
    batch = frames[start:start + batch_size]
    B = len(batch)

    coords = torch.tensor([f.positions for f in batch], dtype=torch.float32)
    numbers = torch.tensor(batch[0].numbers).unsqueeze(0).expand(B, -1)
    charge = torch.zeros(B)

    result = calc({"coord": coords, "numbers": numbers, "charge": charge})
    all_energies.append(result["energy"].cpu())

energies = torch.cat(all_energies)

# --- 4. Rank conformers ---
relative = energies - energies.min()

# Convert to kcal/mol for chemical interpretation
EV_TO_KCAL = 23.0609
relative_kcal = relative * EV_TO_KCAL

# Sort by energy
order = relative_kcal.argsort()
print("\nConformer ranking:")
print(f"{'Rank':<6} {'Index':<8} {'Rel. Energy (kcal/mol)':<24}")
for rank, idx in enumerate(order[:10]):
    print(f"{rank+1:<6} {idx.item():<8} {relative_kcal[idx].item():<24.2f}")

# --- 5. Boltzmann populations at 298 K ---
kT = 0.001987 * 298  # kcal/mol
weights = torch.exp(-relative_kcal / kT)
populations = weights / weights.sum()

print(f"\nTop conformer population: {populations[order[0]].item():.1%}")
print(f"Conformers within 2 kcal/mol: {(relative_kcal < 2.0).sum().item()}")
```

## Step 5: Processing Mixed Datasets with Different Charges

When your dataset contains molecules with different charges, build the `mol_idx` and `charge` tensors carefully:

```python
import torch
from aimnet.calculators import AIMNet2Calculator

calc = AIMNet2Calculator("aimnet2")

# Dataset: neutral water, ammonium cation, hydroxide anion
dataset = [
    {"z": [8, 1, 1],    "pos": [[0,0,0.12], [0,0.76,-0.47], [0,-0.76,-0.47]], "q": 0.0},
    {"z": [7, 1, 1, 1, 1], "pos": [[0,0,0], [.59,.59,.59], [.59,-.59,-.59], [-.59,.59,-.59], [-.59,-.59,.59]], "q": 1.0},
    {"z": [8, 1],        "pos": [[0,0,0], [0.96,0,0]], "q": -1.0},
]

coords, numbers, mol_idx, charges = [], [], [], []
for i, mol in enumerate(dataset):
    n = len(mol["z"])
    coords.extend(mol["pos"])
    numbers.extend(mol["z"])
    mol_idx.extend([i] * n)
    charges.append(mol["q"])

result = calc({
    "coord": torch.tensor(coords, dtype=torch.float32),
    "numbers": torch.tensor(numbers, dtype=torch.long),
    "mol_idx": torch.tensor(mol_idx, dtype=torch.long),
    "charge": torch.tensor(charges, dtype=torch.float32),
}, forces=True)

# Per-molecule energies
for i, mol in enumerate(dataset):
    print(f"Molecule {i} (charge={mol['q']:+.0f}): {result['energy'][i].item():.4f} eV")
```

## Common Pitfalls

!!! warning "Unsorted mol_idx"

````python # WRONG -- mol_idx is not sorted
mol_idx = torch.tensor([0, 1, 0, 1])

    # CORRECT -- atoms grouped by molecule
    mol_idx = torch.tensor([0, 0, 1, 1])
    ```

!!! warning "Scalar charge with batched input"
```python # WRONG -- scalar charge with multiple molecules
charge = 0.0

    # CORRECT -- one charge per molecule
    charge = torch.tensor([0.0, 0.0, 0.0])
    ```

!!! note "Hessian is single-molecule only"
Hessian calculation (`hessian=True`) is not supported for batched inputs. If you
need Hessians, compute them one molecule at a time. Hessian computation also
requires O(N^2) memory, so it is impractical for large molecules.

## What's Next

- [Performance Tuning](performance.md) -- Optimize batch size, compilation, and GPU memory
- [Geometry Optimization](geometry_optimization.md) -- Optimize structures before ranking
- [Conformer Search](../advanced/conformer_search.md) -- Full conformational sampling workflow
````
