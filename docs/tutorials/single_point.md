# Your First Calculation

## What You'll Learn

- How to load an AIMNet2 model and run a single-point energy calculation
- What the output dictionary contains and how to interpret each key
- How to compare results across different AIMNet2 models
- What to expect during first-time warmup (Warp kernel JIT compilation)

## Prerequisites

- Python 3.11 or 3.12
- AIMNet2 installed (`pip install aimnet`)
- A CUDA-capable GPU is recommended but not required

## Step 1: Quick Warmup --- Water Molecule

Start with the smallest useful system: a water molecule (3 atoms). This lets you verify your installation and understand the API before moving to larger molecules.

```python
import torch
from aimnet.calculators import AIMNet2Calculator

# Load the default AIMNet2 model by registry alias.
# The first call downloads the model weights (~50 MB) if not cached locally.
calc = AIMNet2Calculator("aimnet2")

# Water molecule coordinates in Angstroms
coords = torch.tensor([
    [0.0000,  0.0000,  0.1173],  # O
    [0.0000,  0.7572, -0.4692],  # H
    [0.0000, -0.7572, -0.4692],  # H
])

# Run a single-point calculation with forces
result = calc(
    {
        "coord": coords,
        "numbers": torch.tensor([8, 1, 1]),  # Atomic numbers: O, H, H
        "charge": 0.0,                        # Net molecular charge
    },
    forces=True,
)

print(f"Energy: {result['energy'].item():.6f} eV")
print(f"Forces (eV/A):\n{result['forces']}")
print(f"Partial charges (e): {result['charges']}")
```

!!! note "First-run warmup" The very first calculation triggers two one-time costs:

    1. **Warp kernel JIT compilation** (10--30 seconds): AIMNet2 uses NVIDIA Warp
       for GPU-accelerated neighbor list and kernel operations. These kernels are compiled on first use and cached for subsequent runs.
    2. **Model weight download**: If this is your first time using a model name,
       the weights are downloaded from the model registry.

    After the first run, subsequent calculations are fast.

## Step 2: A Real-World Example --- Aspirin

Water is useful for testing, but let's work with something more relevant: aspirin (acetylsalicylic acid, C9H8O4, 21 atoms). This is a typical drug-like molecule that AIMNet2 handles routinely.

```python
# Aspirin (C9H8O4) - 21 atoms
# Coordinates from a DFT-optimized geometry (Angstroms)
aspirin_coords = torch.tensor([
    [-2.3261,  0.4893, -0.0383],  # C  (ring)
    [-1.7264, -0.7538, -0.1530],  # C  (ring)
    [-0.3383, -0.8189, -0.1283],  # C  (ring)
    [ 0.4410,  0.3177,  0.0106],  # C  (ring)
    [-0.1526,  1.5623,  0.1267],  # C  (ring)
    [-1.5376,  1.6331,  0.1005],  # C  (ring)
    [-2.3043, -1.9263, -0.3001],  # C  (ester carbonyl)
    [-3.7879, -1.8267, -0.3170],  # O  (ester bridge)
    [-1.8120, -3.0284, -0.4024],  # O  (ester =O)
    [ 1.8690,  0.2108,  0.0406],  # C  (carboxyl)
    [ 2.4883, -0.8937, -0.3152],  # O  (carboxyl =O)
    [ 2.5299,  1.2674,  0.4682],  # O  (carboxyl -OH)
    [-4.4210, -0.6688, -0.1741],  # C  (methyl)
    [-3.8458,  0.4058,  0.0016],  # H
    [-5.8831, -0.8229, -0.2112],  # H
    [-3.3925,  0.5192, -0.0588],  # H
    [ 0.1114, -1.7963, -0.2185],  # H
    [ 0.4453,  2.4625,  0.2334],  # H
    [-1.9915,  2.6116,  0.1878],  # H
    [-6.1444, -1.8626, -0.3952],  # H
    [-6.3503, -0.1625, -0.9472],  # H
])

aspirin_numbers = torch.tensor([
    6, 6, 6, 6, 6, 6,        # 6 ring C
    6,                        # Ester carbonyl C
    8, 8,                     # Ester O (bridge + =O)
    6,                        # Carboxyl C
    8, 8,                     # Carboxyl O (=O + -OH)
    6,                        # Methyl C
    1, 1, 1, 1, 1, 1, 1, 1,  # 8 H (ring + methyl + OH)
])

# Full calculation: energy, forces, charges, and hessian
result = calc(
    {
        "coord": aspirin_coords,
        "numbers": aspirin_numbers,
        "charge": 0.0,
    },
    forces=True,
    hessian=True,
)

print(f"Energy: {result['energy'].item():.6f} eV")
print(f"Max atomic force (norm): {result['forces'].norm(dim=1).max().item():.6f} eV/A")
print(f"Sum of charges: {result['charges'].sum().item():.4f} e")
print(f"Hessian shape: {result['hessian'].shape}")
```

## Step 3: Understanding the Output Dictionary

Every call to the calculator returns a dictionary of PyTorch tensors. Here is what each key means and when it appears:

| Key | Shape | Units | When present |
| --- | --- | --- | --- |
| `energy` | `(1,)` or `(B,)` | eV | Always |
| `charges` | `(N,)` or `(B,N)` | e | Always |
| `forces` | `(N,3)` or `(B,N,3)` | eV/A | When `forces=True` |
| `stress` | `(3,3)` or `(B,3,3)` | eV/A^3 | When `stress=True` (requires `cell`) |
| `hessian` | `(N,3,N,3)` | eV/A^2 | When `hessian=True` (single molecule only) |

Where `N` is the number of atoms and `B` is the batch size.

Let's inspect the aspirin result in detail:

```python
# Energy is the total potential energy of the molecule
energy_ev = result["energy"].item()
energy_kcal = energy_ev * 23.0609  # 1 eV = 23.0609 kcal/mol
print(f"Energy: {energy_ev:.4f} eV = {energy_kcal:.2f} kcal/mol")

# Forces are the negative gradient of energy with respect to positions.
# Near an energy minimum, forces should be close to zero.
forces = result["forces"]
max_force = forces.norm(dim=1).max().item()
print(f"Max force magnitude: {max_force:.6f} eV/A")

# Charges are atomic partial charges that sum to the net molecular charge.
# They are predicted by the neural network, not from a population analysis.
charges = result["charges"]
print(f"Charge on O atoms: {charges[[7, 8, 10, 11]]}")
print(f"Total charge: {charges.sum().item():.6f} e (should be ~0.0)")

# The hessian is the matrix of second derivatives: d^2E / (dR_i dR_j).
# Shape is (N, 3, N, 3) -- one 3x3 block for each atom pair.
hessian = result["hessian"]
print(f"Hessian shape: {hessian.shape}")  # (21, 3, 21, 3) for aspirin
```

!!! tip "When to request each property" Only request what you need. Computing forces adds one backward pass. Computing the hessian adds `3N` backward passes (one per force component), so it becomes expensive for large molecules. The hessian is also limited to single-molecule calculations.

## Step 4: Comparing Models

AIMNet2 provides several models trained on different DFT functionals. Each is suitable for different chemistry. You can compare them on the same molecule to understand how model choice affects predictions:

```python
model_names = ["aimnet2", "aimnet2_b973c", "aimnet2_2025"]

input_data = {
    "coord": aspirin_coords,
    "numbers": aspirin_numbers,
    "charge": 0.0,
}

print(f"{'Model':<20} {'Energy (eV)':>14} {'Max Force (eV/A)':>18}")
print("-" * 55)

for name in model_names:
    model_calc = AIMNet2Calculator(name)
    res = model_calc(input_data, forces=True)
    e = res["energy"].item()
    f_max = res["forces"].norm(dim=1).max().item()
    print(f"{name:<20} {e:>14.6f} {f_max:>18.6f}")
```

!!! warning "Absolute energies differ between models" Different models use different DFT functionals as training targets, so their absolute energy scales are not comparable. **Relative** energies (e.g., conformer energy differences, reaction energies) are meaningful within the same model but should not be mixed across models.

The available model aliases are:

| Alias           | DFT Functional    | Best for                     |
| --------------- | ----------------- | ---------------------------- |
| `aimnet2`       | wB97M-D3          | General organic chemistry    |
| `aimnet2_2025`  | B97-3c (improved) | Recommended B97-3c model     |
| `aimnet2_b973c` | B97-3c            | Legacy (superseded by 2025)  |
| `aimnet2nse`    | wB97M-D3          | Open-shell / radical systems |
| `aimnet2pd`     | wB97M-D3          | Palladium-containing systems |

Each alias points to ensemble member `_0` by default. For uncertainty estimation, you can load all four ensemble members (e.g., `aimnet2_wb97m_d3_0` through `aimnet2_wb97m_d3_3`) and compare their predictions.

## Step 5: Using `compile_model` for Repeated Calculations

If you plan to run many calculations on similar-sized molecules (e.g., scanning a potential energy surface), enable model compilation for a significant speedup:

```python
import time

calc_compiled = AIMNet2Calculator("aimnet2", compile_model=True)

input_data = {
    "coord": aspirin_coords,
    "numbers": aspirin_numbers,
    "charge": 0.0,
}

# First call: includes compilation overhead
torch.cuda.synchronize()
t0 = time.perf_counter()
_ = calc_compiled(input_data, forces=True)
torch.cuda.synchronize()
t1 = time.perf_counter()
print(f"First call (with compilation): {t1 - t0:.3f} s")

# Subsequent calls: compiled and fast
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(100):
    _ = calc_compiled(input_data, forces=True)
torch.cuda.synchronize()
t1 = time.perf_counter()
print(f"Average over 100 calls: {(t1 - t0) / 100 * 1000:.2f} ms")
```

!!! note "What `compile_model=True` does" This wraps the neural network forward pass with `torch.compile()`. It does **not** compile the neighbor list construction or the external Coulomb/DFTD3 modules. The benefit is greatest for repeated evaluations on the same system size, such as MD trajectories or geometry optimizations.

## What's Next

Now that you can run single-point calculations, continue with:

- **[Geometry Optimization](geometry_optimization.md)** -- Relax structures to their energy minimum using ASE
- **[Molecular Dynamics](molecular_dynamics.md)** -- Run NVT and NPT simulations
- **[Calculator API Reference](../calculator.md)** -- Full details on all constructor parameters and methods
