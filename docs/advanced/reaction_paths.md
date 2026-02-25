# Transition States and Reaction Profiles

## What You Will Learn

- Using PySisyphus with AIMNet2 for reaction path calculations
- Setting up NEB (Nudged Elastic Band) to find transition states
- Characterizing transition states via Hessian analysis (one imaginary frequency)
- Following IRC (Intrinsic Reaction Coordinate) from a transition state
- Worked example: SN2 reaction with charged species and symmetry

## Prerequisites

- Familiarity with the [AIMNet2Calculator API](../calculator.md)
- Understanding of [AIMNet2-NSE for open-shell chemistry](open_shell.md)
- Installation: `pip install "aimnet[ase]" pysisyphus`
- Basic knowledge of transition state theory and reaction coordinates

## Why Use AIMNet2 for Reaction Paths?

Finding transition states with DFT typically requires hundreds to thousands of gradient evaluations, each taking minutes to hours. AIMNet2 provides near-DFT gradients in milliseconds, making reaction path calculations that would take days at the DFT level complete in minutes.

!!! tip "Model selection for reaction paths" Use `aimnet2nse` for reactions involving bond breaking or forming, especially homolytic processes (radical mechanisms, H-atom abstractions). For heterolytic processes (SN2, proton transfers), the standard `aimnet2` model may also work, but `aimnet2nse` is the safer default because it handles both closed-shell and open-shell regions of the potential energy surface.

## PySisyphus Integration

AIMNet2 integrates with PySisyphus through the `AIMNet2Pysis` calculator class. PySisyphus is a Python framework for exploring potential energy surfaces, providing NEB, growing string, and IRC methods.

### The AIMNet2Pysis Calculator

`AIMNet2Pysis` wraps `AIMNet2Calculator` with the PySisyphus calculator interface, handling unit conversions automatically:

- Coordinates: Bohr (PySisyphus) <-> Angstrom (AIMNet2)
- Energy: Hartree (PySisyphus) <-> eV (AIMNet2)
- Forces: Hartree/Bohr (PySisyphus) <-> eV/Angstrom (AIMNet2)
- Hessian: Hartree/Bohr^2 (PySisyphus) <-> eV/Angstrom^2 (AIMNet2)

### Using AIMNet2Pysis Directly

```python
from aimnet.calculators import AIMNet2Pysis, AIMNet2Calculator

# From model name (creates AIMNet2Calculator internally)
calc = AIMNet2Pysis("aimnet2nse", charge=-1, mult=1)

# From existing calculator (reuse across workflows)
base = AIMNet2Calculator("aimnet2nse", compile_model=True)
calc = AIMNet2Pysis(base, charge=-1, mult=1)
```

### The run_pysis() Entry Point

For YAML-driven workflows, `run_pysis()` registers AIMNet2 as a calculator named `"aimnet"` within PySisyphus and launches the standard PySisyphus CLI:

```python
from aimnet.calculators.aimnet2pysis import run_pysis

run_pysis()
```

After calling `run_pysis()`, PySisyphus recognizes `type: aimnet` in YAML configuration files. This is the recommended way to run NEB, TS optimization, and IRC calculations.

## NEB for Finding Transition States

The Nudged Elastic Band method connects reactant and product geometries with a chain of images and optimizes the band to find the minimum energy path and approximate transition state.

### Step 1: Prepare Reactant and Product Geometries

First, optimize the reactant and product structures. For an SN2 reaction (Cl- + CH3Cl -> ClCH3 + Cl-), we need the reactant complex and product complex.

```python
import torch
from ase import Atoms
from ase.optimize import BFGS
from aimnet.calculators import AIMNet2ASE, AIMNet2Calculator

base_calc = AIMNet2Calculator("aimnet2nse", compile_model=True)

# Reactant complex: Cl- approaching CH3Cl from the back side
reactant = Atoms(
    symbols="CClClHHH",
    positions=[
        [ 0.000,  0.000,  0.000],  # C
        [ 1.800,  0.000,  0.000],  # Cl (leaving group)
        [-3.500,  0.000,  0.000],  # Cl- (nucleophile, far)
        [ 0.000,  1.030,  0.000],  # H
        [ 0.000, -0.515,  0.893],  # H
        [ 0.000, -0.515, -0.893],  # H
    ],
)

reactant.calc = AIMNet2ASE(base_calc, charge=-1, mult=1)
opt = BFGS(reactant, logfile="reactant_opt.log")
opt.run(fmax=0.01)

# Product complex: mirror image (Cl and Cl- swap roles)
product = Atoms(
    symbols="CClClHHH",
    positions=[
        [ 0.000,  0.000,  0.000],  # C
        [-1.800,  0.000,  0.000],  # Cl- (now leaving, far)
        [ 3.500,  0.000,  0.000],  # Cl (now bonded, was nucleophile)
        [ 0.000,  1.030,  0.000],  # H
        [ 0.000, -0.515,  0.893],  # H
        [ 0.000, -0.515, -0.893],  # H
    ],
)

product.calc = AIMNet2ASE(base_calc, charge=-1, mult=1)
opt = BFGS(product, logfile="product_opt.log")
opt.run(fmax=0.01)

# Save optimized structures for PySisyphus
from ase.io import write

write("reactant.xyz", reactant)
write("product.xyz", product)
```

### Step 2: PySisyphus YAML Configuration for NEB

Create a YAML file (`neb.yaml`) for the NEB calculation:

```yaml
geom:
  type: cart
  fn:
    - reactant.xyz
    - product.xyz

calc:
  type: aimnet
  model: aimnet2nse
  charge: -1
  mult: 1

cos:
  type: neb
  nimages: 11 # Number of interpolated images between endpoints
  climb: True # Use climbing-image NEB for accurate TS
  k_min: 0.01 # Spring constant range
  k_max: 0.10

opt:
  type: lbfgs
  max_cycles: 200
  thresh: gau_tight # Gaussian tight convergence criteria
```

### Step 3: Run the NEB Calculation

```bash
# Using the PySisyphus CLI with AIMNet2
python -c "from aimnet.calculators.aimnet2pysis import run_pysis; run_pysis()" neb.yaml
```

Or equivalently within a Python script:

```python
from aimnet.calculators.aimnet2pysis import run_pysis
import sys

sys.argv = ["pysis", "neb.yaml"]
run_pysis()
```

PySisyphus writes output files including:

- `cos_energies.dat` -- energies along the reaction path
- `cos_forces.dat` -- force norms per image
- Images as XYZ files for each optimization step

## Transition State Characterization

A true transition state has exactly one imaginary vibrational frequency (negative eigenvalue of the Hessian), corresponding to motion along the reaction coordinate. All other frequencies must be real (positive eigenvalues).

### Computing the Hessian

The AIMNet2 Hessian is computed analytically via double backpropagation through the neural network. It is available for single molecules only.

```python
import torch
import numpy as np
from aimnet.calculators import AIMNet2Calculator

calc = AIMNet2Calculator("aimnet2nse", compile_model=True)

# TS geometry (from NEB climbing image)
# Replace with actual TS coordinates from your NEB result
ts_coords = torch.tensor([
    [ 0.000,  0.000,  0.000],  # C
    [ 2.350,  0.000,  0.000],  # Cl (partially bonded)
    [-2.350,  0.000,  0.000],  # Cl (partially bonded)
    [ 0.000,  1.030,  0.000],  # H
    [ 0.000, -0.515,  0.893],  # H
    [ 0.000, -0.515, -0.893],  # H
])

result = calc({
    "coord": ts_coords,
    "numbers": torch.tensor([6, 17, 17, 1, 1, 1]),
    "charge": torch.tensor(-1.0),
    "mult": torch.tensor(1.0),
}, forces=True, hessian=True)

# Hessian shape: (N, 3, N, 3) -> reshape to (3N, 3N) for eigenvalue analysis
hessian = result["hessian"]
N = ts_coords.shape[0]
hessian_2d = hessian.reshape(3 * N, 3 * N).cpu().numpy()

# Convert to atomic units for frequency calculation
# AIMNet2 Hessian is in eV/Angstrom^2
EV_TO_HARTREE = 1.0 / 27.2114
ANG_TO_BOHR = 1.8897259886
hessian_au = hessian_2d * EV_TO_HARTREE / (ANG_TO_BOHR ** 2)
```

!!! warning "Hessian limitations" The Hessian computation is supported for **single molecules only**. If `mol_idx` indicates multiple molecules, the calculator raises `NotImplementedError`. The Hessian output has shape `(N, 3, N, 3)` and should be flattened to `(3N, 3N)` for eigenvalue analysis.

### Vibrational Frequency Analysis

```python
import numpy as np

# Atomic masses in atomic mass units
ATOMIC_MASSES = {1: 1.008, 6: 12.011, 7: 14.007, 8: 15.999, 17: 35.453}
numbers = [6, 17, 17, 1, 1, 1]
masses = np.array([ATOMIC_MASSES[z] for z in numbers])

# Mass-weighted Hessian
mass_weights = np.repeat(1.0 / np.sqrt(masses), 3)
hessian_mw = hessian_au * np.outer(mass_weights, mass_weights)

# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eigh(hessian_mw)

# Convert eigenvalues to wavenumbers (cm^-1)
# omega = sqrt(|eigenvalue|) in atomic units
# 1 Hartree / (bohr^2 * amu) -> cm^-1 conversion
HARTREE_TO_CM = 219474.63  # Hartree to cm^-1
AMU_TO_ME = 1822.888       # AMU to electron mass

frequencies_cm = []
for ev in eigenvalues:
    ev_scaled = ev * AMU_TO_ME  # Convert mass units
    if ev_scaled < 0:
        freq = -np.sqrt(abs(ev_scaled)) * HARTREE_TO_CM
    else:
        freq = np.sqrt(ev_scaled) * HARTREE_TO_CM
    frequencies_cm.append(freq)

# First 6 modes should be near-zero (translation + rotation)
# The 7th mode should be imaginary (negative) for a TS
print("Vibrational frequencies (cm^-1):")
print("-" * 40)
for i, freq in enumerate(frequencies_cm):
    marker = ""
    if i < 6:
        marker = " (translation/rotation)"
    elif freq < -50:
        marker = " ** IMAGINARY **"
    print(f"  Mode {i+1:3d}: {freq:10.1f}{marker}")

# Check TS criterion: exactly one imaginary frequency
imaginary = [f for f in frequencies_cm[6:] if f < -50]
if len(imaginary) == 1:
    print(f"\nValid transition state: one imaginary frequency at {imaginary[0]:.1f} cm^-1")
elif len(imaginary) == 0:
    print("\nNo imaginary frequencies: this is a minimum, not a TS")
else:
    print(f"\n{len(imaginary)} imaginary frequencies: higher-order saddle point")
```

### PySisyphus Hessian Calculation

PySisyphus can also compute the Hessian through its own workflow. The `AIMNet2Pysis.get_hessian()` method handles unit conversion automatically:

```yaml
# hessian.yaml
geom:
  type: cart
  fn: ts_guess.xyz

calc:
  type: aimnet
  model: aimnet2nse
  charge: -1
  mult: 1

tsopt:
  type: rsirfo # Rational function optimization for TS
  do_hess: True # Compute initial Hessian
  hessian_recalc: 5 # Recompute every 5 steps
  thresh: gau_tight
  max_cycles: 100
```

## IRC Following from Transition State

The Intrinsic Reaction Coordinate traces the steepest descent path from a transition state to both reactant and product minima, confirming that the TS connects the intended reactant and product.

### PySisyphus IRC Configuration

```yaml
# irc.yaml
geom:
  type: cart
  fn: ts_optimized.xyz # Optimized TS geometry

calc:
  type: aimnet
  model: aimnet2nse
  charge: -1
  mult: 1

irc:
  type: eulerpc # Euler predictor-corrector
  rms_grad_thresh: 0.0005 # Convergence criterion
  max_cycles: 100
  step_length: 0.1 # Step size in mass-weighted coordinates
  forward: True # Follow both forward and backward
  backward: True
```

Run the IRC:

```bash
python -c "from aimnet.calculators.aimnet2pysis import run_pysis; run_pysis()" irc.yaml
```

PySisyphus outputs the IRC path as a series of geometries with energies, which can be plotted as a reaction energy profile.

!!! warning "Zero-point energy corrections for barrier heights" Barrier heights computed from electronic energies alone (Delta E‡) do not include zero-point energy corrections. For quantitative comparison with experimental activation energies, compute the Hessian at each stationary point, extract ZPE (excluding the imaginary frequency at the TS), and apply corrections:

    Delta E‡_0 = E(TS) - E(reactant) + ZPE(TS) - ZPE(reactant)

    Note: The TS has one fewer real vibrational mode (3N-7 vs 3N-6) because the imaginary frequency is excluded.

### Plotting the Reaction Profile

```python
import numpy as np
import matplotlib.pyplot as plt

# Load IRC energies from PySisyphus output
# The exact file format depends on PySisyphus version
# Typically found in irc_energies.dat or similar
irc_data = np.loadtxt("irc_energies.dat")
reaction_coord = irc_data[:, 0]  # Mass-weighted distance
energies_hartree = irc_data[:, 1]

# Convert to relative kcal/mol
energies_kcal = (energies_hartree - energies_hartree[0]) * 627.509

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(reaction_coord, energies_kcal, "b-", linewidth=2)
ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")

# Mark key points
ts_idx = np.argmax(energies_kcal)
ax.plot(reaction_coord[ts_idx], energies_kcal[ts_idx], "ro", markersize=10, label="TS")
ax.plot(reaction_coord[0], energies_kcal[0], "gs", markersize=10, label="Reactant")
ax.plot(reaction_coord[-1], energies_kcal[-1], "gs", markersize=10, label="Product")

ax.set_xlabel("Reaction coordinate (amu^(1/2) * bohr)")
ax.set_ylabel("Relative energy (kcal/mol)")
ax.set_title("SN2 Reaction Profile: Cl- + CH3Cl")
ax.legend()
plt.tight_layout()
plt.savefig("reaction_profile.png", dpi=150)
plt.show()
```

## Worked Example: SN2 Reaction (Cl- + CH3Cl)

The Walden inversion SN2 reaction is an ideal test case because it involves:

- **Charged species**: the nucleophile Cl- and the overall -1 charge
- **Bond breaking and forming simultaneously**: the C-Cl bond breaks as the new C-Cl bond forms
- **Symmetry**: for the identity reaction (Cl- + CH3Cl -> ClCH3 + Cl-), the TS should be symmetric with equal C-Cl distances

### Complete Workflow

```python
import torch
from ase import Atoms
from ase.io import write
from ase.optimize import BFGS
from aimnet.calculators import AIMNet2ASE, AIMNet2Calculator

base_calc = AIMNet2Calculator("aimnet2nse", compile_model=True)

# 1. Optimize reactant complex
reactant = Atoms(
    symbols="CClClHHH",
    positions=[
        [ 0.000,  0.000,  0.000],  # C
        [ 1.780,  0.000,  0.000],  # Cl (bonded)
        [-3.200,  0.000,  0.000],  # Cl- (nucleophile)
        [-0.390,  1.010,  0.078],  # H
        [-0.390, -0.342,  0.955],  # H
        [-0.390, -0.668, -1.033],  # H
    ],
)
reactant.calc = AIMNet2ASE(base_calc, charge=-1, mult=1)
opt = BFGS(reactant, logfile=None)
opt.run(fmax=0.005)
E_reactant = reactant.get_potential_energy()
write("sn2_reactant.xyz", reactant)

# 2. Create product (mirror image)
product = reactant.copy()
pos = product.get_positions()
pos[:, 0] = -pos[:, 0]  # Mirror along x-axis
# Swap Cl indices to maintain correct bonding
pos[[1, 2]] = pos[[2, 1]]
product.set_positions(pos)
product.calc = AIMNet2ASE(base_calc, charge=-1, mult=1)
opt = BFGS(product, logfile=None)
opt.run(fmax=0.005)
E_product = product.get_potential_energy()
write("sn2_product.xyz", product)

# 3. Generate TS guess (symmetric, C-Cl distances equal)
ts_guess = Atoms(
    symbols="CClClHHH",
    positions=[
        [ 0.000,  0.000,  0.000],  # C (planar)
        [ 2.350,  0.000,  0.000],  # Cl
        [-2.350,  0.000,  0.000],  # Cl
        [ 0.000,  1.020,  0.150],  # H
        [ 0.000, -0.510,  0.885],  # H
        [ 0.000, -0.510, -0.885],  # H
    ],
)
write("sn2_ts_guess.xyz", ts_guess)

print(f"Reactant energy: {E_reactant:.4f} eV")
print(f"Product energy:  {E_product:.4f} eV")
print(f"Symmetry check:  dE = {abs(E_reactant - E_product) * 23.0609:.2f} kcal/mol")
print("  (Should be ~0 for identity SN2 reaction)")

# 4. Run NEB with PySisyphus (create YAML config)
neb_yaml = """
geom:
  type: cart
  fn:
    - sn2_reactant.xyz
    - sn2_product.xyz

calc:
  type: aimnet
  model: aimnet2nse
  charge: -1
  mult: 1

cos:
  type: neb
  nimages: 11
  climb: True

opt:
  type: lbfgs
  max_cycles: 200
  thresh: gau_tight
"""

with open("sn2_neb.yaml", "w") as f:
    f.write(neb_yaml)

print("\nNEB config written to sn2_neb.yaml")
print("Run with: python -c 'from aimnet.calculators.aimnet2pysis import run_pysis; run_pysis()' sn2_neb.yaml")
```

!!! note "SN2 reactions and model choice" The SN2 reaction involves heterolytic bond breaking (the leaving group departs with both bonding electrons). For heterolytic processes like this, the standard `aimnet2` model may perform adequately. However, `aimnet2nse` is recommended as the default for all reaction path calculations because it handles mixed closed-shell/open-shell regions correctly and introduces no penalty for closed-shell species (`mult=1` reduces to standard behavior).

## Practical Tips

### Convergence Troubleshooting

If the NEB calculation does not converge:

- **Increase the number of images** (`nimages: 15` or `21`) for complex reaction paths
- **Adjust spring constants** (`k_min` and `k_max`) -- stiffer springs keep images evenly spaced
- **Use a better initial path** -- provide an intermediate geometry if linear interpolation creates clashes
- **Check atom ordering** -- reactant and product must have consistent atom indices for interpolation to work

### Hessian Computation Considerations

- The Hessian requires double backpropagation through the neural network, which is more expensive than a single gradient evaluation
- For a system with N atoms, the Hessian has shape `(N, 3, N, 3)` and is computed element-by-element
- GPU memory usage scales as O(N^2) for the Hessian
- For systems larger than approximately 200 atoms, consider numerical Hessians instead

### Multi-Step Reactions

For reactions with multiple transition states (e.g., addition-elimination mechanisms):

1. Optimize all intermediates separately
2. Run NEB between consecutive stationary points (reactant->intermediate, intermediate->product)
3. Verify each TS with Hessian analysis
4. Follow IRC from each TS to confirm connectivity

## What's Next

- [Open-Shell Chemistry](open_shell.md) -- detailed NSE model explanation and radical chemistry workflows
- [AIMNet2-NSE model details](../models/aimnet2nse.md) -- training data and accuracy benchmarks
- [Geometry Optimization tutorial](../tutorials/geometry_optimization.md) -- structure relaxation fundamentals
- [Model Selection Guide](../models/guide.md) -- choosing the right model for your reaction chemistry
