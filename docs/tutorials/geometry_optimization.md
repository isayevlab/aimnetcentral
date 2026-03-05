# Geometry Optimization

## What You'll Learn

- How to relax molecular structures to their energy minimum using ASE and AIMNet2
- How to monitor optimization convergence and inspect trajectories
- How to compute vibrational frequencies from the Hessian to confirm a true minimum
- How to obtain thermochemical quantities (enthalpy, Gibbs free energy) from frequency data

## Prerequisites

- AIMNet2 installed with ASE support: `pip install "aimnet[ase]"`
- Familiarity with [Your First Calculation](single_point.md) (loading models, interpreting output)
- A CUDA-capable GPU (recommended; CPU works but is slower)

## Step 1: Set Up the Calculator and Molecule

The `AIMNet2ASE` class wraps `AIMNet2Calculator` for use with ASE's optimization and dynamics infrastructure. It translates between ASE's `Atoms` object and the tensor-based calculator interface.

```python
from ase import Atoms
from ase.optimize import BFGS
from aimnet.calculators import AIMNet2ASE

# Create the ASE calculator wrapper.
# This loads the default aimnet2 model (wB97M-D3 functional).
calc = AIMNet2ASE("aimnet2")

# Build a water molecule to verify the setup
water = Atoms("OH2", positions=[
    [0.000,  0.000,  0.119],
    [0.000,  0.763, -0.477],
    [0.000, -0.763, -0.477],
])
water.calc = calc

# Check that the calculator works
energy = water.get_potential_energy()
forces = water.get_forces()
print(f"Initial energy: {energy:.6f} eV")
print(f"Max force: {abs(forces).max():.6f} eV/A")
```

!!! tip "Accessing the underlying calculator" The ASE wrapper stores the `AIMNet2Calculator` as `calc.base_calc`. You can use this to configure model settings that are not exposed through the ASE interface:

    ```python
    # Switch long-range Coulomb method (for periodic systems)
    calc.base_calc.set_lrcoulomb_method("dsf")

    # Check what device the model is running on
    print(calc.base_calc.device)
    ```

## Step 2: Optimize a Drug Molecule

Let's optimize aspirin (C9H8O4, 21 atoms) -- a small but realistic drug molecule. In practice, you would load coordinates from a file. Here we build the molecule directly to keep the tutorial self-contained.

```python
from ase import Atoms
from ase.optimize import BFGS
from ase.io import write
from aimnet.calculators import AIMNet2ASE

calc = AIMNet2ASE("aimnet2")

# Aspirin from an approximate (not optimized) geometry.
# In practice you would load from a file: atoms = read("aspirin.xyz")
aspirin = Atoms(
    numbers=[6, 6, 6, 6, 6, 6, 6, 8, 8, 6, 8, 8, 6, 8, 6, 1, 1, 1, 1, 1, 1],
    positions=[
        [-2.326,  0.489, -0.038], [-1.726, -0.754, -0.153],
        [-0.338, -0.819, -0.128], [ 0.441,  0.318,  0.011],
        [-0.153,  1.562,  0.127], [-1.538,  1.633,  0.100],
        [-2.304, -1.926, -0.300], [-3.788, -1.827, -0.317],
        [-1.812, -3.028, -0.402], [ 1.869,  0.211,  0.041],
        [ 2.488, -0.894, -0.315], [ 2.530,  1.267,  0.468],
        [-4.421, -0.669, -0.174], [-3.846,  0.406,  0.002],
        [-5.883, -0.823, -0.211], [-3.393,  0.519, -0.059],
        [ 0.111, -1.796, -0.219], [ 0.445,  2.463,  0.233],
        [-1.992,  2.612,  0.188], [-6.144, -1.863, -0.395],
        [-6.350, -0.163, -0.947],
    ],
)
aspirin.calc = calc

# Run BFGS optimization, saving trajectory for inspection
opt = BFGS(aspirin, trajectory="aspirin_opt.traj")
opt.run(fmax=0.01)  # Converge until max force < 0.01 eV/A

print(f"Optimized energy: {aspirin.get_potential_energy():.6f} eV")
print(f"Steps taken: {opt.nsteps}")

# Save the optimized structure
write("aspirin_optimized.xyz", aspirin)
```

The `fmax` parameter controls the convergence criterion: the optimization stops when the maximum atomic force magnitude (i.e., the largest per-atom force norm sqrt(fx^2+fy^2+fz^2)) drops below this threshold. A value of 0.01 eV/A is a reasonable default for most applications.

!!! note "Why BFGS?" BFGS (Broyden--Fletcher--Goldfarb--Shanno) builds an approximate Hessian from force evaluations, giving superlinear convergence near the minimum. ASE also provides `LBFGS` (lower memory for large systems) and `FIRE` (robust for far-from-minimum starting geometries). For most molecules under ~200 atoms, `BFGS` is the best choice.

## Step 3: Monitor Convergence

Inspecting the optimization trajectory helps verify that the optimization converged smoothly and didn't get stuck in a transition state or saddle point.

```python
from ase.io import read

# Read all frames from the trajectory
traj = read("aspirin_opt.traj", index=":")

print(f"{'Step':>4}  {'Energy (eV)':>14}  {'Max Force (eV/A)':>16}")
print("-" * 40)
for i, frame in enumerate(traj):
    e = frame.get_potential_energy()
    f_max = abs(frame.get_forces()).max()
    print(f"{i:4d}  {e:14.6f}  {f_max:16.6f}")
```

You should see the energy decrease monotonically (or nearly so) and the maximum force drop toward your `fmax` threshold.

!!! warning "Convergence issues" If the optimization oscillates or takes more than ~100 steps for a molecule under 50 atoms, check:

    1. **Starting geometry**: Very distorted structures may need `FIRE` instead
       of `BFGS` for the first phase.
    2. **Charge**: An incorrect net charge leads to wrong forces. Verify with
       `calc.charge`.
    3. **Multiplicity**: For open-shell systems, use `AIMNet2ASE("aimnet2nse")`
       and set `mult` appropriately.

## Step 4: Verify the Minimum with Frequency Analysis

A structure is a true energy minimum only if all vibrational frequencies are real (positive). Imaginary frequencies indicate a saddle point or transition state. We compute frequencies from the Hessian matrix.

```python
import torch
import numpy as np
from ase.units import invcm
from aimnet.calculators import AIMNet2Calculator

# Use the base calculator directly for the Hessian calculation.
# The ASE wrapper does not expose hessian=True, so we call the
# underlying calculator with the optimized coordinates.
base_calc = AIMNet2Calculator("aimnet2")

# Get optimized coordinates from the ASE Atoms object
coords = torch.tensor(aspirin.get_positions(), dtype=torch.float32)
numbers = torch.tensor(aspirin.get_atomic_numbers())

result = base_calc(
    {
        "coord": coords,
        "numbers": numbers,
        "charge": 0.0,
    },
    forces=True,
    hessian=True,
)

# The Hessian has shape (N, 3, N, 3). Reshape to (3N, 3N) for diagonalization.
N = len(aspirin)
hessian = result["hessian"].cpu().numpy().reshape(3 * N, 3 * N)

# Mass-weight the Hessian: H_mw[i,j] = H[i,j] / sqrt(m_i * m_j)
masses = aspirin.get_masses()  # atomic mass units
mass_weights = np.repeat(masses, 3)  # one per Cartesian coordinate
mass_matrix = np.sqrt(np.outer(mass_weights, mass_weights))
hessian_mw = hessian / mass_matrix

# Diagonalize to get eigenvalues (proportional to frequency^2)
eigenvalues, _ = np.linalg.eigh(hessian_mw)

# Convert eigenvalues to frequencies in cm^-1
# eigenvalue units: eV / (A^2 * amu) -> need conversion factors
from ase.units import _hbar, _e, _amu

# omega^2 = eigenvalue * eV / (A^2 * amu), convert to cm^-1
factor = np.sqrt(abs(_e) / (_amu * 1e-20)) / (2 * np.pi * 2.998e10)
frequencies = np.sign(eigenvalues) * np.sqrt(abs(eigenvalues)) * factor

# First 6 values should be near-zero (translations + rotations)
print("Lowest 10 frequencies (cm^-1):")
for i, freq in enumerate(sorted(frequencies)[:10]):
    label = "TR" if i < 6 else "vib"
    print(f"  {i+1:3d}: {freq:10.1f}  ({label})")

# Check for imaginary frequencies (true minimum has none)
vibrational = sorted(frequencies)[6:]  # skip translations/rotations
n_imaginary = sum(1 for f in vibrational if f < -10)  # threshold for numerical noise
if n_imaginary == 0:
    print("\nAll vibrational frequencies are real -> confirmed minimum!")
else:
    print(f"\nWARNING: {n_imaginary} imaginary frequency(ies) found -> not a minimum")
```

!!! warning "Hessian limitations" The Hessian calculation in AIMNet2 is limited to **single molecules** and scales as O(N^2) in memory. It is practical for molecules up to roughly 200 atoms. For larger systems, use finite-difference approaches or specialized phonon tools.

## Step 5: Thermochemistry from Frequencies

Once you have confirmed a true minimum (all real frequencies), you can compute thermochemical quantities -- enthalpy and Gibbs free energy corrections -- using ASE's `IdealGasThermo` module.

```python
from ase.thermochemistry import IdealGasThermo

# Get the electronic energy (potential energy at the minimum)
electronic_energy = aspirin.get_potential_energy()  # eV

# Filter out translations/rotations (keep only vibrational modes)
# Use absolute values and filter small near-zero modes
vib_energies = []
for freq in sorted(frequencies)[6:]:
    if abs(freq) > 10:  # Skip near-zero modes (numerical noise)
        # Convert cm^-1 to eV: E = h * c * nu
        energy_ev = abs(freq) * invcm
        vib_energies.append(energy_ev)

# Create the thermochemistry object
thermo = IdealGasThermo(
    vib_energies=vib_energies,
    potentialenergy=electronic_energy,
    atoms=aspirin,
    geometry="nonlinear",
    symmetrynumber=1,  # aspirin has no rotational symmetry (C1)
    spin=0,            # singlet ground state
)

# Compute thermochemical properties at 298.15 K and 1 atm
T = 298.15  # K
p = 101325  # Pa (1 atm)

H = thermo.get_enthalpy(temperature=T)
G = thermo.get_gibbs_energy(temperature=T, pressure=p)

print(f"Electronic energy:     {electronic_energy:.4f} eV")
print(f"Enthalpy (H, 298 K):   {H:.4f} eV")
print(f"Gibbs free energy (G): {G:.4f} eV")
print(f"Thermal correction:    {(H - electronic_energy) * 23.0609:.2f} kcal/mol")
print(f"-T*S contribution:     {(G - H) * 23.0609:.2f} kcal/mol")
```

This workflow -- optimize, compute Hessian, extract thermochemistry -- is the standard approach for obtaining reaction enthalpies and free energies with AIMNet2.

!!! tip "Thermochemistry for reactions" To compute a reaction enthalpy Delta_H, run this workflow for each reactant and product separately, then take the difference:

    ```
    Delta_H = sum(H_products) - sum(H_reactants)
    ```

    The same applies for Delta_G. Make sure all structures are true minima (no imaginary frequencies) before computing differences.

## Step 6: Scaling Up -- Larger Molecules

For molecules approaching 100+ atoms, consider these adjustments:

```python
from ase.io import read
from ase.optimize import LBFGS
from aimnet.calculators import AIMNet2ASE, AIMNet2Calculator

# For large molecules, compile the model for faster force evaluations
calc = AIMNet2ASE(
    AIMNet2Calculator("aimnet2", compile_model=True)
)

# Load a large molecule (e.g., Taxol with 113 atoms)
# atoms = read("taxol.xyz")
# atoms.calc = calc

# Use LBFGS for lower memory footprint on large systems
# opt = LBFGS(atoms, trajectory="taxol_opt.traj")
# opt.run(fmax=0.01)
```

!!! note "Practical size limits" - **Geometry optimization** works well up to thousands of atoms. - **Hessian computation** is practical up to ~200 atoms due to O(N^2) memory scaling (the full Hessian for 200 atoms requires 200 x 3 x 200 x 3 = 360,000 elements). - For larger systems needing frequencies, use finite-difference approaches with a displacement step.

## What's Next

With optimized structures in hand, you can proceed to:

- **[Molecular Dynamics](molecular_dynamics.md)** -- Run NVT and NPT simulations starting from your optimized geometry
- **[Periodic Systems](periodic_systems.md)** -- Optimize crystal structures with unit cell relaxation
- **[Conformer Search](../advanced/conformer_search.md)** -- Systematically explore conformational space
- **[Reaction Paths](../advanced/reaction_paths.md)** -- Find transition states and compute reaction barriers
