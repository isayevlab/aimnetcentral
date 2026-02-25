# Periodic Systems: Crystals and Surfaces

## What You'll Learn

- How to set up periodic boundary conditions (PBC) with AIMNet2 and ASE
- How the calculator handles long-range electrostatics in periodic systems
- When to use DSF vs Ewald summation for Coulomb interactions
- How to optimize unit cell parameters using the stress tensor

## Prerequisites

- AIMNet2 installed with ASE support (`pip install "aimnet[ase]"`)
- Familiarity with ASE `Atoms` objects (see [Geometry Optimization](geometry_optimization.md))
- Basic understanding of periodic boundary conditions in atomistic simulation

## Step 1: Setting Up a Periodic System

Periodic systems require a unit cell and PBC flags. With ASE, you define these on the `Atoms` object:

```python
from ase import Atoms
from aimnet.calculators import AIMNet2ASE

# Formaldehyde molecular crystal (P2_1/c, simplified)
# 4 molecules in the unit cell
atoms = Atoms(
    symbols="C4O4H8",
    positions=[
        # Molecule 1
        [0.00, 0.00, 0.00],   # C
        [1.20, 0.00, 0.00],   # O
        [-0.55, 0.94, 0.00],  # H
        [-0.55, -0.94, 0.00], # H
        # Molecule 2
        [3.80, 2.50, 1.80],   # C
        [5.00, 2.50, 1.80],   # O
        [3.25, 3.44, 1.80],   # H
        [3.25, 1.56, 1.80],   # H
        # Molecule 3
        [1.90, 5.00, 3.60],   # C
        [3.10, 5.00, 3.60],   # O
        [1.35, 5.94, 3.60],   # H
        [1.35, 4.06, 3.60],   # H
        # Molecule 4
        [5.70, 2.50, 5.40],   # C
        [6.90, 2.50, 5.40],   # O
        [5.15, 3.44, 5.40],   # H
        [5.15, 1.56, 5.40],   # H
    ],
    cell=[7.60, 5.00, 7.20],
    pbc=True,
)

calc = AIMNet2ASE("aimnet2")
atoms.calc = calc
```

!!! note When `pbc=True`, the calculator always uses **sparse mode** with neighbor lists, regardless of system size or `nb_threshold`. This ensures periodic images are handled correctly through cell shift vectors.

## Step 2: Understanding the Coulomb Auto-Switch

When you run a periodic calculation for the first time, the calculator detects that your system has a unit cell and automatically switches the Coulomb method:

```python
energy = atoms.get_potential_energy()
# UserWarning: Switching to DSF Coulomb for PBC
```

**Why does this happen?** The default Coulomb method is `"simple"` (all-pairs 1/r sum), which does not account for periodic images. For periodic systems, this would give incorrect electrostatics. The calculator detects PBC and automatically switches to the **Damped Shifted Force (DSF)** method, which properly handles the infinite lattice sum with a smooth cutoff.

!!! warning The auto-switch warning appears once per calculator when PBC is first detected. To avoid the warning entirely, set the Coulomb method explicitly before your first periodic calculation (see Step 3).

## Step 3: Choosing a Coulomb Method for PBC

For periodic systems, you have two options for long-range electrostatics. Set the method on the underlying `AIMNet2Calculator` via the `base_calc` attribute:

### DSF (Recommended for Most Uses)

DSF (Damped Shifted Force) is the recommended method for routine periodic calculations. It uses a neighbor-list-based cutoff with smooth damping, giving O(N) scaling:

```python
# Set DSF before running any periodic calculation
calc.base_calc.set_lrcoulomb_method("dsf", cutoff=15.0, dsf_alpha=0.2)

energy = atoms.get_potential_energy()
forces = atoms.get_forces()
```

The default parameters (`cutoff=15.0`, `dsf_alpha=0.2`) work well for most molecular crystals. For dense systems, you can reduce the cutoff to 12 Angstrom. For surfaces or dilute systems, consider increasing to 18-20 Angstrom.

### Ewald (High-Accuracy Benchmarks)

Ewald summation splits the Coulomb sum into real-space and reciprocal-space components, giving the most accurate treatment of long-range electrostatics:

```python
calc.base_calc.set_lrcoulomb_method("ewald", ewald_accuracy=1e-8)

energy = atoms.get_potential_energy()
```

!!! warning Ewald summation is currently limited to **single-molecule (single-system) calculations**. It uses a full interaction matrix rather than neighbor lists, so it does not support batched periodic systems. For routine PBC work, use DSF.

**When to use Ewald:**

- Validating DSF results for a new system type
- Computing precise lattice energies for benchmarking
- Systems where electrostatics dominate (ionic molecular crystals)
- When computational cost is acceptable (not long MD trajectories)

### Comparing Methods

You can verify convergence by comparing DSF and Ewald on the same system:

```python
# DSF calculation
calc.base_calc.set_lrcoulomb_method("dsf", cutoff=15.0)
energy_dsf = atoms.get_potential_energy()

# Ewald calculation
calc.base_calc.set_lrcoulomb_method("ewald", ewald_accuracy=1e-8)
energy_ewald = atoms.get_potential_energy()

print(f"DSF energy:   {energy_dsf:.6f} eV")
print(f"Ewald energy: {energy_ewald:.6f} eV")
print(f"Difference:   {abs(energy_dsf - energy_ewald):.6f} eV")
# Typically < 0.01 eV for well-converged cutoffs
```

## Step 4: Computing the Stress Tensor

The stress tensor is essential for optimizing unit cell parameters. Request it via `stress=True` in the ASE calculator:

```python
from ase.constraints import ExpCellFilter
from ase.optimize import BFGS

# Set up the calculator with DSF for PBC
calc = AIMNet2ASE("aimnet2")
calc.base_calc.set_lrcoulomb_method("dsf", cutoff=15.0)
atoms.calc = calc

# Verify stress is available
stress = atoms.get_stress()  # Voigt notation (6,) in ASE
print(f"Stress (Voigt): {stress}")
```

!!! tip ASE returns stress in Voigt notation as a 6-element array (xx, yy, zz, yz, xz, xy). The AIMNet2 calculator internally computes the full 3x3 stress tensor, and ASE handles the conversion.

## Step 5: Cell Optimization

To optimize both atomic positions and cell parameters simultaneously, wrap the `Atoms` object in `ExpCellFilter` (or `FrechetCellFilter`):

```python
from ase.constraints import ExpCellFilter
from ase.optimize import BFGS

# ExpCellFilter allows simultaneous optimization of positions and cell
ecf = ExpCellFilter(atoms)

opt = BFGS(ecf, logfile="cell_opt.log")
opt.run(fmax=0.05)  # eV/Angstrom force convergence

print(f"Optimized cell: {atoms.cell.lengths()}")
print(f"Optimized angles: {atoms.cell.angles()}")
print(f"Final energy: {atoms.get_potential_energy():.6f} eV")
```

**Why `ExpCellFilter`?** It maps cell degrees of freedom onto an expanded coordinate space so that a standard optimizer (BFGS, FIRE) can optimize both positions and cell shape in a single run. The `fmax` criterion applies to both atomic forces and stress-derived cell forces.

## Step 6: Worked Example -- Molecular Crystal Optimization

Here is a complete workflow for optimizing a molecular crystal structure:

```python
from ase import Atoms
from ase.constraints import ExpCellFilter
from ase.optimize import BFGS
from aimnet.calculators import AIMNet2ASE

# --- 1. Build the crystal ---
atoms = Atoms(
    symbols="C4O4H8",
    positions=[
        [0.00, 0.00, 0.00], [1.20, 0.00, 0.00],
        [-0.55, 0.94, 0.00], [-0.55, -0.94, 0.00],
        [3.80, 2.50, 1.80], [5.00, 2.50, 1.80],
        [3.25, 3.44, 1.80], [3.25, 1.56, 1.80],
        [1.90, 5.00, 3.60], [3.10, 5.00, 3.60],
        [1.35, 5.94, 3.60], [1.35, 4.06, 3.60],
        [5.70, 2.50, 5.40], [6.90, 2.50, 5.40],
        [5.15, 3.44, 5.40], [5.15, 1.56, 5.40],
    ],
    cell=[7.60, 5.00, 7.20],
    pbc=True,
)

# --- 2. Attach calculator with DSF Coulomb ---
calc = AIMNet2ASE("aimnet2")
calc.base_calc.set_lrcoulomb_method("dsf", cutoff=15.0)
atoms.calc = calc

# --- 3. Optimize cell + positions ---
ecf = ExpCellFilter(atoms)
opt = BFGS(ecf, logfile="crystal_opt.log")
opt.run(fmax=0.05)

# --- 4. Report results ---
print(f"Converged in {opt.get_number_of_steps()} steps")
print(f"Final energy: {atoms.get_potential_energy():.4f} eV")
print(f"Cell lengths: {atoms.cell.lengths()}")
print(f"Cell angles:  {atoms.cell.angles()}")
```

!!! tip For reading crystal structures from CIF files, use `ase.io.read("structure.cif")`. ASE will parse the cell parameters and symmetry automatically.

## Summary of Coulomb Methods for PBC

| Method | Scaling | PBC Support | Accuracy | Best For |
| --- | --- | --- | --- | --- |
| `simple` | O(N^2) | No | Exact (non-PBC) | Auto-switches to DSF for PBC |
| `dsf` | O(N) | Yes | Good | Routine PBC, MD, optimization |
| `ewald` | O(N^2) | Yes | Highest | Benchmarks, single systems |

## What's Next

- [Batch Processing](batch_processing.md) -- Process multiple structures from datasets
- [Performance Tuning](performance.md) -- Optimize speed for repeated PBC calculations
- [Long-Range Methods](../long_range.md) -- Full reference for DSF, Ewald, and DFTD3 parameters
- [Charged Systems](../advanced/charged_systems.md) -- Handling ions and zwitterions in periodic cells
