# Molecular Dynamics

## What You'll Learn

- How to run NVT (constant temperature) and NPT (constant pressure) molecular dynamics with AIMNet2 and ASE
- How to choose timesteps, equilibration strategies, and thermostat parameters for ML potentials
- How to use `compile_model=True` effectively and understand warmup costs
- How to analyze trajectories with radial distribution functions (RDF)

## Prerequisites

- AIMNet2 installed with ASE support: `pip install "aimnet[ase]"`
- Completed [Your First Calculation](single_point.md) and [Geometry Optimization](geometry_optimization.md)
- A CUDA-capable GPU (strongly recommended for MD)

## Step 1: Prepare the System

Always start MD from an optimized structure. Running dynamics from a poorly relaxed geometry wastes equilibration time and can cause numerical instabilities from large initial forces.

```python
from ase import Atoms
from ase.optimize import BFGS
from aimnet.calculators import AIMNet2ASE, AIMNet2Calculator

# Create the calculator.
# compile_model=True is recommended for MD because the same forward pass
# runs thousands of times. The compilation overhead pays for itself quickly.
base_calc = AIMNet2Calculator("aimnet2", compile_model=True)
calc = AIMNet2ASE(base_calc)

# Build ethanol (C2H5OH) as a simple example
ethanol = Atoms(
    "C2H6O",
    positions=[
        [ 0.000,  0.000,  0.000],  # C
        [ 1.520,  0.000,  0.000],  # C
        [-0.370,  1.040,  0.000],  # H
        [-0.370, -0.520,  0.900],  # H
        [-0.370, -0.520, -0.900],  # H
        [ 1.890, -1.040,  0.000],  # H
        [ 1.890,  0.520,  0.900],  # H
        [ 2.040,  0.520, -0.900],  # O
        [ 1.660,  1.440, -0.900],  # H
    ],
)
ethanol.calc = calc

# Relax before MD
opt = BFGS(ethanol, logfile=None)
opt.run(fmax=0.01)
print(f"Optimized energy: {ethanol.get_potential_energy():.4f} eV")
```

!!! note "What `compile_model=True` actually compiles" The `compile_model=True` flag wraps **only the neural network forward pass** with `torch.compile()`. The neighbor list construction, external Coulomb module, and external DFT-D3 module are **not** compiled. This means:

    - The NN evaluation (the most expensive part) gets compiled and optimized
    - Neighbor lists are rebuilt every step as normal
    - External long-range corrections run uncompiled

    The speedup is typically 1.5--3x for the overall step, depending on system size and what fraction of time is spent in the NN.

## Step 2: NVT Simulation with Langevin Thermostat

NVT (constant number, volume, temperature) is the most common ensemble for studying molecular motion in vacuum or implicit solvent. ASE's Langevin thermostat couples the system to a heat bath with a friction term.

```python
import time
import torch
from ase.md.langevin import Langevin
from ase.io.trajectory import Trajectory
from ase import units

# Set initial velocities from a Maxwell-Boltzmann distribution.
# Use a slightly higher temperature than the target to account for
# kinetic energy redistribution during equilibration.
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

target_temp = 300  # K
MaxwellBoltzmannDistribution(ethanol, temperature_K=target_temp)

# Set up the Langevin thermostat.
# friction=0.01 / units.fs is a moderate coupling strength.
# Too low: slow thermalization. Too high: overdamped, unphysical dynamics.
dyn = Langevin(
    ethanol,
    timestep=0.5 * units.fs,   # 0.5 fs is conservative for ML potentials
    temperature_K=target_temp,
    friction=0.01 / units.fs,
)

# Attach a trajectory writer to save frames every 10 steps
traj = Trajectory("ethanol_nvt.traj", "w", ethanol)
dyn.attach(traj.write, interval=10)

# Attach a logger to monitor temperature and energy
def print_status():
    t = ethanol.get_temperature()
    e_pot = ethanol.get_potential_energy()
    e_kin = ethanol.get_kinetic_energy()
    print(f"Step {dyn.nsteps:5d}  T={t:6.1f} K  "
          f"E_pot={e_pot:10.4f} eV  E_kin={e_kin:8.4f} eV  "
          f"E_tot={e_pot + e_kin:10.4f} eV")

dyn.attach(print_status, interval=50)

# Run the simulation
print("Starting NVT MD...")
torch.cuda.synchronize()
t_start = time.perf_counter()

dyn.run(steps=1000)  # 500 fs at 0.5 fs/step

torch.cuda.synchronize()
t_end = time.perf_counter()
wall_time = t_end - t_start
print(f"\nCompleted 1000 steps in {wall_time:.1f} s "
      f"({wall_time / 1000 * 1000:.2f} ms/step)")

traj.close()
```

!!! warning "First-step warmup" The very first MD step is significantly slower than subsequent steps because of two one-time costs:

    1. **Warp kernel JIT compilation** (10--30 s): NVIDIA Warp compiles GPU
       kernels on first use. These are cached on disk for future sessions.
    2. **torch.compile warmup** (if `compile_model=True`): The first forward
       pass through the compiled model triggers tracing and optimization. This adds 5--30 s depending on model size.

    After the first step, typical step times for a 9-atom molecule on a modern GPU are 1--5 ms. Do not include the first step in performance benchmarks.

!!! tip "Timestep selection for ML potentials" ML potentials like AIMNet2 have smooth, continuous energy surfaces, so they tolerate slightly larger timesteps than ab initio MD. However, the safe range depends on the dynamics:

    - **0.5 fs**: Conservative. Good starting point for any system.
    - **1.0 fs**: Suitable for equilibrium sampling of organic molecules when
      hydrogen motion is not critical.
    - **> 1.0 fs**: Not recommended without careful validation. ML potentials
      do not have the constraint machinery of classical force fields.

    When in doubt, run a short trajectory at 0.5 fs and check energy conservation in NVE (see below).

## Step 3: NPT Simulation with Berendsen Barostat

NPT (constant pressure) is needed for condensed-phase simulations where the volume should fluctuate. ASE provides the `NPT` integrator for this purpose.

!!! note "NPT requires periodic boundary conditions" The Berendsen barostat adjusts the unit cell dimensions, which only makes sense for periodic systems. For an isolated molecule in vacuum, use NVT instead.

```python
from ase.md.npt import NPT as NPTIntegrator
from ase.io import read
from ase import units

# For NPT, you need a periodic system with a unit cell.
# This example shows the setup pattern; replace with your periodic system.
# atoms = read("your_periodic_system.xyz")
# atoms.pbc = True
# atoms.cell = [10.0, 10.0, 10.0]  # Angstroms

# calc = AIMNet2ASE(
#     AIMNet2Calculator("aimnet2", compile_model=True)
# )

# For periodic systems with charges, switch to DSF Coulomb
# calc.base_calc.set_lrcoulomb_method("dsf")

# atoms.calc = calc

# MaxwellBoltzmannDistribution(atoms, temperature_K=300)

# npt = NPTIntegrator(
#     atoms,
#     timestep=0.5 * units.fs,
#     temperature_K=300,
#     externalstress=0.0,  # Target pressure in eV/A^3 (0 = vacuum, no external stress)
#     ttime=25 * units.fs,   # Thermostat time constant
#     pfactor=None,          # Barostat fictitious mass (None = ASE default)
# )

# traj = Trajectory("npt.traj", "w", atoms)
# npt.attach(traj.write, interval=10)
# npt.run(steps=5000)
# traj.close()
```

!!! tip "Equilibration strategy for condensed-phase systems" For reliable results, follow a two-phase protocol:

    1. **NVT equilibration** (1--5 ps): Let the temperature stabilize before
       coupling the barostat. This prevents large pressure spikes from an initial geometry that is far from the equilibrium density.
    2. **NPT production** (10+ ps): Collect data for analysis after the
       volume has stabilized.

    Monitor both temperature and volume/density over time to verify equilibration before collecting production data.

## Step 4: Validate Energy Conservation (NVE Check)

Before trusting production results, verify that your timestep conserves energy in an NVE (microcanonical) ensemble. Energy drift should be small compared to the thermal energy fluctuations.

```python
from ase.md.verlet import VelocityVerlet
from ase import units

# Start from a thermalized configuration
ethanol_nve = ethanol.copy()
ethanol_nve.calc = calc
MaxwellBoltzmannDistribution(ethanol_nve, temperature_K=300)

dyn_nve = VelocityVerlet(ethanol_nve, timestep=0.5 * units.fs)

energies = []

def log_energy():
    e_pot = ethanol_nve.get_potential_energy()
    e_kin = ethanol_nve.get_kinetic_energy()
    energies.append(e_pot + e_kin)

dyn_nve.attach(log_energy, interval=1)
dyn_nve.run(steps=500)

import numpy as np
energies = np.array(energies)
drift_per_step = (energies[-1] - energies[0]) / len(energies)
fluctuation = energies.std()

print(f"Energy drift: {drift_per_step * 1000:.4f} meV/step")
print(f"Energy fluctuation (std): {fluctuation * 1000:.2f} meV")
print(f"Drift/fluctuation ratio: {abs(drift_per_step) / fluctuation:.4f}")
```

A drift-to-fluctuation ratio below 0.01 indicates good energy conservation. If the drift is too large, reduce the timestep.

## Step 5: Trajectory Analysis -- Radial Distribution Function

The radial distribution function (RDF) characterizes the average structure of a system by measuring how atom density varies as a function of distance from a reference atom. ASE provides tools for computing the RDF from trajectories.

```python
import numpy as np
from ase.io import read

# Load the saved NVT trajectory
traj = read("ethanol_nvt.traj", index=":")
print(f"Loaded {len(traj)} frames")

# Compute pairwise distance distribution (simplified RDF for non-periodic)
# For a proper RDF in periodic systems, use ase.geometry.analysis
all_distances = []
for frame in traj:
    pos = frame.get_positions()
    n = len(pos)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pos[i] - pos[j])
            all_distances.append(d)

all_distances = np.array(all_distances)

# Histogram of pairwise distances
bins = np.linspace(0.5, 5.0, 100)
hist, edges = np.histogram(all_distances, bins=bins)
centers = 0.5 * (edges[:-1] + edges[1:])

print("\nPairwise distance distribution:")
print(f"  Most common distance: {centers[hist.argmax()]:.2f} A")
print(f"  Mean distance: {all_distances.mean():.2f} A")

# For periodic systems, use the proper RDF analysis:
# from ase.geometry.analysis import Analysis
# ana = Analysis(traj)
# rdf = ana.get_rdf(rmax=10.0, nbins=200, elements=("O", "H"))
```

!!! tip "For periodic systems" The simplified distance histogram above works for isolated molecules. For proper RDF analysis of periodic (condensed-phase) systems, use the ASE `Analysis` class which correctly handles periodic images:

    ```python
    from ase.geometry.analysis import Analysis

    ana = Analysis(traj)
    rdf = ana.get_rdf(rmax=10.0, nbins=200)
    ```

## Step 6: Performance Tuning

### Benchmarking Correctly

Always synchronize the GPU before timing to get accurate measurements:

```python
import time
import torch

# Warm up (first step includes JIT compilation)
dyn.run(steps=1)

# Benchmark subsequent steps
torch.cuda.synchronize()
t0 = time.perf_counter()
dyn.run(steps=100)
torch.cuda.synchronize()
t1 = time.perf_counter()

ms_per_step = (t1 - t0) / 100 * 1000
print(f"Performance: {ms_per_step:.2f} ms/step")
ns_per_day = (86400 / (ms_per_step / 1000)) * 0.5e-6  # 0.5 fs timestep
print(f"Throughput: {ns_per_day:.1f} ns/day")
```

### Compilation Modes

The default `torch.compile()` behavior is a good starting point. For lower per-step latency (at the cost of longer initial compilation), try `reduce-overhead` mode:

```python
base_calc = AIMNet2Calculator(
    "aimnet2",
    compile_model=True,
    compile_kwargs={"mode": "reduce-overhead"},
)
calc = AIMNet2ASE(base_calc)
```

| Compile mode | Warmup time | Per-step time | Best for |
| --- | --- | --- | --- |
| `False` (no compile) | None | Baseline | Single-point calculations |
| `True` (default) | 5--30 s | ~0.7x baseline | Most MD simulations |
| `reduce-overhead` | 10--60 s | ~0.5x baseline | Long production runs |

!!! warning "Compilation and system size changes" `torch.compile` traces the computation graph for a specific tensor shape. If the system size changes (e.g., atoms are added or removed), the model will be recompiled. For MD of a fixed system this is not an issue, but avoid using `compile_model=True` for workflows that process molecules of varying sizes.

## Practical Recommendations

### Timestep Guidelines

| System type | Recommended timestep | Notes |
| --- | --- | --- |
| Small organic (< 50 atoms) | 0.5--1.0 fs | Validate with NVE check |
| Large biomolecule | 0.5 fs | Hydrogen vibrations limit timestep |
| Condensed phase | 0.5 fs | Start conservative, validate |

### Thermostat Parameters

| Parameter            | Langevin                | Berendsen (NPT)        |
| -------------------- | ----------------------- | ---------------------- |
| Temperature coupling | `friction=0.01/fs`      | `ttime=25*fs`          |
| Pressure coupling    | N/A                     | `pfactor=None` (auto)  |
| Effect on dynamics   | Stochastic, ergodic     | Deterministic, smooth  |
| Good for             | Sampling, equilibration | Pressure equilibration |

### Equilibration Checklist

1. Start from an optimized structure
2. Initialize velocities with `MaxwellBoltzmannDistribution`
3. Run NVT for 1--5 ps to stabilize temperature
4. (If NPT) Switch to NPT for another 1--5 ps to stabilize volume
5. Verify equilibration by checking that temperature and energy fluctuate around stable mean values
6. Begin production run and collect data

## What's Next

With MD simulations running, explore these related topics:

- **[Periodic Systems](periodic_systems.md)** -- Set up condensed-phase simulations with proper PBC and Coulomb methods
- **[Performance Tuning](performance.md)** -- Detailed guide to maximizing throughput for large-scale simulations
- **[Non-Covalent Interactions](../advanced/intermolecular_interactions.md)** -- Study hydrogen bonding and stacking with the AIMNet2-2025 model
