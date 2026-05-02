# OpenMM (openmm-ml)

**Status: supported.**

[`openmm-ml`](https://github.com/openmm/openmm-ml) ships a built-in `AIMNet2PotentialImpl` that wraps `aimnet.calculators.AIMNet2Calculator` behind OpenMM's `MLPotential` API. The integration uses `openmm.PythonForce` -- a Python callback per step -- so it does **not** require a TorchScript export and is unaffected by the [TorchScript-export blockers](gromacs.md) that gate GROMACS and LAMMPS.

## Install

```bash
pip install aimnet openmm openmmml
```

The PyPI distribution name is `openmmml` (no hyphen). Conda-forge ships the same project as `openmm-ml`.

## Quick start

```python
from openmm.app import Topology
from openmmml import MLPotential

potential = MLPotential('aimnet2')
system = potential.createSystem(topology)
```

## QM/MM (mixed system)

`createMixedSystem` lets you replace a subset of atoms with the AIMNet2 potential while keeping the rest of the system on a classical force field:

```python
mlAtoms = [...]   # atom indices belonging to the ML region
system = potential.createMixedSystem(topology, mmSystem, mlAtoms)
```

## Charge and multiplicity

Pass via the `MLPotential` arguments. They are forwarded to the AIMNet2 calculator through the `args` dict:

```python
system = potential.createSystem(topology, charge=-1, multiplicity=1)
```

Defaults: `charge=0`, `multiplicity=1`.

!!! warning "Closed-shell only" `MLPotential('aimnet2')` is hard-wired upstream to the wb97m-d3 closed-shell model (`is_nse=False`). The `multiplicity` argument is silently consumed but does **not** select an open-shell PES -- setting `multiplicity=2` for a doublet returns the closed-shell answer. For radicals / open-shell species use the AIMNet2-NSE family directly via the in-tree [`AIMNet2ASE`](ase.md) calculator; NSE is not exposed by `openmmml` today.

## Periodic systems

`AIMNet2PotentialImpl` automatically passes the periodic box vectors to the calculator when the topology has them set (`topology.getPeriodicBoxVectors() is not None`). The `PythonForce` is flagged with `setUsesPeriodicBoundaryConditions(True)` in that case.

## Mechanism

Internally:

1. `MLPotential('aimnet2')` constructs `AIMNet2Calculator('aimnet2')` -- the published wb97m-d3 model is hard-wired upstream.
2. Each step OpenMM calls a Python function that:
   - reads positions (nm) from the simulation `State`, converts to A,
   - calls `model({"coord", "numbers", "charge", "mult", "cell"?}, forces=True)`,
   - converts energy eV -> kJ/mol and forces eV/A -> kJ/mol/nm,
   - returns `(energy, forces)`.
3. OpenMM consumes those via `PythonForce`.

Because this runs in Python on every MD step, performance is bounded by the same eager-mode AIMNet2 throughput you get from [`AIMNet2ASE`](ase.md).

## Caveats

- **Model coverage in this engine**: only the wb97m-d3 model is exposed upstream as `MLPotential('aimnet2')`. Other variants (B97-3c, NSE, rxn) require per-family changes upstream, not just a model-name string -- e.g. AIMNet2-rxn has `supports_charged_systems=False` and a learned energy scale that makes cross-family `createMixedSystem` energies meaningless. Exposing another variant in `openmmml` is a per-family contract, not a one-line change.
- Performance is the eager Python path -- not the eventual TorchScript fast path tracked in the internal export plan.
