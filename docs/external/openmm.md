# OpenMM (openmm-ml)

**Status: supported.**

[`openmm-ml`](https://github.com/openmm/openmm-ml) ships a built-in
`AIMNet2PotentialImpl` that wraps `aimnet.calculators.AIMNet2Calculator`
behind OpenMM's `MLPotential` API. The integration uses
`openmm.PythonForce` -- a Python callback per step -- so it does **not**
require a TorchScript export and is unaffected by the
[TorchScript-export blockers](gromacs.md) that gate GROMACS and LAMMPS.

## Install

```bash
pip install aimnet openmm openmmml
```

The PyPI distribution name is `openmmml` (no hyphen). Conda-forge ships
the same project as `openmm-ml`.

## Quick start

```python
from openmm.app import Topology
from openmmml import MLPotential

potential = MLPotential('aimnet2')
system = potential.createSystem(topology)
```

## QM/MM (mixed system)

`createMixedSystem` lets you replace a subset of atoms with the AIMNet2
potential while keeping the rest of the system on a classical force
field:

```python
mlAtoms = [...]   # atom indices belonging to the ML region
system = potential.createMixedSystem(topology, mmSystem, mlAtoms)
```

## Charge and multiplicity

Pass via the `MLPotential` arguments. They are forwarded to the AIMNet2
calculator through the `args` dict:

```python
system = potential.createSystem(topology, charge=-1, multiplicity=2)
```

Defaults: `charge=0`, `multiplicity=1`.

## Periodic systems

`AIMNet2PotentialImpl` automatically passes the periodic box vectors to
the calculator when the topology has them set
(`topology.getPeriodicBoxVectors() is not None`). The `PythonForce` is
flagged with `setUsesPeriodicBoundaryConditions(True)` in that case.

## Mechanism

Internally:

1. `MLPotential('aimnet2')` constructs `AIMNet2Calculator('aimnet2')` --
   the published wb97m-d3 model is hard-wired upstream.
2. Each step OpenMM calls a Python function that:
   - reads positions (nm) from the simulation `State`, converts to A,
   - calls `model({"coord", "numbers", "charge", "mult", "cell"?},
     forces=True)`,
   - converts energy eV -> kJ/mol and forces eV/A -> kJ/mol/nm,
   - returns `(energy, forces)`.
3. OpenMM consumes those via `PythonForce`.

Because this runs in Python on every MD step, performance is bounded by
the same eager-mode AIMNet2 throughput you get from
[`AIMNet2ASE`](ase.md).

## Caveats

- The model name is hard-coded to `'aimnet2'` upstream. Other AIMNet2
  variants (b97-3c, NSE, rxn) need a one-line change in
  `openmmml/models/aimnet2potential.py` to expose them via
  `MLPotential('...')`.
- Performance is the eager Python path -- not the eventual TorchScript
  fast path planned in
  [`docs/superpowers/plans/2026-04-26-torchscript-export.md`](../superpowers/plans/2026-04-26-torchscript-export.md).
