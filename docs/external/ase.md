# ASE

AIMNet2 ships an ASE `Calculator` implementation (`AIMNet2ASE`) that plugs
directly into the standard ASE workflows: single-point energies, geometry
optimization, vibrational analysis, MD, and NEB.

## Install

```bash
pip install "aimnet[ase]"
```

## Quick start

```python
from ase.build import molecule
from aimnet.calculators import AIMNet2ASE

atoms = molecule("H2O")
atoms.calc = AIMNet2ASE("aimnet2")           # registry alias
energy = atoms.get_potential_energy()        # eV
forces = atoms.get_forces()                  # eV/A
```

## Charge and multiplicity

`AIMNet2ASE` accepts `charge=` and `mult=` constructor arguments and also
reads `atoms.info["charge"]` / `atoms.info["spin"]` (or `["mult"]`) on
each call, with `atoms.info` taking precedence over the constructor value.

```python
atoms.calc = AIMNet2ASE("aimnet2", charge=-1)
```

## Periodic systems

The calculator handles PBC automatically when `atoms.pbc` is set and the
cell is non-zero. Stress requires `stress=True`-equivalent ASE calls.

See:
- [Single point](../tutorials/single_point.md)
- [Geometry optimization](../tutorials/geometry_optimization.md)
- [Molecular dynamics](../tutorials/molecular_dynamics.md)
- [Periodic systems](../tutorials/periodic_systems.md)
- [Calculator API reference](../calculator.md)

## Implemented properties

`energy`, `forces`, `free_energy`, `charges`, `stress`, `dipole_moment`,
plus `spin_charges` for NSE models.
