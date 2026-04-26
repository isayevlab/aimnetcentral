# External Packages

AIMNet2 is consumed by several external simulation and analysis packages.
This section is the entry point for users coming from one of those tools.

## Available today

| Package | Status | Page |
|---|---|---|
| ASE | Supported (in-tree calculator) | [ASE](ase.md) |
| pysisyphus | Supported (in-tree calculator) | [pysisyphus](pysis.md) |
| OpenMM (`openmmml`) | Supported upstream via `MLPotential('aimnet2')` (Python callback, no TorchScript needed) | [OpenMM](openmm.md) |
| AMBER (`torchani-amber`) | Supported upstream for `sander` and `pmemd`; requires recompiling AmberTools 25/26 with the integration | [AMBER](amber.md) |
| SCM AMS (`MLPotential` engine) | Supported upstream from AMS2024.1 via `Model AIMNet2-wB97MD3` / `AIMNet2-B973c` | [SCM AMS](ams.md) |
| ORCA (`!ExtOpt`) | Supported upstream from ORCA 6.1 via the ORCA-External-Tools `aimnet2` wrapper | [ORCA](orca.md) |

## In progress / blocked

| Package | Status | Page |
|---|---|---|
| GROMACS NNPot (>=2025.0) | Blocked on TorchScript export | [GROMACS](gromacs.md) |
| LAMMPS (`pair_style mliap`) | Blocked on TorchScript export | [LAMMPS](lammps.md) |

The TorchScript-export blockers shared by GROMACS and LAMMPS are tracked
in
[`docs/superpowers/plans/2026-04-26-torchscript-export.md`](../superpowers/plans/2026-04-26-torchscript-export.md).
Note: AMBER also consumes a TorchScript `.pt` (via `model_type = "..."`
in the `&ani` namelist), so the same export work would let us ship a
self-contained AIMNet2 jit asset for `torchani-amber` too.
