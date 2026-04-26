# External Packages

AIMNet2 is consumed by several external simulation and analysis packages.
This section is the entry point for users coming from one of those tools.
Each package's per-page banner is the canonical status; the table here
is a navigation aid only.

| Package | Status | Page |
|---|---|---|
| ASE | Supported | [ASE](ase.md) |
| pysisyphus | Supported | [pysisyphus](pysis.md) |
| OpenMM (`openmmml`) | Supported upstream | [OpenMM](openmm.md) |
| AMBER (`torchani-amber`) | Supported upstream | [AMBER](amber.md) |
| SCM AMS (`MLPotential` engine) | Supported upstream | [SCM AMS](ams.md) |
| ORCA (`!ExtOpt`) | Supported upstream | [ORCA](orca.md) |
| GROMACS NNPot | Blocked on TorchScript export | [GROMACS](gromacs.md) |
| LAMMPS (`pair_style mliap`) | Blocked on TorchScript export | [LAMMPS](lammps.md) |

The TorchScript-export blockers shared by GROMACS and LAMMPS are tracked
in
[`docs/superpowers/plans/2026-04-26-torchscript-export.md`](../superpowers/plans/2026-04-26-torchscript-export.md).
AMBER also benefits from that work -- `torchani-amber` accepts a
TorchScript `.pt` via the `model_type` keyword, so the same export
pathway would let us ship a self-contained AIMNet2 jit asset.
