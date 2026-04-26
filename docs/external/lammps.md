# LAMMPS

**Status: not currently supported.**

LAMMPS exposes TorchScript-based ML potentials via
[`pair_style mliap`](https://docs.lammps.org/pair_mliap.html) with the
`mliappy` model (built into LAMMPS; requires the `ML-IAP` and `PYTHON`
packages at build time). LAMMPS supplies the neighbor list to the
Python/TorchScript model rather than the model building its own.

AIMNet2 cannot currently produce a `.pt` that this route accepts, for
the same reasons documented in [GROMACS NNPot](gromacs.md):

1. The shipped v2 `.pt` files are state-dict archives, not TorchScript.
2. `torch.jit.script` on the core `AIMNet2` module fails on
   `tensor.data_ptr()` in `aimnet/nbops.py`.
3. `torch.jit.script` on the external `DFTD3` module fails on its
   custom autograd signature.

In addition, LAMMPS expects the scripted model to **accept a
LAMMPS-supplied neighbor list** (atom indices, and for some pair styles
edge-index + cell-shift tensors) rather than build its own. The
all-pairs neighbor list used in the parked GROMACS wrapper would not be
performant for typical LAMMPS box sizes. A LAMMPS-targeted wrapper
would need to consume the engine-supplied neighbor list inside the
scripted module instead.

The shared remediation plan lives at
[`docs/superpowers/plans/2026-04-26-torchscript-export.md`](../superpowers/plans/2026-04-26-torchscript-export.md).
