# GROMACS (NNPot)

**Status: not currently supported.**

GROMACS 2025.0 introduced the [NNPot](https://manual.gromacs.org/nightly/reference-manual/special/nnpot.html) interface, which loads a TorchScript `.pt` file via the `nnpot-modelfile` mdp option for QM-region energies and forces in QM/MM simulations. The 2026 series expanded model compatibility.

AIMNet2 cannot currently produce a `.pt` that the NNPot interface accepts. The reasons are upstream-internal, not a packaging gap:

1. The shipped v2 `.pt` assets in `aimnet/calculators/assets/` are `torch.save` state-dict archives, **not** TorchScript archives. They are loaded into a Python `nn.Module` by `aimnet.models.base.load_model`.
2. `torch.jit.script` on the in-memory `AIMNet2` model fails in `aimnet/nbops.py` -- the code uses `tensor.data_ptr()` as a neighbor-cache key, which TorchScript rejects.
3. The external DFTD3/Coulomb modules call nvalchemiops Python APIs that are not TorchScript export targets.
4. The published `aimnet2` (wb97m-d3) model needs both external Coulomb and external D3 added on top of the core, so even if the core scripted, the wrapper would have to bundle all three pieces to match `AIMNet2Calculator` energies.

## Tracking

A starter wrapper is parked at `aimnet/interfaces/gromacs.py` -- the forward signature, unit conversions (nm -> A on input; eV -> kJ/mol on output), and pure-PyTorch all-pairs neighbor list have been verified to `torch.jit.script` cleanly with a dummy inner model and to round-trip via `jit.save` / `jit.load`. It will become functional once the blockers above are resolved.

The remediation plan is tracked internally with the TorchScript export work.

## What works today instead

For QM/MM-style workflows that do not require GROMACS specifically:

- `AIMNet2ASE` + ASE's MD drivers (Langevin, NVT, NVE) for pure ML trajectories, with [pysisyphus](pysis.md) for path following.
- [OpenMM via openmm-ml](openmm.md) -- pip-installable, runs the AIMNet2 calculator behind `openmm.PythonForce` (no TorchScript needed); supports periodic systems and `createMixedSystem` for QM/MM.
- [AMBER via torchani-amber](amber.md) -- compiled-in C++/Fortran integration for `sander` and `pmemd`; supports both full-ML and ML/MM modes via the `&extpot` and `&qmmm` mdin namelists.

A future GROMACS wrapper must include AIMNet2's D3 dispersion (the embedded D3 in the wb97m-d3 model is part of the published level of theory). A no-dispersion shortcut would silently shift conformer rankings, intermolecular binding, and torsion barriers by 1-10 kcal/mol vs the published numbers.
