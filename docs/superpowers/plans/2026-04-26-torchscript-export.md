# TorchScript export pathway for external MD engines

**Status:** Open / parked. Created 2026-04-26 from issue #19 (GROMACS) follow-up.

## Goal

Make AIMNet2 models exportable as a single self-contained TorchScript `.pt`
file usable by:

- **GROMACS NNPot** (`nnpot-modelfile`, GROMACS >= 2025.0)
- **LAMMPS** (`pair_pytorch` plugin)
- Any other engine that loads via `torch.jit.load`

Note: OpenMM is **not** in this list. `openmm-ml` already ships an
`AIMNet2PotentialImpl` that wraps `AIMNet2Calculator` behind
`openmm.PythonForce`, sidestepping TorchScript entirely. A TorchScript
path would still benefit OpenMM by removing the per-step Python callback
overhead, but it is not a prerequisite for OpenMM support.

Today: **none of the shipped `aimnet/calculators/assets/*.pt` files are
loadable as TorchScript.** They are `torch.save` state-dict archives, not
TorchScript archives.

## Concrete upstream blockers

1. **`aimnet/nbops.py:51`** uses `tensor.data_ptr()` as a cache key for
   neighbor-list reuse. `torch.jit.script` errors on this:
   ```
   'Tensor' object has no attribute or method 'data_ptr'.
   ```
   Fix options:
   - Replace `data_ptr` cache key with the `id(tensor)` Python builtin
     (forbidden in script too) -- not viable.
   - Drop the cache from the script path entirely; only use it when running
     in eager Python. Refactor so the script path recomputes neighbor
     pieces unconditionally.
   - Pass cache-bust hints in via the input dict (e.g. an explicit
     `data["_recompute_nb"]` bool) and do membership checks on a dict key.

2. **`aimnet/modules/lr.py` DFTD3 autograd** -- the `DFTD3` class
   (`aimnet/modules/lr.py:356`) calls `torch.autograd.grad` at line 596
   with a signature TorchScript cannot match. Verified runtime error:
   `Expected a value of type 'List[Tensor]' for argument 'outputs' but
   instead found type 'Tensor'.`. Fix options:
   - Wrap in a proper `torch.autograd.Function` with explicit `forward` /
     `backward` static methods (TorchScript supports these).
   - Split into a script-only forward path + a Python-only training path,
     since GROMACS only needs forces via autograd of energy w.r.t.
     positions.
   - Note: `aimnet/modules/ops.py` contains a separate `_DFTD3Function`
     wrapping `nvalchemiops` -- that one is unrelated to the script
     failure.

3. **Bundled-export wrapper** -- once 1 and 2 are fixed, wrap
   `(model, external_coulomb, external_dftd3)` together so a single
   scripted module reproduces full `AIMNet2Calculator.eval()` energies
   without `nvalchemiops` (which is NVIDIA-only and not jit-able anyway).
   This means writing a TorchScript-friendly all-pairs / cutoff neighbor
   list inside the wrapper.

## Already done (parked)

- `aimnet/interfaces/__init__.py` -- new package, no public exports until
  the wrapper actually works.
- `aimnet/interfaces/gromacs.py` -- starter wrapper with the GROMACS NNPot
  forward signature, unit conversions, and an all-pairs neighbor-list
  construction. Verified to `jit.script` cleanly with a *dummy* inner
  ScriptModule, save+reload round-trip OK, autograd-derived forces OK.
  `build_gromacs_nnpot_model("aimnet2")` correctly raises for every shipped
  model today (model isn't a ScriptModule + has external LR).

## Per-engine status

| Engine | Needs TorchScript? | Status |
|---|---|---|
| GROMACS NNPot | yes | Blocked (this plan) |
| LAMMPS `pair_style mliap` (mliappy) | yes | Blocked (this plan) |
| OpenMM (openmm-ml) | no -- uses `openmm.PythonForce` with a Python callback | **Already supported upstream** via `MLPotential('aimnet2')`. A TorchScript path would still cut per-step Python overhead. |
| AMBER `sander` / `pmemd` (`torchani-amber`) | yes (jit-compiled `.pt`), but with a different harness | **Supported upstream** via the [`torchani-amber`](https://github.com/roitberg-group/torchani-amber) integration, which ships pre-built support for AimNet2 (and Nutmeg). Requires recompiling AmberTools 25/26 with the integration; not pip-installable. Models are jit-compiled `.pt` files passed as `model_type` -- the same TorchScript-export work in this plan unblocks shipping a self-contained AIMNet2 jit asset for that path. |

## Suggested ordering

1. Refactor `nbops.py` to remove `data_ptr` from the script path.
2. Refactor `DFTD3` autograd into `torch.autograd.Function`.
3. Land the bundled `GromacsNNPotWrapper` and an export CLI
   (`examples/export_gromacs.py`, removed from this branch).
4. Land docs/external/gromacs.md as a real "how to use" page (not a
   blocker page).
5. LAMMPS pair_pytorch then follows with a thin per-engine wrapper
   around the same scripted core. OpenMM gets an optional TorchScript
   fast-path that replaces the existing `PythonForce` callback in
   `openmm-ml`.

## Out of scope here

- PBC support inside the scripted wrapper (nvalchemiops neighbor lists
  are NVIDIA-only and not script-able). PBC AIMNet2 stays on the Python
  ASE path.
- NSE / open-shell models. Separate model family, separate wrapper later.
