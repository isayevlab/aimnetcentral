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

(File:line references current as of commit 9b89166; symbol references
are intended to survive line drift.)

1. **The neighbor-cache `data_ptr()` call in `aimnet/nbops.py`** uses
   `tensor.data_ptr()` as a cache key for neighbor-list reuse.
   `torch.jit.script` errors on this with:
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

2. **The `DFTD3` external module's autograd in `aimnet/modules/lr.py`**
   calls `torch.autograd.grad` with a signature TorchScript cannot match.
   Verified runtime error:
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

- `aimnet/interfaces/__init__.py` -- new namespace; no public exports.
- `aimnet/interfaces/gromacs.py` -- starter `AIMNet2Gromacs` class with the
  GROMACS NNPot forward signature, unit conversions, and all-pairs
  neighbor-list construction. Verified to `jit.script` cleanly with a
  *dummy* inner ScriptModule, save+reload round-trip OK, autograd-derived
  forces OK. `build_gromacs_nnpot_model()` is a stub that raises
  `NotImplementedError` unconditionally; the class definition is preserved
  as a starting point.

## Per-engine status

See [`docs/external/`](../../external/index.md) for the canonical
per-engine status. Summary in this plan: GROMACS NNPot and LAMMPS
`pair_style mliap` are blocked here. OpenMM is supported today via
`openmm-ml` (Python callback, no TorchScript). AMBER is supported today
via `torchani-amber` (compiled-in C++/Fortran, accepts a jit `.pt` via
`model_type`). Resolving this plan also enables shipping a
self-contained jit AIMNet2 for AMBER and an optional TorchScript
fast-path for OpenMM.

## Suggested ordering

1. Refactor `nbops.py` to remove `data_ptr` from the script path.
2. Refactor `DFTD3` autograd into `torch.autograd.Function`.
3. Promote `aimnet/interfaces/gromacs.py` from parked to functional
   (drop the `NotImplementedError` stub, add tests, ship a CLI driver
   under `examples/`).
4. Rewrite `docs/external/gromacs.md` as a real "how to use" page.
5. Surface a documented "external input contract" on the calculator
   side (e.g. `AIMNet2Calculator.build_input_dict(coord, numbers, charge)`)
   so future export wrappers depend on a documented surface rather than
   reaching into private helpers like `_add_padding_row`.
6. LAMMPS `pair_style mliap` then follows with a thin per-engine wrapper.
   OpenMM gets an optional TorchScript fast-path that replaces the
   existing `PythonForce` callback in `openmm-ml`.
7. Once a second TorchScript export wrapper exists, factor out a shared
   base (unit conversions + neighbor-list builder + padding-row
   construction) under `aimnet/interfaces/_torchscript_base.py` rather
   than duplicating across wrappers.

## Out of scope here

- PBC support inside the scripted wrapper (nvalchemiops neighbor lists
  are NVIDIA-only and not script-able). PBC AIMNet2 stays on the Python
  ASE path.
- NSE / open-shell models. Separate model family, separate wrapper later.
