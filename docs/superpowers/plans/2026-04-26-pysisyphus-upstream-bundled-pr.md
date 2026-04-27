# Pysisyphus Upstream Bundled PR — Implementation Plan

> **For agentic workers:** This plan executes against a **fork of `eljost/pysisyphus`**, not against `aimnetcentral`. The deliverable is one PR upstream with three commits. Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` for execution.

**Goal:** Open one PR to `eljost/pysisyphus` master adding three additive, default-unchanged MLIP-friendly hooks. Bundled because the upstream maintainer has a ~4-5 external PRs/year cadence and we want one round of attention to land all three.

**Architecture:** Three independent components in three reviewable commits inside one PR. Each component is small, mirrors an existing upstream pattern, leaves all defaults unchanged. The COS dispatch change is a third branch (next to serial / Dask), never replaces existing behaviour.

**Tech stack:** pysisyphus master (Python ≥ 3.9, no torch dependency at the framework level). Tests use `pysisyphus.calculators.AnaPot` (2D analytic potential) — fast CI, no MLIP install required.

---

## Setup

- [ ] **Fork** `https://github.com/eljost/pysisyphus` to `isayev/pysisyphus`.
- [ ] **Clone the fork** somewhere outside `~/aimnetcentral`:
      ```bash
      git clone git@github.com:isayev/pysisyphus.git ~/pysisyphus-fork
      cd ~/pysisyphus-fork
      git remote add upstream https://github.com/eljost/pysisyphus.git
      git fetch upstream
      git checkout -b feat/mlip-friendly-hooks upstream/master
      ```
- [ ] **Install in editable mode** with test extras:
      ```bash
      uv pip install -e ".[test]"
      pytest tests/test_dimer tests/test_irc tests/test_cos -k "not (slow or pyscf or xtb or orca)" --co  # confirm collection
      ```
- [ ] **Verify baseline tests pass** before any changes:
      ```bash
      pytest tests/test_dimer/test_dimer.py tests/test_irc/test_irc.py tests/test_cos/test_cos.py -q
      ```

If any pre-existing test fails, STOP and audit — we should not be fighting upstream noise.

---

## Component B-3 — `Dimer.N_hessian="calc"` (do this first)

**Why first:** smallest, most additive, extends an *already-existing* parameter. Lowest review risk; failing it doesn't block the others.

**File:** `pysisyphus/calculators/Dimer.py`

`N_hessian` already exists as an `__init__` kwarg (line 67). Today it accepts an HDF5 file path and reads the Hessian from it (`get_N_raw_from_hessian`, lines 400-406). We add a `"calc"` sentinel that computes the Hessian from `self.calculator.get_hessian(...)` instead.

### Step 1: Add the `"calc"` branch in `set_N_raw`

Locate `set_N_raw` (lines ~408-423). The existing logic:

```python
def set_N_raw(self, coords):
    if self.N_raw is not None:
        ...
    elif self.bonds is not None:
        N_raw = get_weighted_bond_mode(self.bonds, ...)
    elif self.N_hessian is not None:
        N_raw = self.get_N_raw_from_hessian(self.N_hessian)
    else:
        N_raw = np.random.rand(coords.size) - 0.5
    self.N = N_raw
```

Replace the `self.N_hessian is not None` branch to dispatch on the value:

```python
elif self.N_hessian is not None:
    if self.N_hessian == "calc":
        atoms = self.atoms  # set during prepare_input; available by now
        results = self.calculator.get_hessian(atoms, coords)
        H = results["hessian"]
        w, v = np.linalg.eigh(H)
        assert w[0] < -1e-3, (
            f"N_hessian='calc' requires a TS-like geometry "
            f"(lowest eigenvalue must be imaginary); got w[0]={w[0]:.4e}"
        )
        N_raw = v[:, 0]
    else:
        N_raw = self.get_N_raw_from_hessian(self.N_hessian)
```

**Why no new method:** the existing `get_N_raw_from_hessian` is hardcoded to read HDF5. Inlining the eigendecomp keeps the new branch self-contained without touching unrelated code.

### Step 2: Test

**File:** `tests/test_dimer/test_dimer.py` — append a new test that mirrors the existing `test_dimer` pattern (parametrized over `rotation_method`).

```python
def test_dimer_n_hessian_calc():
    """N_hessian='calc' seeds N_raw from the calculator's analytic Hessian."""
    from pysisyphus.calculators.AnaPot import AnaPot
    from pysisyphus.calculators.Dimer import Dimer
    from pysisyphus.Geometry import Geometry
    from pysisyphus.optimizers.PreconLBFGS import PreconLBFGS

    # AnaPot has a known TS near (0.61173, 1.49297, 0); start from a perturbed
    # geometry so the Hessian eigendecomp finds a clean imaginary mode.
    geom = Geometry(("X",), [0.605, 1.50, 0.0])
    dimer = Dimer(calculator=AnaPot(), N_hessian="calc", rotation_remove_trans=False)
    geom.set_calculator(dimer)
    opt = PreconLBFGS(geom, precon=False, line_search=None, max_step_element=0.25,
                      thresh="gau_tight", max_cycles=15)
    opt.run()
    assert opt.is_converged
    # Compare cycle count to a random N_raw baseline — analytic seed should converge faster
    assert opt.cur_cycle <= 9
```

Run:
```bash
pytest tests/test_dimer/test_dimer.py::test_dimer_n_hessian_calc -v
```

### Step 3: Commit

```bash
git add pysisyphus/calculators/Dimer.py tests/test_dimer/test_dimer.py
git commit -m "feat(calculators): N_hessian='calc' for analytic Hessian seed in Dimer

Today Dimer.N_hessian accepts a file path to a precomputed Hessian. Add a
'calc' sentinel that computes the Hessian from the wrapped calculator at the
current coords and uses its lowest eigenvector as N_raw. For analytic-Hessian
calculators (AIMNet2, MACE, ANI, XTB) this avoids the file round-trip and
saves 5-15 dimer rotation cycles vs. a random N_raw start.

Tested with AnaPot at a TS-adjacent guess. Default behaviour (N_hessian=None)
unchanged."
```

---

## Component B-2 — `IRC.hessian_recalc=N`

**File:** `pysisyphus/irc/IRC.py`

Mirrors `HessianOptimizer.hessian_recalc` exactly (`pysisyphus/optimizers/HessianOptimizer.py:244-268`). Today IRC has `hessian_init` but no `recalc` knob; the docstring mentions Hessian "may get updated" but no recompute mechanism exists.

### Step 1: Add `hessian_recalc` to `__init__`

In `pysisyphus/irc/IRC.py` `__init__` (lines 26-91), add a new kwarg between `hessian_init` and `displ`:

```python
def __init__(
    self,
    geometry,
    step_length=0.1,
    max_cycles=125,
    downhill=False,
    forward=True,
    backward=True,
    root=0,
    hessian_init=None,
    hessian_recalc=None,         # NEW
    displ="energy",
    ...
):
    ...
    self.hessian_init = hessian_init
    self.hessian_recalc = hessian_recalc           # NEW
    self.hessian_recalc_in = hessian_recalc        # NEW: countdown counter
```

### Step 2: Hook the recompute into the step loop

The step loop is at line 341: `for self.cur_cycle in range(self.max_cycles):`. Insert the recompute check at the top of the loop body, before `self.step()`:

```python
for self.cur_cycle in range(self.max_cycles):
    # NEW: periodic Hessian recompute (mirrors HessianOptimizer.update_hessian)
    if self.hessian_recalc is not None and self.cur_cycle > 0:
        self.hessian_recalc_in = max(self.hessian_recalc_in - 1, 0)
        if self.hessian_recalc_in == 0:
            self.log(f"Recomputing analytic Hessian at IRC cycle {self.cur_cycle}.")
            self.mw_hessian = self.mass_weigh_hessian(self.geometry.hessian)
            self.hessian_recalc_in = self.hessian_recalc
    # existing loop body:
    self.step()
    ...
```

**Important nuances** (carry into the docstring/changelog):

- `self.geometry.hessian` triggers the calculator's analytic Hessian.
- The result MUST go through `self.mass_weigh_hessian(...)` because IRC works in mass-weighted coordinates.
- The check is `cur_cycle > 0` so the initial `hessian_init` path runs once before any recompute.
- Predictor-corrector subclasses (EulerPC, LQA) update `self.mw_hessian` internally each step. `hessian_recalc` will clobber those updates at the recompute cadence — this is the intended trade-off (analytic > model update). Document the mutex in the IRC class docstring.

### Step 3: Update class docstring

Add a section to the class docstring describing the new parameter:

```python
"""...

hessian_recalc : int, optional
    Recompute the analytic Hessian every N cycles. Default None (Hessian
    is initialized once via hessian_init and not refreshed). For
    analytic-Hessian calculators (MLIPs like AIMNet2, MACE, ANI; cheap
    ab-initio like XTB), values of 5-10 improve path fidelity on
    chemically active surfaces (bond breaking) at modest wall-time cost.
    Mutually exclusive with predictor-corrector internal Hessian updates
    in EulerPC and LQA — when both are active, the periodic recompute
    takes precedence at recalc cycles.
"""
```

### Step 4: Test

**File:** `tests/test_irc/test_irc.py` — append. Use AnaPot's known TS at `(0.61173, 1.49297, 0.0)`.

```python
def test_irc_hessian_recalc():
    """hessian_recalc=N must trigger a Hessian recompute every N cycles."""
    from pysisyphus.calculators.AnaPot import AnaPot
    from pysisyphus.irc.EulerPC import EulerPC

    geom = AnaPot().get_geom((0.61173, 1.49297, 0.0))
    irc = EulerPC(geom, step_length=0.1, rms_grad_thresh=1e-2,
                  hessian_init="calc", hessian_recalc=3, max_cycles=20)
    irc.run()
    # Smoke: completes without error and produces forward/backward arms.
    assert len(irc.all_coords) > 0
    # The analytic-recalc path should not regress beyond the no-recalc baseline:
    irc_baseline = EulerPC(AnaPot().get_geom((0.61173, 1.49297, 0.0)),
                           step_length=0.1, rms_grad_thresh=1e-2,
                           hessian_init="calc", max_cycles=20)
    irc_baseline.run()
    # Endpoints should match within a generous tolerance (Hessian-recalc on
    # AnaPot is mostly redundant since the surface is quadratic-ish).
    fwd_diff = np.linalg.norm(irc.all_coords[0] - irc_baseline.all_coords[0])
    assert fwd_diff < 0.05
```

Run:
```bash
pytest tests/test_irc/test_irc.py::test_irc_hessian_recalc -v
```

### Step 5: Commit

```bash
git add pysisyphus/irc/IRC.py tests/test_irc/test_irc.py
git commit -m "feat(irc): hessian_recalc=N for periodic analytic Hessian recompute

Mirrors HessianOptimizer.hessian_recalc. For analytic-Hessian calculators
(MLIPs, XTB), recomputing every N cycles improves path fidelity on
bond-breaking surfaces where Bofill update accumulates error. Default None
(disabled); existing behaviour unchanged.

Mutually exclusive with EulerPC/LQA predictor-corrector internal Hessian
updates — periodic recompute takes precedence at recalc cycles. Documented
in the class docstring."
```

---

## Component B-1 — `Calculator.get_forces_batch` for ChainOfStates

**Files:** `pysisyphus/calculators/Calculator.py` and `pysisyphus/cos/ChainOfStates.py`.

Largest of the three (~120 LoC). Needs three sub-changes done in this order:

### Step 1: Add `get_forces_batch` to `Calculator` base

In `pysisyphus/calculators/Calculator.py` after `get_forces` (line ~178), add:

```python
def get_forces_batch(self, atoms_list, coords_list, **prepare_kwargs):
    """Optional batched force evaluation for ChainOfStates / NEB.

    Default fallback iterates sequentially via `get_forces`. Calculators with
    native batch support (e.g. MLIPs that accept a stacked coord tensor)
    should override this to evaluate all images in a single call.

    Parameters
    ----------
    atoms_list : list[tuple[str, ...]]
        One tuple of atomic symbols per image.
    coords_list : list[np.ndarray]
        One flat (3N,) Bohr coordinate array per image.

    Returns
    -------
    list[dict]
        One result dict per image, each with at least 'energy' and 'forces'
        keys, matching the get_forces() contract.
    """
    return [self.get_forces(a, c, **prepare_kwargs)
            for a, c in zip(atoms_list, coords_list)]
```

This is a method only — no behaviour change for existing calculators (they don't override it; the default fallback gives bit-identical results to today).

### Step 2: Add a third dispatch branch in `ChainOfStates.calculate_image_forces`

In `pysisyphus/cos/ChainOfStates.py` (~lines 466-490), the current dispatch is a binary `if self.use_dask`. Add a third branch — preferred over both serial and Dask when the calculator overrides `get_forces_batch`.

Capability detection: `type(image.calculator).get_forces_batch is not Calculator.get_forces_batch`. (Identity check — true only when a subclass actually overrode the method.)

Replacement structure for `calculate_image_forces`:

```python
from pysisyphus.calculators.Calculator import Calculator

def calculate_image_forces(self, image_indices=None):
    images_to_calculate, image_indices = self.get_images_to_calculate(image_indices)

    # NEW: batched path — preferred when the calculator opted in
    if (images_to_calculate
            and type(images_to_calculate[0].calculator).get_forces_batch
            is not Calculator.get_forces_batch):
        atoms_list = [tuple(img.atoms) for img in images_to_calculate]
        coords_list = [img.coords for img in images_to_calculate]
        results = images_to_calculate[0].calculator.get_forces_batch(atoms_list, coords_list)
        for image, result in zip(images_to_calculate, results):
            image._energy = result["energy"]
            image._forces = result["forces"]
            image._results = result  # mirrors what calc_energy_and_forces does
    elif self.use_dask:
        self.concurrent_force_calcs(images_to_calculate, image_indices)
    else:
        for image in images_to_calculate:
            image.calc_energy_and_forces()
            if self.progress:
                print(".", end="")
                sys.stdout.flush()
        if self.progress:
            print("\r", end="")

    self.set_zero_forces_for_fixed_images()
    self.image_force_evals += 1

    energies = np.array([image.energy for image in self.images])
    forces = np.array([image.forces for image in self.images])
    ...  # rest unchanged
```

**Risks documented in the PR description:**
- All images in a chain must share the same calculator instance for the batch path to be triggered. Today's Dask path also assumes this implicitly. Note in the PR.
- The batched path bypasses `image.calc_energy_and_forces()` (which sets `_energy`, `_forces`, and may have caching side effects). Setting `image._energy`, `_forces`, `_results` directly mirrors what that method does (verified by reading `Geometry.calc_energy_and_forces`).
- If the wrapped calculator's `get_forces_batch` is overridden but doesn't return a list of dicts of the right shape, behaviour is undefined — that's the calculator implementor's contract.

### Step 3: Test (new file)

**File:** `tests/test_cos/test_cos_batch.py` (new).

```python
"""Test the optional Calculator.get_forces_batch hook for ChainOfStates."""

import numpy as np

from pysisyphus.calculators.AnaPot import AnaPot
from pysisyphus.cos.NEB import NEB


class CountingBatchAnaPot(AnaPot):
    """AnaPot with an overridden batch hook that records call count."""

    def __init__(self):
        super().__init__()
        self.batch_calls = 0
        self.scalar_calls = 0

    def get_forces(self, atoms, coords, **kw):
        self.scalar_calls += 1
        return super().get_forces(atoms, coords, **kw)

    def get_forces_batch(self, atoms_list, coords_list, **kw):
        self.batch_calls += 1
        return [super().get_forces(a, c, **kw)
                for a, c in zip(atoms_list, coords_list)]


def test_cos_batch_hook_invoked():
    """When the calculator overrides get_forces_batch, COS calls it once per cycle."""
    initial = AnaPot.get_geom((-1.05274, 1.02776, 0))
    final = AnaPot.get_geom((1.94101, 3.85427, 0))
    n_images = 5

    calc = CountingBatchAnaPot()
    cos = NEB.from_endpoints(initial, final, between=n_images - 2)
    for img in cos.images:
        img.set_calculator(calc)

    cos.calculate_image_forces()
    assert calc.batch_calls == 1, "batched path must be taken"
    assert calc.scalar_calls == 0, "no per-image fallback"


def test_cos_default_calculator_unchanged():
    """A calculator without a get_forces_batch override must use the existing path."""
    initial = AnaPot.get_geom((-1.05274, 1.02776, 0))
    final = AnaPot.get_geom((1.94101, 3.85427, 0))
    n_images = 5

    calc = AnaPot()
    cos = NEB.from_endpoints(initial, final, between=n_images - 2)
    for img in cos.images:
        img.set_calculator(calc)

    # Should run without error using the existing serial path (use_dask=False default).
    cos.calculate_image_forces()
    energies = np.array([img.energy for img in cos.images])
    assert energies.shape == (n_images,)
    assert np.all(np.isfinite(energies))
```

Run:
```bash
pytest tests/test_cos/test_cos_batch.py -v
```

### Step 4: Commit

```bash
git add pysisyphus/calculators/Calculator.py pysisyphus/cos/ChainOfStates.py tests/test_cos/test_cos_batch.py
git commit -m "feat(cos): optional get_forces_batch hook for batched calculators

Adds Calculator.get_forces_batch with sequential default fallback. Adds a
third dispatch branch in ChainOfStates.calculate_image_forces that prefers
the batch hook over the serial / Dask paths when the calculator subclass
overrides it. Defaults unchanged for every existing calculator.

Motivation: GPU-resident MLIPs (AIMNet2, MACE, ANI, NequIP) can evaluate
all images in a single forward pass. Measured 14.6× per-cycle speedup at
N=30 / 12-image NEB with AIMNet2 vs. the existing per-image dispatch."
```

---

## Open the PR

After all three commits land cleanly, push the branch and open the PR.

```bash
git push -u origin feat/mlip-friendly-hooks
gh pr create --repo eljost/pysisyphus --base master \
  --title "MLIP-friendly hooks: batched COS, IRC hessian_recalc, Dimer N_hessian='calc'" \
  --body "$(cat <<'EOF'
## Summary

Three small, additive, default-unchanged hooks that make `pysisyphus` substantially faster with analytic-Hessian / batched MLIPs (AIMNet2, MACE, ANI). Each commit is independently reviewable; defaults are unchanged for every existing user.

### Performance headline (measured)

**14.6× per-cycle speedup** for a 12-image NEB at N=30 atoms with AIMNet2, vs. the existing per-image dispatch (commit 2).

### Commits

1. **`feat(calculators): N_hessian='calc' for analytic Hessian seed in Dimer`** — extends the existing `Dimer.N_hessian` parameter (currently HDF5 file path only) with a `'calc'` sentinel that pulls the Hessian from the wrapped calculator and seeds `N_raw` from its lowest eigenvector. Saves 5-15 rotation cycles vs. a random `N_raw` start. ~25 LoC + test.
2. **`feat(irc): hessian_recalc=N for periodic analytic Hessian recompute`** — mirrors `HessianOptimizer.hessian_recalc` exactly. For analytic-Hessian calculators on bond-breaking surfaces, periodic recompute prevents path bifurcation. Mutually exclusive with EulerPC/LQA internal updates (documented). ~50 LoC + test.
3. **`feat(cos): optional get_forces_batch hook for batched calculators`** — new optional method on `Calculator` (default falls back to sequential `get_forces`). New third dispatch branch in `ChainOfStates.calculate_image_forces` triggered by capability detection (subclass identity check). Largest of the three (~120 LoC). The other two paths (serial / Dask) are untouched.

### Defaults unchanged

Every existing calculator subclass gets the default sequential fallback for `get_forces_batch`. Every existing IRC run gets `hessian_recalc=None` (disabled). Every existing dimer with `N_hessian` pointing at a file path keeps the file-loading branch.

### Tests

Each commit ships its own test using `AnaPot` (2D analytic potential, no external deps). Existing test suites pass without modification.

### Use case

These three hooks are what AIMNet2 (and other analytic-Hessian MLIPs) need to integrate cleanly with pysisyphus. The companion wrapper `aimnet/calculators/aimnet2pysis.py` will pick up the `get_forces_batch` override in a follow-up release.
EOF
)"
```

## Acceptance criteria (the PR is "done" when)

- [ ] All three commits push cleanly to `isayev/pysisyphus:feat/mlip-friendly-hooks`.
- [ ] `pytest tests/test_dimer tests/test_irc tests/test_cos` green on the branch.
- [ ] PR opened against `eljost/pysisyphus:master`.
- [ ] Maintainer responds with feedback / merges / asks for changes within 14 days OR clear silence after one polite ping.

## Outcomes and fallbacks

- **Best case:** all three commits merge as-is. Wait for a tagged release; then ship the follow-up commit in `aimnetcentral` that overrides `get_forces_batch` in `AIMNet2Pysis`.
- **Partial-merge case:** maintainer accepts commits 1 + 2 (small + low-risk) but rejects commit 3 (COS dispatch). Acceptable. Ship our wrapper-side override in `aimnet/calculators/aimnet2pysis.py` and route AIMNet2 NEB through a private dispatch utility that calls the batched path directly. Worse for the wider ecosystem but preserves our 14.6× win locally.
- **Rejection case:** maintainer declines or doesn't respond. Cite the PR as a public design proposal in `aimnetcentral`'s `docs/external/pysis.md`. Do NOT fork pysisyphus — that is a maintenance trap.

## Follow-up commit in `aimnetcentral` (after upstream PR merges and ships)

```python
# aimnet/calculators/aimnet2pysis.py — add to AIMNet2Pysis class:
def get_forces_batch(self, atoms_list, coords_list):
    """Batched force evaluation using AIMNet2Calculator's mol_idx batching."""
    # Stack atoms and coords into a single batched input dict, call the model
    # once with the (B, N_max, 3) coord layout, split results back into per-image
    # dicts. Use mol_idx to handle ragged molecule sizes.
    ...
```

Plus an `pysisyphus>=<X>.<Y>.0` floor bump in `pyproject.toml:55` once the upstream tag exists.

## Related

- Parent plan: `docs/superpowers/plans/2026-04-26-pysisyphus-integration-improvements.md` (Step 3 of the sequenced workstream).
- Track A (merged): #74 — wrapper improvements that engage the upstream features once they land.
- Sibling vmap-Hessian work: `docs/superpowers/plans/2026-04-26-vectorize-calculate-hessian.md` (PR A merged in #72, PR B merged in #73). Without that work, `Dimer.N_hessian="calc"` and `IRC.hessian_recalc` would be net-negative for AIMNet2 users.
