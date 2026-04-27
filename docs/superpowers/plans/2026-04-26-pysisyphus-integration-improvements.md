# Pysisyphus Integration Improvement Plan

> **Status:** Plan only. Single sequenced workstream. Upstream pysisyphus is poorly maintained (~4-5 external PRs/year), so we deliberately ship **one** upstream PR — the highest-leverage one — instead of spreading bets.

## Goal

Bring `AIMNet2Pysis` to the same quality bar as the Sella integration (#71) and add the single upstream feature that makes AIMNet2 NEB substantially faster — without shipping anything that regresses today's users.

## Sequencing

**Step 1 (precondition).** Land **PR B of the vmap-Hessian plan** (`docs/superpowers/plans/2026-04-26-vectorize-calculate-hessian.md`). Drops `AIMNet2Calculator.calculate_hessian` from the row-wise Python loop (~1,211 ms at N=24) to `torch.func.vmap`-over-vjp (~50 ms target). Without this, every `hessian_init=calc` recommendation in this plan is a regression for users.

**Step 2.** Ship **all of Track A below** in `aimnetcentral`. Six small, independent commits.

**Step 3.** Open **one** upstream PR to `eljost/pysisyphus` bundling **three perf-themed MLIP hooks** under a single coherent narrative ("MLIP-friendly performance hooks for analytic-Hessian and batched calculators"). All three pieces are additive, default-unchanged, and small individually; bundling maximizes the value extracted from one round of maintainer attention. Drop the entry-points proposal (B1 in earlier drafts) — it's the only candidate that's not perf-themed AND has a wrapper-side workaround already shipping (`run_pysis`).

---

## Track A — `aimnetcentral` changes (after Step 1 lands)

Six commits, ~1 day total. None depend on each other.

| # | Change | File | Notes |
| --- | --- | --- | --- |
| A1 | Delete dead `implemented_properties` ClassVar (and `from typing import ClassVar`). Pre-flight: `grep -r implemented_properties` to confirm no reflective consumers; if any, leave an empty list instead. | `aimnet/calculators/aimnet2pysis.py:19` | Hygiene; pysisyphus's `Calculator` ignores it. |
| A2 | **Result-level cache** keyed on `coord.tobytes()` so the `get_energy → get_forces` double-call pattern (AFIR / some IRC paths) doesn't run the model twice. Cache entry holds the full `results` dict from a `forces=True` call; `get_energy` serves from it; invalidate on coord change. | `aimnet/calculators/aimnet2pysis.py:55-74` | Saves one forward (~6.9 ms = 32% per pair). The "input-tensor cache" idea from earlier drafts is YAGNI — measured 0.48% win at N=24, not worth the state. |
| A3 | CPU-side coord cast + `BOHR2ANG` premultiply, then `.to(device, non_blocking=True)`. Pattern: `torch.from_numpy(np.asarray(coord, dtype=np.float32) * BOHR2ANG).view(-1, 3).to(device, non_blocking=True)`. | `aimnet/calculators/aimnet2pysis.py:33` | Halves H2D bandwidth, eliminates one GPU kernel launch (~20 µs/call). Zero correctness risk: `as_tensor` already copies on dtype mismatch. |
| A4 | Add `pysisyphus>=1.0.0` floor to the `pysis` extra (matches Sella's `>=2.4.0` discipline; required by A5's YAML keys). | `pyproject.toml:54-55` | Closes contract gap. |
| A5 | Inline YAML examples in `docs/external/pysis.md` for **geom-opt** (with `opt: { hessian_init: calc, hessian_recalc: 5 }`) and **TS** (with `tsopt: { type: rsprfo, hessian_init: calc, hessian_recalc: 3 }`). Add a `force_num_hess: true` paragraph for FD validation. | `docs/external/pysis.md` | Now safe to recommend `hessian_init: calc` because Step 1 made the analytic Hessian cheap. Single source of truth — no `examples/*.yaml` files to bitrot against the doc. |
| A6 | One smoke per recommended config: water → geom-opt convergence; HCN → CNH TS via rsprfo. Module-level `pytestmark = pytest.mark.pysis` + `pytest.importorskip("pysisyphus")` matching `tests/test_sella.py`. | `tests/test_pysis.py` (new) | Regression guard for the YAML configs A5 ships. The IRC smoke is skipped — `hessian_recalc` for IRC requires upstream pysisyphus support that this plan deliberately does not pursue. |

**Explicitly NOT changing:**

- The `aimnet2pysis` console script (`pyproject.toml:79`) and `run_pysis()` shim (`aimnet/calculators/aimnet2pysis.py:77-79`). Documented public interface; users follow `docs/advanced/reaction_paths.md`. Removal would be a breaking change with no upstream replacement (we're explicitly not pursuing the entry-points PR).
- `_prepare_input` argument signature. Any internal refactor stays behind the existing call surface so pysisyphus's optimizer code paths see the same wrapper.

---

## Track B — single bundled upstream PR to `eljost/pysisyphus`

**Title:** "MLIP-friendly hooks: batched ChainOfStates, IRC `hessian_recalc`, Dimer analytic-Hessian seed"

**Three components, four files, ~195 LoC total, three independent commits inside one PR:**

| Component | File | LoC | Measured / estimated win | Why bundled |
| --- | --- | --- | --- | --- |
| **B-1: batched COS** | `pysisyphus/calculators/Calculator.py` (base method) + `pysisyphus/cos/ChainOfStates.py` (dispatch site) | ~120 | **14.6× per NEB cycle** at N=30 / 12 images | Headline win; no AIMNet2-side workaround |
| **B-2: IRC `hessian_recalc=N`** | `pysisyphus/irc/IRC.py` | ~50 | Path fidelity (prevents bifurcation on bond-breaking surfaces); ~2 s wall cost per 200-step IRC at `recalc=5` after Step 1 | Mirrors existing `HessianOptimizer.hessian_recalc` exactly |
| **B-3: Dimer `N_hessian="calc"`** | `pysisyphus/calculators/Dimer.py` | ~25 | ~150-450 ms saved per saddle search (after Step 1's vmap-Hessian lands) | Tiny additive change; same MLIP-friendly theme |

**Components dropped from earlier drafts:**

| Dropped | Why |
| --- | --- |
| Entry-points calculator registry (closes #180) | Different theme (UX, not perf); our `run_pysis()` shim already works; bundling it dilutes the perf narrative |

### Why bundle (instead of one-PR-per-feature)

- Maintainer external-PR cadence is ~4-5/year. Splitting into three PRs raises the chance that two of them sit indefinitely. One bundled PR with a coherent theme either lands all three or bounces all three with single feedback round.
- All three components are additive, default-unchanged, and individually small. The largest single risk is B-1's COS dispatch change.
- Single CI run, single review pass, single PR description. Lower maintainer cognitive load per item.

### Commit structure inside the PR

Three commits, reviewable independently:

1. `feat(calculator): optional get_forces_batch hook for batched evaluation` — B-1, base class only (no COS change yet). Default `get_forces_batch` falls back to sequential `get_forces`.
2. `feat(cos): use batched calculator hook when available` — B-1, the dispatch-site change. Behind capability detection; existing calculators unaffected.
3. `feat(tsoptimizers): IRC hessian_recalc + Dimer N_hessian="calc"` — B-2 + B-3 combined. Both mirror existing patterns (`HessianOptimizer.hessian_recalc` and `Dimer.N_raw` precomputed-from-file path) and total ~75 LoC.

If the maintainer pushes back on commit 2 (COS dispatch), commits 1, 3 can be cherry-picked alone.

### Sketches

**B-1 — `Calculator.get_forces_batch`:**

```python
# pysisyphus/calculators/Calculator.py — optional method on the base
def get_forces_batch(self, atoms_list, coords_list, **prepare_kwargs):
    """Optional batched hook. Default: sequential fallback."""
    return [self.get_forces(a, c, **prepare_kwargs) for a, c in zip(atoms_list, coords_list)]

# pysisyphus/cos/ChainOfStates.py — prefer batched when overridden
def calculate_forces_chain(self, ...):
    if type(self.calculator).get_forces_batch is not Calculator.get_forces_batch:
        results = self.calculator.get_forces_batch(atoms_list, coords_list)
    else:
        # existing per-image dispatch path (Dask or sequential)
        ...
```

**B-2 — `IRC.hessian_recalc`:**

```python
# pysisyphus/irc/IRC.py
def __init__(self, ..., hessian_recalc: int | None = None, ...):
    self.hessian_recalc = hessian_recalc
    ...

# inside the IRC step loop:
if self.hessian_recalc and self.cur_cycle and (self.cur_cycle % self.hessian_recalc == 0):
    self.mw_hessian = self.geometry.mw_hessian  # triggers calc.get_hessian
```

**B-3 — `Dimer.N_hessian="calc"`:**

```python
# pysisyphus/calculators/Dimer.py — in __init__, after parsing N_raw source
if N_hessian == "calc":
    H = geometry.cart_hessian
    eigvals, eigvecs = np.linalg.eigh(H)
    N_raw = eigvecs[:, 0]  # lowest mode
```

### Tests (one per component)

- **B-1:** synthetic calculator records call count; 4-image COS asserts one batched call per cycle (not 4 sequential).
- **B-2:** IRC on bundled HCN test fixture with `hessian_recalc=5` runs to completion; energy profile within tolerance of the no-recalc baseline (path-fidelity assertion is qualitative — the test confirms the hook doesn't break correctness).
- **B-3:** synthetic analytic-Hessian fake calc; dimer with `N_hessian="calc"` converges in fewer rotation cycles than `N_hessian=None` (random) on the same surface. Quantitative.

### PR description structure

- **Lead with the 14.6× NEB measurement** — concrete number first.
- One paragraph per component, framed as "MLIP-friendly hooks." Cite AIMNet2 as the use case for all three.
- All defaults unchanged statement.
- Note that B-2 mirrors `HessianOptimizer.hessian_recalc` and B-3 mirrors the existing `Dimer.N_raw` file-load path (precedent in their own codebase).
- Total LoC: ~195 across 4 files; smaller than several recent merged PRs (e.g., #265 DMA was ~600).
- Keep the body tight — this maintainer prefers concrete + minimal.

### Follow-up `aimnet2pysis.py` changes once the PR ships in a release

- Override `get_forces_batch` in `AIMNet2Pysis` to use `AIMNet2Calculator`'s existing batched `(B, N, 3)` coord path via `mol_idx`. ~15 LoC.
- Add `irc: { hessian_recalc: 5 }` and `dimer: { N_hessian: calc }` to the YAML examples in `docs/external/pysis.md` (extends Track A item A5).

Not part of this plan; opens after the upstream PR is in a tagged pysisyphus release.

### Risk and fallback

- **Highest risk: B-1's COS dispatch change.** Touches the recently-refactored (2024-06-26) chain-of-states code. Maintainer may push back.
- **Fallback if maintainer rejects B-1's COS change but accepts the base-class method:** keep commit 1 and commits 3 (B-2 + B-3); drop commit 2. Ship our wrapper-side override and route AIMNet2 NEB through a private dispatch utility that calls the batched path directly. Worse for the wider ecosystem but preserves our 14.6× win locally.
- **Fallback if maintainer rejects everything:** the work is still useful as a public design proposal documenting what MLIP-friendly hooks should look like; cite from our docs as "what we'd ship if upstream were responsive." Don't fork pysisyphus — that's a maintenance trap.

---

## Out of scope (and why)

- **`hessian_init=calc` default override in `AIMNet2Pysis.__init__`.** Tempting alternative to A5 documentation — but pysisyphus reads `hessian_init` from the YAML, not our class. Setting it programmatically requires either monkey-patching pysisyphus internals or a `**kwargs`-passthrough that doesn't currently exist. Documentation is the simplest correct surface.
- **Entry-points calculator registry (closes upstream issue #180).** Different theme (UX/plugins, not perf); our `run_pysis()` shim already works. Including it in the bundled PR would dilute the perf narrative. Revisit only if the bundled PR lands cleanly AND the maintainer engages on follow-ups.
- **A `has_analytic_hessian` flag + auto-`hessian_init=calc` default override in `HessianOptimizer`.** A higher-merge-risk "sticky default" change that touches a heavily-used optimizer base. Excluded from the bundle to keep review surface bounded.
- **i-PI bridge.** v2 protocol does not transport Hessians, defeating the analytic-Hessian advantage that motivates this work. Considered, rejected.
- **`FakeASE`-style two-way bridge.** Wrapper-of-wrapper. Sella's win taught us direct `Calculator` subclasses are simpler.
- **Console-script removal / `run_pysis()` deprecation.** Documented public interface; no upstream replacement is being pursued, so the shim stays.

## Acceptance criteria

- **Step 1 (precondition):** `AIMNet2Calculator.calculate_hessian` swapped to `torch.func.vmap`-over-vjp; `pytest tests/test_calculator.py -v` green; benchmark on caffeine N=24 GPU shows ≥10× speedup vs. the loop (target: 20-25×).
- **Step 2 (Track A):** Six commits land in `aimnetcentral` `main`; `pytest tests/test_pysis.py -m pysis` green with `pysisyphus` installed; `pytest tests/test_model_registry.py tests/test_hf_hub.py` green (CLAUDE.md gate); `mkdocs serve` clean.
- **Step 3 (upstream bundled PR):** Single PR open at `eljost/pysisyphus` with three commits (B-1 batched COS, B-2 IRC `hessian_recalc`, B-3 Dimer `N_hessian="calc"`); 14.6× measurement headlined in the description; one round of maintainer feedback received. Acceptable outcomes: full merge / partial merge with the COS dispatch change rejected (commits 1 + 3 land) / outright rejection (cite the design as documentation, do not fork).

## Related

- Sibling Sella integration: #71, plan at `docs/superpowers/plans/2026-04-26-sella-integration.md`.
- **Step-1 prerequisite plan:** `docs/superpowers/plans/2026-04-26-vectorize-calculate-hessian.md` (PR A merged in #72; **PR B pending** — must land before Track A here).
- Wrapper source: `aimnet/calculators/aimnet2pysis.py` (80 LoC).
- Upstream `Calculator` base: `pysisyphus/calculators/Calculator.py` (890 LoC, 2024-08-30); `ChainOfStates`: `pysisyphus/cos/ChainOfStates.py` (refactored 2024-06-26).
