# Pysisyphus Integration Improvement Plan

> **Status:** Plan only. Single sequenced workstream. Upstream pysisyphus is poorly maintained (~4-5 external PRs/year), so we deliberately ship **one** upstream PR — the highest-leverage one — instead of spreading bets.

## Goal

Bring `AIMNet2Pysis` to the same quality bar as the Sella integration (#71) and add the single upstream feature that makes AIMNet2 NEB substantially faster — without shipping anything that regresses today's users.

## Sequencing

**Step 1 (precondition).** Land **PR B of the vmap-Hessian plan** (`docs/superpowers/plans/2026-04-26-vectorize-calculate-hessian.md`). Drops `AIMNet2Calculator.calculate_hessian` from the row-wise Python loop (~1,211 ms at N=24) to `torch.func.vmap`-over-vjp (~50 ms target). Without this, every `hessian_init=calc` recommendation in this plan is a regression for users.

**Step 2.** Ship **all of Track A below** in `aimnetcentral`. Six small, independent commits.

**Step 3.** Open **one** upstream PR to `eljost/pysisyphus`: a batched-evaluation hook for ChainOfStates / NEB. Measured 14.6× per-cycle speedup at N=30 / 12 images — the largest single win available, and the only one that needs upstream code (everything else has either a wrapper-side workaround, deferred value, or marginal benefit).

---

## Track A — `aimnetcentral` changes (after Step 1 lands)

Six commits, ~1 day total. None depend on each other.

| # | Change | File | Notes |
|---|---|---|---|
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

## Track B — single upstream PR to `eljost/pysisyphus`

### B1 — `Calculator.get_forces_batch` for ChainOfStates / NEB

**Files:**
- `pysisyphus/cos/ChainOfStates.py` (per-image dispatch site)
- `pysisyphus/calculators/Calculator.py` (optional method on the base; default = sequential fallback)

**Why this PR, and only this PR:**

| Candidate | Measured / estimated | LoC | AIMNet2-side workaround? | Decision |
|---|---|---|---|---|
| Batched NEB hook | **14.6× per cycle** at N=30 / 12 images | ~120 | None — COS dispatches per-image at the framework level | **Submit** |
| Entry-points calculator registry (closes #180) | Quality-of-life only | ~20 | Yes — our `run_pysis()` shim ships and works | Drop |
| `IRC.hessian_recalc=N` | Path fidelity, not wall time | ~50 | None, but recommend in a wrapper-side YAML config block once accepted | Drop |
| `Dimer.N_hessian="calc"` | ~5-15 dimer rotation cycles saved (~150-450 ms) | ~25 | None, but limited audience | Drop |

The maintainer's external-PR cadence is ~4-5/year. Splitting attention across multiple asks risks none landing. The 14.6× NEB win is the only one with no AIMNet2-side workaround AND a defensible-by-the-numbers performance case.

**Sketch:**

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

**Test (one):** Use a synthetic calculator that records call count. Run a 4-image COS with the synthetic; assert `get_forces_batch` invoked once per cycle (not 4 sequential `get_forces`).

**PR description:** Lead with the **14.6× measured number**. Cite AIMNet2 as the use case. Note that defaults are unchanged — every existing calculator still uses the per-image dispatch path. Keep the body short — this maintainer prefers concrete + minimal.

**`aimnet2pysis.py` change once the PR merges and ships:** override `get_forces_batch` to use `AIMNet2Calculator`'s existing batched `(B, N, 3)` coord path via `mol_idx`. ~15 LoC follow-up commit. Not part of this plan; opens after the upstream PR is in a tagged release.

**Risk:** B1 is the largest of the candidates and touches the recently-refactored COS path. Maintainer may push back on the COS dispatch change. Fallback if rejected: drop the COS-side change, keep only the base-class `get_forces_batch` method as documentation; ship our wrapper-side override anyway and route AIMNet2 NEB through a private utility that calls the batched path directly. That's worse for the rest of the ecosystem but preserves our 14.6× win locally.

---

## Out of scope (and why)

- **`hessian_init=calc` default override in `AIMNet2Pysis.__init__`.** Tempting alternative to A5 documentation — but pysisyphus reads `hessian_init` from the YAML, not our class. Setting it programmatically requires either monkey-patching pysisyphus internals or a `**kwargs`-passthrough that doesn't currently exist. Documentation is the simplest correct surface.
- **Entry-points calculator registry / `IRC.hessian_recalc` / `Dimer.N_hessian="calc"`.** Each is a real upstream improvement, but the user constraint is "one upstream PR" given the project's maintenance pace. See the comparison table above. If the batched-NEB PR lands cleanly and the maintainer is responsive, revisit *one* of these as a follow-up.
- **i-PI bridge.** v2 protocol does not transport Hessians, defeating the analytic-Hessian advantage that motivates this work. Considered, rejected.
- **`FakeASE`-style two-way bridge.** Wrapper-of-wrapper. Sella's win taught us direct `Calculator` subclasses are simpler.
- **Console-script removal / `run_pysis()` deprecation.** Documented public interface; no upstream replacement is being pursued, so the shim stays.

## Acceptance criteria

- **Step 1 (precondition):** `AIMNet2Calculator.calculate_hessian` swapped to `torch.func.vmap`-over-vjp; `pytest tests/test_calculator.py -v` green; benchmark on caffeine N=24 GPU shows ≥10× speedup vs. the loop (target: 20-25×).
- **Step 2 (Track A):** Six commits land in `aimnetcentral` `main`; `pytest tests/test_pysis.py -m pysis` green with `pysisyphus` installed; `pytest tests/test_model_registry.py tests/test_hf_hub.py` green (CLAUDE.md gate); `mkdocs serve` clean.
- **Step 3 (upstream B1):** PR open at `eljost/pysisyphus` with the 14.6× measurement in the description; one round of maintainer feedback received.

## Related

- Sibling Sella integration: #71, plan at `docs/superpowers/plans/2026-04-26-sella-integration.md`.
- **Step-1 prerequisite plan:** `docs/superpowers/plans/2026-04-26-vectorize-calculate-hessian.md` (PR A merged in #72; **PR B pending** — must land before Track A here).
- Wrapper source: `aimnet/calculators/aimnet2pysis.py` (80 LoC).
- Upstream `Calculator` base: `pysisyphus/calculators/Calculator.py` (890 LoC, 2024-08-30); `ChainOfStates`: `pysisyphus/cos/ChainOfStates.py` (refactored 2024-06-26).
