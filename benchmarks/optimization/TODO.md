# GPU Throughput Follow-Ups

## Archive Decision

This branch is archived as a non-go production path for moving-geometry neighbor reuse. The implemented
`neighbor_skin` prototype is numerically correct in the tested non-PBC CUDA cases, but L40S benchmarks show only
about 5-7% wall-time improvement at 5k atoms because DFTD3/model work dominates once neighbor-list rebuilds are
removed. Do not promote this as a default or headline optimization without a larger end-to-end win.

Keep the useful result: exact same-geometry caching remains valuable for property grids where the geometry is fixed
and charge/multiplicity changes between evaluations.

## Same-Geometry Charge/Multiplicity Property Grids

Use case: AIMNet2-NSE and constrained-DFT-style reactivity descriptors often evaluate the same geometry many times
while varying total charge and/or spin multiplicity. This is a strong fit for `cache_static` because neighbor matrices
and external DFTD3 terms are geometry-dependent, while the neural network must still be recomputed for each
`(charge, mult)` state.

Follow-up tasks:

- Add a benchmark workload for same-geometry charge/multiplicity sweeps, including AIMNet2-NSE when available.
- Add examples/docs showing `cache_static=True` for descriptor grids such as neutral/cation/anion and
  singlet/doublet/triplet evaluations.
- Add regression tests proving cached geometry work is reused while charge/multiplicity-dependent model outputs are
  recomputed.
- Consider a small convenience helper for property grids only if it removes boilerplate without hiding the explicit
  `(charge, mult)` states.

Acceptance target:

- Meaningful speedup on large fixed geometries with multiple charge/multiplicity states, with identical
  energies/charges/forces to uncached execution for each state.

## Geometry Optimization And MD

Low-priority micro-optimizations should be ignored unless they compound with a larger win. The main target is reusable
geometry topology, not exact static DFTD3 reuse.

Implemented first low-risk step:

- `AIMNet2ASE(..., compute_forces_for_energy=True)` computes forces during energy requests, so a follow-up force request
  at the same geometry is served from ASE's normal results cache. This is opt-in because it slows true energy-only
  screening.
- `AIMNet2Calculator(..., neighbor_skin=...)` adds opt-in Verlet-style sparse neighbor-list reuse for large non-PBC CUDA
  systems. It rebuilds with `cutoff + skin` and reuses while all atoms stay within `skin / 2` of the reference geometry.
  The neural network and external physics terms still run every step.

Candidate work:

- Validate `neighbor_skin` on longer ASE geometry optimization and NVE trajectories; tune recommended skin values by
  observed rebuild frequency and force/energy stability.
- Persistent GPU-resident calculator state for optimizers and MD drivers.
- Exact last-result cache for duplicate energy/force requests at the same geometry.
- CUDA graph replay between neighbor-list rebuilds if tensor shapes remain fixed.

Acceptance target:

- At least 20-30% wall-time improvement on representative large-system MD or geometry optimization, with unchanged
  forces and stable NVE energy drift versus rebuild-every-step baselines.

## Hessian

Hessian speedups should avoid detached force reuse. The useful cache target is fixed or conservative neighbor topology.

Candidate work:

- Finite-difference Hessian mode using one cutoff-plus-displacement-margin neighbor superset across all displaced force
  evaluations.
- HVP-first workflows for large systems instead of dense Hessian construction.
- Active-region Hessian workflows with frozen environment topology.

Acceptance target:

- Large reduction in neighbor-list and DFTD3 overhead for finite-difference/HVP workflows without changing the
  mathematical Hessian for the selected approximation.
