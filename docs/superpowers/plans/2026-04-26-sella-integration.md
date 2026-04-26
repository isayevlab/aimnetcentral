# Sella Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add first-class Sella saddle-point optimizer support to AIMNet2 by exposing analytic Hessians through the existing `AIMNet2ASE` calculator and shipping the optional extra, smoke test, example, and docs.

**Architecture:** Sella is an ASE-native optimizer; no wrapper class is required. The single load-bearing change is adding `get_hessian(atoms=None) -> ndarray (3N, 3N)` to `AIMNet2ASE` so users can pass `hessian_function=atoms.calc.get_hessian` to `Sella(...)`. This bypasses Sella's expensive Davidson finite-difference loop (10–30 extra gradient calls per refinement) and was validated by Schreiner et al. (Nature Comms 2024) to cut step count by 2–3×. Surrounding tasks add the optional dependency, pytest marker, smoke test, example script, and docs page that mirror the existing `pysisyphus` integration footprint.

**Tech Stack:** Python 3.11+, PyTorch (existing AIMNet2 autograd Hessian path at `aimnet/calculators/calculator.py:1135-1142`), ASE ≥ 3.27, Sella ≥ 2.4.0 (required — v2.4.0 March 2026 added MLIP-targeted vectorization that gave ~22× wall-clock improvement on a 50-atom test case).

**Out of scope (explicitly deferred):**
- Vectorizing `AIMNet2Calculator.calculate_hessian` with `torch.func.hessian` — orthogonal optimization that benefits both Sella and pysisyphus; track as a separate plan once we have a Sella-driven benchmark.
- Wrapping the energy-only path in `torch.inference_mode()` — micro-optimization, not on Sella's hot path (Sella always asks for forces).
- A batched many-Sella driver — workflow-level project, not a Sella code change.
- PBC TS searches — Sella's `internal=True` path assumes molecular topology; gas-phase only for v1.

---

## File Structure

| File | Status | Responsibility |
|------|--------|----------------|
| `pyproject.toml` | modify | Add `sella` extra, pytest marker, deptry ignore entries |
| `aimnet/calculators/aimnet2ase.py` | modify | Add `get_hessian(atoms=None) -> np.ndarray` method |
| `tests/test_ase.py` | modify | Add Hessian unit test (no Sella dep) |
| `tests/test_sella.py` | create | Sella smoke test gated by `pytest.mark.sella` |
| `examples/sella_ts.py` | create | End-to-end TS search example with analytic Hessian callback |
| `docs/external/sella.md` | create | User-facing Sella integration guide |
| `docs/external/index.md` | modify | Add Sella row to external-package overview |
| `mkdocs.yml` | modify | Add Sella entry to External Packages nav |
| `docs/index.md` | modify | Add `pip install "aimnet[sella]"` install line |
| `README.md` | modify | Add Sella to features list and install matrix |

---

## Task 1: Wire `sella` into project metadata

**Files:**
- Modify: `pyproject.toml:47-56` (optional-dependencies)
- Modify: `pyproject.toml:199-206` (pytest markers)
- Modify: `pyproject.toml:226-230` (deptry per-rule ignores)

- [ ] **Step 1: Add the `sella` optional extra**

Edit `pyproject.toml` so the `[project.optional-dependencies]` block reads:

```toml
[project.optional-dependencies]
# ASE calculator integration
ase = [
    "ase>=3.27.0,<4",
]

# PySisyphus calculator integration
pysis = [
    "pysisyphus",
]

# Sella saddle-point optimizer integration (requires ASE)
sella = [
    "ase>=3.27.0,<4",
    "sella>=2.4.0",
]
```

Why pin `sella>=2.4.0`: pre-v2.4.0 the optimizer is bottlenecked by NumPy/SciPy linear algebra; v2.4.0 (released 2026-03-28) ships the MLIP-targeted vectorization that makes the integration worthwhile. Why duplicate `ase>=3.27.0,<4` rather than depend on the `ase` extra: PEP 631 doesn't standardize extra-of-extra; duplicating the pin is the portable form.

- [ ] **Step 2: Add the `sella` pytest marker**

Update the `markers` list to include the Sella marker (alphabetical insertion between `pysis` and `train`):

```toml
markers = [
    "ase: marks tests that require ASE (deselect with: -m 'not ase')",
    "gpu: marks tests that require GPU/CUDA (deselect with: -m 'not gpu')",
    "hf: marks tests that require Hugging Face extras (deselect with: -m 'not hf')",
    "network: marks tests that hit external services (deselect with: -m 'not network')",
    "pysis: marks tests that require PySisyphus (deselect with: -m 'not pysis')",
    "sella: marks tests that require Sella (deselect with: -m 'not sella')",
    "train: marks tests that require training dependencies (deselect with: -m 'not train')",
]
```

- [ ] **Step 3: Add `sella` to deptry ignore lists**

Update the `[tool.deptry.per_rule_ignores]` block:

```toml
[tool.deptry.per_rule_ignores]
# Optional extras (ase, pysisyphus, sella) — installed via [project.optional-dependencies]; deptry can't see them statically
DEP001 = ["ase", "pysisyphus", "sella"]
DEP003 = ["ase", "pysisyphus", "sella"]
DEP004 = ["ase", "pysisyphus", "sella"]
```

- [ ] **Step 4: Verify metadata is parseable**

Run: `python -c "import tomllib; print(tomllib.load(open('pyproject.toml','rb'))['project']['optional-dependencies']['sella'])"`
Expected: `['ase>=3.27.0,<4', 'sella>=2.4.0']`

Run: `pytest --collect-only -m sella tests/ 2>&1 | tail -5`
Expected: `0 tests collected` and **no warning about unknown marker `sella`**.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add sella optional extra and pytest marker"
```

---

## Task 2: Add `get_hessian` to `AIMNet2ASE` (TDD)

**Files:**
- Modify: `aimnet/calculators/aimnet2ase.py:1-174`
- Test: `tests/test_ase.py` (append new test class)

The underlying `AIMNet2Calculator.__call__(..., hessian=True)` already returns a `(N, 3, N, 3)` Cartesian Hessian in eV/Å² (`aimnet/calculators/calculator.py:1131-1142`). The pysisyphus wrapper already consumes it (`aimnet/calculators/aimnet2pysis.py:46-53, 68-74`). ASE-internal Hessian convention is also eV/Å² so no unit conversion is needed — only a `(N, 3, N, 3) → (3N, 3N)` flatten. The method must be assignable as a `Sella(..., hessian_function=callable)` callback, which has signature `(atoms) -> ndarray`.

- [ ] **Step 1: Write the failing test**

Append this class to `tests/test_ase.py` (anywhere after `TestBasicCalculator`):

```python
class TestHessian:
    """Hessian property — used by Sella analytic-Hessian callback."""

    def test_hessian_shape_and_finite(self):
        pytest.importorskip("ase", reason="ASE not installed")
        from ase.io import read

        from aimnet.calculators import AIMNet2ASE

        atoms = read(CAFFEINE_FILE)
        atoms.calc = AIMNet2ASE("aimnet2")

        H = atoms.calc.get_hessian(atoms)
        N = len(atoms)
        assert H.shape == (3 * N, 3 * N)
        assert np.isfinite(H).all()

    def test_hessian_symmetric(self):
        pytest.importorskip("ase", reason="ASE not installed")
        from ase.io import read

        from aimnet.calculators import AIMNet2ASE

        atoms = read(CAFFEINE_FILE)
        atoms.calc = AIMNet2ASE("aimnet2")

        H = atoms.calc.get_hessian(atoms)
        # Cartesian Hessian must be symmetric to autograd noise
        assert np.max(np.abs(H - H.T)) < 5e-3

    def test_hessian_callback_signature(self):
        """Must be usable as Sella's hessian_function=callable callback."""
        pytest.importorskip("ase", reason="ASE not installed")
        import ase
        from ase import Atoms

        from aimnet.calculators import AIMNet2ASE

        # Tiny water for speed
        atoms = Atoms("OH2", positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
        atoms.calc = AIMNet2ASE("aimnet2")

        callback = atoms.calc.get_hessian
        assert callable(callback)
        H = callback(atoms)
        assert H.shape == (9, 9)
        assert isinstance(H, np.ndarray)

    def test_hessian_default_atoms(self):
        """get_hessian() with no argument should use the attached atoms."""
        pytest.importorskip("ase", reason="ASE not installed")
        from ase import Atoms

        from aimnet.calculators import AIMNet2ASE

        atoms = Atoms("OH2", positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
        calc = AIMNet2ASE("aimnet2")
        atoms.calc = calc
        # Trigger an energy calc so calc.atoms is populated
        atoms.get_potential_energy()
        H = calc.get_hessian()
        assert H.shape == (9, 9)
```

- [ ] **Step 2: Run the test to confirm it fails**

Run: `pytest tests/test_ase.py::TestHessian -x -v 2>&1 | tail -20`
Expected: FAIL with `AttributeError: 'AIMNet2ASE' object has no attribute 'get_hessian'`

- [ ] **Step 3: Implement `get_hessian` on `AIMNet2ASE`**

Edit `aimnet/calculators/aimnet2ase.py`. Add `get_hessian` as a new method right after `get_spin_charges` (around line 126), before `calculate`. Insert this exact block:

```python
    def get_hessian(self, atoms=None):
        """Return Cartesian Hessian as a (3N, 3N) ndarray in eV/Å^2.

        Designed for use as ``Sella(atoms, hessian_function=atoms.calc.get_hessian)``.
        Computed via double-backward through the AIMNet2 energy graph; cost scales
        as O(3N) backward passes. Not supported when ``compile_model=True`` or
        for batched / multi-molecule input.
        """
        if atoms is None:
            atoms = getattr(self, "atoms", None)
            if atoms is None:
                raise PropertyNotImplementedError(
                    "get_hessian() requires an attached Atoms object or an explicit argument."
                )
        if atoms.pbc.any():
            raise PropertyNotImplementedError(
                "Hessian for periodic systems is not supported by AIMNet2ASE.get_hessian()."
            )

        self._update_charge_spin_from_info()
        self.update_tensors()
        if self._t_numbers is None or self._t_numbers.shape[0] != len(atoms):
            self._t_numbers = torch.tensor(
                atoms.numbers, dtype=torch.int64, device=self.base_calc.device
            )

        coord = torch.tensor(
            atoms.positions, dtype=torch.float32, device=self.base_calc.device
        ).unsqueeze(0)
        _in = {
            "coord": coord,
            "numbers": self._t_numbers.unsqueeze(0),
            "charge": self._t_charge.unsqueeze(0),
            "mult": self._t_mult.unsqueeze(0),
        }

        results = self.base_calc(
            _in,
            forces=True,
            hessian=True,
            validate_species=self.validate_species,
        )
        H = results["hessian"].detach()  # (N, 3, N, 3)
        N = H.shape[0]
        return H.reshape(N * 3, N * 3).cpu().numpy()
```

The `coord` tensor is built fresh (not reused from `_in` in `calculate`) because `set_grad_tensors` mutates `requires_grad` on it for the double-backward path — sharing it across an energy call and a Hessian call invites stale-graph bugs.

- [ ] **Step 4: Run the test to confirm it passes**

Run: `pytest tests/test_ase.py::TestHessian -x -v 2>&1 | tail -20`
Expected: 4 passed.

- [ ] **Step 5: Run the existing ASE test suite to confirm no regressions**

Run: `pytest tests/test_ase.py -m ase 2>&1 | tail -5`
Expected: all previously-passing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add aimnet/calculators/aimnet2ase.py tests/test_ase.py
git commit -m "feat(ase): expose analytic Hessian via AIMNet2ASE.get_hessian"
```

---

## Task 3: Add a Sella smoke test

**Files:**
- Create: `tests/test_sella.py`

The point of this test is twofold: (1) verify that the `hessian_function=atoms.calc.get_hessian` wiring actually flows through Sella without exception, and (2) act as a contract test that Sella ≥ 2.4.0 still accepts our callback signature.

- [ ] **Step 1: Write the test file**

Create `tests/test_sella.py` with this exact content:

```python
"""Sella saddle-point optimizer smoke tests.

These tests verify that AIMNet2's analytic Hessian (exposed via
AIMNet2ASE.get_hessian) is callable from Sella's hessian_function= hook.
They are gated by both the ``sella`` and ``ase`` markers; CI without Sella
installed will deselect them automatically.
"""

import numpy as np
import pytest

pytestmark = [pytest.mark.sella, pytest.mark.ase]


def _h2o2_guess():
    """A trivially perturbed H2O2 starting from a non-stationary geometry."""
    from ase import Atoms

    return Atoms(
        "H2O2",
        positions=[
            [0.000, 0.700, 0.350],
            [0.000, -0.700, 0.350],
            [0.000, 0.700, -0.350],
            [0.000, -0.700, -0.350],
        ],
    )


class TestSellaIntegration:
    def test_callback_consumed_by_sella(self):
        """Sella(..., hessian_function=callback) must run at least one step."""
        pytest.importorskip("ase", reason="ASE not installed")
        sella = pytest.importorskip("sella", reason="Sella not installed")

        from aimnet.calculators import AIMNet2ASE

        atoms = _h2o2_guess()
        atoms.calc = AIMNet2ASE("aimnet2")

        dyn = sella.Sella(
            atoms,
            order=0,
            internal=True,
            hessian_function=atoms.calc.get_hessian,
        )
        # Two steps is enough to prove the Hessian callback is consumed without
        # error; we are not benchmarking convergence here.
        dyn.run(fmax=0.05, steps=2)

        # Energy must be finite and forces present afterwards.
        e = atoms.get_potential_energy()
        f = atoms.get_forces()
        assert np.isfinite(e)
        assert np.isfinite(f).all()

    def test_default_sella_no_hessian_callback(self):
        """Sella without a Hessian callback must also work via standard ASE."""
        pytest.importorskip("ase", reason="ASE not installed")
        sella = pytest.importorskip("sella", reason="Sella not installed")

        from aimnet.calculators import AIMNet2ASE

        atoms = _h2o2_guess()
        atoms.calc = AIMNet2ASE("aimnet2")

        dyn = sella.Sella(atoms, order=0, internal=True)
        dyn.run(fmax=0.05, steps=2)

        assert np.isfinite(atoms.get_potential_energy())
```

- [ ] **Step 2: Verify the test is collected and skipped without Sella installed**

Run: `pytest tests/test_sella.py --collect-only 2>&1 | tail -10`
Expected: collection shows the two tests under `TestSellaIntegration`.

Run (without `sella` installed, default state): `pytest tests/test_sella.py -v 2>&1 | tail -10`
Expected: both tests are SKIPPED with reason `Sella not installed`.

- [ ] **Step 3: Run the test with Sella installed (optional local check)**

If `sella` and `ase` are available locally:

Run: `pip install "sella>=2.4.0" && pytest tests/test_sella.py -m sella -v 2>&1 | tail -20`
Expected: 2 passed.

If Sella is not available, skip this step — CI will not run Sella tests by default since `pytestmark = [pytest.mark.sella, pytest.mark.ase]` deselects them unless `-m sella` is requested.

- [ ] **Step 4: Commit**

```bash
git add tests/test_sella.py
git commit -m "test(sella): smoke test analytic Hessian callback through Sella"
```

---

## Task 4: Add a Sella TS example

**Files:**
- Create: `examples/sella_ts.py`

The example must be runnable, reproducible, and demonstrate the analytic Hessian callback (the whole point of the integration). It mirrors `examples/ase_opt.py` for tone and structure.

- [ ] **Step 1: Write the example**

Create `examples/sella_ts.py` with this exact content:

```python
"""Sella transition-state search using AIMNet2's analytic Hessian.

Demonstrates the recommended configuration for using Sella with AIMNet2:
- Minimum-mode following on internal coordinates (order=1, internal=True).
- Analytic Hessian callback via AIMNet2ASE.get_hessian — replaces Sella's
  iterative Davidson finite-difference loop with a single double-backward
  pass through the AIMNet2 energy graph.

Reference: Schreiner et al., Nature Communications 2024
(https://www.nature.com/articles/s41467-024-52481-5) showed that providing
analytic ML Hessians to Sella reduces step count by 2-3x.

Requires: pip install "aimnet[sella]"
"""

from time import perf_counter

import ase.io
from sella import Sella

from aimnet.calculators import AIMNet2ASE


def main(xyz_path: str, fmax: float = 0.01, max_steps: int = 200) -> None:
    atoms = ase.io.read(xyz_path)
    atoms.calc = AIMNet2ASE("aimnet2")

    dyn = Sella(
        atoms,
        order=1,
        internal=True,
        hessian_function=atoms.calc.get_hessian,
    )

    print(f"Sella TS search for {len(atoms)} atoms; fmax={fmax} eV/A.")
    t0 = perf_counter()
    dyn.run(fmax=fmax, steps=max_steps)
    t1 = perf_counter()

    nsteps = dyn.nsteps
    print(f"Converged in {nsteps} steps ({t1 - t0:.1f} s, {(t1 - t0) / max(nsteps, 1):.3f} s/step).")
    print(f"Final energy: {atoms.get_potential_energy():.6f} eV")

    ase.io.write("ts_optimized.xyz", atoms)
    print("Optimized geometry written to ts_optimized.xyz.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python sella_ts.py <ts_guess.xyz>")
        sys.exit(1)
    main(sys.argv[1])
```

- [ ] **Step 2: Smoke-check syntax**

Run: `python -c "import ast; ast.parse(open('examples/sella_ts.py').read()); print('ok')"`
Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add examples/sella_ts.py
git commit -m "docs(examples): add Sella TS search example with analytic Hessian"
```

---

## Task 5: Add the Sella docs page and link it from navigation

**Files:**
- Create: `docs/external/sella.md`
- Modify: `docs/external/index.md`
- Modify: `mkdocs.yml:36-46` (External Packages nav block)
- Modify: `docs/index.md:79-82` (install commands)
- Modify: `README.md:61-64` (install matrix)

- [ ] **Step 1: Write the Sella docs page**

Create `docs/external/sella.md` with this exact content:

```markdown
# Sella

**Status: supported (uses AIMNet2's existing ASE calculator).**

[Sella](https://github.com/zadorlab/sella) is a saddle-point optimizer for ASE that targets transition states (`order=1`) and minima (`order=0`) with partitioned rational function optimization on internal coordinates. Sella consumes any ASE `Calculator`, so AIMNet2 works through the existing `AIMNet2ASE` class. The optional extra installs Sella ≥ 2.4.0 alongside ASE.

## Install

```bash
pip install "aimnet[sella]"
```

`sella>=2.4.0` is required. The 2.4.0 release (March 2026) introduced MLIP-targeted vectorization that made Sella usable for large systems (~22x wall-clock improvement on a 50-atom benchmark, [PR #64](https://github.com/zadorlab/sella/pull/64)).

## Recommended configuration

```python
import ase.io
from sella import Sella
from aimnet.calculators import AIMNet2ASE

atoms = ase.io.read("ts_guess.xyz")
atoms.calc = AIMNet2ASE("aimnet2")

dyn = Sella(
    atoms,
    order=1,                                    # 1 = saddle, 0 = minimum
    internal=True,                              # internal coordinates (recommended)
    hessian_function=atoms.calc.get_hessian,    # analytic Hessian callback
)
dyn.run(fmax=0.01)
```

## Why pass `hessian_function`

By default Sella refines its Hessian via an iterative Davidson eigensolver that costs ~10–30 extra gradient calls per refinement (every `nsteps_per_diag=3` steps). Wiring `AIMNet2ASE.get_hessian` into `hessian_function=` replaces that loop with a single double-backward through the AIMNet2 energy graph. This pattern was validated by [Schreiner et al. (Nature Comms 2024)](https://www.nature.com/articles/s41467-024-52481-5) for NewtonNet, where it cut TS optimization step counts by 2–3x.

The Hessian is computed in eV/Å² and shaped `(3N, 3N)` to match Sella's convention.

## Limitations

- Gas-phase only. The `internal=True` path assumes molecular topology; periodic TS searches are not supported by `AIMNet2ASE.get_hessian`.
- `compile_model=True` is incompatible with the Hessian path — Dynamo + double-backward through GELU hangs (`AIMNet2Calculator` raises `RuntimeError` if you combine them).
- Multi-molecule batching is not available for the Hessian; each `Sella` instance must hold one structure.

## Minima with Sella

For minima (not TS), the [Rowan optimizer benchmark](https://rowansci.com/blog/which-optimizer-should-you-use-with-nnps) (September 2025) recommends `Sella(order=0, internal=True)` as a strong plug-and-play optimizer for AIMNet2 — often converging in fewer steps than LBFGS, and without requiring an analytic Hessian.

## See also

- [ASE calculator interface](ase.md)
- [pysisyphus integration](pysis.md) — alternative for IRC, NEB, growing-string
- [Reaction paths and transition states](../advanced/reaction_paths.md)
- [Sella v2.4.0 release notes](https://github.com/zadorlab/sella/releases/tag/v2.4.0)
- [Sella JCTC 2022 paper](https://pubs.acs.org/doi/10.1021/acs.jctc.2c00395)
```

- [ ] **Step 2: Add Sella to the External Packages overview index**

Read the current state of `docs/external/index.md` to find where ASE / pysisyphus rows are described, then add a row for Sella using the same wording style. The exact prose should match the patterns already in that file. If `index.md` uses a table format, append:

```markdown
| [Sella](sella.md) | TS / minima optimizer (saddle-point search) | supported |
```

If `index.md` uses a bullet list, append:

```markdown
- **[Sella](sella.md)** — saddle-point and minima optimizer; uses the AIMNet2 ASE calculator.
```

(Match whichever format already exists. Do not invent a new structure.)

- [ ] **Step 3: Add Sella to the mkdocs nav**

Edit `mkdocs.yml`. In the `External Packages:` block (currently `mkdocs.yml:36-46`), insert a Sella entry directly after pysisyphus so the block reads:

```yaml
    - External Packages:
          - Overview: external/index.md
          - ASE: external/ase.md
          - pysisyphus: external/pysis.md
          - Sella: external/sella.md
          - OpenMM (openmm-ml): external/openmm.md
          - AMBER (torchani-amber): external/amber.md
          - SCM AMS (MLPotential): external/ams.md
          - ORCA (ExtOpt): external/orca.md
          - GROMACS (NNPot): external/gromacs.md
          - LAMMPS (mliap): external/lammps.md
```

- [ ] **Step 4: Add the install line to `docs/index.md`**

Find the install block in `docs/index.md` (around line 79–82) that currently shows `pip install "aimnet[ase]"` and `pip install "aimnet[pysis]"`. Add a Sella line directly below them so the block reads:

```bash
pip install "aimnet[ase]"             # ASE calculator interface
pip install "aimnet[pysis]"           # PySisyphus reaction path calculator
pip install "aimnet[sella]"           # Sella TS optimizer (includes ASE)
```

- [ ] **Step 5: Add the install line to `README.md`**

Find the install matrix in `README.md` (around line 61–64). Add a Sella line:

```bash
pip install "aimnet[ase]"             # ASE calculator interface
pip install "aimnet[pysis]"           # PySisyphus reaction path calculator
pip install "aimnet[sella]"           # Sella TS optimizer (includes ASE)
pip install "aimnet[ase,pysis,sella,train]" # All extras
```

(The previous `aimnet[ase,pysis,train]` "all extras" line should be replaced with the form above; also add `sella` between `pysis` and `train`.)

- [ ] **Step 6: Build the docs locally to validate the nav**

Run: `mkdocs build --strict 2>&1 | tail -20`
Expected: build succeeds with no warnings about missing pages or broken links. If `mkdocs` is not installed, skip this step — CI will catch it.

- [ ] **Step 7: Commit**

```bash
git add docs/external/sella.md docs/external/index.md docs/index.md mkdocs.yml README.md
git commit -m "docs(sella): add Sella integration page and navigation"
```

---

## Self-Review

Spec coverage check:

- ✅ Win 1 — analytic Hessian callback wiring → Task 2 (`get_hessian` on `AIMNet2ASE`).
- ✅ Win 2 — pin `sella>=2.4.0` and document recommended config → Task 1 (extra), Task 5 (docs).
- ✅ Smoke test that the wiring is consumed by Sella → Task 3.
- ✅ Example demonstrating recommended config → Task 4.
- ✅ Discoverability (nav, install line, README) → Task 5.
- ✅ Defer-list documented in plan header (vectorize `calculate_hessian`, `inference_mode` wrap, batched many-Sella driver, PBC TS).

Type-consistency check: `get_hessian(atoms=None) -> np.ndarray` of shape `(3N, 3N)` is used identically in Task 2 (definition + tests), Task 3 (Sella callback), Task 4 (example), Task 5 (docs). Method name does not collide with any ASE Calculator base-class method (ASE's base `Calculator` has no `get_hessian`).

No placeholders. No TBDs. All exact paths are real (verified against the working tree at plan-write time).
