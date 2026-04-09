# Repository Audit Remediation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Address all findings from the 5-agent repository audit (CI/CD, tests, packaging, docs, codebase) conducted on 2026-04-08.

**Architecture:** Fixes are self-contained and ordered by blast radius — workflow/CI first (no code risk), critical bugs second, codebase cleanup third, tests/docs last. Each task can be committed independently.

**Tech Stack:** Python 3.11+, PyTorch, pytest, GitHub Actions, uv, hatchling, MkDocs, ruff

---

## File Map

**Delete:**
- `.github/workflows/cursor-code-review.yml` — broken, CURSOR_API_KEY secret missing
- `aimnet/base.py` — dead code, duplicate of `aimnet/models/base.py`

**Modify:**
- `.github/workflows/main.yml` — add `tests-hf` job, fix action pins, add timeouts, `fetch-depth: 0` in tests-core
- `.github/workflows/gpu-tests.yml` — add `concurrency` limit, fix action pins
- `.github/workflows/on-release-main.yml` — remove broken `branches` filter from `release` event
- `aimnet/calculators/calculator.py:248-281` — fix routing bug (relative path fallthrough)
- `aimnet/calculators/model_registry.py:13` — fix `registry_file` param ignored in `load_model_registry()`
- `aimnet/calculators/model_registry.py:57` — fix `logging.warn` → `logging.warning`
- `aimnet/calculators/aimnet2ase.py:36` — fix species validation (`hasattr` always False)
- `aimnet/calculators/hf_hub.py:274-276` — fix `needs_coulomb`/`needs_dispersion` not using `_cfg()`
- `aimnet/models/base.py:20-44` — add `has_embedded_lr` field to `ModelMetadata`
- `tests/conftest.py` — add session-scoped `aimnet2_calc` fixture
- `pyproject.toml` — add `warp-lang` upper bound, Python 3.13 classifiers, S101 per-file ignore
- `docs/calculator.md` — add missing `ensemble_member`, `revision`, `token`, `HF repo` constructor params
- `docs/models/aimnet2pd.md` — fix DFT functional description

---

## Task 1: Delete cursor-code-review.yml

**Files:**
- Delete: `.github/workflows/cursor-code-review.yml`

- [ ] **Step 1: Confirm the file is standalone (nothing references it)**

```bash
grep -r "cursor-code-review" .github/
```

Expected: only finds the file itself (no caller).

- [ ] **Step 2: Delete the file**

```bash
git rm .github/workflows/cursor-code-review.yml
```

- [ ] **Step 3: Commit**

```bash
git commit -m "ci: remove cursor-code-review workflow (missing CURSOR_API_KEY secret)"
```

---

## Task 2: Fix CI workflow — add tests-hf job + minor fixes to main.yml

**Files:**
- Modify: `.github/workflows/main.yml`

The `tests-hf` job is missing, so HF-marked tests never run in CI. The `tests-core` job also lacks `fetch-depth: 0` (needed for `hatch-vcs` in some edge cases) and job-level `timeout-minutes`.

- [ ] **Step 1: Read the current main.yml**

Read `.github/workflows/main.yml` lines 1–178 (full file).

- [ ] **Step 2: Add timeout-minutes to all jobs and add tests-hf job**

In `.github/workflows/main.yml`:

a) Add `timeout-minutes: 20` to each existing job block (`quality`, `tests-core`, `tests-ase`, `tests-train`, `tests-pysis`, `check-docs`).

b) Add `fetch-depth: 0` to the `tests-core` checkout step:
```yaml
            - name: Check out
              uses: actions/checkout@v6
              with:
                  fetch-depth: 0
```

c) Add a new `tests-hf` job after `tests-pysis`:
```yaml
    tests-hf:
        runs-on: ubuntu-latest
        timeout-minutes: 20
        strategy:
            matrix:
                python-version: ["3.11"]
            fail-fast: false
        defaults:
            run:
                shell: bash
        steps:
            - name: Check out
              uses: actions/checkout@v6

            - name: Set up the environment
              uses: ./.github/actions/setup-uv-env
              with:
                  python-version: ${{ matrix.python-version }}
                  groups: dev
                  extras: hf

            - name: Run hf-marked tests (allow empty)
              run: |
                  set -e
                  uv run pytest tests -m hf || rc=$?
                  if [ "${rc:-0}" -eq 5 ]; then
                    echo "No hf-marked tests collected; treating as success."
                    exit 0
                  fi
                  exit "${rc:-0}"
```

d) Add `tests-hf` to the `deploy-docs` needs list:
```yaml
    deploy-docs:
        if: github.event_name == 'push' && github.ref == 'refs/heads/main'
        needs:
            [
                quality,
                tests-core,
                tests-ase,
                tests-train,
                tests-pysis,
                tests-hf,
                check-docs,
            ]
```

- [ ] **Step 3: Run mkdocs build to ensure no docs breakage**

```bash
uv run mkdocs build -s 2>&1 | tail -5
```

Expected: `INFO - Documentation built in X.X seconds`

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/main.yml
git commit -m "ci: add tests-hf job, timeouts, and fetch-depth to main workflow"
```

---

## Task 3: Fix on-release-main.yml — remove broken branches filter

**Files:**
- Modify: `.github/workflows/on-release-main.yml`

The `branches: [main]` under `on: release:` is silently ignored by GitHub Actions (the `branches` filter only applies to `push`/`pull_request` events, not `release`). It's dead config that creates false confidence about branch scoping.

- [ ] **Step 1: Read the current on-release-main.yml**

Read `.github/workflows/on-release-main.yml`.

- [ ] **Step 2: Remove the branches filter**

Change:
```yaml
on:
    release:
        types: [published]
        branches: [main]
```

To:
```yaml
on:
    release:
        types: [published]
```

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/on-release-main.yml
git commit -m "ci: remove invalid branches filter from release event trigger"
```

---

## Task 4: Fix GPU tests workflow — add concurrency limit

**Files:**
- Modify: `.github/workflows/gpu-tests.yml`

Without a `concurrency` group, multiple pushes can queue simultaneous GPU jobs, tying up the self-hosted runner for long periods.

- [ ] **Step 1: Read the current gpu-tests.yml**

Read `.github/workflows/gpu-tests.yml`.

- [ ] **Step 2: Add concurrency group at the workflow level**

Add after the `on:` block:
```yaml
concurrency:
    group: gpu-${{ github.ref }}
    cancel-in-progress: true
```

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/gpu-tests.yml
git commit -m "ci: add concurrency limit to GPU tests workflow"
```

---

## Task 5: Fix AIMNet2Calculator routing bug (relative two-segment paths)

**Files:**
- Modify: `aimnet/calculators/calculator.py:248-281`

**Bug:** When `model` is a relative path like `"subdir/mymodel.pt"` (exactly one `/`, no directory exists, but file exists), it matches `_HF_ID_RE` but `is_hf_repo_id()` returns `False` (because `Path(model).exists()`) and `_is_hf_dir` is `False`. We enter the outer `if` but skip the inner `if`, so `self.model` is never assigned → `AttributeError` on first call.

**Fix:** Add an `else` branch inside the outer block to fall through to `get_model_path`.

- [ ] **Step 1: Write the failing test**

In `tests/test_calculator.py`, add to the `TestModelLoading` class (or create one):

```python
def test_relative_path_with_slash_loads_correctly(tmp_path):
    """Relative two-segment paths like 'subdir/model.pt' must not be misrouted to HF."""
    from aimnet.calculators.model_registry import get_model_path
    import shutil

    # Copy the real model to a nested path
    real_path = get_model_path("aimnet2")
    subdir = tmp_path / "mymodels"
    subdir.mkdir()
    dest = subdir / "aimnet2.pt"
    shutil.copy(real_path, dest)

    calc = AIMNet2Calculator(str(dest))
    assert hasattr(calc, "model")
    assert calc.cutoff > 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_calculator.py::test_relative_path_with_slash_loads_correctly -v
```

Expected: `AttributeError: 'AIMNet2Calculator' object has no attribute 'model'` or similar.

- [ ] **Step 3: Read calculator.py lines 244–285**

Read `aimnet/calculators/calculator.py` offset 244 limit 45.

- [ ] **Step 4: Apply the fix**

Change the inner block from:
```python
            if is_hf_repo_id(model) or _is_hf_dir:
                _model, metadata = load_from_hf_repo(
                    model,
                    ensemble_member=ensemble_member,
                    device=self.device,
                    revision=revision,
                    token=token,
                )
                self.model = _model
                self.cutoff = metadata["cutoff"]
        else:
            p = get_model_path(model)
            self.model, metadata = load_model(p, device=self.device)
            self.cutoff = metadata["cutoff"]
```

To:
```python
            if is_hf_repo_id(model) or _is_hf_dir:
                _model, metadata = load_from_hf_repo(
                    model,
                    ensemble_member=ensemble_member,
                    device=self.device,
                    revision=revision,
                    token=token,
                )
                self.model = _model
                self.cutoff = metadata["cutoff"]
            else:
                # _looks_like_hf was True but it's actually a local file path
                p = get_model_path(model)
                self.model, metadata = load_model(p, device=self.device)
                self.cutoff = metadata["cutoff"]
        else:
            p = get_model_path(model)
            self.model, metadata = load_model(p, device=self.device)
            self.cutoff = metadata["cutoff"]
```

- [ ] **Step 5: Run test to verify it passes**

```bash
uv run pytest tests/test_calculator.py::test_relative_path_with_slash_loads_correctly -v
```

Expected: PASS

- [ ] **Step 6: Run core tests to check for regressions**

```bash
uv run pytest tests -m "not ase and not pysis and not train and not hf and not gpu" -x -q
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add aimnet/calculators/calculator.py tests/test_calculator.py
git commit -m "fix: handle relative two-segment paths in AIMNet2Calculator model routing"
```

---

## Task 6: Fix load_model_registry() ignoring its registry_file parameter

**Files:**
- Modify: `aimnet/calculators/model_registry.py:11-14`

**Bug:** `registry_file` param is computed but `open()` always uses the hardcoded path.

- [ ] **Step 1: Write the failing test**

In `tests/test_model_registry.py` (or create it):

```python
import os
import yaml
import pytest
from aimnet.calculators.model_registry import load_model_registry

def test_load_model_registry_respects_registry_file_param(tmp_path):
    """registry_file param should override the default path."""
    fake = {"aliases": {}, "models": {"fake_model": {"file": "x.pt", "url": "http://x"}}}
    registry_path = tmp_path / "my_registry.yaml"
    registry_path.write_text(yaml.dump(fake))

    result = load_model_registry(str(registry_path))
    assert "fake_model" in result["models"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_model_registry.py::test_load_model_registry_respects_registry_file_param -v
```

Expected: FAIL (returns default registry, not fake one).

- [ ] **Step 3: Apply the fix**

In `aimnet/calculators/model_registry.py`, change:
```python
def load_model_registry(registry_file: str | None = None) -> dict[str, str]:
    registry_file = registry_file or os.path.join(os.path.dirname(__file__), "model_registry.yaml")
    with open(os.path.join(os.path.dirname(__file__), "model_registry.yaml")) as f:
        return yaml.load(f, Loader=yaml.SafeLoader)
```

To:
```python
def load_model_registry(registry_file: str | None = None) -> dict[str, str]:
    registry_file = registry_file or os.path.join(os.path.dirname(__file__), "model_registry.yaml")
    with open(registry_file) as f:
        return yaml.load(f, Loader=yaml.SafeLoader)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_model_registry.py::test_load_model_registry_respects_registry_file_param -v
```

Expected: PASS

- [ ] **Step 5: Fix logging.warn → logging.warning in the same file**

In `aimnet/calculators/model_registry.py` line 57, change:
```python
            logging.warn(f"Removing {fil}")
```
To:
```python
            logging.warning(f"Removing {fil}")
```

- [ ] **Step 6: Commit**

```bash
git add aimnet/calculators/model_registry.py tests/test_model_registry.py
git commit -m "fix: load_model_registry() now uses its registry_file parameter; fix deprecated logging.warn"
```

---

## Task 7: Fix AIMNet2ASE species validation (hasattr always False)

**Files:**
- Modify: `aimnet/calculators/aimnet2ase.py:36-39`

**Bug:** `AIMNet2Calculator` stores implemented species as `self.implemented_species` (a tensor), but `AIMNet2ASE.__init__` checks `hasattr(base_calc, "implemented_species")` — this is `True` when a tensor attribute exists and `False` when it doesn't, so the `hasattr` check itself isn't the problem. The bug reported was that the attribute is set on `base_calc` but the check always returns `False`.

Let's verify by reading the relevant section of calculator.py to confirm where `implemented_species` is set.

- [ ] **Step 1: Find where implemented_species is set in calculator.py**

```bash
grep -n "implemented_species" aimnet/calculators/calculator.py | head -20
```

- [ ] **Step 2: Read the result and determine the actual fix**

If `AIMNet2Calculator` sets `self.implemented_species` only when metadata is available, and the attribute is a `Tensor`, the check `hasattr(base_calc, "implemented_species")` should work. If the audit found it always returns `False`, it may be that the attribute name doesn't match or is set on a different object.

Check the actual attribute name:
```bash
grep -n "implemented_species" aimnet/calculators/calculator.py aimnet/calculators/aimnet2ase.py
```

- [ ] **Step 3: Write a test that validates species filtering works**

In `tests/test_ase.py` (or `tests/test_aimnet2ase.py`), add:

```python
@pytest.mark.ase
def test_ase_species_validation_raises_for_unsupported():
    """AIMNet2ASE should raise ValueError when unsupported elements are used."""
    import numpy as np
    from ase import Atoms
    from aimnet.calculators import AIMNet2Calculator
    from aimnet.calculators.aimnet2ase import AIMNet2ASE

    base_calc = AIMNet2Calculator("aimnet2")
    ase_calc = AIMNet2ASE(base_calc)

    if ase_calc.implemented_species is not None:
        # Gold (Au, Z=79) is not in aimnet2 implemented species
        atoms = Atoms("Au", positions=[[0, 0, 0]])
        atoms.calc = ase_calc
        with pytest.raises(ValueError, match="not implemented"):
            atoms.get_potential_energy()
    else:
        pytest.skip("implemented_species not set on this calculator")
```

- [ ] **Step 4: Apply the fix if needed**

After step 2, if the attribute is named differently in `AIMNet2Calculator` (e.g., it's on `metadata`), fix `aimnet2ase.py:36` accordingly. For example if the fix is to read from metadata:

```python
        if base_calc.implemented_species is not None:
            self.implemented_species = base_calc.implemented_species
```

(This step requires reading the grep output from step 2 to determine exact fix.)

- [ ] **Step 5: Commit**

```bash
git add aimnet/calculators/aimnet2ase.py tests/
git commit -m "fix: AIMNet2ASE species validation now correctly reads implemented_species"
```

---

## Task 8: Add has_embedded_lr to ModelMetadata + fix hf_hub.py needs_coulomb fallback

**Files:**
- Modify: `aimnet/models/base.py:20-44`
- Modify: `aimnet/calculators/hf_hub.py:274-276`

**Bug 1:** `ModelMetadata` TypedDict is missing `has_embedded_lr` field, causing type checkers to flag usages in `hf_hub.py` and `calculator.py`.

**Bug 2:** In `hf_hub.py` lines 274-276, `needs_coulomb` and `needs_dispersion` are read directly from `config.get()` instead of using `_cfg()`, so GCS fallback values are silently ignored.

- [ ] **Step 1: Add has_embedded_lr to ModelMetadata**

In `aimnet/models/base.py`, change the `ModelMetadata` TypedDict to add the field after `d3_params`:
```python
    # Dispersion parameters (optional)
    d3_params: NotRequired[dict | None]  # {s8, a1, a2, s6} if needs_dispersion=True

    has_embedded_lr: NotRequired[bool]  # True if model has embedded LR (legacy or D3TS)
    implemented_species: list[int]  # Supported atomic numbers
```

- [ ] **Step 2: Fix needs_coulomb/needs_dispersion in hf_hub.py**

In `aimnet/calculators/hf_hub.py`, change lines 274-276 from:
```python
        "needs_coulomb": config.get("needs_coulomb", False),
        "needs_dispersion": config.get("needs_dispersion", False),
        "coulomb_mode": config.get("coulomb_mode", "none"),
```

To:
```python
        "needs_coulomb": _cfg("needs_coulomb", False),
        "needs_dispersion": _cfg("needs_dispersion", False),
        "coulomb_mode": _cfg("coulomb_mode", "none"),
```

- [ ] **Step 3: Run ruff to verify no type errors**

```bash
uv run ruff check aimnet/models/base.py aimnet/calculators/hf_hub.py
```

Expected: no errors.

- [ ] **Step 4: Run hf tests**

```bash
uv run pytest tests/test_hf_hub.py -v
```

Expected: all pass (or `importorskip` skips if safetensors not installed).

- [ ] **Step 5: Commit**

```bash
git add aimnet/models/base.py aimnet/calculators/hf_hub.py
git commit -m "fix: add has_embedded_lr to ModelMetadata; fix hf_hub needs_coulomb/dispersion fallback"
```

---

## Task 9: Delete aimnet/base.py (dead code)

**Files:**
- Delete: `aimnet/base.py`

`aimnet/base.py` is a duplicate of an earlier version of `aimnet/models/base.py`. It exports `AIMNet2Base` (a different class from `AIMNet2` in `models/`), but nothing imports it anywhere.

- [ ] **Step 1: Verify nothing imports aimnet.base**

```bash
grep -r "from aimnet.base" . --include="*.py"
grep -r "import aimnet.base" . --include="*.py"
grep -r "aimnet/base" . --include="*.py" --include="*.yaml" --include="*.toml"
```

Expected: no results (only `aimnet/models/base.py` references).

- [ ] **Step 2: Check __init__.py doesn't re-export it**

```bash
grep -n "base" aimnet/__init__.py
```

- [ ] **Step 3: Delete the file**

```bash
git rm aimnet/base.py
```

- [ ] **Step 4: Run core tests to check nothing broke**

```bash
uv run pytest tests -m "not ase and not pysis and not train and not hf and not gpu" -x -q
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git commit -m "chore: delete aimnet/base.py (dead code, superseded by aimnet/models/base.py)"
```

---

## Task 10: Add session-scoped model fixture to conftest.py

**Files:**
- Modify: `tests/conftest.py`

Many tests each load the `aimnet2` model independently, which means N model downloads/loads per test run. A session-scoped fixture loads it once per session, cutting test suite time significantly.

- [ ] **Step 1: Read conftest.py lines 237–350**

Read `tests/conftest.py` offset 237 limit 120.

- [ ] **Step 2: Add session-scoped calculator fixture**

In `tests/conftest.py`, after the `device` fixture, add:

```python
@pytest.fixture(scope="session")
def aimnet2_calc():
    """Session-scoped AIMNet2Calculator with the default aimnet2 model.

    Loaded once per test session. Use this instead of constructing
    AIMNet2Calculator("aimnet2") inside individual tests.
    """
    from aimnet.calculators import AIMNet2Calculator
    return AIMNet2Calculator("aimnet2")
```

- [ ] **Step 3: Run tests to verify the fixture is discoverable**

```bash
uv run pytest tests -m "not ase and not pysis and not train and not hf and not gpu" --collect-only 2>&1 | head -30
```

Expected: no collection errors.

- [ ] **Step 4: Commit**

```bash
git add tests/conftest.py
git commit -m "test: add session-scoped aimnet2_calc fixture to reduce model load overhead"
```

---

## Task 11: Add missing constructor params to calculator.md

**Files:**
- Modify: `docs/calculator.md`

The constructor docs at line 66-77 list 7 parameters but are missing `ensemble_member`, `revision`, `token` (added with HF integration) and the HF repo ID model source.

- [ ] **Step 1: Read calculator.md lines 65–145**

Read `docs/calculator.md` offset 65 limit 80.

- [ ] **Step 2: Update constructor signature block**

Change the constructor block from:
```python
AIMNet2Calculator(
    model: str | nn.Module = "aimnet2",
    nb_threshold: int = 120,
    needs_coulomb: bool | None = None,
    needs_dispersion: bool | None = None,
    device: str | None = None,
    compile_model: bool = False,
    compile_kwargs: dict | None = None,
)
```

To:
```python
AIMNet2Calculator(
    model: str | nn.Module = "aimnet2",
    nb_threshold: int = 120,
    needs_coulomb: bool | None = None,
    needs_dispersion: bool | None = None,
    device: str | None = None,
    compile_model: bool = False,
    compile_kwargs: dict | None = None,
    ensemble_member: int = 0,
    revision: str | None = None,
    token: str | None = None,
)
```

- [ ] **Step 3: Add documentation sections for the new params**

After the `compile_kwargs` section, add:

```markdown
#### `ensemble_member`

Which ensemble member to load when loading from a Hugging Face repo. Default: `0`.

Ensemble members are indexed 0–3. Only applies when `model` is a HF repo ID or a local HF-style directory (containing `config.json` + `ensemble_N.safetensors`).

#### `revision`

HF repo revision (branch, tag, or commit hash) to load from. Default: `None` (latest).

Only applies when `model` is a HF repo ID.

#### `token`

HF API token for accessing private or gated repositories. Default: `None`.

Set via environment variable `HF_TOKEN` as an alternative to passing it directly.

Only applies when `model` is a HF repo ID.
```

- [ ] **Step 4: Update the model parameter table to include HF repo IDs**

In the `#### model` section, extend the table:

| Type | Behavior |
| --- | --- |
| `str` (registry name) | Loads from model registry (e.g., `"aimnet2"`), downloading if needed |
| `str` (file path) | Loads from `.pt` (v2) or `.jpt` (v1 legacy) file if the path exists |
| `str` (HF repo ID) | Loads from Hugging Face Hub (e.g., `"isayevlab/aimnet2-wb97m-d3"`); requires `aimnet[hf]` |
| `str` (local HF dir) | Loads from a local directory with `config.json` + `ensemble_N.safetensors` |
| `torch.nn.Module` | Uses provided module directly |

- [ ] **Step 5: Run mkdocs build**

```bash
uv run mkdocs build -s 2>&1 | tail -5
```

Expected: clean build.

- [ ] **Step 6: Commit**

```bash
git add docs/calculator.md
git commit -m "docs: add ensemble_member, revision, token params and HF repo ID model source to calculator.md"
```

---

## Task 12: Packaging fixes (pyproject.toml)

**Files:**
- Modify: `pyproject.toml`

Three fixes: (1) add `warp-lang` upper bound to avoid API breakage on next major version; (2) add Python 3.13 classifiers; (3) add S101 per-file ignore for test files so `assert` in tests doesn't trigger the bandit rule.

- [ ] **Step 1: Read pyproject.toml lines 33–50 and 128–180**

Read `pyproject.toml` lines 33–50 (dependencies) and 165–180 (ruff ignore section).

- [ ] **Step 2: Add warp-lang upper bound**

Change:
```toml
    "warp-lang>=1.11",
```
To:
```toml
    "warp-lang>=1.11,<2",
```

- [ ] **Step 3: Add Python 3.13 classifier**

In the `classifiers` list, add after `"Programming Language :: Python :: 3.12"`:
```toml
    "Programming Language :: Python :: 3.13",
```

- [ ] **Step 4: Add S101 per-file ignore for tests**

In `pyproject.toml`, after the `[tool.ruff.lint]` section's `ignore` list, add:

```toml
[tool.ruff.lint.per-file-ignores]
"tests/**/*.py" = ["S101"]
```

(Remove `"S101"` from the global `ignore` list if present — it currently is, so this also tightens the rule for non-test code.)

Actually, since S101 is already in the global ignore list, the cleaner approach is to keep it there. Only do this step if the audit specifically identified non-test assert misuse. Skip this step if S101 global ignore is intentional.

- [ ] **Step 5: Run ruff check**

```bash
uv run ruff check aimnet/ --select S101
```

Expected: no violations (or only in tests if global ignore was removed).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add warp-lang upper bound, Python 3.13 classifier"
```

---

## Task 13: Run full verification

This is a final sanity check after all tasks are complete.

- [ ] **Step 1: Run make check**

```bash
make check
```

Expected: all pre-commit hooks pass (ruff, markdownlint, prettier, codespell, deptry).

If prettier reformats any docs files, stage the reformatted versions and amend/commit:
```bash
git add docs/
git commit -m "chore: apply prettier formatting"
```

- [ ] **Step 2: Run full test suite**

```bash
uv run pytest tests -m "not ase and not pysis and not train and not hf and not gpu" --tb=short -q
```

Expected: all pass, no regressions.

- [ ] **Step 3: Run mkdocs build**

```bash
uv run mkdocs build -s
```

Expected: clean build with no warnings.

---

## Self-Review

**Spec coverage check:**

| Finding | Task |
|---------|------|
| Delete cursor-code-review.yml | Task 1 |
| Add tests-hf CI job | Task 2 |
| CI timeouts | Task 2 |
| fetch-depth in tests-core | Task 2 |
| Remove broken branches filter from release event | Task 3 |
| GPU runner concurrency | Task 4 |
| Relative path routing bug in AIMNet2Calculator | Task 5 |
| load_model_registry() param bug | Task 6 |
| logging.warn → logging.warning | Task 6 |
| AIMNet2ASE species validation bug | Task 7 |
| has_embedded_lr missing from ModelMetadata | Task 8 |
| needs_coulomb/dispersion not using _cfg() in hf_hub.py | Task 8 |
| Delete aimnet/base.py dead code | Task 9 |
| Session-scoped model fixture | Task 10 |
| Missing HF params in calculator.md | Task 11 |
| warp-lang upper bound | Task 12 |
| Python 3.13 classifiers | Task 12 |

**Out of scope for this plan** (deferred — low risk, no user-visible bugs):
- Codecov token secret (needs repo admin access)
- Branch protection rules (repo settings, not code)
- PyPI publish gate environment (needs repo admin access)
- Move `docs/superpowers/` out of published docs (separate housekeeping PR)
- Add CHANGELOG to mkdocs nav (requires CHANGELOG file to exist)
- api/modules.md LR/AEV coverage (docs enhancement, not a bug)
- pysis/train smoke tests (test enhancement)
- validate-codecov path filter (minor CI polish)
