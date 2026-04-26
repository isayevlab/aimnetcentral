# aimnet2-rxn Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the `aimnet2-rxn` model family in `aimnetcentral` (registry + architecture YAML + load path), wired up with the chemistry safeguards rxn requires (charge validation, AFV row sanitization, Coulomb-cutoff lock, cross-family mixing detection, Hessian + `torch.compile` guard) and three orthogonal cleanups surfaced during the integration (`load_v1_model` `.double()` ordering, calculator `metadata` property, ASE/Pysis `validate_species` propagation).

**Architecture:** Two-layer change. **Bottom layer**: `aimnet/models/base.py::ModelMetadata` and the two metadata-construction sites (`base.py::load_model` and `hf_hub.py::load_from_hf_repo`) gain optional `family` and `supports_charged_systems` fields. `aimnet/models/utils.py::load_v1_model` gains keyword-only `implemented_species`, `family`, `supports_charged_systems` overrides plus AFV-row NaN-padding when species are declared. **Top layer**: `AIMNet2Calculator` reads the new metadata fields via a new `metadata` property and enforces the safeguards inside `eval` and `set_lrcoulomb_method`. Wrappers (`AIMNet2ASE`, `AIMNet2Pysis`) propagate `validate_species`. Registry and architecture YAML are pure additions.

**Tech Stack:** Python 3.11+, PyTorch, pytest with `@pytest.mark.hf` marker (existing), YAML configs read via `yaml.safe_load`.

**Reference spec:** `docs/superpowers/specs/2026-04-25-aimnet2-rxn-integration-design.md`.

---

## File Structure

| Path | Action | Responsibility |
| --- | --- | --- |
| `aimnet/models/base.py` | Modify | Extend `ModelMetadata` TypedDict; propagate `family` + `supports_charged_systems` in `load_model`. |
| `aimnet/calculators/hf_hub.py` | Modify | Propagate `family` + `supports_charged_systems` in `load_from_hf_repo`. |
| `aimnet/models/utils.py` | Modify | Add `implemented_species`/`family`/`supports_charged_systems` kwargs to `load_v1_model`; AFV-row NaN-padding when species declared; reorder `.double()` before `load_state_dict`. Add docstring example. |
| `aimnet/models/aimnet2_rxn.yaml` | Create | Architecture YAML for the rxn model family. |
| `aimnet/calculators/model_registry.yaml` | Modify | 4 entries + 1 alias for aimnet2-rxn. |
| `aimnet/calculators/calculator.py` | Modify | Add `metadata` property; `_was_compiled` flag; `_constructed_families` ClassVar; safeguards in `eval`; rxn-family warning in `set_lrcoulomb_method`. |
| `aimnet/calculators/aimnet2ase.py` | Modify | Propagate `validate_species` kwarg through to base calculator. |
| `aimnet/calculators/aimnet2pysis.py` | Modify | Propagate `validate_species` kwarg through to base calculator. |
| `tests/test_model.py` | Modify | Add `test_load_v1_model_species_override_nan_pads_other_rows` (also covers `.double()` ordering). |
| `tests/test_calculator.py` | Modify | Add 7 tests for safeguards + alias E2E + metadata property coverage. |
| `tests/test_hf_hub.py` | Modify | Add `test_aimnet2_rxn_hf_load_matches_gcs_metadata` (network-gated). |
| `docs/models/aimnet2_rxn.md` | Create | ≤30-line stub model card with canonical link to HF README. |
| `mkdocs.yml` | Modify | One nav entry pointing at the new docs page. |

---

## Task 0: Preparation — feature branch

**Files:** none (git only).

- [ ] **Step 1: Confirm clean working tree on `main`**

Run: `git -C /home/olexandr/aimnetcentral status` Expected: `nothing to commit, working tree clean` (the spec commits should already be on main).

- [ ] **Step 2: Create and switch to a feature branch**

Run:

```bash
git -C /home/olexandr/aimnetcentral switch -c feat/aimnet2-rxn-integration
```

Expected: `Switched to a new branch 'feat/aimnet2-rxn-integration'`.

- [ ] **Step 3: Verify the branch**

Run: `git -C /home/olexandr/aimnetcentral branch --show-current` Expected: `feat/aimnet2-rxn-integration`.

---

## Task 1: Extend `ModelMetadata` TypedDict

**Files:**

- Modify: `aimnet/models/base.py:20-45` (TypedDict)
- Test: `tests/test_model.py` (append a new test function)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_model.py`:

```python
def test_model_metadata_typeddict_includes_family_and_charge_fields():
    """ModelMetadata must declare family and supports_charged_systems as optional fields."""
    from typing import get_type_hints
    from aimnet.models.base import ModelMetadata

    hints = get_type_hints(ModelMetadata, include_extras=False)
    assert "family" in hints, "ModelMetadata missing 'family' field"
    assert "supports_charged_systems" in hints, (
        "ModelMetadata missing 'supports_charged_systems' field"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_model.py::test_model_metadata_typeddict_includes_family_and_charge_fields -v` Expected: FAIL with `AssertionError: ModelMetadata missing 'family' field`.

- [ ] **Step 3: Add the new fields to `ModelMetadata`**

Edit `aimnet/models/base.py`. Locate the `class ModelMetadata(TypedDict):` block (around line 20). After the existing `implemented_species: list[int]  # Supported atomic numbers` line, add:

```python
    family: NotRequired[str | None]                       # e.g. "rxn"; None for legacy/families that don't declare
    supports_charged_systems: NotRequired[bool | None]    # False for rxn; None for legacy
```

Verify `NotRequired` is already imported at the top of `base.py` (it is — used by other fields like `coulomb_sr_rc`).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_model.py::test_model_metadata_typeddict_includes_family_and_charge_fields -v` Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git -C /home/olexandr/aimnetcentral add aimnet/models/base.py tests/test_model.py
git -C /home/olexandr/aimnetcentral commit -m "feat(models): add family and supports_charged_systems to ModelMetadata"
```

---

## Task 2: Propagate new metadata fields in `load_model`

**Files:**

- Modify: `aimnet/models/base.py:111-122` (the v2-format metadata construction inside `load_model`)
- Test: `tests/test_model.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_model.py`:

```python
def test_load_model_propagates_family_and_charge_fields(tmp_path):
    """load_model must include family/supports_charged_systems in metadata when present in .pt."""
    import torch
    from aimnet.calculators.model_registry import get_model_path
    from aimnet.models.base import load_model

    # Take the existing aimnet2 .pt as a template; add the new fields; save; reload.
    src = get_model_path("aimnet2")
    raw = torch.load(src, map_location="cpu", weights_only=False)
    raw["family"] = "test-family"
    raw["supports_charged_systems"] = False

    out = tmp_path / "with_family.pt"
    torch.save(raw, str(out))

    _, metadata = load_model(str(out), device="cpu")
    assert metadata.get("family") == "test-family"
    assert metadata.get("supports_charged_systems") is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_model.py::test_load_model_propagates_family_and_charge_fields -v` Expected: FAIL — `metadata.get("family")` returns `None` because `load_model` does not yet propagate the field.

- [ ] **Step 3: Update the `load_model` v2 branch**

Edit `aimnet/models/base.py`. Locate the v2-format `metadata: ModelMetadata = { ... }` block (around line 111). Add two lines just before the closing brace, immediately after `"implemented_species": data.get("implemented_species", []),`:

```python
            "family": data.get("family"),
            "supports_charged_systems": data.get("supports_charged_systems"),
```

The full block now looks like:

```python
        metadata: ModelMetadata = {
            "format_version": data.get("format_version", 2),
            "cutoff": data["cutoff"],
            "needs_coulomb": data.get("needs_coulomb", False),
            "needs_dispersion": data.get("needs_dispersion", False),
            "coulomb_mode": data.get("coulomb_mode", "none"),
            "coulomb_sr_rc": data.get("coulomb_sr_rc"),
            "coulomb_sr_envelope": data.get("coulomb_sr_envelope"),
            "d3_params": data.get("d3_params"),
            "has_embedded_lr": data.get("has_embedded_lr", False),
            "implemented_species": data.get("implemented_species", []),
            "family": data.get("family"),
            "supports_charged_systems": data.get("supports_charged_systems"),
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_model.py::test_load_model_propagates_family_and_charge_fields -v` Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git -C /home/olexandr/aimnetcentral add aimnet/models/base.py tests/test_model.py
git -C /home/olexandr/aimnetcentral commit -m "feat(models): propagate family/supports_charged_systems in load_model"
```

---

## Task 3: Propagate new metadata fields in `load_from_hf_repo`

**Files:**

- Modify: `aimnet/calculators/hf_hub.py:270-282` (metadata construction in `load_from_hf_repo`)
- Test: `tests/test_hf_hub.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_hf_hub.py` (keep the existing `fake_hf_repo` fixture pattern; create a separate fixture that injects the new fields):

```python
@pytest.fixture
def fake_hf_repo_with_family(tmp_path):
    """A fake HF repo whose config.json declares family + supports_charged_systems."""
    from aimnet.calculators.model_registry import get_model_path

    pt_path = get_model_path("aimnet2")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", ".*weights_only.*")
        raw = torch.load(pt_path, map_location="cpu", weights_only=False)

    state_dict = raw["state_dict"]
    save_file(state_dict, str(tmp_path / "ensemble_0.safetensors"))

    config = {
        "config_schema_version": 1,
        "family_name": "fake-family",
        "ensemble_size": 1,
        "member_names": ["fake_0"],
        "cutoff": float(raw["cutoff"]),
        "needs_coulomb": raw.get("needs_coulomb", False),
        "needs_dispersion": raw.get("needs_dispersion", False),
        "coulomb_mode": raw.get("coulomb_mode", "none"),
        "implemented_species": raw.get("implemented_species", []),
        "model_yaml": raw["model_yaml"],
        "format_version": 2,
        "coulomb_sr_rc": raw.get("coulomb_sr_rc"),
        "coulomb_sr_envelope": raw.get("coulomb_sr_envelope"),
        "has_embedded_lr": raw.get("has_embedded_lr", False),
        # NEW fields under test:
        "family": "test-family",
        "supports_charged_systems": False,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    return tmp_path


def test_load_from_hf_repo_propagates_family_and_charge_fields(fake_hf_repo_with_family):
    _, metadata = load_from_hf_repo(str(fake_hf_repo_with_family), ensemble_member=0, device="cpu")
    assert metadata.get("family") == "test-family"
    assert metadata.get("supports_charged_systems") is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_hf_hub.py::test_load_from_hf_repo_propagates_family_and_charge_fields -v` Expected: FAIL — `metadata.get("family")` returns `None`.

- [ ] **Step 3: Update the metadata construction in `hf_hub.py`**

Edit `aimnet/calculators/hf_hub.py`. Locate `metadata: ModelMetadata = { ... }` (around line 271). Add two lines after `"implemented_species": _cfg("implemented_species", []),`:

```python
        "family": _cfg("family"),
        "supports_charged_systems": _cfg("supports_charged_systems"),
```

The full block now reads:

```python
    metadata: ModelMetadata = {
        "format_version": _cfg("format_version", 2),
        "cutoff": config["cutoff"],
        "needs_coulomb": _cfg("needs_coulomb", False),
        "needs_dispersion": _cfg("needs_dispersion", False),
        "coulomb_mode": _cfg("coulomb_mode", "none"),
        "coulomb_sr_rc": coulomb_sr_rc,
        "coulomb_sr_envelope": coulomb_sr_envelope,
        "d3_params": _cfg("d3_params"),
        "has_embedded_lr": _cfg("has_embedded_lr", False),
        "implemented_species": _cfg("implemented_species", []),
        "family": _cfg("family"),
        "supports_charged_systems": _cfg("supports_charged_systems"),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_hf_hub.py::test_load_from_hf_repo_propagates_family_and_charge_fields -v` Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git -C /home/olexandr/aimnetcentral add aimnet/calculators/hf_hub.py tests/test_hf_hub.py
git -C /home/olexandr/aimnetcentral commit -m "feat(hf_hub): propagate family/supports_charged_systems in load_from_hf_repo"
```

---

## Task 4: Extend `load_v1_model` with override kwargs + AFV sanitization

**Files:**

- Modify: `aimnet/models/utils.py:568-735` (`load_v1_model` function)
- Test: `tests/test_model.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_model.py`:

```python
def test_load_v1_model_species_override_nan_pads_other_rows():
    """load_v1_model with implemented_species override must NaN-pad AFV rows
    for elements outside the supported set, write family/supports_charged_systems
    to metadata, AND keep atomic_shift in float64 (regression on .double() ordering)."""
    import math
    from pathlib import Path

    import pytest
    import torch

    src_jpt = Path("_tmp/model_1.jpt")
    src_yaml = Path("_tmp/config.yaml")
    if not src_jpt.exists() or not src_yaml.exists():
        pytest.skip("aimnet2-rxn JIT source not available in _tmp/; skipping override test")

    from aimnet.models.utils import load_v1_model

    species = [1, 6, 7, 8]
    model, metadata = load_v1_model(
        str(src_jpt),
        str(src_yaml),
        output_path=None,
        implemented_species=species,
        family="rxn",
        supports_charged_systems=False,
        verbose=False,
    )

    assert metadata["implemented_species"] == species
    assert metadata.get("family") == "rxn"
    assert metadata.get("supports_charged_systems") is False

    afv = model.afv.weight.data
    species_set = set(species)
    for z in range(1, afv.shape[0]):
        row = afv[z]
        if z in species_set:
            assert not torch.isnan(row).any(), f"row {z} (in species) must not be NaN"
        else:
            assert torch.isnan(row).all(), f"row {z} (out of species) must be all-NaN"

    # .double() ordering regression: atomic_shift dtype is float64.
    assert hasattr(model, "outputs") and hasattr(model.outputs, "atomic_shift")
    assert model.outputs.atomic_shift.shifts.weight.dtype == torch.float64


def test_load_v1_model_without_overrides_is_backward_compatible():
    """Existing four families' conversions must still work when the new kwargs are omitted."""
    from pathlib import Path

    import pytest

    src_jpt = Path("_tmp/model_1.jpt")
    src_yaml = Path("_tmp/config.yaml")
    if not src_jpt.exists() or not src_yaml.exists():
        pytest.skip("JIT source not available")

    from aimnet.models.utils import load_v1_model

    _, metadata = load_v1_model(str(src_jpt), str(src_yaml), output_path=None, verbose=False)
    # No overrides → species derived from afv.weight (which for rxn means [1..63]).
    assert isinstance(metadata["implemented_species"], list)
    assert metadata.get("family") is None
    assert metadata.get("supports_charged_systems") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_model.py::test_load_v1_model_species_override_nan_pads_other_rows tests/test_model.py::test_load_v1_model_without_overrides_is_backward_compatible -v` Expected: FAIL on the override test (`unexpected keyword argument 'implemented_species'`); the backward-compat test should pass already.

- [ ] **Step 3: Add the new kwargs and AFV sanitization**

Edit `aimnet/models/utils.py`. Locate the `load_v1_model` signature at line ~568:

Replace:

```python
def load_v1_model(
    jpt_path: str,
    yaml_config_path: str,
    output_path: str | None = None,
    verbose: bool = True,
) -> tuple[nn.Module, dict]:
```

With:

```python
def load_v1_model(
    jpt_path: str,
    yaml_config_path: str,
    output_path: str | None = None,
    *,
    implemented_species: list[int] | None = None,
    family: str | None = None,
    supports_charged_systems: bool | None = None,
    verbose: bool = True,
) -> tuple[nn.Module, dict]:
```

Then locate the species extraction (around line 634):

Replace:

```python
    cutoff = float(jit_model.cutoff)
    implemented_species = extract_species(jit_model)
```

With:

```python
    cutoff = float(jit_model.cutoff)
    _species_kwarg = implemented_species  # rename to avoid shadowing the metadata field below
    if _species_kwarg is not None:
        implemented_species_list = sorted(set(_species_kwarg))
    else:
        implemented_species_list = extract_species(jit_model)
```

Then locate the metadata construction near the end of the function (around line 710):

Replace:

```python
    metadata = {
        "format_version": 2,
        "model_yaml": core_yaml_str,
        "cutoff": cutoff,
        "needs_coulomb": needs_coulomb,
        "needs_dispersion": needs_dispersion,
        "coulomb_mode": coulomb_mode,
        "coulomb_sr_rc": coulomb_sr_rc if needs_coulomb else None,
        "coulomb_sr_envelope": coulomb_sr_envelope if needs_coulomb else None,
        "d3_params": d3_params if needs_dispersion else None,
        "has_embedded_lr": has_embedded_lr,
        "implemented_species": implemented_species,
    }
```

With:

```python
    metadata = {
        "format_version": 2,
        "model_yaml": core_yaml_str,
        "cutoff": cutoff,
        "needs_coulomb": needs_coulomb,
        "needs_dispersion": needs_dispersion,
        "coulomb_mode": coulomb_mode,
        "coulomb_sr_rc": coulomb_sr_rc if needs_coulomb else None,
        "coulomb_sr_envelope": coulomb_sr_envelope if needs_coulomb else None,
        "d3_params": d3_params if needs_dispersion else None,
        "has_embedded_lr": has_embedded_lr,
        "implemented_species": implemented_species_list,
    }
    if family is not None:
        metadata["family"] = family
    if supports_charged_systems is not None:
        metadata["supports_charged_systems"] = bool(supports_charged_systems)

    # AFV row sanitization: when the caller declares the supported species
    # explicitly, NaN-pad rows for elements outside that set so
    # validate_species=False at inference time produces NaN-propagation
    # instead of plausible-looking garbage from populated-but-untrained rows.
    if _species_kwarg is not None:
        species_set = set(implemented_species_list)
        afv = core_model.afv.weight.data
        for z in range(1, afv.shape[0]):
            if z not in species_set:
                afv[z] = float("nan")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_model.py::test_load_v1_model_species_override_nan_pads_other_rows tests/test_model.py::test_load_v1_model_without_overrides_is_backward_compatible -v` Expected: PASS for both (or `SKIPPED` if `_tmp/` is not present — that's acceptable).

- [ ] **Step 5: Commit**

```bash
git -C /home/olexandr/aimnetcentral add aimnet/models/utils.py tests/test_model.py
git -C /home/olexandr/aimnetcentral commit -m "feat(models): add implemented_species/family/supports_charged_systems overrides to load_v1_model with AFV sanitization"
```

---

## Task 5: Reorder `.double()` in `load_v1_model`

**Files:**

- Modify: `aimnet/models/utils.py:697-704` (the atomic_shift block inside `load_v1_model`)
- Test: covered by `test_load_v1_model_species_override_nan_pads_other_rows` (Task 4)

- [ ] **Step 1: Verify the test from Task 4 still asserts float64**

Run: `pytest tests/test_model.py::test_load_v1_model_species_override_nan_pads_other_rows -v` Expected: PASS (the existing pre-fix code happens to leave atomic_shift as float64 because it calls `.double()` after `load_state_dict`).

- [ ] **Step 2: Reorder `.double()` to run BEFORE `load_state_dict`**

Edit `aimnet/models/utils.py`. Locate the section around line 686-704 inside `load_v1_model`:

Replace:

```python
    # Load weights from JIT model
    jit_sd = jit_model.state_dict()
    load_result = core_model.load_state_dict(jit_sd, strict=False)

    # Validate keys
    real_missing, real_unexpected = validate_state_dict_keys(load_result.missing_keys, load_result.unexpected_keys)
    if real_missing:
        print(f"WARNING: Unexpected missing keys: {real_missing}")
    if real_unexpected:
        print(f"WARNING: Unexpected extra keys: {real_unexpected}")
    if not real_missing and not real_unexpected and verbose:
        print("Loaded weights successfully")

    # Convert atomic_shift to float64 to preserve SAE precision
    if hasattr(core_model, "outputs") and hasattr(core_model.outputs, "atomic_shift"):
        core_model.outputs.atomic_shift.double()
        atomic_shift_key = "outputs.atomic_shift.shifts.weight"
        if atomic_shift_key in jit_sd:
            core_model.outputs.atomic_shift.shifts.weight.data.copy_(jit_sd[atomic_shift_key])
            if verbose:
                print("  Atomic shift converted to float64")
```

With:

```python
    # Cast atomic_shift to float64 BEFORE load_state_dict so the destination
    # buffer can hold full precision when load_state_dict's internal copy_ runs.
    # Order matters: doing this after load_state_dict + a redundant copy_ from
    # the still-float32 source is the bug this fixes.
    if hasattr(core_model, "outputs") and hasattr(core_model.outputs, "atomic_shift"):
        core_model.outputs.atomic_shift.double()
        if verbose:
            print("  Atomic shift cast to float64 before load_state_dict")

    # Load weights from JIT model
    jit_sd = jit_model.state_dict()
    load_result = core_model.load_state_dict(jit_sd, strict=False)

    # Validate keys
    real_missing, real_unexpected = validate_state_dict_keys(load_result.missing_keys, load_result.unexpected_keys)
    if real_missing:
        print(f"WARNING: Unexpected missing keys: {real_missing}")
    if real_unexpected:
        print(f"WARNING: Unexpected extra keys: {real_unexpected}")
    if not real_missing and not real_unexpected and verbose:
        print("Loaded weights successfully")
```

- [ ] **Step 3: Run the regression test**

Run: `pytest tests/test_model.py::test_load_v1_model_species_override_nan_pads_other_rows -v` Expected: PASS — atomic_shift dtype is still float64; numerical behavior is byte-identical for current JIT models (atomic_shift in source is float32).

- [ ] **Step 4: Run the full test_model.py to catch any unrelated regressions**

Run: `pytest tests/test_model.py -v` Expected: All tests PASS or `SKIPPED` (no FAIL).

- [ ] **Step 5: Commit**

```bash
git -C /home/olexandr/aimnetcentral add aimnet/models/utils.py
git -C /home/olexandr/aimnetcentral commit -m "fix(models): cast atomic_shift to float64 before load_state_dict in load_v1_model"
```

---

## Task 6: Add docstring example to `load_v1_model`

**Files:**

- Modify: `aimnet/models/utils.py` (the docstring of `load_v1_model`)

- [ ] **Step 1: Locate the docstring**

Open `aimnet/models/utils.py` and find the `load_v1_model` docstring (starts around line 574, ends around line 615 with the `Warnings` section).

- [ ] **Step 2: Add an Examples section**

Just before the existing `Warnings` section in the docstring, insert:

```python
    Examples
    --------
    Convert a single legacy JIT model to the v2 .pt format:

    >>> from aimnet.models.utils import load_v1_model
    >>> model, metadata = load_v1_model("model.jpt", "config.yaml")

    Convert the four aimnet2-rxn ensemble members with explicit species,
    family, and charge-support declarations (writes AFV-sanitized .pt files
    with rows for elements outside [1, 6, 7, 8] NaN-padded):

    >>> for i in range(4):
    ...     load_v1_model(
    ...         f"_tmp/model_{i+1}.jpt",
    ...         "aimnet/models/aimnet2_rxn.yaml",
    ...         output_path=f"aimnet2_rxn_{i}.pt",
    ...         implemented_species=[1, 6, 7, 8],
    ...         family="rxn",
    ...         supports_charged_systems=False,
    ...     )

```

- [ ] **Step 3: Verify the docstring renders without raising**

Run: `python -c "from aimnet.models.utils import load_v1_model; help(load_v1_model)"` Expected: docstring prints with no exception; `Examples` section visible.

- [ ] **Step 4: Commit**

```bash
git -C /home/olexandr/aimnetcentral add aimnet/models/utils.py
git -C /home/olexandr/aimnetcentral commit -m "docs(models): add aimnet2-rxn conversion example to load_v1_model docstring"
```

---

## Task 7: Add architecture YAML for aimnet2-rxn

**Files:**

- Create: `aimnet/models/aimnet2_rxn.yaml`
- Test: `tests/test_model.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_model.py`:

```python
def test_aimnet2_rxn_yaml_builds():
    """The architecture YAML for aimnet2-rxn must be loadable and produce a real AIMNet2 module."""
    import importlib.resources
    import yaml

    from aimnet.config import build_module

    yaml_text = importlib.resources.files("aimnet.models").joinpath("aimnet2_rxn.yaml").read_text()
    cfg = yaml.safe_load(yaml_text)
    model = build_module(cfg)

    # The reconstructed module must have the rxn output heads.
    assert hasattr(model, "outputs")
    assert hasattr(model.outputs, "energy_mlp")
    assert hasattr(model.outputs, "atomic_shift")
    assert hasattr(model.outputs, "atomic_sum")
    assert hasattr(model.outputs, "dipole")
    assert hasattr(model.outputs, "quadrupole")
    assert hasattr(model.outputs, "lrcoulomb")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_model.py::test_aimnet2_rxn_yaml_builds -v` Expected: FAIL — `aimnet2_rxn.yaml` does not exist yet.

- [ ] **Step 3: Create the YAML**

Create `aimnet/models/aimnet2_rxn.yaml` with this exact content (this is the same architecture already embedded in the per-member `.pt` `model_yaml` field and in the HF `config.json`):

```yaml
class: aimnet.models.AIMNet2
kwargs:
  nfeature: 16
  d2features: true
  ncomb_v: 12
  hidden:
    - [512, 380]
    - [512, 380]
    - [512, 380, 380]
  aim_size: 256
  num_charge_channels: 1
  aev:
    rc_s: 5.0
    nshifts_s: 16
  outputs:
    energy_mlp:
      class: aimnet.modules.Output
      kwargs:
        n_in: 256
        n_out: 1
        key_in: aim
        key_out: energy
        mlp:
          activation_fn: torch.nn.GELU
          last_linear: true
          hidden: [128, 128]
    atomic_shift:
      class: aimnet.modules.AtomicShift
      kwargs:
        key_in: energy
        key_out: energy
    atomic_sum:
      class: aimnet.modules.AtomicSum
      kwargs:
        key_in: energy
        key_out: energy
    dipole:
      class: aimnet.modules.Dipole
      kwargs:
        key_in: charges
        key_out: dipole
    quadrupole:
      class: aimnet.modules.Quadrupole
      kwargs:
        key_in: charges
        key_out: quadrupole
    lrcoulomb:
      class: aimnet.modules.LRCoulomb
      kwargs:
        rc: 4.6
        key_in: charges
        key_out: energy
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_model.py::test_aimnet2_rxn_yaml_builds -v` Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git -C /home/olexandr/aimnetcentral add aimnet/models/aimnet2_rxn.yaml tests/test_model.py
git -C /home/olexandr/aimnetcentral commit -m "feat(models): add aimnet2_rxn.yaml architecture descriptor"
```

---

## Task 8: Add registry entries for aimnet2-rxn

**Files:**

- Modify: `aimnet/calculators/model_registry.yaml` (append entries + alias)
- Test: `tests/test_model_registry.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_model_registry.py`:

```python
def test_aimnet2rxn_registry_entries_and_alias():
    """All four aimnet2-rxn members must be registered with the canonical GCS URL,
    and the `aimnet2rxn` alias must resolve to member 0."""
    from aimnet.calculators.model_registry import load_model_registry

    registry = load_model_registry()
    models = registry["models"]

    base_url = "https://storage.googleapis.com/aimnetcentral/aimnet2v2/AIMNet2rxn"
    for i in range(4):
        name = f"aimnet2_rxn_{i}"
        assert name in models, f"missing registry entry: {name}"
        entry = models[name]
        assert entry["file"] == f"{name}.pt"
        assert entry["url"] == f"{base_url}/{name}.pt"

    aliases = registry["aliases"]
    assert aliases.get("aimnet2rxn") == "aimnet2_rxn_0"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_model_registry.py::test_aimnet2rxn_registry_entries_and_alias -v` Expected: FAIL — `KeyError` or `assertion` because the entries don't exist.

- [ ] **Step 3: Add the entries to `model_registry.yaml`**

Edit `aimnet/calculators/model_registry.yaml`. Locate the last `aimnet2-pd_3:` entry (around line 60-62). Add immediately after it (before the `# map model alias` comment):

```yaml
aimnet2_rxn_0:
  file: aimnet2_rxn_0.pt
  url: https://storage.googleapis.com/aimnetcentral/aimnet2v2/AIMNet2rxn/aimnet2_rxn_0.pt
aimnet2_rxn_1:
  file: aimnet2_rxn_1.pt
  url: https://storage.googleapis.com/aimnetcentral/aimnet2v2/AIMNet2rxn/aimnet2_rxn_1.pt
aimnet2_rxn_2:
  file: aimnet2_rxn_2.pt
  url: https://storage.googleapis.com/aimnetcentral/aimnet2v2/AIMNet2rxn/aimnet2_rxn_2.pt
aimnet2_rxn_3:
  file: aimnet2_rxn_3.pt
  url: https://storage.googleapis.com/aimnetcentral/aimnet2v2/AIMNet2rxn/aimnet2_rxn_3.pt
```

In the `aliases:` section at the end, add the alias right after `aimnet2_2025: aimnet2_b973c_2025_d3_0`:

```yaml
aimnet2rxn: aimnet2_rxn_0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_model_registry.py::test_aimnet2rxn_registry_entries_and_alias -v` Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git -C /home/olexandr/aimnetcentral add aimnet/calculators/model_registry.yaml tests/test_model_registry.py
git -C /home/olexandr/aimnetcentral commit -m "feat(registry): register aimnet2-rxn family (4 GCS entries + aimnet2rxn alias)"
```

---

## Task 9: Add `metadata` property and `_was_compiled` flag to `AIMNet2Calculator`

**Files:**

- Modify: `aimnet/calculators/calculator.py` (`__init__` + new property)
- Test: `tests/test_calculator.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_calculator.py`:

```python
def test_calculator_metadata_property_returns_model_metadata():
    """AIMNet2Calculator.metadata must return the same dict as model._metadata."""
    from aimnet.calculators import AIMNet2Calculator

    calc = AIMNet2Calculator("aimnet2", device="cpu")
    assert calc.metadata is calc.model._metadata
    # Existing aimnet2 family declares neither family nor supports_charged_systems.
    assert calc.metadata.get("family") is None
    assert calc.metadata.get("supports_charged_systems") is None


def test_calculator_was_compiled_flag_default_false():
    from aimnet.calculators import AIMNet2Calculator

    calc = AIMNet2Calculator("aimnet2", device="cpu")
    assert calc._was_compiled is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_calculator.py::test_calculator_metadata_property_returns_model_metadata tests/test_calculator.py::test_calculator_was_compiled_flag_default_false -v` Expected: FAIL — `metadata` does not exist as a property; `_was_compiled` not set.

- [ ] **Step 3: Add the `_was_compiled` flag in `__init__`**

Edit `aimnet/calculators/calculator.py`. Locate the constructor block around line 286-291 where `compile_model` is handled:

Replace:

```python
        # Compile model if requested
        if compile_model:
            kwargs = compile_kwargs or {}
            self.model = torch.compile(self.model, **kwargs)
```

With:

```python
        # Compile model if requested
        self._was_compiled = bool(compile_model)
        if compile_model:
            kwargs = compile_kwargs or {}
            self.model = torch.compile(self.model, **kwargs)
```

- [ ] **Step 4: Add the `metadata` property**

In the same file, locate the existing `@property` for `has_external_coulomb` (around line 417). Just before it, add:

```python
    @property
    def metadata(self) -> dict | None:
        """Read-only view of the model's metadata dict.

        Returns the same object as ``model._metadata`` for v2 .pt models,
        or ``None`` for raw ``nn.Module`` inputs that don't carry metadata.
        Downstream consumers should prefer this accessor over reaching into
        the private ``model._metadata`` attribute.
        """
        return getattr(self.model, "_metadata", None)
```

- [ ] **Step 5: Migrate existing reads to use the property (lines 490 and 515)**

Locate the two existing `getattr(self.model, "_metadata", None)` calls outside `__init__`:

At around line 490, replace `meta = getattr(self.model, "_metadata", None)` with `meta = self.metadata`.

At around line 515, replace `meta = getattr(self.model, "_metadata", None)` with `meta = self.metadata`.

**Do NOT change** the `getattr(self.model, "_metadata", None)` at around line 284 — that one is inside `__init__` and runs before `self.model` is fully usable as a property surface; leaving it as `getattr` keeps the constructor independent of the property.

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_calculator.py::test_calculator_metadata_property_returns_model_metadata tests/test_calculator.py::test_calculator_was_compiled_flag_default_false -v` Expected: PASS.

- [ ] **Step 7: Run the full calculator test suite to catch regressions**

Run: `pytest tests/test_calculator.py -v -m "not gpu and not hf"` Expected: All non-skipped tests PASS.

- [ ] **Step 8: Commit**

```bash
git -C /home/olexandr/aimnetcentral add aimnet/calculators/calculator.py tests/test_calculator.py
git -C /home/olexandr/aimnetcentral commit -m "feat(calculator): add metadata property and _was_compiled flag"
```

---

## Task 10: Add species-validation guard in `eval`

**Files:**

- Modify: `aimnet/calculators/calculator.py:724` (`eval` method signature + new validation block at top)
- Test: `tests/test_calculator.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_calculator.py`:

```python
def test_calculator_rejects_unsupported_species():
    """Calling the calculator with an unsupported atomic number must raise ValueError
    with chemistry context and pointers to alternative models."""
    import pytest
    import torch

    from aimnet.calculators import AIMNet2Calculator

    calc = AIMNet2Calculator("aimnet2", device="cpu")
    # aimnet2's implemented_species does NOT include Z=92 (uranium).
    coords = torch.tensor([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
    numbers = torch.tensor([1, 92])  # H + U; U is unsupported
    data = {"coord": coords, "numbers": numbers, "charge": torch.tensor(0.0)}

    with pytest.raises(ValueError, match=r"implemented_species"):
        calc(data)


def test_calculator_validate_species_false_bypasses():
    """Passing validate_species=False must skip the species check (no ValueError)."""
    import torch

    from aimnet.calculators import AIMNet2Calculator

    calc = AIMNet2Calculator("aimnet2", device="cpu")
    # H is supported by aimnet2, so this should not raise even without bypass;
    # the test asserts the kwarg flows through and does not itself raise.
    coords = torch.tensor([[0.0, 0.0, 0.0]])
    numbers = torch.tensor([1])
    data = {"coord": coords, "numbers": numbers, "charge": torch.tensor(0.0)}

    # Both calls should succeed; validate_species=False is the explicit bypass path.
    calc(data, validate_species=True)
    calc(data, validate_species=False)
```

- [ ] **Step 2: Run tests to verify the first fails**

Run: `pytest tests/test_calculator.py::test_calculator_rejects_unsupported_species tests/test_calculator.py::test_calculator_validate_species_false_bypasses -v` Expected: FIRST test FAILs (calculator silently produces output for U); the bypass test may FAIL with "got unexpected keyword argument 'validate_species'".

- [ ] **Step 3: Add `validate_species` to the `eval` signature**

Edit `aimnet/calculators/calculator.py`. Locate `def eval(self, data: dict[str, Any], forces=False, stress=False, hessian=False)` at line 724.

Replace with:

```python
    def eval(self, data: dict[str, Any], forces=False, stress=False, hessian=False,
             *, validate_species: bool = True) -> dict[str, Tensor]:
```

- [ ] **Step 4: Add the species-validation block at the top of `eval`**

Immediately after the new signature line and BEFORE the existing `data = self.prepare_input(data)` line at line 725, insert:

```python
        # Species validation — opt-out via validate_species=False.
        # Silent no-op for models that did not declare implemented_species (older .pt,
        # raw nn.Module).
        if validate_species:
            impl = (self.metadata or {}).get("implemented_species") or []
            if impl:
                seen = {int(z) for z in data["numbers"].flatten().tolist() if int(z) > 0}
                unsupported = sorted(seen - set(impl))
                if unsupported:
                    raise ValueError(
                        f"Atomic numbers {unsupported} are not in this model's "
                        f"implemented_species {sorted(impl)}. This model was trained on "
                        f"a restricted element set; passing other elements yields undefined "
                        f"output. For broader element coverage on equilibrium structures use "
                        f"`isayevlab/aimnet2-wb97m-d3`; for radicals/open-shell systems use "
                        f"`isayevlab/aimnet2-nse`. Pass validate_species=False to bypass."
                    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_calculator.py::test_calculator_rejects_unsupported_species tests/test_calculator.py::test_calculator_validate_species_false_bypasses -v` Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git -C /home/olexandr/aimnetcentral add aimnet/calculators/calculator.py tests/test_calculator.py
git -C /home/olexandr/aimnetcentral commit -m "feat(calculator): add validate_species ValueError on unsupported atomic numbers"
```

---

## Task 11: Add charge-guard for `supports_charged_systems: false` models

**Files:**

- Modify: `aimnet/calculators/calculator.py` (`eval` — append a second guard inside the `if validate_species` block)
- Test: `tests/test_calculator.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_calculator.py`:

```python
def test_calculator_rejects_charged_input_when_unsupported(monkeypatch):
    """When metadata declares supports_charged_systems=False, a non-zero charge raises."""
    import pytest
    import torch

    from aimnet.calculators import AIMNet2Calculator

    calc = AIMNet2Calculator("aimnet2", device="cpu")
    # Synthetically inject the family-narrowing metadata (aimnet2 does not declare it
    # natively; this mirrors what an aimnet2-rxn .pt would carry).
    calc.model._metadata = dict(calc.model._metadata)
    calc.model._metadata["supports_charged_systems"] = False

    coords = torch.tensor([[0.0, 0.0, 0.0]])
    numbers = torch.tensor([1])
    data = {"coord": coords, "numbers": numbers, "charge": torch.tensor(-1.0)}

    with pytest.raises(ValueError, match=r"net-charged systems"):
        calc(data)

    # Bypass works
    calc(data, validate_species=False)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_calculator.py::test_calculator_rejects_charged_input_when_unsupported -v` Expected: FAIL — no charge guard yet.

- [ ] **Step 3: Add the charge guard inside `eval`**

Edit `aimnet/calculators/calculator.py`. Locate the species-validation block added in Task 10 (still inside the `if validate_species:` clause). After the species check raises, append (still inside the `if validate_species:` outer block):

```python
            meta = self.metadata or {}
            if meta.get("supports_charged_systems") is False:
                charge_val = float(data.get("charge", 0.0))
                if abs(charge_val) > 1e-6:
                    raise ValueError(
                        f"This model does not support net-charged systems "
                        f"(got charge={charge_val}). Net-neutral zwitterions are supported. "
                        f"For ions use `isayevlab/aimnet2-wb97m-d3`. "
                        f"Pass validate_species=False to bypass."
                    )
```

(`charge` may arrive as a tensor; `float(tensor)` returns a Python float for 0-d tensors. For batched 1-d charge tensors, `float(...)` will raise — that's correct behavior because per-system charge mixing requires per-system validation, which is out of scope for this guard.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_calculator.py::test_calculator_rejects_charged_input_when_unsupported -v` Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git -C /home/olexandr/aimnetcentral add aimnet/calculators/calculator.py tests/test_calculator.py
git -C /home/olexandr/aimnetcentral commit -m "feat(calculator): add charge guard for models that declare supports_charged_systems=False"
```

---

## Task 12: Add `hessian + compile_model` guard

**Files:**

- Modify: `aimnet/calculators/calculator.py` (`eval` — guard added after the validate_species block)
- Test: `tests/test_calculator.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_calculator.py`:

```python
def test_hessian_with_compile_raises():
    """Calling with hessian=True on a calculator constructed with compile_model=True
    must raise RuntimeError instead of hanging (Dynamo + double-backward on GELU)."""
    import pytest
    import torch

    from aimnet.calculators import AIMNet2Calculator

    calc = AIMNet2Calculator("aimnet2", device="cpu")
    # Don't actually torch.compile (slow + may need GPU); just flip the flag.
    calc._was_compiled = True

    coords = torch.tensor([[0.0, 0.0, 0.0]])
    numbers = torch.tensor([1])
    data = {"coord": coords, "numbers": numbers, "charge": torch.tensor(0.0)}

    with pytest.raises(RuntimeError, match=r"Hessian computation is incompatible with compile_model=True"):
        calc(data, hessian=True)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_calculator.py::test_hessian_with_compile_raises -v` Expected: FAIL — no guard yet (test hangs OR proceeds to the hessian path).

- [ ] **Step 3: Add the guard in `eval`**

Edit `aimnet/calculators/calculator.py`. The Task-11 charge-guard block ends inside `if validate_species:`. Just AFTER that whole `if validate_species:` block (i.e. unconditional — the Hessian guard always fires regardless of `validate_species`), and BEFORE `data = self.prepare_input(data)`, insert:

```python
        # Hessian + torch.compile is known to hang on the double-backward
        # path through GELU activations. Fail fast instead.
        if hessian and getattr(self, "_was_compiled", False):
            raise RuntimeError(
                "Hessian computation is incompatible with compile_model=True "
                "(Dynamo + double-backward through GELU hangs). Reconstruct calculator "
                "with compile_model=False."
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_calculator.py::test_hessian_with_compile_raises -v` Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git -C /home/olexandr/aimnetcentral add aimnet/calculators/calculator.py tests/test_calculator.py
git -C /home/olexandr/aimnetcentral commit -m "feat(calculator): raise RuntimeError on hessian=True + compile_model=True (was: silent hang)"
```

---

## Task 13: Add `set_lrcoulomb_method` rxn-family cutoff guard

**Files:**

- Modify: `aimnet/calculators/calculator.py:601-670` (`set_lrcoulomb_method`)
- Test: `tests/test_calculator.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_calculator.py`:

```python
def test_set_lrcoulomb_method_warns_on_rxn_cutoff_change():
    """For family='rxn', changing the coulomb cutoff away from coulomb_sr_rc
    (4.6 A) must emit a UserWarning about SR/LR matching."""
    import pytest
    import warnings

    from aimnet.calculators import AIMNet2Calculator

    calc = AIMNet2Calculator("aimnet2", device="cpu")
    calc.model._metadata = dict(calc.model._metadata)
    calc.model._metadata["family"] = "rxn"
    calc.model._metadata["coulomb_sr_rc"] = 4.6

    with pytest.warns(UserWarning, match=r"SR/LR"):
        calc.set_lrcoulomb_method("dsf", cutoff=10.0)


def test_set_lrcoulomb_method_no_warn_on_matching_cutoff():
    """No warning when cutoff matches coulomb_sr_rc, or for non-rxn families."""
    import warnings

    from aimnet.calculators import AIMNet2Calculator

    # Non-rxn family: never warn about SR/LR.
    calc1 = AIMNet2Calculator("aimnet2", device="cpu")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        calc1.set_lrcoulomb_method("dsf", cutoff=10.0)
    sr_lr_warnings = [w for w in caught if "SR/LR" in str(w.message)]
    assert sr_lr_warnings == []
```

- [ ] **Step 2: Run tests to verify the first fails**

Run: `pytest tests/test_calculator.py::test_set_lrcoulomb_method_warns_on_rxn_cutoff_change tests/test_calculator.py::test_set_lrcoulomb_method_no_warn_on_matching_cutoff -v` Expected: First test FAILs (no warning raised); second test PASSes already (no warning is the current behavior).

- [ ] **Step 3: Add the rxn guard at the top of `set_lrcoulomb_method`**

Edit `aimnet/calculators/calculator.py`. Locate `def set_lrcoulomb_method(...)` at line 601. After the existing `if method not in (...): raise ValueError` validation (around line 635-636), and before the `# Warn if model has embedded Coulomb` block (around line 638), insert:

```python
        # rxn-family guard: the 4.6 A SR/LR cancellation point is physically
        # frozen for this family. Changing the cutoff silently breaks matching.
        meta = self.metadata or {}
        if meta.get("family") == "rxn":
            sr_rc = meta.get("coulomb_sr_rc")
            if sr_rc is not None and method in ("dsf", "ewald") and abs(cutoff - float(sr_rc)) > 1e-6:
                warnings.warn(
                    f"Setting Coulomb {method} cutoff to {cutoff} A on aimnet2-rxn breaks "
                    f"the SR/LR cancellation matching (this family was trained with a "
                    f"physically frozen crossover at coulomb_sr_rc={sr_rc} A). Use the "
                    f"matching cutoff or revert to the default external Coulomb.",
                    UserWarning,
                    stacklevel=2,
                )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_calculator.py::test_set_lrcoulomb_method_warns_on_rxn_cutoff_change tests/test_calculator.py::test_set_lrcoulomb_method_no_warn_on_matching_cutoff -v` Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git -C /home/olexandr/aimnetcentral add aimnet/calculators/calculator.py tests/test_calculator.py
git -C /home/olexandr/aimnetcentral commit -m "feat(calculator): warn on set_lrcoulomb_method cutoff mismatch for rxn family"
```

---

## Task 14: Add cross-family mixing detection

**Files:**

- Modify: `aimnet/calculators/calculator.py` (add ClassVar + record-and-warn at end of `__init__`)
- Test: `tests/test_calculator.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_calculator.py`:

```python
def test_constructing_two_families_warns_once(monkeypatch):
    """Constructing calculators from two different families in one process must
    emit a UserWarning about energy-scale incompatibility."""
    import pytest
    import warnings

    from aimnet.calculators import AIMNet2Calculator

    # Reset the class-level set so test order does not pollute it.
    AIMNet2Calculator._constructed_families.clear()
    monkeypatch.delenv("AIMNET_QUIET_FAMILY_MIX", raising=False)

    # First calculator: synthetic 'family-A'.
    calc_a = AIMNet2Calculator("aimnet2", device="cpu")
    calc_a.model._metadata = dict(calc_a.model._metadata)
    calc_a.model._metadata["family"] = "family-A"
    AIMNet2Calculator._constructed_families.add("family-A")  # mimic what __init__ does

    # Second calculator: a different family — should warn.
    with pytest.warns(UserWarning, match=r"different families"):
        calc_b = AIMNet2Calculator("aimnet2", device="cpu")
        # The constructor itself must add the (synthetic) family. Since we cannot
        # change metadata before __init__ finishes, simulate by constructing then
        # injecting+re-running the warn step. For this PR's purposes the warn
        # logic lives in __init__ AFTER load. Test it by direct invocation:
        calc_b.model._metadata = dict(calc_b.model._metadata)
        calc_b.model._metadata["family"] = "family-B"
        # Call the production helper that __init__ calls (see Step 3).
        calc_b._maybe_warn_family_mix("family-B")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_calculator.py::test_constructing_two_families_warns_once -v` Expected: FAIL — `_constructed_families` and `_maybe_warn_family_mix` do not exist.

- [ ] **Step 3: Add the ClassVar and helper, and call from `__init__`**

Edit `aimnet/calculators/calculator.py`. Inside the `class AIMNet2Calculator:` body, find the existing class-level declarations (e.g. the `_HF_ID_RE` line is local to `__init__` — the right place is alongside other class attributes). Add right above `def __init__(`:

```python
    _constructed_families: ClassVar[set[str]] = set()
```

Then add the helper method just after the `metadata` property (added in Task 9):

```python
    def _maybe_warn_family_mix(self, family: str | None) -> None:
        """If multiple distinct families have been constructed in this process,
        emit a one-time UserWarning about energy-scale incompatibility.

        Bypass: set the AIMNET_QUIET_FAMILY_MIX environment variable to '1'.
        """
        if family is None:
            return
        if os.environ.get("AIMNET_QUIET_FAMILY_MIX") == "1":
            self._constructed_families.add(family)
            return
        already_warned = family in self._constructed_families
        self._constructed_families.add(family)
        if not already_warned and len(self._constructed_families) > 1:
            others = sorted(self._constructed_families - {family})
            warnings.warn(
                f"AIMNet2Calculator instances from different families have been "
                f"constructed in this process: {sorted(self._constructed_families)}. "
                f"Energy scales differ across families (e.g. rxn uses a learned "
                f"shifted-electronic scale; aimnet2-wb97m-d3 uses absolute "
                f"electronic energies on the ~-1100 eV scale). Do not mix or compare "
                f"energies across families. Set AIMNET_QUIET_FAMILY_MIX=1 to silence.",
                UserWarning,
                stacklevel=2,
            )
```

Then near the END of `__init__` (after metadata is fully resolved, e.g. just before any final return), add:

```python
        self._maybe_warn_family_mix((metadata or {}).get("family") if metadata else None)
```

The right place is at the very end of `__init__` — find the last line of the constructor body and append the call after it. Ensure `import os` exists at module top (if not, add `import os` to the import block).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_calculator.py::test_constructing_two_families_warns_once -v` Expected: PASS.

- [ ] **Step 5: Run the full test_calculator.py to catch regressions**

Run: `pytest tests/test_calculator.py -v -m "not gpu and not hf"` Expected: All non-skipped tests PASS.

- [ ] **Step 6: Commit**

```bash
git -C /home/olexandr/aimnetcentral add aimnet/calculators/calculator.py tests/test_calculator.py
git -C /home/olexandr/aimnetcentral commit -m "feat(calculator): warn once on cross-family construction (energy-scale incompatibility)"
```

---

## Task 15: Propagate `validate_species` through `AIMNet2ASE`

**Files:**

- Modify: `aimnet/calculators/aimnet2ase.py:24-45` (`__init__`) and `aimnet2ase.py:121-145` (`calculate`)
- Test: `tests/test_ase.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ase.py` (do this only if ASE is importable — guard with the existing pattern in the file):

```python
@pytest.mark.ase
def test_aimnet2ase_propagates_validate_species(monkeypatch):
    """AIMNet2ASE.calculate(..., validate_species=False) must propagate the kwarg
    through to the underlying AIMNet2Calculator.__call__."""
    from ase import Atoms
    from aimnet.calculators.aimnet2ase import AIMNet2ASE

    calc = AIMNet2ASE("aimnet2")
    # H2O — only H, O which are supported. Use validate_species=False to verify the
    # kwarg flows through (the call should succeed regardless).
    atoms = Atoms("H2O", positions=[[0,0,0],[0,0,1],[0,1,0]])
    atoms.calc = calc

    # Fast path: just access energy. This exercises the full pipeline.
    _ = atoms.get_potential_energy()  # baseline default validate_species=True
    # Re-run with the explicit kwarg via the constructor escape hatch:
    calc2 = AIMNet2ASE("aimnet2", validate_species=False)
    atoms.calc = calc2
    _ = atoms.get_potential_energy()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ase.py::test_aimnet2ase_propagates_validate_species -v` Expected: FAIL — `AIMNet2ASE.__init__` does not accept `validate_species`.

- [ ] **Step 3: Add `validate_species` to `AIMNet2ASE.__init__`**

Edit `aimnet/calculators/aimnet2ase.py`. Locate `def __init__(self, base_calc: AIMNet2Calculator | str = "aimnet2", charge=0, mult=1):` at line 24.

Replace with:

```python
    def __init__(
        self,
        base_calc: AIMNet2Calculator | str = "aimnet2",
        charge=0,
        mult=1,
        validate_species: bool = True,
    ):
```

After the existing assignment of `base_calc` to `self.base_calc` (around line 27-28), add:

```python
        self.validate_species = validate_species
```

- [ ] **Step 4: Pass it through in `calculate`**

Locate the `results = self.base_calc(_in, forces="forces" in properties, stress="stress" in properties)` line (around line 145).

Replace with:

```python
        results = self.base_calc(
            _in,
            forces="forces" in properties,
            stress="stress" in properties,
            validate_species=self.validate_species,
        )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_ase.py::test_aimnet2ase_propagates_validate_species -v` Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git -C /home/olexandr/aimnetcentral add aimnet/calculators/aimnet2ase.py tests/test_ase.py
git -C /home/olexandr/aimnetcentral commit -m "feat(ase): propagate validate_species kwarg through AIMNet2ASE"
```

---

## Task 16: Propagate `validate_species` through `AIMNet2Pysis`

**Files:**

- Modify: `aimnet/calculators/aimnet2pysis.py:21-65` (`__init__` + the call site that invokes `self.base_calc`)
- Test: NOT added — pysisyphus is heavy and behind a marker; manual smoke is sufficient. Skip a test for this wrapper.

- [ ] **Step 1: Inspect the current pysisyphus call site**

Run: `grep -n "self.base_calc\|self\.calc\|base_calc(" /home/olexandr/aimnetcentral/aimnet/calculators/aimnet2pysis.py`

Read the printed lines to identify the single point where the wrapper invokes the underlying calculator.

- [ ] **Step 2: Add `validate_species` to `__init__`**

Edit `aimnet/calculators/aimnet2pysis.py`. Locate `def __init__(self, model: AIMNet2Calculator | str = "aimnet2", charge=0, mult=1, **kwargs):` at line 21.

Replace with:

```python
    def __init__(self, model: AIMNet2Calculator | str = "aimnet2", charge=0, mult=1,
                 validate_species: bool = True, **kwargs):
```

Right after the existing model assignment block (look for `self.base_calc = model` or equivalent — around line 24-25), add:

```python
        self.validate_species = validate_species
```

- [ ] **Step 3: Pass `validate_species=self.validate_species` at the call site**

Find the location identified in Step 1 (the single `self.base_calc(...)` invocation). Add `validate_species=self.validate_species` to its kwargs.

For example, if the call looks like `results = self.base_calc(in_data, forces=True)`, change to `results = self.base_calc(in_data, forces=True, validate_species=self.validate_species)`.

- [ ] **Step 4: Smoke-import the module to catch syntax errors**

Run: `python -c "from aimnet.calculators.aimnet2pysis import AIMNet2Pysis; print('OK')"` Expected: `OK` (no exception).

- [ ] **Step 5: Commit**

```bash
git -C /home/olexandr/aimnetcentral add aimnet/calculators/aimnet2pysis.py
git -C /home/olexandr/aimnetcentral commit -m "feat(pysis): propagate validate_species kwarg through AIMNet2Pysis"
```

---

## Task 17: Alias end-to-end test for `aimnet2rxn`

**Files:**

- Test: `tests/test_calculator.py` (append)

This test exercises Tasks 7 + 8 + 9 together: the alias resolves, the .pt downloads (or hits cache), the calculator constructs cleanly, and the metadata carries the rxn-specific fields. Network-gated because it needs the GCS object — only meaningful AFTER the maintainer has uploaded the re-converted .pt files.

- [ ] **Step 1: Write the test (with skip if GCS file is missing)**

Append to `tests/test_calculator.py`:

```python
@pytest.mark.network
def test_aimnet2rxn_alias_calculator_e2e():
    """Alias 'aimnet2rxn' must resolve through the registry, the .pt must load,
    and the calculator must expose rxn-specific metadata fields."""
    import urllib.error
    import pytest

    from aimnet.calculators import AIMNet2Calculator

    try:
        calc = AIMNet2Calculator("aimnet2rxn", device="cpu")
    except (urllib.error.HTTPError, urllib.error.URLError) as e:
        pytest.skip(f"GCS .pt for aimnet2rxn not yet uploaded: {e}")

    assert calc.metadata is not None
    assert calc.metadata.get("family") == "rxn"
    assert calc.metadata.get("supports_charged_systems") is False
    assert calc.metadata.get("implemented_species") == [1, 6, 7, 8]
    assert abs(calc.metadata.get("cutoff") - 5.0) < 1e-6
```

- [ ] **Step 2: Add `network` marker to `pyproject.toml` if not present**

Check: `grep -n "network" /home/olexandr/aimnetcentral/pyproject.toml` If `"network: ..."` is not in the markers list, add to the `markers = [...]` block in `[tool.pytest.ini_options]`:

```toml
    "network: marks tests that hit external services (deselect with: -m 'not network')",
```

- [ ] **Step 3: Run the test (will skip if GCS not yet populated)**

Run: `pytest tests/test_calculator.py::test_aimnet2rxn_alias_calculator_e2e -v` Expected: SKIPPED with "GCS .pt for aimnet2rxn not yet uploaded" (acceptable for CI before maintainer uploads), or PASS if GCS objects already exist.

- [ ] **Step 4: Commit**

```bash
git -C /home/olexandr/aimnetcentral add tests/test_calculator.py pyproject.toml
git -C /home/olexandr/aimnetcentral commit -m "test(calculator): add aimnet2rxn alias end-to-end test (network-gated)"
```

---

## Task 18: HF end-to-end test for aimnet2-rxn

**Files:**

- Test: `tests/test_hf_hub.py` (append)

- [ ] **Step 1: Write the test**

Append to `tests/test_hf_hub.py`:

```python
@pytest.mark.hf
def test_aimnet2_rxn_hf_load_matches_gcs_metadata():
    """Loading aimnet2-rxn from the HF repo must produce metadata equivalent
    to what the GCS .pt path would produce: same implemented_species, cutoff,
    family, and supports_charged_systems.

    This test EXPECTS the HF repo's config.json to have been updated with
    `family: rxn` and `supports_charged_systems: false` (out-of-band task)."""
    from aimnet.calculators import AIMNet2Calculator

    calc = AIMNet2Calculator("isayevlab/aimnet2-rxn", ensemble_member=0, device="cpu")

    assert calc.metadata.get("implemented_species") == [1, 6, 7, 8]
    assert abs(calc.metadata.get("cutoff") - 5.0) < 1e-6
    assert calc.metadata.get("coulomb_mode") == "sr_embedded"
    assert calc.metadata.get("needs_coulomb") is True
    assert calc.metadata.get("needs_dispersion") is False

    # The next two assertions REQUIRE the HF config.json to have the new fields.
    # If they fail with None, the maintainer needs to update the HF config.json.
    assert calc.metadata.get("family") == "rxn", (
        "HF config.json missing `family: rxn` — maintainer must update HF repo."
    )
    assert calc.metadata.get("supports_charged_systems") is False, (
        "HF config.json missing `supports_charged_systems: false` — maintainer must update HF repo."
    )
```

- [ ] **Step 2: Run the test (network-gated, requires HF deps)**

Run: `pytest tests/test_hf_hub.py::test_aimnet2_rxn_hf_load_matches_gcs_metadata -v -m hf` Expected: PASS if HF repo's `config.json` has been updated; otherwise FAIL with the explicit "maintainer must update HF repo" message — that failure is informative, not a real bug.

- [ ] **Step 3: Commit**

```bash
git -C /home/olexandr/aimnetcentral add tests/test_hf_hub.py
git -C /home/olexandr/aimnetcentral commit -m "test(hf_hub): add aimnet2-rxn HF end-to-end metadata test"
```

---

## Task 19: Add docs page for aimnet2-rxn

**Files:**

- Create: `docs/models/aimnet2_rxn.md`
- Modify: `mkdocs.yml` (add nav entry)

- [ ] **Step 1: Create the docs page**

Create `docs/models/aimnet2_rxn.md` with this content:

````markdown
# AIMNet2-rxn

A neural network interatomic potential specialized for **closed-shell organic reactions** (H, C, N, O), trained on ~4.7M reaction-relevant geometries at ωB97M-V/def2-TZVPP. Use for transition-state searches, NEB / batched-NEB, IRC profiles, and reaction-coordinate energy work.

## Loading

```python
from aimnet.calculators import AIMNet2Calculator

# From the GCS-backed registry (alias):
calc = AIMNet2Calculator("aimnet2rxn", ensemble_member=0)

# From Hugging Face Hub:
calc = AIMNet2Calculator("isayevlab/aimnet2-rxn", ensemble_member=0)
```
````

Both paths produce equivalent calculators. The HF path requires `pip install "aimnet[hf]"`.

## Calculator-enforced safeguards (this family)

The calculator applies the following checks automatically when `validate_species=True` (the default). Each can be bypassed with `validate_species=False`:

- **Element scope**: input atomic numbers must be a subset of `[1, 6, 7, 8]`. Other elements raise `ValueError` with pointers to alternative families.
- **Net charge**: only net-neutral systems (zwitterions OK). Non-zero `charge` raises `ValueError` pointing at `aimnet2-wb97m-d3` for ions.
- **AFV row sanitization**: at conversion time, atomic-feature-vector rows for elements outside `[1, 6, 7, 8]` are NaN-padded so `validate_species=False` produces NaN-propagation rather than plausible-looking nonsense.

Two further safeguards fire regardless of `validate_species`:

- **Hessian + `torch.compile`**: setting both raises `RuntimeError` (Dynamo + double-backward through GELU is known to hang). Reconstruct with `compile_model=False` for TS / IRC / vibrational work.
- **Coulomb cutoff lock**: calling `set_lrcoulomb_method(method, cutoff=…)` with a cutoff different from the model's `coulomb_sr_rc` (4.6 Å) emits a `UserWarning` because the SR/LR cancellation point was physically frozen during training.

A separate one-time `UserWarning` fires if the same Python process constructs calculators from two different AIMNet2 families (rxn vs. wb97m-d3 etc.), because the energy scales are not comparable.

## Canonical model card

Full content (energy convention, training data details, full limitations list, citation) lives at the Hugging Face model card:

[https://huggingface.co/isayevlab/aimnet2-rxn](https://huggingface.co/isayevlab/aimnet2-rxn)

The HF README is the canonical source — this page summarizes only the integration mechanics.

````

- [ ] **Step 2: Add the nav entry to `mkdocs.yml`**

Open `mkdocs.yml`. Locate the existing model entries (around lines 15-19):

```yaml
          - AIMNet2 (wB97M-D3): models/aimnet2.md
          - AIMNet2-B97-3c: models/aimnet2_b973c.md
          - AIMNet2-2025: models/aimnet2_2025.md
          - AIMNet2-NSE: models/aimnet2nse.md
          - AIMNet2-Pd: models/aimnet2pd.md
````

Append immediately after the `AIMNet2-Pd` line, with the same indentation:

```yaml
- AIMNet2-rxn: models/aimnet2_rxn.md
```

- [ ] **Step 3: Verify mkdocs builds**

Run: `mkdocs build --strict --site-dir /tmp/aimnet-mkdocs-test` Expected: build succeeds, no warnings, no broken links. The test directory `/tmp/aimnet-mkdocs-test` will contain the generated site; you can `rm -rf /tmp/aimnet-mkdocs-test` afterward.

- [ ] **Step 4: Commit**

```bash
git -C /home/olexandr/aimnetcentral add docs/models/aimnet2_rxn.md mkdocs.yml
git -C /home/olexandr/aimnetcentral commit -m "docs(models): add aimnet2-rxn page (stub linking to HF model card)"
```

---

## Task 20: Final test suite + branch summary

**Files:** none (verification only).

- [ ] **Step 1: Run the entire affected test surface**

Run:

```bash
pytest tests/test_model.py tests/test_calculator.py tests/test_model_registry.py tests/test_hf_hub.py tests/test_ase.py -v -m "not gpu"
```

Expected: All non-skipped tests PASS. Skipped tests are acceptable (`hf`, `network`, `pysis`, missing `_tmp/` data, missing GCS upload).

- [ ] **Step 2: Run the full suite once to catch unrelated regressions**

Run: `pytest -v -m "not gpu and not hf and not network and not pysis"` Expected: green or pre-existing-yellow only.

- [ ] **Step 3: Show the branch's commit graph**

Run: `git -C /home/olexandr/aimnetcentral log main..HEAD --oneline` Expected: ~17-18 commits, each labeled with `feat(...)`, `fix(...)`, `test(...)`, or `docs(...)`. Verify no commit message contains "Claude", "Anthropic", or "Co-Authored-By".

- [ ] **Step 4: Stop and surface the out-of-band tasks before any push/PR**

The following must be done by the maintainer before the PR can be considered ready:

1. Upload `_tmp/aimnet2_rxn_v2_gcs.zip`'s four `.pt` files to `gs://aimnetcentral/aimnet2v2/AIMNet2rxn/` (sha256s: see `_tmp/aimnet2_rxn_v2_gcs.zip`).
2. Update the HF repo `isayevlab/aimnet2-rxn`'s `config.json` to add `"family": "rxn"` and `"supports_charged_systems": false`.

After both: `test_aimnet2rxn_alias_calculator_e2e` and `test_aimnet2_rxn_hf_load_matches_gcs_metadata` will both PASS in their network-gated environments.

- [ ] **Step 5: Hand back to the user**

Print: "Branch `feat/aimnet2-rxn-integration` ready. ~18 commits applied. Out-of-band tasks pending: GCS upload + HF config.json update."

---

## Spec Coverage Self-Review

| Spec section | Covered by |
| --- | --- |
| Component 1 — Registry entries | Task 8 |
| Component 2 — Architecture YAML | Task 7 |
| Component 3 — `load_v1_model` overrides + AFV sanitization + .double() reorder + ModelMetadata schema | Tasks 1, 2, 3, 4, 5 |
| Component 4 — Calculator validation API (species + charge + Hessian) | Tasks 10, 11, 12 |
| Component 5 — ASE / Pysis wrapper propagation | Tasks 15, 16 |
| Component 6 — `set_lrcoulomb_method` rxn guard | Task 13 |
| Component 7 — Cross-family mixing detection | Task 14 |
| Component 8 — `metadata` property | Task 9 |
| Component 9 — Conversion (docstring example) | Task 6 |
| Docs page | Task 19 |
| Out-of-band tasks (GCS upload, HF config update) | Surfaced in Task 20 |

All nine components and the docs page have a dedicated implementing task. The two out-of-band steps are flagged as maintainer actions, not part of the executable plan.
