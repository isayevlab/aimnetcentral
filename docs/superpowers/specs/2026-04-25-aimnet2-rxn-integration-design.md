# aimnet2-rxn Integration — Design Spec

**Date:** 2026-04-25
**Status:** Approved (pending implementation plan)
**Scope:** One bundled PR registering the `aimnet2-rxn` model family in `aimnetcentral`, with the orthogonal cleanups required to make it correct end-to-end.

## Goal

Land `aimnet2-rxn` as a first-class member of the model registry, available via two distribution channels:

- **GCS** as `.pt` files at `gs://aimnetcentral/aimnet2v2/AIMNet2rxn/aimnet2_rxn_{0..3}.pt`, loadable via `AIMNet2Calculator("aimnet2rxn")` (alias) or `AIMNet2Calculator("aimnet2_rxn_<i>")`.
- **HF** as `safetensors` at `isayevlab/aimnet2-rxn`, loadable via `AIMNet2Calculator("isayevlab/aimnet2-rxn", ensemble_member=<i>)`.

GCS load automatically falls back to HF if the GCS object cannot be retrieved or read.

In the same PR, fix the upstream gaps that the rxn integration surfaced: the `extract_species` heuristic doesn't work for this model, the calculator silently produces undefined output for unsupported elements, the `load_v1_model` `.double()` call ordering is wrong, and external consumers reach into private attributes (`model._metadata`) and private modules (`aimnet.models.utils.load_v1_model`).

## Non-goals

- A first-class ensemble-aggregation API (mean/std across members). Existing convention: 4 separate calculators, caller aggregates. Out of scope.
- A new `aimnet.deployment` namespace. Top-level re-export of `load_v1_model` is sufficient for now.
- HF-as-canonical-distribution policy. Both channels remain supported.
- Migration of other families to the `hf_fallback` mechanism. Per-entry opt-in; only rxn declares it in this PR.
- GCS upload of the produced `.pt` files. Performed manually by the maintainer; not a CI step.

## Background — what the rxn JIT looks like

Probed `_tmp/model_1.jpt` directly:

- `cutoff = 5.0`; one `lrcoulomb` child module; no `dftd3` / `d3bj` / `d3ts`.
- `afv.weight` shape `(64, 256)`. Rows 1–63 are **all populated** (no NaN, no all-zero). Only row 0 is zero (the placeholder index).
- The existing four families (`aimnet2_wb97m_d3_0` etc.) NaN-pad unused rows: 49 NaN rows + 1 zero row + 14 trained rows. `extract_species` correctly returns the 14-element list.
- For rxn, `extract_species` would return `[1..63]` — wrong. The true scope is `[1, 6, 7, 8]`.

**Implication:** implemented species for rxn cannot be derived from `afv.weight`. It must be passed explicitly at conversion time.

## Components

### 1. Registry entries

`aimnet/calculators/model_registry.yaml` gains four `aimnet2_rxn_{0..3}` entries and one alias `aimnet2rxn → aimnet2_rxn_0`. Each entry declares an `hf_fallback` block:

```yaml
aimnet2_rxn_0:
    file: aimnet2_rxn_0.pt
    url: https://storage.googleapis.com/aimnetcentral/aimnet2v2/AIMNet2rxn/aimnet2_rxn_0.pt
    hf_fallback:
        repo_id: isayevlab/aimnet2-rxn
        ensemble_member: 0
```

`hf_fallback` is an optional, schema-additive field. Existing entries are untouched.

### 2. Architecture YAML

`aimnet/models/aimnet2_rxn.yaml` is a new file co-located with `aimnet2.yaml` and `aimnet2_dftd3_wb97m.yaml`. Its body matches the `model_yaml` already embedded in the per-member `.pt` metadata and in the HF `config.json`. The file lives in the upstream package so any future change to `aimnet.models.AIMNet2`'s constructor is caught at upstream PR time.

### 3. `extract_species` — `sentinel="auto"`

`aimnet/models/utils.py::extract_species` gains a keyword-only `sentinel` parameter:

```python
def extract_species(model: nn.Module, *, sentinel: str = "auto") -> list[int]:
```

Values:
- `"nan"` — legacy: row is unused if all-NaN.
- `"zero"` — row is unused if all-zero.
- `"auto"` (new default) — row is unused if all-NaN OR all-zero.

For the existing four families, `auto` returns the same set as `nan`. For aimnet2-rxn, `auto` still returns `[1..63]` (no row is empty by either criterion); the override below covers that case.

### 4. `load_v1_model` — explicit species override + `.double()` ordering fix

Two changes in `aimnet/models/utils.py::load_v1_model`:

```python
def load_v1_model(jpt_path, yaml_config_path, output_path=None, *,
                  implemented_species: list[int] | None = None,
                  verbose=True) -> tuple[nn.Module, dict]:
    ...
    if implemented_species is not None:
        metadata["implemented_species"] = list(implemented_species)
    else:
        metadata["implemented_species"] = extract_species(jit_model)
    ...
```

And the `.double()` block is reordered:

```python
# Cast atomic_shift to float64 BEFORE load_state_dict so the destination
# can hold full precision when copy_ runs internally.
if hasattr(core_model, "outputs") and hasattr(core_model.outputs, "atomic_shift"):
    core_model.outputs.atomic_shift.double()

load_result = core_model.load_state_dict(jit_sd, strict=False)
# (drop the redundant .copy_ block — load_state_dict now preserves precision)
```

For current JIT models (atomic_shift is float32 in source), the numerical output is byte-identical. For any future JIT shipping float64 atomic_shift, precision is preserved.

### 5. Calculator validation API

The check goes in `aimnet/calculators/calculator.py::AIMNet2Calculator.eval` (the implementation `__call__` delegates to via `*args, **kwargs`). New parameter is keyword-only — existing `forces`/`stress`/`hessian` stay positional-OK so the HF README example `calc(data, forces=True)` and any existing positional callers are unaffected:

```python
def eval(self, data: dict[str, Any], forces=False, stress=False, hessian=False,
         *, validate_species: bool = True) -> dict[str, Tensor]:
    if validate_species:
        impl = (self.metadata or {}).get("implemented_species") or []
        if impl:
            seen = {int(z) for z in data["numbers"].flatten().tolist() if int(z) > 0}
            unsupported = sorted(seen - set(impl))
            if unsupported:
                raise ValueError(
                    f"Atomic numbers {unsupported} are not in this model's "
                    f"implemented_species {sorted(impl)}. Pass validate_species=False "
                    f"to bypass (output will be undefined)."
                )
    data = self.prepare_input(data)
    # ... existing forward path unchanged ...
```

`__call__` is the existing one-line `return self.eval(*args, **kwargs)`; the new kwarg flows through automatically. Default-on; silent no-op when the model didn't declare any species (older `.pt`, raw `nn.Module`); padding zeros are filtered out before the diff. The single `.flatten().tolist()` GPU→CPU sync fires once per call when enabled.

### 6. Public API

**6a. `AIMNet2Calculator.metadata` property** — a read-only view replacing the three internal `getattr(self.model, "_metadata", None)` reads at `calculator.py:284, 490, 515`. External consumers stop reaching into `model._metadata`:

```python
@property
def metadata(self) -> dict | None:
    return getattr(self.model, "_metadata", None)
```

**6b. Top-level re-export** in `aimnet/__init__.py`:

```python
from aimnet.models.utils import load_v1_model
__all__ = ["__version__", "load_v1_model"]
```

External consumers write `from aimnet import load_v1_model` and pin to a release tag instead of reaching into a `utils` module.

### 7. GCS → HF fallback

New function in `aimnet/calculators/model_registry.py`:

```python
def load_model_with_fallback(name: str, device: str = "cpu") -> tuple[nn.Module, dict]:
    registry = load_model_registry()
    resolved = registry["aliases"].get(name, name)
    entry = registry["models"].get(resolved)
    if entry is None:
        raise KeyError(f"Unknown model '{name}' (resolved to '{resolved}').")

    try:
        path = get_model_path(resolved)
        return load_model(path, device=device)
    except (urllib.error.URLError, urllib.error.HTTPError,
            OSError, RuntimeError) as gcs_err:
        fb = entry.get("hf_fallback")
        if fb is None:
            raise
        warnings.warn(
            f"GCS load failed for '{resolved}' ({gcs_err.__class__.__name__}: {gcs_err}); "
            f"falling back to HF repo '{fb['repo_id']}' member {fb['ensemble_member']}.",
            UserWarning, stacklevel=2,
        )
        from aimnet.calculators.hf_hub import load_from_hf_repo
        return load_from_hf_repo(
            fb["repo_id"],
            ensemble_member=fb.get("ensemble_member", 0),
            device=device,
        )
```

`AIMNet2Calculator.__init__`'s non-HF branch (currently `calculator.py:277-280`) calls this in place of the direct `get_model_path` + `load_model` pair. The HF-repo-id branch (`isayevlab/aimnet2-rxn` as input) is unchanged.

**Failure-mode boundary:**

- **Caught (triggers fallback):** `URLError`/`HTTPError` (network down, 404, 5xx); `OSError` (corrupt cache); `RuntimeError` (`torch.load` on a bad file).
- **Not caught:** `KeyError` (unknown model), `ImportError` (`huggingface_hub` not installed — re-raises with install hint), errors after a successful GCS load (those are real bugs).
- **One warning per fallback.** No retry loop. No sticky state — next call retries GCS first.

### 8. Conversion artifact

`scripts/convert_aimnet2_rxn.py` — one-shot maintainer script. Reads `_tmp/model_{1..4}.jpt` + `_tmp/config.yaml`, calls `load_v1_model(..., implemented_species=[1, 6, 7, 8])`, writes `aimnet2_rxn_{0..3}.pt`. Not imported by package code; not part of the public API. Documented in the script's docstring with the GCS upload paths.

The four `.pt` files for the initial GCS upload were already produced during this design session (sha256s in the conversion log) and bundled into `_tmp/aimnet2_rxn_v2_gcs.zip`.

## Data flow

**GCS path (default for registry-known names):**

```
"aimnet2rxn" → alias → "aimnet2_rxn_0"
            → load_model_with_fallback
            → get_model_path → ~/.cache/aimnet/aimnet2_rxn_0.pt (download from GCS if missing)
            → load_model → (model, metadata)
            → metadata.implemented_species = [1, 6, 7, 8]
            → AIMNet2Calculator state set up
```

**HF path (explicit repo ID input):**

```
"isayevlab/aimnet2-rxn" → AIMNet2Calculator detects org/name pattern
                       → load_from_hf_repo
                       → snapshot_download (config.json + ensemble_<i>.safetensors)
                       → build_module + load_state_dict
                       → metadata.implemented_species = [1, 6, 7, 8]  (from config.json)
                       → AIMNet2Calculator state set up
```

**Fallback path (GCS unreachable):**

```
"aimnet2rxn" → load_model_with_fallback
            → get_model_path raises HTTPError
            → entry has hf_fallback → warn + load_from_hf_repo("isayevlab/aimnet2-rxn", member=0)
            → returns (model, metadata)
```

## Error handling

| Condition | Behavior |
| --- | --- |
| Unknown model name | `KeyError` from `load_model_with_fallback` |
| GCS down, no fallback declared | original `URLError`/`HTTPError` propagates |
| GCS down, fallback declared, HF also down | `UserWarning` for the GCS failure, then HF exception propagates |
| `huggingface_hub` not installed, fallback declared | `ImportError` with install hint |
| Unsupported atomic numbers, `validate_species=True` | `ValueError` listing unsupported numbers and the supported set |
| Unsupported atomic numbers, `validate_species=False` | undefined output (documented escape hatch) |
| Model with no `implemented_species` field | validation is a silent no-op |

## Testing

| Concern | Test | File |
| --- | --- | --- |
| Registry alias resolves; .pt loads | `test_aimnet2_rxn_alias_resolves`, `test_aimnet2_rxn_pt_loads` | `tests/test_model_registry.py` |
| Architecture YAML builds | `test_aimnet2_rxn_yaml_loads` | `tests/test_model.py` |
| `extract_species(sentinel="auto")` | `test_extract_species_handles_zero_init`, `test_extract_species_handles_nan` | `tests/test_model.py` |
| `load_v1_model` species override | `test_load_v1_model_species_override` (skipped if `_tmp/model_1.jpt` absent) | `tests/test_model.py` |
| Calculator validates species | `test_calculator_rejects_unsupported_species`, `test_calculator_validate_species_false_bypasses` | `tests/test_calculator.py` |
| `metadata` property | `test_calculator_metadata_property_round_trips` | `tests/test_calculator.py` |
| Top-level re-export | `test_load_v1_model_importable_from_top` | `tests/test_model.py` |
| `.double()` ordering | `test_load_v1_model_atomic_shift_is_float64` | `tests/test_model.py` |
| Fallback on GCS failure | `test_load_model_with_fallback_uses_hf_on_failure` (monkeypatch `get_model_path` to raise) | `tests/test_model_registry.py` |
| Fallback not declared | `test_load_model_with_fallback_propagates_when_no_fallback` | `tests/test_model_registry.py` |
| HF end-to-end | `test_aimnet2_rxn_hf_load_matches_gcs` (network-gated `@pytest.mark.hf`) | `tests/test_hf_hub.py` |

Network-gated tests use the existing `@pytest.mark.hf` marker (already excluded from CI). Tests requiring a JIT model gracefully `pytest.skip` when `_tmp/` is absent — fresh clones stay green.

## Docs

- `docs/models/aimnet2_rxn.md` — mirror of the HF model card at `https://huggingface.co/isayevlab/aimnet2-rxn`. The HF README is the canonical user-facing content (scope, energy convention, dispersion handling, dual-source loading, ensemble usage, the `compile_model=False` Hessian caveat, citation). The docs page is a copy maintained in sync with the HF README — no new authoring beyond a one-line "this model is also distributed via GCS as `aimnet2rxn`" addendum to make the dual-source story discoverable from the docs site.
- `mkdocs.yml` nav — one entry pointing at the new page.

No tutorial — existing `docs/tutorials/*.md` examples don't change.

## Resolved facts grounded in the HF model card

- **Element scope:** H, C, N, O — drives the `validate_species` `ValueError` content.
- **Cutoff:** 5.0 Å — matches the JIT probe.
- **Coulomb:** SR embedded ≤4.6 Å + external LR ≥4.6 Å. The 4.6 Å crossover is physically frozen (SR/LR cancellation point used during training); calculator must not let users alter `coulomb_sr_rc` for this family.
- **Dispersion:** none added at inference. VV10 is baked into the ωB97M-V reference; `needs_dispersion: false` in `config.json`. No `external_dftd3` plumbing needed.
- **Hessian / TS workflow:** requires `compile_model=False` (Dynamo + double-backward through GELU). The docs page surfaces this; no code change in this PR.
- **Net charge:** zwitterions in scope; net anions/cations out of scope (calculator does not need a charge-validation guard for this PR).

## Open assumptions to verify in implementation

- Whether `AIMNet2ASE` and `AIMNet2Pysis` need their own `validate_species=` knob. Assumed: no for this PR; users disable by calling `AIMNet2Calculator` directly if needed.

## Out-of-band tasks

- Upload `aimnet2_rxn_{0..3}.pt` to `gs://aimnetcentral/aimnet2v2/AIMNet2rxn/`. Done by maintainer before merge so the registry URLs resolve.
- Verify HF fallback works in the field by deleting the local GCS cache and running `AIMNet2Calculator("aimnet2rxn")` once.
