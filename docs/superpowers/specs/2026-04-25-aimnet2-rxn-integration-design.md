# aimnet2-rxn Integration — Design Spec

**Date:** 2026-04-25
**Status:** Revised after multi-review (2 critical, 10 major, 5 minor findings addressed)
**Scope:** One bundled PR registering the `aimnet2-rxn` model family in `aimnetcentral`, with the orthogonal cleanups required to make it correct end-to-end. Engineering speculation cut; chemistry safeguards added in their lightest form.

## Goal

Land `aimnet2-rxn` as a first-class member of the model registry. The model is distributed via two channels:

- **GCS** as `.pt` files at `gs://aimnetcentral/aimnet2v2/AIMNet2rxn/aimnet2_rxn_{0..3}.pt`, loaded via `AIMNet2Calculator("aimnet2rxn")` (alias) or `AIMNet2Calculator("aimnet2_rxn_<i>")`.
- **HF** as `safetensors` at `isayevlab/aimnet2-rxn`, loaded via `AIMNet2Calculator("isayevlab/aimnet2-rxn", ensemble_member=<i>)`. The HF path already works on `main` (verified).

The two channels are independent. Users select per call; there is no automatic fallback.

In the same PR, fix the upstream gaps that the rxn integration surfaced (silent garbage on out-of-scope inputs, `.double()` ordering bug in `load_v1_model`, calculator's private-attribute coupling) and add the chemistry safeguards rxn requires (charge validation, AFV row sanitization, Coulomb-cutoff lock, cross-family mixing detection, Hessian + `torch.compile` interaction guard).

## Non-goals

- A first-class ensemble-aggregation API (mean/std across members). Existing convention: 4 separate calculators, caller aggregates.
- A new `aimnet.deployment` namespace; or a top-level re-export of `load_v1_model`. Both deferred until a real external consumer asks.
- An `extract_species(sentinel="auto")` parameter. The `implemented_species` override on `load_v1_model` is sufficient for rxn; no other model in this PR benefits from a sentinel change.
- A GCS→HF automatic fallback. The HF path already works for users who pass the repo ID directly; auto-fallback added significant code surface (broad exception catches, `hf_fallback` registry schema, lazy imports) for marginal benefit.
- HF-as-canonical-distribution policy. Both channels remain supported.
- GCS upload of the produced `.pt` files. Performed manually by the maintainer; not a CI step.

## Background — what the rxn JIT looks like

Probed `_tmp/model_1.jpt` directly:

- `cutoff = 5.0`; one `lrcoulomb` child module; no `dftd3` / `d3bj` / `d3ts`.
- `afv.weight` shape `(64, 256)`. Rows 1–63 are **all populated** (no NaN, no all-zero). Only row 0 is zero (the placeholder index).
- The existing four families (`aimnet2_wb97m_d3_0` etc.) NaN-pad unused rows: 49 NaN rows + 1 zero row + 14 trained rows. `extract_species` correctly returns the 14-element list.
- For rxn, `extract_species` returns `[1..63]` — wrong. The true scope is `[1, 6, 7, 8]`.

**Implication:** implemented species for rxn cannot be derived from `afv.weight`. It must be passed explicitly at conversion time. Worse, the populated-but-untrained rows are a **chemistry footgun**: with `validate_species=False`, a user passing P (Z=15) gets numerically plausible nonsense from a populated AFV row, instead of an immediate NaN-poisoning. The conversion-time AFV sanitization (Component 5) addresses this by NaN-padding rows ∉ implemented_species, restoring the fail-fast property the other families have by construction.

## Components

### 1. Registry entries

`aimnet/calculators/model_registry.yaml` gains four `aimnet2_rxn_{0..3}` entries and one alias `aimnet2rxn → aimnet2_rxn_0`. Mirrors the schema of the four existing families exactly:

```yaml
aimnet2_rxn_0:
    file: aimnet2_rxn_0.pt
    url: https://storage.googleapis.com/aimnetcentral/aimnet2v2/AIMNet2rxn/aimnet2_rxn_0.pt
# _1, _2, _3 mirror with member index.

aliases:
    aimnet2rxn: aimnet2_rxn_0
```

No new YAML fields. No `hf_fallback` block (deferred — see Non-goals).

### 2. Architecture YAML

`aimnet/models/aimnet2_rxn.yaml` — co-located with `aimnet2.yaml` and `aimnet2_dftd3_wb97m.yaml`. Body matches the `model_yaml` already embedded in the per-member `.pt` metadata and in the HF `config.json`. No top-level extras (architecture-only); family-specific facts (`family`, `implemented_species`, `supports_charged_systems`) are passed at conversion time as explicit kwargs to `load_v1_model` and persisted to `.pt` metadata. Three copies of the architecture across `.pt` / HF `config.json` / upstream YAML is a known cost; the upstream copy is the tripwire that catches `aimnet.models.AIMNet2.__init__` signature drift at upstream PR time, before deploy.

### 3. `load_v1_model` — species override + AFV sanitization + family metadata + `.double()` fix

Three changes in `aimnet/models/utils.py::load_v1_model`. The species override and AFV sanitization are coupled by design — declaring scope and enforcing scope happen in one call:

```python
def load_v1_model(jpt_path, yaml_config_path, output_path=None, *,
                  implemented_species: list[int] | None = None,
                  family: str | None = None,
                  supports_charged_systems: bool | None = None,
                  verbose=True) -> tuple[nn.Module, dict]:
    ...
    if implemented_species is not None:
        species_set = set(implemented_species)
        metadata["implemented_species"] = sorted(species_set)
        # Sanitize AFV rows for elements outside the supported set: NaN-pad them
        # so validate_species=False still produces NaN-propagating output instead
        # of plausible-looking nonsense from populated-but-untrained rows.
        afv = core_model.afv.weight.data
        for z in range(1, afv.shape[0]):
            if z not in species_set:
                afv[z] = float("nan")
    else:
        metadata["implemented_species"] = extract_species(jit_model)

    if family is not None:
        metadata["family"] = family
    if supports_charged_systems is not None:
        metadata["supports_charged_systems"] = bool(supports_charged_systems)
    ...
```

All three new kwargs are keyword-only and default `None` — backward-compatible for the existing four families' conversions, which omit them and continue to use `extract_species(jit_model)` and no family/charge metadata. The HF `config.json` for rxn carries the same fields (`family`, `supports_charged_systems`) so both load paths produce equivalent metadata.

**Schema propagation (mandatory companion change):** `aimnet/models/base.py::ModelMetadata` (TypedDict) and `load_model` (which constructs the dict and assigns it to `model._metadata`) silently drop unknown fields. Two additions are required so the calculator can actually read `family` and `supports_charged_systems`:

```python
class ModelMetadata(TypedDict):
    ...
    implemented_species: list[int]
    family: NotRequired[str | None]                       # NEW
    supports_charged_systems: NotRequired[bool | None]    # NEW

# Inside load_model's v2-format branch:
metadata: ModelMetadata = {
    ...
    "implemented_species": data.get("implemented_species", []),
    "family": data.get("family"),
    "supports_charged_systems": data.get("supports_charged_systems"),
}
```

The same two-line addition goes into `aimnet/calculators/hf_hub.py::load_from_hf_repo` where it constructs its `ModelMetadata`. Both load paths converge on equivalent metadata for rxn.

`.double()` ordering fix (unchanged from review):

```python
# Cast atomic_shift to float64 BEFORE load_state_dict so the destination can
# hold full precision when copy_ runs internally.
if hasattr(core_model, "outputs") and hasattr(core_model.outputs, "atomic_shift"):
    core_model.outputs.atomic_shift.double()

load_result = core_model.load_state_dict(jit_sd, strict=False)
# (drop the redundant .copy_ block — load_state_dict now preserves precision)
```

For current JIT models the numerical output is byte-identical (atomic_shift in source is float32). For any future float64 JIT, precision is preserved.

### 4. Calculator validation API

In `aimnet/calculators/calculator.py::AIMNet2Calculator.eval` (where `__call__` delegates via `*args, **kwargs` at line 414). New parameter is keyword-only — existing `forces`/`stress`/`hessian` stay positional-OK so the HF README example `calc(data, forces=True)` keeps working:

```python
def eval(self, data, forces=False, stress=False, hessian=False,
         *, validate_species: bool = True) -> dict[str, Tensor]:
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
    # Charge guard — defensive for models that declare narrow charge support.
    if validate_species:
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
    # Hessian + torch.compile guard — fails fast instead of hanging.
    if hessian and getattr(self, "_was_compiled", False):
        raise RuntimeError(
            "Hessian computation is incompatible with compile_model=True "
            "(Dynamo + double-backward through GELU hangs). Reconstruct calculator "
            "with compile_model=False."
        )
    data = self.prepare_input(data)
    # ... existing forward path unchanged ...
```

`__call__` is the existing one-line `return self.eval(*args, **kwargs)`; the new kwarg flows through automatically. The single `.flatten().tolist()` GPU→CPU sync fires once per call when validation is enabled.

`AIMNet2Calculator.__init__` sets `self._was_compiled = bool(compile_model)` so the Hessian guard has a flag to read.

`rxn`-family `.pt` metadata declares `supports_charged_systems: false`. Other families do not declare the field; the guard is a silent no-op for them.

### 5. ASE / Pysis wrapper propagation

`AIMNet2ASE.calculate` and `AIMNet2Pysis.eval_orca` (or equivalent) gain a `validate_species: bool = True` kwarg that passes through to the underlying `AIMNet2Calculator.__call__`. Resolves the open-assumption footgun (a user hitting `ValueError` via ASE has no escape hatch). Two-line change per wrapper.

### 6. `set_lrcoulomb_method` rxn-family guard

`AIMNet2Calculator.set_lrcoulomb_method(method, cutoff=...)` (existing API at `calculator.py:601`) emits a `UserWarning` when `family == "rxn"` and the requested cutoff differs from `coulomb_sr_rc` (4.6 Å for rxn). The 4.6 Å SR/LR cancellation point is physically frozen for this family; replacing it silently breaks the matching. One-line check using `self.metadata`.

### 7. Cross-family mixing detection

`AIMNet2Calculator` gains a class-level `_constructed_families: ClassVar[set[str]] = set()`. The constructor records the loaded family. When `len(_constructed_families) > 1`, the constructor emits a one-time `UserWarning` per unique family addition explaining that energy scales differ across families (rxn uses a learned shifted-electronic scale; wb97m-d3 family uses absolute electronic energies on the −1100 eV scale). ~10 lines total.

The warning fires on construction, not on `eval`, so the cost is paid once per calculator. Bypass: set `family = None` in metadata or set `os.environ["AIMNET_QUIET_FAMILY_MIX"] = "1"`.

### 8. `metadata` property

`AIMNet2Calculator.metadata` — read-only view replacing the `getattr(self.model, "_metadata", None)` reads at `calculator.py:490, 515`. Line 284 stays as `getattr` (it's inside `__init__` before the property is meaningful):

```python
@property
def metadata(self) -> dict | None:
    return getattr(self.model, "_metadata", None)
```

Externalises a stable accessor so downstream consumers (`aimnet-solvate`, `protonator`, `LoQI`) stop reaching into `model._metadata`.

### 9. Conversion (no script artifact)

The four `.pt` files are produced by a four-line example in `load_v1_model`'s docstring:

```python
>>> from aimnet.models.utils import load_v1_model
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

No `scripts/convert_aimnet2_rxn.py` is added — single-use scripts in `scripts/` rot. The conversion is reproducible from the docstring + the `_tmp/` source files.

**The four `.pt` files produced earlier in this design session DO NOT yet have AFV row sanitization or `family: rxn` metadata.** They need re-conversion under the revised spec before GCS upload. See "Out-of-band tasks" below.

## Data flow

**GCS path (default for registry-known names):**

```
"aimnet2rxn" → alias → "aimnet2_rxn_0"
            → get_model_path → ~/.cache/aimnet/aimnet2_rxn_0.pt (download from GCS if missing)
            → load_model → (model, metadata)
            → metadata = {implemented_species: [1,6,7,8], family: "rxn",
                          supports_charged_systems: false, coulomb_sr_rc: 4.6, ...}
            → AIMNet2Calculator state set up; _constructed_families.add("rxn")
```

**HF path (explicit repo ID input):**

```
"isayevlab/aimnet2-rxn" → AIMNet2Calculator detects org/name pattern
                       → load_from_hf_repo
                       → snapshot_download (config.json + ensemble_<i>.safetensors)
                       → build_module + load_state_dict
                       → metadata read from config.json (must include the same
                          family / supports_charged_systems fields — see below)
                       → AIMNet2Calculator state set up
```

**HF config.json update (out-of-band):** the existing HF repo's `config.json` needs `family: "rxn"` and `supports_charged_systems: false` added so the HF path's safety nets match the GCS path's. One-time edit by maintainer; both paths converge.

## Error handling

| Condition | Behavior |
| --- | --- |
| Unknown model name | `KeyError` from `get_model_path` |
| GCS down | `URLError`/`HTTPError` propagates. Workaround: pass `"isayevlab/aimnet2-rxn"` directly. |
| Unsupported atomic numbers, `validate_species=True` | `ValueError` with chemistry context + alternative model pointers |
| Unsupported atomic numbers, `validate_species=False` | NaN propagation from sanitized AFV rows (immediate NaN, not silent garbage) |
| `charge != 0` on rxn (`supports_charged_systems: false`), `validate_species=True` | `ValueError` pointing at `aimnet2-wb97m-d3` |
| `hessian=True` with `compile_model=True` | `RuntimeError` with reconstruct hint (instead of hanging) |
| `set_lrcoulomb_method(cutoff != 4.6)` on rxn | `UserWarning` about SR/LR matching |
| Two different families constructed in one process | `UserWarning` once about cross-family energy-scale incompatibility |
| Model with no `implemented_species` field | validation is a silent no-op |

## Testing

| Concern | Test | File |
| --- | --- | --- |
| Registry alias resolves; `.pt` loads via calculator end-to-end | `test_aimnet2rxn_alias_calculator_e2e` | `tests/test_calculator.py` |
| `load_v1_model` species override sanitizes AFV rows | `test_load_v1_model_species_override_nan_pads_other_rows` (uses `_tmp/model_1.jpt` if present, else `pytest.skip`) | `tests/test_model.py` |
| Calculator rejects unsupported species; bypass works | `test_calculator_rejects_unsupported_species`, `test_calculator_validate_species_false_bypasses_with_nan_propagation` | `tests/test_calculator.py` |
| Calculator rejects `charge != 0` for `supports_charged_systems: false` | `test_calculator_rejects_charged_input_for_rxn` | `tests/test_calculator.py` |
| `hessian=True + compile_model=True` raises | `test_hessian_with_compile_raises` | `tests/test_calculator.py` |
| `set_lrcoulomb_method(cutoff != 4.6)` on rxn warns | `test_set_lrcoulomb_method_warns_on_rxn_cutoff_change` | `tests/test_calculator.py` |
| Cross-family construction warns once | `test_constructing_two_families_warns_once` | `tests/test_calculator.py` |
| `.double()` ordering preserves precision (regression-only assertion) | folded into `test_load_v1_model_species_override_nan_pads_other_rows` | `tests/test_model.py` |
| HF end-to-end (matches GCS metadata for rxn) | `test_aimnet2_rxn_hf_load_matches_gcs_metadata` (network-gated `@pytest.mark.hf`) | `tests/test_hf_hub.py` |

Network-gated tests use the existing `@pytest.mark.hf` marker (already excluded from CI). Tests requiring a JIT model gracefully `pytest.skip` when `_tmp/` is absent — fresh clones stay green.

Test count: **9** (down from 11 in the original spec). Cuts: `test_extract_species_handles_zero_init` (component dropped), `test_load_v1_model_importable_from_top` (component dropped), the two fallback tests (component dropped), the dedicated `metadata` property test (a one-line `@property` wrapping `getattr` does not need its own test — coverage from `test_aimnet2rxn_alias_calculator_e2e` is sufficient), the dedicated `.double()` test (folded into the species-override test). Adds: alias E2E, charge guard, Hessian+compile guard, Coulomb cutoff guard, family-mixing warn.

## Docs

`docs/models/aimnet2_rxn.md` — short stub page (≤30 lines):

- One-line description of scope (H/C/N/O closed-shell organic reactions).
- The two load snippets (`AIMNet2Calculator("aimnet2rxn")` and `AIMNet2Calculator("isayevlab/aimnet2-rxn")`).
- The list of safety nets the calculator enforces for this family (so users know what protections they have without reading the code).
- A canonical link to the HF model card (`https://huggingface.co/isayevlab/aimnet2-rxn`) for full content (energy convention, training data, citation, full limitations list).

No mirroring. The HF README is the canonical user-facing copy; duplicating it in repo docs would drift within a release cycle.

`mkdocs.yml` nav — one entry pointing at the new page.

No tutorial — existing `docs/tutorials/*.md` examples don't change.

## Resolved facts grounded in the HF model card

Each fact below is enforced in code unless marked **(documented only)**:

- **Element scope:** H, C, N, O. Enforced via `validate_species` (Component 4) AND AFV row NaN-padding at conversion time (Component 3).
- **Net charge:** zwitterions in scope; net anions/cations out of scope. Enforced via `supports_charged_systems: false` metadata + charge guard in `eval` (Component 4).
- **Coulomb crossover:** 4.6 Å SR/LR cancellation point physically frozen. Enforced via `set_lrcoulomb_method` warning (Component 6).
- **Cross-family energy-scale mixing:** rxn uses a learned shifted-electronic scale (~few eV for methane); wb97m-d3 family uses absolute electronic energies (~−1100 eV). Detected via `_constructed_families` warning (Component 7).
- **Hessian + `torch.compile`:** Dynamo + double-backward through GELU hangs. Enforced via `RuntimeError` in `eval` when `hessian=True` and `_was_compiled=True` (Component 4).
- **Cutoff:** 5.0 Å — matches the JIT probe. Read from `.pt` metadata; no code-level enforcement needed (the value is data, not a contract).
- **Dispersion:** none added at inference. VV10 baked into ωB97M-V; `needs_dispersion: false` in metadata. The existing calculator code respects this cleanly (`external_dftd3` is not constructed when `needs_dispersion=False`); confirmed during review. **(no-op — no change required)**

## Open assumptions

None remaining. The ASE/Pysis `validate_species` propagation is now Component 5.

## Out-of-band tasks

- **Re-convert the four `.pt` files** under the revised `load_v1_model` (with `implemented_species=[1,6,7,8]`, `family="rxn"`, `supports_charged_systems=False`, AFV sanitization). The files produced earlier in this design session (`_tmp/aimnet2_rxn_v2_gcs.zip`, sha256 `beca299c…`) are stale relative to the revised spec.
- **Update the HF repo's `config.json`** to add `family: "rxn"` and `supports_charged_systems: false`, so the HF path enforces the same safeguards as the GCS path.
- **Upload the re-converted `.pt` files** to `gs://aimnetcentral/aimnet2v2/AIMNet2rxn/`.

## Architectural note (deferred decision)

The architect-reviewer recommended splitting this PR into two: PR-A (registration-only: registry, YAML, `load_v1_model` overrides, conversion, docs) and PR-B (cleanups: validation, metadata property, ASE/Pysis propagation, the four chemistry safeguards). The user's chosen approach is bundled (Approach A from brainstorming). The bundle is now smaller after the multi-review cuts; if reviewer load proves blocking, splitting per the architect's suggestion is a clean fallback.

## Revisions from multi-review

Removed (engineering speculation, no current consumer): `extract_species(sentinel="auto")`, top-level `load_v1_model` re-export, GCS→HF auto-fallback (with its registry schema, lazy import, broad exception catch), `scripts/convert_aimnet2_rxn.py`, docs page mirroring of the HF README.

Added (chemistry safety, smallest viable form): AFV row NaN-padding at conversion (Critical — fail-fast for `validate_species=False`), `supports_charged_systems` charge guard (Critical — silent garbage on ions), `set_lrcoulomb_method` cutoff guard (Major — SR/LR matching), cross-family mixing warning (Major — energy scale incompatibility), Hessian + `torch.compile` raise instead of hang (Minor — silent process hang), improved `validate_species` error message with alternative-model pointers (Major).

Fixed: clarified that `metadata` property replaces `getattr` only at lines 490/515 (line 284 keeps `getattr`); added the alias-end-to-end test that was missing; resolved the ASE/Pysis open assumption inline as Component 5.

Net effect: spec is leaner (8 tests vs 11; 9 components vs 8 components but each smaller), more correct (chemistry safeguards close 4 silent-failure modes), and avoids 3 over-engineered features that had no concrete consumer.
