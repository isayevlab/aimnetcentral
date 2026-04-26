# Project rules for Claude Code

These rules cover invariants and conventions that are not obvious from reading the code. The general code style is established by the codebase itself; only project-wide rules and back-compat anchors live here.

## Model registry naming convention

The registry at `aimnet/calculators/model_registry.yaml` uses a strict shape for every entry under `models:`:

```
aimnet2-<family>_<ensemble-member>
```

- A dash separates `aimnet2` from the family tag, and is the only separator allowed inside the family tag.
- The trailing `_<int>` is reserved for the ensemble member index.
- Family tags must NOT contain `_` and must NOT end in `_<int>`, so `key.rsplit("_", 1)` always yields `(family, member)` unambiguously.
- The convention applies to entries under `models:` only. Entries under `aliases:` are not parsed.

When adding a new model family:

1. Add the canonical key under `models:` following the convention above. The `file:` field can keep any historical filename — file fields are not constrained by the convention.
2. Add a canonical short alias `aimnet2-<family>` → `aimnet2-<family>_0` under the `# canonical short aliases` block in `aliases:`.
3. If you also publish any legacy form (snake-form, no-separator), add it under `# legacy short aliases` and `# legacy member-level aliases`. **Once a name has been released, never remove it.**
4. Update `tests/test_model_registry.py::test_canonical_keys_for_all_families` to include the new family.
5. Update `docs/models/guide.md` (Quick-Pick table, Loading Models examples, Aliases Reference table, per-family download table) and `README.md` (Available Models table).
6. If the family ships a per-model doc page, write it under `docs/models/` and add a "Legacy names" admonition listing any prior published aliases.

## Single-hop alias rule (load-bearing invariant)

`aimnet/calculators/model_registry.py:get_registry_model_path` does **exactly one** alias hop, then a `models:` lookup:

```python
if model_name in aliases:
    model_name = aliases[model_name]      # ONE hop only
if model_name not in models:
    raise ValueError(...)
cfg = models[model_name]
```

So every alias value MUST be a key in `models:`, never another alias. `tests/test_model_registry.py::test_no_alias_to_alias_chains` enforces this — do not bypass it.

## Backwards-compat horizon

The legacy alias section in `model_registry.yaml` covers every name released in **v0.1.0 onward**. Anything published since v0.1.0 must continue to resolve. Names that existed only in pre-v0.1.0 commits (e.g., `aimnet2_qr`, `wb97m_cpcms_v2_*`) are intentionally not preserved.

When in doubt about whether a name is "released," check `git log -- aimnet/calculators/model_registry.yaml` and the v0.1.0 tag (`23e33f2`) onward.

## Tests that should run before any registry change

```
pytest tests/test_model_registry.py
pytest tests/test_hf_hub.py
```

Both must pass. The full suite (`pytest`) requires GPU/network for some tests and is not always practical locally.

## Network-marked tests

Calculator tests that download `.pt` files from GCS are marked `@pytest.mark.network` and skipped by default. They do exercise the registry, so they implicitly cover legacy aliases (`aimnet2nse`, `aimnet2rxn`, etc.) — when changing the registry, do not break the strings these tests reference.
