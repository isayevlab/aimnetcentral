import yaml

from aimnet.calculators.model_registry import load_model_registry


def test_load_model_registry_respects_registry_file_param(tmp_path):
    """registry_file param should override the default path."""
    fake = {"aliases": {}, "models": {"fake_model": {"file": "x.pt", "url": "http://x"}}}
    registry_path = tmp_path / "my_registry.yaml"
    registry_path.write_text(yaml.dump(fake))

    result = load_model_registry(str(registry_path))
    assert "fake_model" in result["models"]


def test_load_model_registry_uses_default_when_no_param():
    """When called with no args, loads the default registry with known models."""
    result = load_model_registry()
    assert "models" in result
    assert "aimnet2" in result.get("aliases", {})


def test_default_aimnet2_alias_resolves_to_wb97m_d3():
    """The bare `aimnet2` alias must resolve to the canonical wb97m-d3 member 0."""
    registry = load_model_registry()
    assert registry["aliases"]["aimnet2"] == "aimnet2-wb97m-d3_0"
    assert "aimnet2-wb97m-d3_0" in registry["models"]


def test_short_alias_forms_match():
    """Each model family's short alias forms (dash-canonical plus any legacy
    forms that have shipped publicly) must all resolve to the same canonical
    member-0 model key."""
    registry = load_model_registry()
    aliases = registry["aliases"]

    # (canonical_alias, [legacy_aliases], expected_target)
    expectations = [
        ("aimnet2-nse", ["aimnet2nse"], "aimnet2-nse_0"),
        ("aimnet2-pd", ["aimnet2pd"], "aimnet2-pd_0"),
        ("aimnet2-rxn", ["aimnet2rxn"], "aimnet2-rxn_0"),
        ("aimnet2-wb97m", ["aimnet2_wb97m"], "aimnet2-wb97m-d3_0"),
        ("aimnet2-b973c", ["aimnet2_b973c"], "aimnet2-b973c-d3_0"),
        ("aimnet2-2025", ["aimnet2_2025"], "aimnet2-b973c-2025-d3_0"),
    ]
    for canonical_alias, legacy_aliases, expected_target in expectations:
        assert aliases.get(canonical_alias) == expected_target, (
            f"{canonical_alias} should resolve to {expected_target}"
        )
        for legacy in legacy_aliases:
            assert aliases.get(legacy) == expected_target, (
                f"legacy alias {legacy} should resolve to {expected_target}"
            )


def test_canonical_keys_for_all_families():
    """Every model family must have its four ensemble members registered under
    the canonical dash-form key, mapped to the original (unchanged) GCS files."""
    registry = load_model_registry()
    models = registry["models"]

    expected = [
        # (canonical_key_template, file_template, gcs_subdir)
        ("aimnet2-wb97m-d3_{i}", "aimnet2_wb97m_d3_{i}.pt", "AIMNet2"),
        ("aimnet2-b973c-d3_{i}", "aimnet2_b973c_d3_{i}.pt", "AIMNet2"),
        ("aimnet2-b973c-2025-d3_{i}", "aimnet2_2025_b973c_d3_{i}.pt", "AIMNet2"),
        ("aimnet2-nse_{i}", "aimnet2nse_wb97m_{i}.pt", "AIMNet2NSE"),
        ("aimnet2-pd_{i}", "aimnet2-pd_{i}.pt", "AIMNet2Pd"),
        ("aimnet2-rxn_{i}", "aimnet2_rxn_{i}.pt", "AIMNet2rxn"),
    ]
    base = "https://storage.googleapis.com/aimnetcentral/aimnet2v2"
    for key_tmpl, file_tmpl, subdir in expected:
        for i in range(4):
            key = key_tmpl.format(i=i)
            file = file_tmpl.format(i=i)
            assert key in models, f"missing model key: {key}"
            entry = models[key]
            assert entry["file"] == file, f"{key}: file mismatch"
            assert entry["url"] == f"{base}/{subdir}/{file}", f"{key}: url mismatch"


def test_legacy_member_aliases_resolve_via_loader(monkeypatch):
    """End-to-end: every legacy member-level key must resolve through
    get_registry_model_path's alias indirection to the canonical model's file.
    The download step is stubbed so the test exercises the lookup logic only."""
    from aimnet.calculators import model_registry as mr

    # stub the download step so the loader collapses to pure lookup; return
    # the expected on-disk path get_registry_model_path would normally hand back
    monkeypatch.setattr(
        mr, "_maybe_download_asset",
        lambda file, url: f"/assets/{file}",
    )

    registry = mr.load_model_registry()
    legacy_keys = [
        # underscore-form legacy keys (the previous shape of every default model)
        "aimnet2_wb97m_d3_0", "aimnet2_wb97m_d3_1",
        "aimnet2_wb97m_d3_2", "aimnet2_wb97m_d3_3",
        "aimnet2_b973c_d3_0", "aimnet2_b973c_d3_1",
        "aimnet2_b973c_d3_2", "aimnet2_b973c_d3_3",
        "aimnet2_b973c_2025_d3_0", "aimnet2_b973c_2025_d3_1",
        "aimnet2_b973c_2025_d3_2", "aimnet2_b973c_2025_d3_3",
        "aimnet2_rxn_0", "aimnet2_rxn_1", "aimnet2_rxn_2", "aimnet2_rxn_3",
        # no-separator-form legacy keys
        "aimnet2nse_0", "aimnet2nse_1", "aimnet2nse_2", "aimnet2nse_3",
    ]
    for legacy in legacy_keys:
        path = mr.get_registry_model_path(legacy)
        canonical = registry["aliases"][legacy]
        expected_file = registry["models"][canonical]["file"]
        assert path == f"/assets/{expected_file}", (
            f"{legacy} resolved to {path}, expected /assets/{expected_file}"
        )


def test_no_alias_to_alias_chains():
    """Single-hop invariant: every alias value must be a real model key, never
    another alias. This makes the get_registry_model_path one-hop lookup
    mechanically enforced rather than maintained by hand."""
    registry = load_model_registry()
    models = registry["models"]
    aliases = registry["aliases"]

    for src, dst in aliases.items():
        assert dst in models, f"alias {src!r} -> {dst!r} is not a model entry"
        assert dst not in aliases, (
            f"alias {src!r} -> {dst!r} is itself an alias (would require >1 hop)"
        )
