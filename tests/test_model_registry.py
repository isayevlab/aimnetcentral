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
