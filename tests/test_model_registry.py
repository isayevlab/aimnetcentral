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
