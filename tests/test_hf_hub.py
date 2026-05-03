"""Test Hugging Face Hub integration."""

import json
import warnings

import pytest
import torch

pytest.importorskip("safetensors")
from safetensors.torch import save_file

from aimnet.calculators.hf_hub import _validate_model_yaml, is_hf_repo_id, load_from_hf_repo


@pytest.fixture
def fake_hf_repo(tmp_path):
    """Create a fake HF repo directory with safetensors + config.json.

    IMPORTANT: Uses torch.load() on raw .pt to get model_yaml,
    NOT load_model() which consumes model_yaml internally.
    """
    from aimnet.calculators.model_registry import get_model_path

    pt_path = get_model_path("aimnet2")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", ".*weights_only.*")
        raw_data = torch.load(pt_path, map_location="cpu", weights_only=False)

    state_dict = raw_data["state_dict"]
    save_file(state_dict, str(tmp_path / "ensemble_0.safetensors"))

    config = {
        "config_schema_version": 1,
        "format_version": raw_data.get("format_version", 2),
        "model_yaml": raw_data["model_yaml"],
        "cutoff": float(raw_data["cutoff"]),
        "needs_coulomb": raw_data.get("needs_coulomb", False),
        "needs_dispersion": raw_data.get("needs_dispersion", False),
        "coulomb_mode": raw_data.get("coulomb_mode", "none"),
        "coulomb_sr_rc": raw_data.get("coulomb_sr_rc"),
        "coulomb_sr_envelope": raw_data.get("coulomb_sr_envelope"),
        "d3_params": raw_data.get("d3_params"),
        "has_embedded_lr": raw_data.get("has_embedded_lr", False),
        "implemented_species": raw_data.get("implemented_species", []),
        "ensemble_size": 4,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))

    return tmp_path


@pytest.mark.hf
def test_is_hf_repo_id():
    """Test HF repo ID detection."""
    assert is_hf_repo_id("isayevlab/aimnet2-wb97m-d3")
    assert not is_hf_repo_id("aimnet2")
    assert not is_hf_repo_id("path/to/model/file.pt")  # >1 slash
    assert not is_hf_repo_id("")


@pytest.mark.hf
def test_validate_model_yaml_allows_aimnet():
    """Test that aimnet classes are allowed."""
    yaml_str = "class: aimnet.models.AIMNet2\nkwargs:\n  outputs:\n    energy:\n      class: aimnet.modules.Output\n"
    _validate_model_yaml(yaml_str)  # Should not raise


@pytest.mark.hf
def test_validate_model_yaml_blocks_untrusted():
    """Test that non-aimnet classes are blocked."""
    yaml_str = "class: os.system\nkwargs: {}"
    with pytest.raises(ValueError, match="Untrusted class"):
        _validate_model_yaml(yaml_str)


@pytest.mark.hf
def test_load_from_hf_repo_local(fake_hf_repo):
    """Test loading model from a local directory mimicking HF repo structure."""
    model, metadata = load_from_hf_repo(str(fake_hf_repo), ensemble_member=0)
    assert model is not None
    assert metadata["cutoff"] > 0
    assert len(metadata["implemented_species"]) > 0


@pytest.mark.hf
def test_calculator_with_hf_repo(fake_hf_repo):
    """Test that AIMNet2Calculator accepts a local HF-style directory."""
    import numpy as np

    from aimnet.calculators import AIMNet2Calculator

    calc = AIMNet2Calculator(str(fake_hf_repo))

    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.09, 0.0, 0.0],
        [-0.36, 1.03, 0.0],
        [-0.36, -0.52, 0.89],
        [-0.36, -0.52, -0.89],
    ])
    numbers = np.array([6, 1, 1, 1, 1])

    results = calc({"coord": coords, "numbers": numbers, "charge": 0.0}, forces=True)
    assert "energy" in results
    assert "forces" in results
    assert "charges" in results
    assert results["forces"].shape == (5, 3)


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


@pytest.mark.hf
def test_load_from_hf_repo_propagates_family_and_charge_fields(fake_hf_repo_with_family):
    _, metadata = load_from_hf_repo(str(fake_hf_repo_with_family), ensemble_member=0, device="cpu")
    assert metadata.get("family") == "test-family"
    assert metadata.get("supports_charged_systems") is False


@pytest.mark.hf
@pytest.mark.network
def test_aimnet2_rxn_hf_load_matches_gcs_metadata():
    """Loading aimnet2-rxn from the HF repo must produce the expected calculator metadata:
    GCS/HF structural metadata plus posthoc wB97M-D3 dispersion defaults.

    This test EXPECTS the HF repo's config.json to have been updated with
    `family: rxn` and `supports_charged_systems: false` (out-of-band task)."""
    from aimnet.calculators import AIMNet2Calculator

    calc = AIMNet2Calculator("isayevlab/aimnet2-rxn", ensemble_member=0, device="cpu")

    assert calc.metadata.get("implemented_species") == [1, 6, 7, 8]
    assert abs(calc.metadata.get("cutoff") - 5.0) < 1e-6
    assert calc.metadata.get("coulomb_mode") == "sr_embedded"
    assert calc.metadata.get("needs_coulomb") is True
    assert calc.metadata.get("needs_dispersion") is True
    assert calc.metadata.get("d3_params") == {"s6": 1.0, "s8": 0.3908, "a1": 0.566, "a2": 3.128}
    assert calc.external_dftd3 is not None

    # The next two assertions REQUIRE the HF config.json to have the new fields.
    # If they fail with None, the maintainer needs to update the HF config.json.
    assert calc.metadata.get("family") == "rxn", (
        "HF config.json missing `family: rxn` — maintainer must update HF repo."
    )
    assert calc.metadata.get("supports_charged_systems") is False, (
        "HF config.json missing `supports_charged_systems: false` — maintainer must update HF repo."
    )
