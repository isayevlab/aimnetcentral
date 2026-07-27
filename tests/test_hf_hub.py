"""Test Hugging Face Hub integration."""

import json
from unittest.mock import Mock

import pytest
import torch

pytest.importorskip("safetensors")
from safetensors.torch import save_file

from aimnet.calculators import hf_hub
from aimnet.calculators.hf_hub import (
    _fetch_pt_metadata_from_registry,
    is_hf_repo_id,
    load_from_hf_repo,
)
from aimnet.models.artifact_validation import validate_model_yaml
from aimnet.modules import AtomicShift

pytestmark = pytest.mark.hf


@pytest.fixture
def fake_hf_repo(tmp_path):
    """Create a fake HF repo directory with safetensors + config.json.

    IMPORTANT: Uses torch.load() on raw .pt to get model_yaml,
    NOT load_model() which consumes model_yaml internally.
    """
    from aimnet.calculators.model_registry import get_model_path

    pt_path = get_model_path("aimnet2")

    raw_data = torch.load(pt_path, map_location="cpu", weights_only=True)

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
    validate_model_yaml(yaml_str)  # Should not raise


@pytest.mark.hf
def test_validate_model_yaml_blocks_untrusted():
    """Test that non-aimnet classes are blocked."""
    yaml_str = "class: os.system\nkwargs: {}"
    with pytest.raises(ValueError, match="Untrusted import path"):
        validate_model_yaml(yaml_str)


@pytest.mark.hf
def test_hf_metadata_fallback_accepts_registry_names_only(monkeypatch, tmp_path):
    from aimnet.calculators import hf_hub

    local_path = str(tmp_path / "attacker.pt")
    registry_path = Mock(side_effect=ValueError("not a registry model"))
    monkeypatch.setattr(hf_hub, "get_registry_model_path", registry_path)

    with pytest.raises(ValueError, match="not a registry model"):
        _fetch_pt_metadata_from_registry(
            {"member_names": [local_path]},
            "attacker/repository",
            0,
        )

    registry_path.assert_called_once_with(local_path)


@pytest.mark.hf
def test_hf_rejects_malicious_yaml_before_build_module(monkeypatch, tmp_path):
    from aimnet.calculators import hf_hub

    save_file({}, str(tmp_path / "ensemble_0.safetensors"))
    (tmp_path / "config.json").write_text(
        json.dumps({"model_yaml": "class: os.system", "cutoff": 5.0, "format_version": 2})
    )
    build_module = Mock(side_effect=AssertionError("build_module must not be called"))
    monkeypatch.setattr(hf_hub, "build_module", build_module)

    with pytest.raises(ValueError, match="Untrusted import path"):
        load_from_hf_repo(str(tmp_path))

    build_module.assert_not_called()


@pytest.mark.hf
def test_hf_rejects_invalid_format_version(tmp_path):
    save_file({}, str(tmp_path / "ensemble_0.safetensors"))
    (tmp_path / "config.json").write_text(
        json.dumps({"model_yaml": "class: aimnet.models.AIMNet2", "cutoff": 5.0, "format_version": "2"})
    )

    with pytest.raises(ValueError, match="format_version"):
        load_from_hf_repo(str(tmp_path))


@pytest.mark.parametrize(
    ("model_import_paths", "model_import_mode"),
    [
        ({"my_package.CustomAIMNet"}, "extend"),
        (None, "replace"),
        (None, "unsafe"),
    ],
)
def test_hf_registry_fallback_rejects_custom_import_settings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    model_import_paths,
    model_import_mode,
):
    (tmp_path / "config.json").write_text(json.dumps({"member_names": ["aimnet2"]}))
    fetch = Mock(side_effect=AssertionError("registry fetch must not be called"))
    monkeypatch.setattr("aimnet.calculators.hf_hub._fetch_pt_metadata_from_registry", fetch)

    with pytest.raises(ValueError, match=r"registry HF fallback|replace|unsafe"):
        load_from_hf_repo(
            str(tmp_path),
            model_import_paths=model_import_paths,
            model_import_mode=model_import_mode,
        )
    fetch.assert_not_called()


def test_hf_registry_fallback_uses_shared_allowlist(monkeypatch: pytest.MonkeyPatch, tmp_path):
    from aimnet.calculators import hf_hub

    pt_path = tmp_path / "registry.pt"
    torch.save(
        {
            "model_yaml": "class: torch.hub.load",
            "state_dict": {},
            "cutoff": 5.0,
            "format_version": 2,
        },
        pt_path,
    )
    monkeypatch.setattr(hf_hub, "get_registry_model_path", lambda _: str(pt_path))
    with pytest.raises(ValueError, match="Untrusted import path"):
        _fetch_pt_metadata_from_registry({"member_names": ["aimnet2"]}, "repo", 0)


def test_hf_rejects_invalid_metadata_before_weights_or_construction(monkeypatch: pytest.MonkeyPatch, tmp_path):
    save_file({}, str(tmp_path / "ensemble_0.safetensors"))
    (tmp_path / "config.json").write_text(
        json.dumps({
            "model_yaml": "class: aimnet.models.AIMNet2",
            "cutoff": 5.0,
            "format_version": 2,
            "needs_dispersion": True,
        })
    )
    load_file = Mock(side_effect=AssertionError("weights must not be loaded"))
    build_module = Mock(side_effect=AssertionError("model must not be built"))
    monkeypatch.setattr(hf_hub, "_load_safetensors_file", load_file)
    monkeypatch.setattr(hf_hub, "build_module", build_module)

    with pytest.raises(ValueError, match="d3_params"):
        load_from_hf_repo(str(tmp_path))

    load_file.assert_not_called()
    build_module.assert_not_called()


def test_hf_fallback_uses_validated_registry_cutoff(monkeypatch: pytest.MonkeyPatch, tmp_path):
    save_file({}, str(tmp_path / "ensemble_0.safetensors"))
    (tmp_path / "config.json").write_text(json.dumps({"member_names": ["aimnet2"]}))
    monkeypatch.setattr(
        hf_hub,
        "_fetch_pt_metadata_from_registry",
        Mock(
            return_value=(
                {
                    "model_yaml": "class: aimnet.models.AIMNet2",
                    "cutoff": 5.0,
                    "format_version": 2,
                },
                {"class": "aimnet.models.AIMNet2"},
            )
        ),
    )
    model = torch.nn.Identity()
    monkeypatch.setattr(hf_hub, "build_module", Mock(return_value=model))

    loaded, metadata = load_from_hf_repo(str(tmp_path))

    assert loaded is model
    assert metadata["cutoff"] == 5.0


def test_hf_rejects_non_mapping_config_root(tmp_path):
    (tmp_path / "config.json").write_text("[]")

    with pytest.raises(TypeError, match="mapping"):
        load_from_hf_repo(str(tmp_path))


def test_hf_remote_weights_wait_for_config_validation(monkeypatch: pytest.MonkeyPatch, tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_yaml": "class: os.system", "cutoff": 5.0, "format_version": 2})
    )
    snapshot_download = Mock(return_value=str(tmp_path))
    monkeypatch.setattr(hf_hub, "_snapshot_download", snapshot_download)

    with pytest.raises(ValueError, match="Untrusted import path"):
        load_from_hf_repo("org/repository")

    snapshot_download.assert_called_once()
    assert snapshot_download.call_args.kwargs["allow_patterns"] == ["config.json"]


def test_hf_rejects_non_module_construction(monkeypatch: pytest.MonkeyPatch, tmp_path):
    save_file({}, str(tmp_path / "ensemble_0.safetensors"))
    (tmp_path / "config.json").write_text(
        json.dumps({"model_yaml": "class: aimnet.models.AIMNet2", "cutoff": 5.0, "format_version": 2})
    )
    monkeypatch.setattr(hf_hub, "build_module", Mock(return_value=object()))

    with pytest.raises(TypeError, match=r"nn\.Module"):
        load_from_hf_repo(str(tmp_path))


@pytest.mark.parametrize(
    ("model_import_paths", "model_import_mode"),
    [
        ({"my_package.CustomAIMNet"}, "extend"),
        ({"my_package.CustomAIMNet"}, "replace"),
        (None, "unsafe"),
    ],
)
def test_hf_forwards_direct_import_options(
    monkeypatch: pytest.MonkeyPatch, tmp_path, model_import_paths, model_import_mode
):
    from aimnet.calculators import hf_hub

    save_file({}, str(tmp_path / "ensemble_0.safetensors"))
    (tmp_path / "config.json").write_text(
        json.dumps({"model_yaml": "class: my_package.CustomAIMNet", "cutoff": 5.0, "format_version": 2})
    )
    validate = Mock(return_value={"class": "my_package.CustomAIMNet"})
    model = torch.nn.Identity()
    monkeypatch.setattr(hf_hub, "validate_model_yaml", validate)
    monkeypatch.setattr(hf_hub, "build_module", Mock(return_value=model))

    loaded, _ = load_from_hf_repo(
        str(tmp_path),
        model_import_paths=model_import_paths,
        model_import_mode=model_import_mode,
    )

    assert loaded is model
    assert validate.call_args.kwargs == {
        "model_import_paths": model_import_paths,
        "model_import_mode": model_import_mode,
    }


def test_hf_loads_weights_on_cpu_and_moves_once(monkeypatch: pytest.MonkeyPatch, tmp_path):
    class SpyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.to_devices: list[str] = []

        def to(self, *args: object, **kwargs: object) -> "SpyModel":
            device = kwargs.get("device", args[0] if args else None)
            self.to_devices.append(str(device))
            return self

    save_file({}, str(tmp_path / "ensemble_0.safetensors"))
    (tmp_path / "config.json").write_text(
        json.dumps({"model_yaml": "class: torch.nn.Identity", "cutoff": 5.0, "format_version": 2})
    )
    model = SpyModel()
    load_file = Mock(return_value={})
    monkeypatch.setattr(hf_hub, "_load_safetensors_file", load_file, raising=False)
    monkeypatch.setattr(hf_hub, "build_module", Mock(return_value=model))

    loaded, _ = load_from_hf_repo(
        str(tmp_path),
        device="cuda",
        model_import_paths={"torch.nn.Identity"},
    )

    assert loaded is model
    load_file.assert_called_once_with(str(tmp_path / "ensemble_0.safetensors"), device="cpu")
    assert model.to_devices == ["cuda"]


def test_hf_preserves_float64_atomic_shifts(monkeypatch: pytest.MonkeyPatch, tmp_path):
    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.outputs = torch.nn.Module()
            self.outputs.atomic_shift = AtomicShift("energy", "shifted")
            self.to_devices: list[str] = []

        def to(self, *args: object, **kwargs: object) -> "Model":
            device = kwargs.get("device", args[0] if args else None)
            self.to_devices.append(str(device))
            return self

    values = torch.zeros(64, 1, dtype=torch.float64)
    values[1, 0] = 1.0000000000000002
    values[2, 0] = 2.0000000000000004
    model = Model()
    save_file({}, str(tmp_path / "ensemble_0.safetensors"))
    (tmp_path / "config.json").write_text(
        json.dumps({"model_yaml": "class: aimnet.models.AIMNet2", "cutoff": 5.0, "format_version": 2})
    )
    load_file = Mock(return_value={"outputs.atomic_shift.shifts.weight": values})
    monkeypatch.setattr(hf_hub, "_load_safetensors_file", load_file)
    monkeypatch.setattr(hf_hub, "build_module", Mock(return_value=model))

    loaded, _ = load_from_hf_repo(str(tmp_path), device="cuda")

    assert loaded is model
    assert model.outputs.atomic_shift.shifts.weight.dtype is torch.float64
    assert torch.equal(model.outputs.atomic_shift.shifts.weight.detach(), values)
    assert model.to_devices == ["cuda"]


def test_hf_complete_artifact_warns_on_unexpected_key(monkeypatch: pytest.MonkeyPatch, tmp_path):
    model = torch.nn.Linear(2, 2)
    state_dict = {**model.state_dict(), "extra": torch.zeros(1)}
    save_file(state_dict, str(tmp_path / "ensemble_0.safetensors"))
    (tmp_path / "config.json").write_text(
        json.dumps({
            "model_yaml": "class: torch.nn.Linear\nkwargs:\n  in_features: 2\n  out_features: 2\n",
            "cutoff": 5.0,
            "format_version": 2,
        })
    )

    with pytest.warns(UserWarning, match=r"Unexpected model parameters.*extra"):
        loaded, _ = load_from_hf_repo(
            str(tmp_path),
            model_import_paths={"torch.nn.Linear"},
        )

    assert isinstance(loaded, torch.nn.Linear)


def test_hf_artifact_fails_on_missing_key(tmp_path):
    model = torch.nn.Linear(2, 2)
    save_file({"weight": model.weight.detach().clone()}, str(tmp_path / "ensemble_0.safetensors"))
    (tmp_path / "config.json").write_text(
        json.dumps({
            "model_yaml": "class: torch.nn.Linear\nkwargs:\n  in_features: 2\n  out_features: 2\n",
            "cutoff": 5.0,
            "format_version": 2,
        })
    )

    with pytest.raises(RuntimeError, match=r"Missing model parameters.*bias"):
        load_from_hf_repo(
            str(tmp_path),
            model_import_paths={"torch.nn.Linear"},
        )


def test_hf_registry_fallback_fails_on_unexpected_key(monkeypatch: pytest.MonkeyPatch, tmp_path):
    state_dict = {
        "weight": torch.zeros(2, 2),
        "bias": torch.zeros(2),
        "extra": torch.zeros(1),
    }
    save_file(state_dict, str(tmp_path / "ensemble_0.safetensors"))
    (tmp_path / "config.json").write_text(json.dumps({"member_names": ["aimnet2"]}))
    monkeypatch.setattr(
        hf_hub,
        "_fetch_pt_metadata_from_registry",
        Mock(
            return_value=(
                {
                    "model_yaml": "class: aimnet.models.AIMNet2",
                    "cutoff": 5.0,
                    "format_version": 2,
                },
                {"class": "aimnet.models.AIMNet2"},
            )
        ),
    )
    monkeypatch.setattr(hf_hub, "build_module", Mock(return_value=torch.nn.Identity()))

    with pytest.raises(RuntimeError, match=r"Unexpected model parameters.*extra"):
        load_from_hf_repo(str(tmp_path))


@pytest.mark.hf
def test_hf_custom_model_does_not_expand_sidecar_yaml(tmp_path):
    sidecar = tmp_path / "sidecar.yaml"
    sidecar.write_text("class: os.system\n", encoding="utf-8")
    save_file({}, str(tmp_path / "ensemble_0.safetensors"))
    (tmp_path / "config.json").write_text(
        json.dumps({
            "model_yaml": f"class: torch.nn.Identity\nsidecar: {sidecar}\n",
            "cutoff": 5.0,
            "format_version": 2,
            "implemented_species": [],
        })
    )

    model, metadata = load_from_hf_repo(
        str(tmp_path),
        model_import_mode="extend",
        model_import_paths={"torch.nn.Identity"},
    )

    assert isinstance(model, torch.nn.Identity)
    assert metadata["format_version"] == 2


@pytest.mark.slow
@pytest.mark.hf
def test_load_from_hf_repo_local(fake_hf_repo):
    """Test loading model from a local directory mimicking HF repo structure."""
    model, metadata = load_from_hf_repo(str(fake_hf_repo), ensemble_member=0)
    assert model is not None
    assert metadata["cutoff"] > 0
    assert len(metadata["implemented_species"]) > 0


@pytest.mark.slow
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
    raw = torch.load(pt_path, map_location="cpu", weights_only=True)

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
        "d3_params": raw.get("d3_params"),
        "coulomb_mode": raw.get("coulomb_mode", "none"),
        "implemented_species": raw.get("implemented_species", []),
        "model_yaml": raw["model_yaml"],
        "format_version": 2,
        "coulomb_sr_rc": raw.get("coulomb_sr_rc"),
        "coulomb_sr_envelope": raw.get("coulomb_sr_envelope"),
        "has_embedded_lr": raw.get("has_embedded_lr", False),
        "has_embedded_d3ts": raw.get("has_embedded_d3ts", False),
        # NEW fields under test:
        "family": "test-family",
        "supports_charged_systems": False,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    return tmp_path


@pytest.mark.slow
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
