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
from aimnet.models import base as model_base
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

    save_file({}, str(tmp_path / "ensemble_0.safetensors"))
    (tmp_path / "config.json").write_text(
        json.dumps({"model_yaml": "class: os.system", "cutoff": 5.0, "format_version": 2})
    )
    build_module = Mock(side_effect=AssertionError("build_module must not be called"))
    monkeypatch.setattr(model_base, "build_module", build_module)

    with pytest.raises(ValueError, match="Untrusted import path"):
        load_from_hf_repo(str(tmp_path))

    build_module.assert_not_called()


@pytest.mark.hf
@pytest.mark.parametrize("format_version", [True, 1, "2"])
def test_hf_rejects_invalid_format_version(tmp_path, format_version):
    save_file({}, str(tmp_path / "ensemble_0.safetensors"))
    (tmp_path / "config.json").write_text(
        json.dumps({
            "model_yaml": "class: aimnet.models.AIMNet2",
            "cutoff": 5.0,
            "format_version": format_version,
        })
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


@pytest.mark.parametrize("ensemble_member", [True, -1, 1.0, "0"])
def test_hf_rejects_invalid_ensemble_member_before_repo_access(
    monkeypatch: pytest.MonkeyPatch,
    ensemble_member,
) -> None:
    resolve_repo = Mock(side_effect=AssertionError("repository must not be accessed"))
    monkeypatch.setattr(hf_hub, "_resolve_repo", resolve_repo)

    with pytest.raises(ValueError, match="ensemble_member"):
        load_from_hf_repo("org/repository", ensemble_member=ensemble_member)

    resolve_repo.assert_not_called()


@pytest.mark.parametrize("member_names", [[], "aimnet2", ["aimnet2", 1]])
def test_hf_rejects_invalid_member_names_before_weight_access(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    member_names,
) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({
            "model_yaml": "class: aimnet.models.AIMNet2",
            "cutoff": 5.0,
            "format_version": 2,
            "member_names": member_names,
        })
    )
    resolve_repo = Mock(return_value=tmp_path)
    monkeypatch.setattr(hf_hub, "_resolve_repo", resolve_repo)

    with pytest.raises(ValueError, match="member_names"):
        load_from_hf_repo("org/repository")

    assert resolve_repo.call_count == 1
    assert resolve_repo.call_args.kwargs["include_weights"] is False


def test_hf_rejects_out_of_range_member_name_before_weight_access(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({
            "model_yaml": "class: aimnet.models.AIMNet2",
            "cutoff": 5.0,
            "format_version": 2,
            "member_names": ["aimnet2"],
        })
    )
    resolve_repo = Mock(return_value=tmp_path)
    monkeypatch.setattr(hf_hub, "_resolve_repo", resolve_repo)

    with pytest.raises(ValueError, match=r"ensemble_member.*member_names"):
        load_from_hf_repo("org/repository", ensemble_member=1)

    assert resolve_repo.call_count == 1
    assert resolve_repo.call_args.kwargs["include_weights"] is False


def test_hf_registry_family_members_never_fall_back_to_member_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(hf_hub, "get_family_policy", Mock(return_value=Mock(members=("aimnet2",))))
    registry_path = Mock(side_effect=AssertionError("registry member zero must not be used"))
    monkeypatch.setattr(hf_hub, "get_registry_model_path", registry_path)

    with pytest.raises(ValueError, match=r"ensemble_member.*family"):
        _fetch_pt_metadata_from_registry({"family_name": "aimnet2"}, "org/aimnet2", 1)

    registry_path.assert_not_called()


def test_hf_allows_incomplete_external_dispersion_metadata_for_calculator_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    save_file({}, str(tmp_path / "ensemble_0.safetensors"))
    (tmp_path / "config.json").write_text(
        json.dumps({
            "model_yaml": "class: aimnet.models.AIMNet2",
            "cutoff": 5.0,
            "format_version": 2,
            "needs_dispersion": True,
        })
    )
    load_file = Mock(return_value={})
    build_module = Mock(return_value=torch.nn.Identity())
    monkeypatch.setattr(hf_hub, "_load_safetensors_file", load_file)
    monkeypatch.setattr(model_base, "build_module", build_module)

    _, metadata = load_from_hf_repo(str(tmp_path))

    assert metadata["needs_dispersion"] is True
    load_file.assert_called_once()
    build_module.assert_called_once()


def test_hf_complete_config_derives_sr_metadata_before_loading_weights(monkeypatch: pytest.MonkeyPatch, tmp_path):
    save_file({}, str(tmp_path / "ensemble_0.safetensors"))
    (tmp_path / "config.json").write_text(
        json.dumps({
            "model_yaml": """
class: aimnet.models.AIMNet2
kwargs:
  outputs:
    coulomb:
      class: aimnet.modules.SRCoulomb
      kwargs:
        rc: 4.6
        envelope: cosine
""",
            "cutoff": 5.0,
            "format_version": 2,
            "needs_coulomb": True,
            "coulomb_mode": "sr_embedded",
            "has_embedded_lr": True,
        })
    )
    load_file = Mock(return_value={})
    monkeypatch.setattr(hf_hub, "_load_safetensors_file", load_file)
    monkeypatch.setattr(model_base, "build_module", Mock(return_value=torch.nn.Identity()))

    _, metadata = load_from_hf_repo(str(tmp_path))

    assert metadata["coulomb_sr_rc"] == 4.6
    assert metadata["coulomb_sr_envelope"] == "cosine"
    load_file.assert_called_once()


def test_hf_complete_config_accepts_duplicate_identical_srcoulomb_pairs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    save_file({}, str(tmp_path / "ensemble_0.safetensors"))
    (tmp_path / "config.json").write_text(
        json.dumps({
            "model_yaml": """
class: aimnet.models.AIMNet2
kwargs:
  outputs:
    - class: aimnet.modules.SRCoulomb
      kwargs: {rc: 4.6, envelope: exp}
    - nested:
        class: custom.SRCoulomb
        kwargs: {rc: 4.6, envelope: exp}
""",
            "cutoff": 5.0,
            "format_version": 2,
            "needs_coulomb": True,
            "coulomb_mode": "sr_embedded",
            "has_embedded_lr": True,
        })
    )
    monkeypatch.setattr(hf_hub, "_load_safetensors_file", Mock(return_value={}))
    monkeypatch.setattr(model_base, "build_module", Mock(return_value=torch.nn.Identity()))

    _, metadata = load_from_hf_repo(
        str(tmp_path),
        model_import_paths={"custom.SRCoulomb"},
    )

    assert (metadata["coulomb_sr_rc"], metadata["coulomb_sr_envelope"]) == (4.6, "exp")


def test_hf_complete_config_rejects_ambiguous_srcoulomb_pairs_before_weight_access(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({
            "model_yaml": """
class: aimnet.models.AIMNet2
kwargs:
  outputs:
    - class: aimnet.modules.SRCoulomb
      kwargs: {rc: 4.6, envelope: exp}
    - class: aimnet.modules.SRCoulomb
      kwargs: {rc: 4.5, envelope: cosine}
""",
            "cutoff": 5.0,
            "format_version": 2,
            "needs_coulomb": True,
            "coulomb_mode": "sr_embedded",
            "has_embedded_lr": True,
        })
    )
    resolve_repo = Mock(return_value=tmp_path)
    monkeypatch.setattr(hf_hub, "_resolve_repo", resolve_repo)

    with pytest.raises(ValueError, match=r"ambiguous.*SRCoulomb"):
        load_from_hf_repo("org/repository")

    assert resolve_repo.call_count == 1
    assert resolve_repo.call_args.kwargs["include_weights"] is False


def test_hf_complete_config_rejects_invalid_srcoulomb_pair_before_weight_access(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({
            "model_yaml": """
class: aimnet.models.AIMNet2
kwargs:
  outputs:
    class: aimnet.modules.SRCoulomb
    kwargs: {rc: 4.6, envelope: [exp]}
""",
            "cutoff": 5.0,
            "format_version": 2,
            "coulomb_mode": "sr_embedded",
            "has_embedded_lr": True,
        })
    )
    resolve_repo = Mock(return_value=tmp_path)
    monkeypatch.setattr(hf_hub, "_resolve_repo", resolve_repo)

    with pytest.raises(ValueError, match="coulomb_sr_envelope"):
        load_from_hf_repo("org/repository")

    assert resolve_repo.call_count == 1
    assert resolve_repo.call_args.kwargs["include_weights"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("coulomb_sr_rc", 4.5),
        ("coulomb_sr_envelope", "cosine"),
    ],
)
def test_hf_complete_config_rejects_srcoulomb_metadata_conflicts_before_weight_access(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    field: str,
    value,
) -> None:
    config = {
        "model_yaml": """
class: aimnet.models.AIMNet2
kwargs:
  outputs:
    coulomb:
      class: aimnet.modules.SRCoulomb
      kwargs: {rc: 4.6, envelope: exp}
""",
        "cutoff": 5.0,
        "format_version": 2,
        "needs_coulomb": True,
        "coulomb_mode": "sr_embedded",
        "coulomb_sr_rc": 4.6,
        "coulomb_sr_envelope": "exp",
        "has_embedded_lr": True,
    }
    config[field] = value
    (tmp_path / "config.json").write_text(json.dumps(config))
    resolve_repo = Mock(return_value=tmp_path)
    monkeypatch.setattr(hf_hub, "_resolve_repo", resolve_repo)

    with pytest.raises(ValueError, match=field):
        load_from_hf_repo("org/repository")

    assert resolve_repo.call_count == 1
    assert resolve_repo.call_args.kwargs["include_weights"] is False


def test_hf_complete_config_validates_derived_sr_metadata_before_loading_weights(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    (tmp_path / "config.json").write_text(
        json.dumps({
            "model_yaml": """
class: aimnet.models.AIMNet2
kwargs:
  outputs:
    coulomb:
      class: aimnet.modules.SRCoulomb
      kwargs:
        rc: 4.6
        envelope: exp
""",
            "cutoff": 5.0,
            "format_version": 2,
            "needs_coulomb": True,
            "coulomb_mode": "sr_embedded",
            "has_embedded_lr": False,
        })
    )
    load_file = Mock(side_effect=AssertionError("weights must not be loaded"))
    monkeypatch.setattr(hf_hub, "_load_safetensors_file", load_file)

    with pytest.raises(ValueError, match="embedded LR"):
        load_from_hf_repo(str(tmp_path))

    load_file.assert_not_called()


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
    monkeypatch.setattr(model_base, "build_module", Mock(return_value=model))

    loaded, metadata = load_from_hf_repo(str(tmp_path))

    assert loaded is model
    assert metadata["cutoff"] == 5.0


def test_hf_fallback_accepts_matching_artifact_metadata_duplicates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    save_file({}, str(tmp_path / "ensemble_0.safetensors"))
    registry_metadata = {
        "model_yaml": "class: aimnet.models.AIMNet2",
        "cutoff": 5.0,
        "format_version": 2,
        "needs_coulomb": False,
        "needs_dispersion": False,
        "coulomb_mode": "none",
        "coulomb_sr_rc": None,
        "coulomb_sr_envelope": None,
        "d3_params": None,
        "has_embedded_lr": False,
        "has_embedded_d3ts": False,
        "implemented_species": [1, 6],
        "family": "test",
        "supports_charged_systems": True,
    }
    (tmp_path / "config.json").write_text(
        json.dumps({
            "config_schema_version": 1,
            "member_names": ["aimnet2"],
            **{key: value for key, value in registry_metadata.items() if key != "model_yaml"},
        })
    )
    monkeypatch.setattr(
        hf_hub,
        "_fetch_pt_metadata_from_registry",
        Mock(return_value=(registry_metadata, {"class": "aimnet.models.AIMNet2"})),
    )
    monkeypatch.setattr(hf_hub, "_load_safetensors_file", Mock(return_value={}))
    monkeypatch.setattr(model_base, "build_module", Mock(return_value=torch.nn.Identity()))

    _, metadata = load_from_hf_repo(str(tmp_path))

    assert metadata["cutoff"] == registry_metadata["cutoff"]
    assert metadata["family"] == registry_metadata["family"]


@pytest.mark.parametrize(
    ("field", "conflicting_value"),
    [
        ("format_version", 1),
        ("model_yaml", None),
        ("cutoff", 6.0),
        ("needs_coulomb", True),
        ("needs_dispersion", True),
        ("coulomb_mode", "full_embedded"),
        ("coulomb_sr_rc", 4.5),
        ("coulomb_sr_envelope", "cosine"),
        ("d3_params", {"s8": 1.0, "a1": 1.0, "a2": 1.0}),
        ("has_embedded_lr", True),
        ("has_embedded_d3ts", True),
        ("implemented_species", [8]),
        ("family", "other"),
        ("supports_charged_systems", False),
    ],
)
def test_hf_fallback_rejects_conflicting_artifact_metadata_before_weight_access(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    field: str,
    conflicting_value,
) -> None:
    registry_metadata = {
        "model_yaml": "class: aimnet.models.AIMNet2",
        "cutoff": 5.0,
        "format_version": 2,
        "needs_coulomb": False,
        "needs_dispersion": False,
        "coulomb_mode": "none",
        "coulomb_sr_rc": None,
        "coulomb_sr_envelope": None,
        "d3_params": None,
        "has_embedded_lr": False,
        "has_embedded_d3ts": False,
        "implemented_species": [1, 6],
        "family": "test",
        "supports_charged_systems": True,
    }
    family_config = {
        "config_schema_version": 1,
        "member_names": ["aimnet2"],
        field: conflicting_value,
    }
    (tmp_path / "config.json").write_text(json.dumps(family_config))
    monkeypatch.setattr(
        hf_hub,
        "_fetch_pt_metadata_from_registry",
        Mock(return_value=(registry_metadata, {"class": "aimnet.models.AIMNet2"})),
    )
    resolve_repo = Mock(return_value=tmp_path)
    monkeypatch.setattr(hf_hub, "_resolve_repo", resolve_repo)

    with pytest.raises(ValueError, match=field):
        load_from_hf_repo("org/repository")

    assert resolve_repo.call_count == 1
    assert resolve_repo.call_args.kwargs["include_weights"] is False


def test_hf_fallback_rejects_nonrouting_family_config_before_weight_access(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({
            "config_schema_version": 1,
            "member_names": ["aimnet2"],
            "architectures": ["AIMNet2"],
        })
    )
    registry_metadata = {
        "model_yaml": "class: aimnet.models.AIMNet2",
        "cutoff": 5.0,
        "format_version": 2,
    }
    monkeypatch.setattr(
        hf_hub,
        "_fetch_pt_metadata_from_registry",
        Mock(return_value=(registry_metadata, {"class": "aimnet.models.AIMNet2"})),
    )
    resolve_repo = Mock(return_value=tmp_path)
    monkeypatch.setattr(hf_hub, "_resolve_repo", resolve_repo)

    with pytest.raises(ValueError, match=r"architectures.*routing"):
        load_from_hf_repo("org/repository")

    assert resolve_repo.call_count == 1
    assert resolve_repo.call_args.kwargs["include_weights"] is False


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


def test_hf_complete_config_requires_cutoff_before_weight_access(monkeypatch: pytest.MonkeyPatch, tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_yaml": "class: aimnet.models.AIMNet2", "format_version": 2})
    )
    resolve_repo = Mock(return_value=tmp_path)
    monkeypatch.setattr(hf_hub, "_resolve_repo", resolve_repo)

    with pytest.raises(ValueError, match="cutoff"):
        load_from_hf_repo("org/repository")

    assert resolve_repo.call_count == 1
    assert resolve_repo.call_args.kwargs["include_weights"] is False


def test_hf_remote_config_and_weights_use_one_immutable_snapshot(monkeypatch: pytest.MonkeyPatch, tmp_path):
    config_commit = "a" * 40
    changed_commit = "b" * 40
    snapshots = tmp_path / "models--org--repository" / "snapshots"
    config_snapshot = snapshots / config_commit
    changed_snapshot = snapshots / changed_commit
    config_snapshot.mkdir(parents=True)
    changed_snapshot.mkdir()
    config = {
        "model_yaml": "class: torch.nn.Linear\nkwargs:\n  in_features: 2\n  out_features: 2\n",
        "cutoff": 5.0,
        "format_version": 2,
    }
    (config_snapshot / "config.json").write_text(json.dumps(config))
    save_file(
        {"weight": torch.ones(2, 2), "bias": torch.ones(2)},
        str(config_snapshot / "ensemble_0.safetensors"),
    )
    save_file(
        {"weight": torch.full((2, 2), 2.0), "bias": torch.full((2,), 2.0)},
        str(changed_snapshot / "ensemble_0.safetensors"),
    )

    def snapshot_download(**kwargs):
        if kwargs["allow_patterns"] == ["config.json"]:
            return str(config_snapshot)
        if kwargs["revision"] == config_commit:
            return str(config_snapshot)
        return str(changed_snapshot)

    download = Mock(side_effect=snapshot_download)
    monkeypatch.setattr(hf_hub, "_snapshot_download", download)

    model, _ = load_from_hf_repo(
        "org/repository",
        revision="main",
        model_import_paths={"torch.nn.Linear"},
    )

    torch.testing.assert_close(model.weight, torch.ones(2, 2))
    torch.testing.assert_close(model.bias, torch.ones(2))
    assert download.call_count == 2
    assert download.call_args_list[1].kwargs["revision"] == config_commit


def test_hf_rejects_non_module_construction(monkeypatch: pytest.MonkeyPatch, tmp_path):
    save_file({}, str(tmp_path / "ensemble_0.safetensors"))
    (tmp_path / "config.json").write_text(
        json.dumps({"model_yaml": "class: aimnet.models.AIMNet2", "cutoff": 5.0, "format_version": 2})
    )
    monkeypatch.setattr(model_base, "build_module", Mock(return_value=object()))

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
    monkeypatch.setattr(model_base, "build_module", Mock(return_value=model))

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
    monkeypatch.setattr(model_base, "build_module", Mock(return_value=model))

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
    monkeypatch.setattr(model_base, "build_module", Mock(return_value=model))

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
    monkeypatch.setattr(model_base, "build_module", Mock(return_value=torch.nn.Identity()))

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
