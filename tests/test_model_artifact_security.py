"""Security and dispatch tests for serialized model artifacts."""

from __future__ import annotations

import asyncio
import json
import threading
import warnings
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from aimnet import config
from aimnet.calculators import hf_hub
from aimnet.models import artifact_validation
from aimnet.models import base as model_base
from aimnet.models import utils as model_utils
from aimnet.models.artifact_validation import (
    ALLOWED_MODEL_IMPORT_PATHS,
    ModelImportPolicy,
    validate_model_metadata,
    validate_model_yaml,
    validate_registry_v2_artifact,
    validate_runtime_model_metadata,
    validate_v2_artifact,
)
from aimnet.modules import AtomicShift


def _v2_data(model_yaml: str, **overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "model_yaml": model_yaml,
        "state_dict": {},
        "cutoff": 5.0,
        "format_version": 2,
    }
    data.update(overrides)
    return data


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("model.jpt", True),
        ("MODEL.JPT", True),
        (".jpt", True),
        ("model.pt", False),
    ],
)
def test_legacy_jit_path_routing(path: str, expected: bool) -> None:
    assert artifact_validation.is_legacy_jit_path(path) is expected


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/models/model.pt", True),
        ("./model.pt", True),
        ("../model.pt", True),
        ("model.pt", False),
        ("org/model", False),
    ],
)
def test_explicit_local_path_routing(path: str, expected: bool) -> None:
    assert artifact_validation.is_explicit_local_path(path) is expected


@pytest.mark.parametrize(
    ("paths", "mode", "expected"),
    [
        (None, "extend", True),
        ((), "extend", False),
        (None, "replace", False),
        (None, "unsafe", False),
    ],
)
def test_default_model_import_settings(paths: object, mode: str, expected: bool) -> None:
    assert artifact_validation.uses_default_model_import_settings(paths, mode) is expected


def test_renamed_loading_helpers_keep_monkeypatch_aliases() -> None:
    assert artifact_validation._REGISTRY_IMPORT_POLICY is artifact_validation.REGISTRY_IMPORT_POLICY
    assert artifact_validation._validate_registry_v2_artifact is artifact_validation.validate_registry_v2_artifact
    assert model_base._REGISTRY_IMPORT_POLICY is artifact_validation.REGISTRY_IMPORT_POLICY
    assert model_base._load_registry_model is model_base.load_registry_model


def test_assemble_v2_model_applies_shared_construction_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = AtomicShift("energy", "first_shifted")
            self.second = AtomicShift("energy", "second_shifted")
            self.to_devices: list[str] = []

        def to(self, *args: object, **kwargs: object) -> Model:
            device = kwargs.get("device", args[0] if args else None)
            self.to_devices.append(str(device))
            return self

    model = Model()
    config = {"class": "torch.nn.Identity", "nested": {"value": "original"}}
    metadata: model_base.ModelMetadata = {
        "format_version": 2,
        "cutoff": 5.0,
        "needs_coulomb": False,
        "needs_dispersion": False,
        "coulomb_mode": "none",
        "d3_params": {"s8": 1.0, "a1": 2.0, "a2": 3.0},
        "implemented_species": [],
    }
    seen: dict[str, object] = {}

    def build(model_config: dict, **kwargs: object) -> Model:
        seen["device"] = torch.empty(1).device.type
        seen["allow_file_references"] = kwargs["allow_file_references"]
        kwargs["import_authorizer"]("torch.nn.Identity", "class")  # type: ignore[operator]
        model_config["nested"]["value"] = "changed"
        return model

    monkeypatch.setattr(model_base, "build_module", build)
    values = torch.zeros(64, 1, dtype=torch.float64)
    values[1, 0] = 1.0000000000000002
    state_dict = {
        "first.shifts.weight": values,
        "second.shifts.weight": values + 1,
    }
    policy = ModelImportPolicy(
        class_paths=frozenset({"torch.nn.Identity"}),
        activation_paths=frozenset(),
        initializer_paths=frozenset(),
    )

    loaded = model_base.assemble_v2_model(
        config,
        state_dict,
        metadata,
        policy=policy,
        device="cuda",
        source="model.pt",
        unexpected="warn",
    )

    assert loaded is model
    assert seen == {"device": "cpu", "allow_file_references": False}
    assert config["nested"]["value"] == "original"
    assert model.first.shifts.weight.dtype is torch.float64
    assert model.second.shifts.weight.dtype is torch.float64
    assert torch.equal(model.first.shifts.weight.detach(), values)
    assert torch.equal(model.second.shifts.weight.detach(), values + 1)
    assert model.to_devices == ["cuda"]
    assert model.training is True
    assert model._metadata == metadata
    assert model._metadata is not metadata
    assert model._metadata["d3_params"] is not metadata["d3_params"]


def test_structural_validation_allows_disabled_incomplete_external_dispersion() -> None:
    metadata = {
        "format_version": 2,
        "cutoff": 5.0,
        "needs_dispersion": True,
        "d3_params": None,
    }
    validate_model_metadata(metadata, require_cutoff=True, require_structural_consistency=True)
    validate_runtime_model_metadata(metadata, needs_coulomb=False, needs_dispersion=False)
    with pytest.raises(ValueError, match="d3_params"):
        validate_runtime_model_metadata(metadata, needs_coulomb=False, needs_dispersion=True)


def test_structural_validation_cannot_disable_invalid_sr_coulomb() -> None:
    metadata = {
        "format_version": 2,
        "cutoff": 5.0,
        "coulomb_mode": "sr_embedded",
        "coulomb_sr_rc": None,
        "coulomb_sr_envelope": None,
        "has_embedded_lr": True,
    }
    with pytest.raises(ValueError, match="sr_embedded"):
        validate_model_metadata(metadata, require_cutoff=True, require_structural_consistency=True)


def test_default_extend_allows_official_and_exact_torch_paths() -> None:
    parsed = validate_model_yaml(
        """
class: aimnet.models.AIMNet2
kwargs:
  in_features: 2
  out_features: 2
activation_fn: torch.nn.GELU
weight_init_fn: torch.nn.init.xavier_normal_
""",
    )
    assert parsed["class"] == "aimnet.models.AIMNet2"


@pytest.mark.parametrize("path", ["torch.nn.Linear", "torch.nn.ReLU", "torch.nn.init.uniform_"])
def test_default_extend_rejects_unlisted_torch_paths(path: str) -> None:
    with pytest.raises(ValueError, match="Untrusted"):
        validate_model_yaml(f"class: {path}")


def test_custom_torch_paths_can_be_explicitly_extended() -> None:
    validate_model_yaml("class: torch.nn.Linear", model_import_paths={"torch.nn.Linear"})


def test_default_import_paths_are_role_specific() -> None:
    with pytest.raises(ValueError, match="Untrusted"):
        validate_model_yaml("class: torch.nn.GELU")
    with pytest.raises(ValueError, match="Untrusted"):
        validate_model_yaml("activation_fn: torch.nn.init.xavier_normal_")
    with pytest.raises(ValueError, match="Untrusted"):
        validate_model_yaml("weight_init_fn: torch.nn.GELU")

    validate_model_yaml("activation_fn: torch.nn.GELU")
    validate_model_yaml("weight_init_fn: torch.nn.init.xavier_normal_")


@pytest.mark.parametrize(
    ("path", "owner"),
    [
        ("aimnet.models.AIMNet2", "class"),
        ("aimnet.models.aimnet2.AIMNet2", "class"),
        ("aimnet.modules.AtomicShift", "class"),
        ("aimnet.modules.AtomicSum", "class"),
        ("aimnet.modules.Dipole", "class"),
        ("aimnet.modules.Output", "class"),
        ("aimnet.modules.Quadrupole", "class"),
        ("aimnet.modules.SRCoulomb", "class"),
        ("torch.nn.GELU", "activation"),
        ("torch.nn.init.xavier_normal_", "initializer"),
    ],
)
@pytest.mark.parametrize("role", ["class", "activation", "initializer"])
def test_default_import_paths_reject_every_wrong_role(path: str, owner: str, role: str) -> None:
    key_by_role = {"class": "class", "activation": "activation_fn", "initializer": "weight_init_fn"}

    if role == owner:
        validate_model_yaml(f"{key_by_role[role]}: {path}")
    else:
        with pytest.raises(ValueError, match="Untrusted"):
            validate_model_yaml(f"{key_by_role[role]}: {path}")


def test_default_extend_allows_torch_nn_init() -> None:
    yaml_text = "class: aimnet.models.AIMNet2\nweight_init_fn: torch.nn.init.xavier_normal_"
    validate_model_yaml(yaml_text)


def test_extend_allows_namespace_and_exact_additions() -> None:
    paths = {"tests.custom_models.CustomAIMNet", "my_package.models.*", "pkg.*"}
    validate_model_yaml("class: tests.custom_models.CustomAIMNet", model_import_paths=paths)
    validate_model_yaml("class: my_package.models.CustomAIMNet", model_import_paths=paths)
    validate_model_yaml("class: pkg.CustomAIMNet", model_import_paths=paths)
    with pytest.raises(ValueError, match="OtherModel"):
        validate_model_yaml("class: my_package.models2.OtherModel", model_import_paths=paths)


@pytest.mark.parametrize("path", ["torch.hub.load", "torch.utils.data.DataLoader"])
def test_default_extend_rejects_non_nn_torch_paths(path: str) -> None:
    with pytest.raises(ValueError, match="torch"):
        validate_model_yaml(f"class: {path}")


@pytest.mark.parametrize(
    "path",
    [
        "",
        " ",
        " pkg.Class ",
        "pkg",
        "pkg.class",
        "pkg.bad-name",
        "pkg..Class",
        "pkg.*.Class",
        "pkg.**",
        "pkg.*.*",
        "*",
        "pkg.?*",
        "pkg.[module]",
        "torch.*",
        "torch.hub.*",
        "pkg.",
    ],
)
def test_rejects_malformed_import_patterns(path: str) -> None:
    with pytest.raises(ValueError, match="import path"):
        validate_model_yaml("class: aimnet.models.AIMNet2", model_import_paths={path})


@pytest.mark.parametrize(
    "paths",
    ["pkg.Class", b"pkg.Class", {"pkg.Class": 1}, iter(["pkg.Class"]), {"pkg.Class", 1}, 1],
)
def test_rejects_invalid_import_path_collections(paths: object) -> None:
    with pytest.raises(ValueError, match="collection"):
        validate_model_yaml("class: aimnet.models.AIMNet2", model_import_paths=paths)  # type: ignore[arg-type]


def test_replace_isolated_and_extend_noop() -> None:
    validate_model_yaml("class: tests.custom_models.CustomAIMNet", model_import_paths={"tests.custom_models.*"})
    with pytest.raises(ValueError, match="Untrusted"):
        validate_model_yaml(
            "class: aimnet.models.AIMNet2", model_import_mode="replace", model_import_paths={"tests.custom_models.*"}
        )
    validate_model_yaml("class: torch.nn.Linear", model_import_mode="replace", model_import_paths={"torch.nn.*"})
    validate_model_yaml("class: aimnet.models.AIMNet2", model_import_mode="extend", model_import_paths=())


@pytest.mark.parametrize("mode", ["invalid", [], {}])
def test_invalid_mode_combinations(mode: object) -> None:
    with pytest.raises(ValueError, match="mode"):
        validate_model_yaml("class: aimnet.models.AIMNet2", model_import_mode=mode)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty"):
        validate_model_yaml("class: aimnet.models.AIMNet2", model_import_mode="replace")
    with pytest.raises(ValueError, match="unsafe"):
        validate_model_yaml(
            "class: aimnet.models.AIMNet2", model_import_mode="unsafe", model_import_paths={"pkg.Class"}
        )


def test_unsafe_bypasses_only_known_path_matching() -> None:
    validate_model_yaml("class: os.system", model_import_mode="unsafe")
    with pytest.raises(ValueError, match="string"):
        validate_model_yaml("class: 1", model_import_mode="unsafe")


def test_import_path_normalization_does_not_mutate_caller() -> None:
    paths = {"tests.custom_models.CustomAIMNet"}
    validate_model_yaml("class: tests.custom_models.CustomAIMNet", model_import_paths=paths)
    assert paths == {"tests.custom_models.CustomAIMNet"}


def test_custom_paths_apply_uniformly_to_all_import_keys() -> None:
    paths = {"my_package.CustomAIMNet"}
    validate_model_yaml(
        """
class: my_package.CustomAIMNet
activation_fn: my_package.CustomAIMNet
weight_init_fn: my_package.CustomAIMNet
""",
        model_import_paths=paths,
    )


def test_unsafe_preserves_yaml_and_envelope_checks() -> None:
    with pytest.raises(ValueError, match="Invalid model_yaml"):
        validate_model_yaml(
            "class: os.system\nvalue: !!python/object/apply:os.system ['true']",
            model_import_mode="unsafe",
        )
    with pytest.raises(ValueError, match="cycle"):
        validate_model_yaml("root: &cycle\n  child: *cycle", model_import_mode="unsafe")
    with pytest.raises(ValueError, match="state_dict"):
        validate_v2_artifact(
            _v2_data("class: os.system", state_dict={"bad": "not a tensor"}),
            model_import_mode="unsafe",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("format_version", 1),
        ("cutoff", True),
        ("needs_coulomb", "yes"),
    ],
)
def test_unsafe_preserves_v2_metadata_validation(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        validate_v2_artifact(
            _v2_data("class: os.system", **{field: value}),
            model_import_mode="unsafe",
        )


@pytest.mark.parametrize("key", ["fn", "trainer", "evaluator"])
def test_custom_policy_rejects_training_keys(key: str) -> None:
    with pytest.raises(ValueError, match=key):
        validate_model_yaml(
            f"{key}: tests.custom_models.TrainingEntryPoint",
            model_import_mode="unsafe",
        )


def test_rejects_nested_malicious_path() -> None:
    with pytest.raises(ValueError, match=r"os\.system"):
        validate_model_yaml(
            """
class: aimnet.models.AIMNet2
kwargs:
  nested:
    - class: os.system
"""
        )


def test_rejects_cyclic_yaml_alias() -> None:
    with pytest.raises(ValueError, match="cycle"):
        validate_model_yaml("root: &cycle\n  child: *cycle")


def test_rejects_non_mapping_model_yaml() -> None:
    with pytest.raises(ValueError, match="mapping"):
        validate_model_yaml("- class: aimnet.models.AIMNet2")


def test_rejects_explicit_non_v2_format_version() -> None:
    with pytest.raises(ValueError, match="format_version"):
        validate_v2_artifact(_v2_data("class: aimnet.models.AIMNet2", format_version=1))


def test_validate_v2_artifact_rejects_invalid_cutoff() -> None:
    with pytest.raises(ValueError, match="cutoff"):
        validate_v2_artifact(_v2_data("class: aimnet.models.AIMNet2", cutoff=True))


def test_validate_v2_artifact_requires_cutoff() -> None:
    data = _v2_data("class: aimnet.models.AIMNet2")
    del data["cutoff"]
    with pytest.raises(ValueError, match="cutoff"):
        validate_v2_artifact(data)


@pytest.mark.parametrize(
    "field,value",
    [
        ("needs_coulomb", "false"),
        ("implemented_species", "H"),
        ("coulomb_mode", "arbitrary"),
    ],
)
def test_validate_v2_artifact_rejects_invalid_runtime_metadata(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        validate_v2_artifact(_v2_data("class: aimnet.models.AIMNet2", **{field: value}))


def test_registry_v2_artifact_requires_complete_d3_params() -> None:
    with pytest.raises(ValueError, match="d3_params"):
        validate_registry_v2_artifact(
            _v2_data(
                "class: aimnet.models.AIMNet2",
                needs_dispersion=True,
                d3_params={"s8": 1.0},
            )
        )


def test_validate_v2_artifact_rejects_incomplete_sr_coulomb_metadata() -> None:
    with pytest.raises(ValueError, match="sr_embedded"):
        validate_v2_artifact(
            _v2_data(
                "class: aimnet.models.AIMNet2",
                needs_coulomb=True,
                coulomb_mode="sr_embedded",
                coulomb_sr_rc=None,
                coulomb_sr_envelope=None,
            )
        )


def test_registry_v2_artifact_rejects_sr_coulomb_without_external_coulomb() -> None:
    with pytest.raises(ValueError, match="external Coulomb"):
        validate_registry_v2_artifact(
            _v2_data(
                "class: aimnet.models.AIMNet2",
                needs_coulomb=False,
                coulomb_mode="sr_embedded",
                coulomb_sr_rc=4.6,
                coulomb_sr_envelope="exp",
                has_embedded_lr=True,
            )
        )


def test_validate_v2_artifact_requires_embedded_lr_metadata() -> None:
    with pytest.raises(ValueError, match="embedded LR"):
        validate_v2_artifact(
            _v2_data(
                "class: aimnet.models.AIMNet2",
                has_embedded_d3ts=True,
                has_embedded_lr=False,
            )
        )


def test_validate_v2_artifact_rejects_sr_coulomb_without_embedded_lr() -> None:
    with pytest.raises(ValueError, match="embedded LR"):
        validate_v2_artifact(
            _v2_data(
                "class: aimnet.models.AIMNet2",
                needs_coulomb=True,
                coulomb_mode="sr_embedded",
                coulomb_sr_rc=4.6,
                coulomb_sr_envelope="exp",
                has_embedded_lr=False,
            )
        )


def test_validate_v2_artifact_rejects_sr_cutoff_above_model_cutoff() -> None:
    with pytest.raises(ValueError, match="coulomb_sr_rc"):
        validate_v2_artifact(
            _v2_data(
                "class: aimnet.models.AIMNet2",
                needs_coulomb=True,
                coulomb_mode="sr_embedded",
                coulomb_sr_rc=6.0,
                coulomb_sr_envelope="exp",
                has_embedded_lr=True,
            )
        )


def test_pt_restricted_failure_is_not_retried(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from aimnet.models import base

    load = Mock(side_effect=RuntimeError("restricted load failed"))
    monkeypatch.setattr(base.torch, "load", load)

    with pytest.raises(RuntimeError, match="restricted load failed"):
        base.load_model(str(tmp_path / "model.pt"))

    load.assert_called_once_with(str(tmp_path / "model.pt"), map_location="cpu", weights_only=True)


def test_missing_pt_is_not_retried(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from aimnet.models import base

    load = Mock(side_effect=FileNotFoundError(str(tmp_path / "missing.pt")))
    monkeypatch.setattr(base.torch, "load", load)

    with pytest.raises(FileNotFoundError):
        base.load_model(str(tmp_path / "missing.pt"))

    load.assert_called_once()


def test_jpt_routes_only_to_torch_jit_load(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from aimnet.models import base

    scripted = Mock()
    scripted.cutoff = 5.0
    jit_load = Mock(return_value=scripted)
    torch_load = Mock(side_effect=AssertionError("torch.load must not be called"))
    monkeypatch.setattr(base.torch.jit, "load", jit_load)
    monkeypatch.setattr(base.torch, "load", torch_load)
    monkeypatch.setattr(base, "extract_species", lambda _: [1, 6])
    monkeypatch.setattr(base, "has_externalizable_dftd3", lambda _: False)

    real_load_model = base.load_model.__wrapped__
    model, metadata = real_load_model(str(tmp_path / "MODEL.JPT"))

    assert model is scripted
    assert metadata["format_version"] == 1
    jit_load.assert_called_once_with(str(tmp_path / "MODEL.JPT"), map_location="cpu")
    torch_load.assert_not_called()


def test_pt_torchscript_archive_does_not_route_to_legacy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from aimnet.models import base

    load = Mock(side_effect=RuntimeError("not a restricted archive"))
    jit_load = Mock(side_effect=AssertionError("JIT fallback is forbidden"))
    monkeypatch.setattr(base.torch, "load", load)
    monkeypatch.setattr(base.torch.jit, "load", jit_load)

    with pytest.raises(RuntimeError, match="not a restricted archive"):
        base.load_model(str(tmp_path / "scripted.pt"))

    load.assert_called_once()
    jit_load.assert_not_called()


@pytest.mark.parametrize(
    "model_yaml",
    [
        "class: os.system",
        "class: aimnet.models.AIMNet2\nactivation_fn: os.system",
    ],
)
def test_local_pt_rejects_malicious_yaml_before_construction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, model_yaml: str
) -> None:
    from aimnet.models import base

    build_module = Mock(side_effect=AssertionError("build_module must not be called"))
    monkeypatch.setattr(base, "build_module", build_module)
    path = tmp_path / "malicious.pt"
    torch.save(_v2_data(model_yaml), path)

    with pytest.raises(ValueError):
        base.load_model.__wrapped__(str(path))

    build_module.assert_not_called()


def test_local_pt_does_not_expand_sidecar_yaml(
    tmp_path: Path,
) -> None:
    from aimnet.models import base

    sidecar = tmp_path / "sidecar.yaml"
    sidecar.write_text("class: os.system\n", encoding="utf-8")
    path = tmp_path / "sidecar-reference.pt"
    torch.save(
        _v2_data(
            f"""
class: torch.nn.Identity
sidecar: {sidecar}
""",
        ),
        path,
    )

    model, _ = base.load_model.__wrapped__(
        str(path),
        model_import_mode="unsafe",
    )

    assert isinstance(model, torch.nn.Identity)


def test_local_pt_custom_import_paths_reach_construction(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from aimnet.models import base

    model = torch.nn.Linear(2, 2)
    build_module = Mock(return_value=model)
    monkeypatch.setattr(base, "build_module", build_module)
    path = tmp_path / "custom.pt"
    torch.save(
        _v2_data(
            """
class: my_package.CustomAIMNet
""",
            state_dict=model.state_dict(),
        ),
        path,
    )

    loaded, _ = base.load_model.__wrapped__(
        str(path),
        model_import_paths={"my_package.CustomAIMNet"},
    )

    assert loaded is model
    build_module.assert_called_once()


def test_unsafe_local_load_keeps_restricted_deserialization_and_sidecar_suppression(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from aimnet.models import base

    path = tmp_path / "unsafe.pt"
    torch.save(_v2_data("class: os.system", state_dict={}), path)
    model = torch.nn.Identity()
    real_load = base.torch.load
    load = Mock(wraps=real_load)
    build_module = Mock(return_value=model)
    monkeypatch.setattr(base.torch, "load", load)
    monkeypatch.setattr(base, "build_module", build_module)

    loaded, _ = base.load_model.__wrapped__(str(path), model_import_mode="unsafe")

    assert loaded is model
    load.assert_called_once_with(str(path), map_location="cpu", weights_only=True)
    assert build_module.call_args.kwargs["allow_file_references"] is False


def test_model_import_policy_is_exported_as_stable_type() -> None:
    import aimnet.models

    assert hasattr(aimnet.models, "ModelImportPolicy")
    assert not hasattr(aimnet.models, "custom_model_import_policy")


def test_allows_frozen_model_paths() -> None:
    validate_model_yaml(
        """
class: aimnet.models.AIMNet2
kwargs:
  outputs:
    class: aimnet.modules.Output
    kwargs:
      activation_fn: torch.nn.GELU
"""
    )


@pytest.mark.parametrize(
    ("model_import_paths", "model_import_mode"),
    [
        ({"my_package.CustomAIMNet"}, "extend"),
        (None, "replace"),
        (None, "unsafe"),
    ],
)
def test_registry_load_rejects_custom_import_settings(
    monkeypatch: pytest.MonkeyPatch,
    model_import_paths: object,
    model_import_mode: str,
) -> None:
    from aimnet.calculators import resolve

    monkeypatch.setattr(resolve, "try_resolve_registry_model_name", lambda _: "aimnet2-wb97m-d3_0")
    with pytest.raises(ValueError, match=r"registry|replace|unsafe"):
        resolve.resolve_model(
            "aimnet2",
            device="cpu",
            model_import_paths=model_import_paths,  # type: ignore[arg-type]
            model_import_mode=model_import_mode,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("model_name", ["aimnet2", "aimnet2-wb97m-d3_0"])
def test_registry_names_and_aliases_use_strict_loader(monkeypatch: pytest.MonkeyPatch, model_name: str) -> None:
    from aimnet.calculators import resolve

    model = torch.nn.Identity()
    metadata = {
        "cutoff": 5.0,
        "needs_coulomb": False,
        "needs_dispersion": False,
        "coulomb_mode": "none",
        "implemented_species": [],
    }
    strict_loader = Mock(return_value=(model, metadata))
    monkeypatch.setattr(resolve, "try_resolve_registry_model_name", lambda _: "aimnet2-wb97m-d3_0")
    monkeypatch.setattr(resolve, "get_registry_model_family", lambda _: None)
    monkeypatch.setattr(resolve, "get_model_path", lambda _: "/registry/model.pt")
    monkeypatch.setattr(resolve, "_load_registry_model", strict_loader)

    loaded, _, cutoff = resolve.resolve_model(model_name, device="cpu")

    assert loaded is model
    assert cutoff == 5.0
    strict_loader.assert_called_once_with("/registry/model.pt", device="cpu")


def test_raw_module_rejects_custom_import_settings_before_movement() -> None:
    from aimnet.calculators import resolve

    model = torch.nn.Identity()
    with pytest.raises(ValueError, match=r"raw nn\.Module"):
        resolve.resolve_model(model, device="cpu", model_import_mode="unsafe")


def test_jpt_rejects_custom_import_settings_before_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    from aimnet.calculators import resolve

    load_model = Mock(side_effect=AssertionError("loader must not be called"))
    monkeypatch.setattr(resolve, "load_model", load_model)
    with pytest.raises(ValueError, match=r"\.jpt"):
        resolve.resolve_model("custom.JPT", device="cpu", model_import_mode="unsafe")
    load_model.assert_not_called()


def test_explicit_local_file_does_not_route_to_hf(monkeypatch: pytest.MonkeyPatch) -> None:
    from aimnet.calculators import resolve

    load_model = Mock(return_value=(torch.nn.Identity(), {"cutoff": 5.0}))
    hf_loader = Mock(side_effect=AssertionError("HF loader must not be called"))
    monkeypatch.setattr(resolve, "load_model", load_model)
    monkeypatch.setattr(resolve, "try_resolve_registry_model_name", lambda _: None)
    monkeypatch.setattr(
        "aimnet.calculators.hf_hub.load_from_hf_repo",
        hf_loader,
    )

    resolve.resolve_model("./model.pt", device="cpu")

    load_model.assert_called_once()
    hf_loader.assert_not_called()


def test_unknown_model_uses_registry_resolution_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from aimnet.calculators import resolve

    get_model_path = Mock(side_effect=ValueError("Model 'missing' not found in the registry."))
    monkeypatch.setattr(resolve, "try_resolve_registry_model_name", lambda _: None)
    monkeypatch.setattr(resolve, "get_model_path", get_model_path)

    with pytest.raises(ValueError, match="not found in the registry"):
        resolve.resolve_model("missing", device="cpu")

    get_model_path.assert_called_once_with("missing")


def test_allowed_import_paths_are_immutable() -> None:
    assert isinstance(ALLOWED_MODEL_IMPORT_PATHS, frozenset)
    assert "torch.nn.GELU" in ALLOWED_MODEL_IMPORT_PATHS
    assert "torch.nn.init.xavier_normal_" in ALLOWED_MODEL_IMPORT_PATHS
    assert "torch.nn.*" not in ALLOWED_MODEL_IMPORT_PATHS
    assert all("*" not in path for path in ALLOWED_MODEL_IMPORT_PATHS)
    with pytest.raises(AttributeError):
        ALLOWED_MODEL_IMPORT_PATHS.add("os.system")  # type: ignore[attr-defined]


def test_load_state_dict_checked_fails_on_missing_parameters() -> None:
    model = torch.nn.Linear(2, 2)
    state_dict = {"weight": torch.zeros_like(model.weight)}

    with pytest.raises(RuntimeError, match=r"Missing model parameters.*bias"):
        model_utils.load_state_dict_checked(model, state_dict, source="local.pt")


def test_load_state_dict_checked_warns_on_unexpected_parameters() -> None:
    model = torch.nn.Linear(2, 2)
    state_dict = {
        "weight": torch.zeros_like(model.weight),
        "bias": torch.zeros_like(model.bias),
        "extra": torch.zeros(1),
    }

    with pytest.warns(UserWarning, match=r"Unexpected model parameters.*extra"):
        model_utils.load_state_dict_checked(model, state_dict, source="local.pt")


def test_load_state_dict_checked_can_fail_on_unexpected_parameters() -> None:
    model = torch.nn.Linear(2, 2)
    state_dict = {
        "weight": torch.zeros_like(model.weight),
        "bias": torch.zeros_like(model.bias),
        "extra": torch.zeros(1),
    }

    with pytest.raises(RuntimeError, match=r"Unexpected model parameters.*extra"):
        model_utils.load_state_dict_checked(model, state_dict, source="registry.pt", unexpected="error")


def test_load_state_dict_checked_filters_migration_keys() -> None:
    class Srcoulomb(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = torch.nn.Parameter(torch.zeros(1))

    class Outputs(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.srcoulomb = Srcoulomb()

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.outputs = Outputs()
            self.to_devices: list[str] = []

        def to(self, *args: object, **kwargs: object) -> Model:
            device = kwargs.get("device", args[0] if args else None)
            self.to_devices.append(str(device))
            return self

    model = Model()
    state_dict = {
        "outputs.lrcoulomb.embedding": torch.zeros(1),
        "outputs.dftd3.embedding": torch.zeros(1),
        "outputs.d3bj.embedding": torch.zeros(1),
        "outputs.dipole.mass": torch.zeros(1),
        "outputs.quadrupole.mass": torch.zeros(1),
    }

    with warnings.catch_warnings(record=True) as caught:
        model_utils.load_state_dict_checked(model, state_dict, source="converted.pt")

    assert caught == []


def test_local_v2_loader_fails_on_missing_parameters(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 2)
    path = tmp_path / "missing.pt"
    torch.save(
        _v2_data(
            "class: torch.nn.Linear\nkwargs:\n  in_features: 2\n  out_features: 2\n",
            state_dict={"weight": model.weight.detach().clone()},
        ),
        path,
    )

    with pytest.raises(RuntimeError, match=r"Missing model parameters.*bias"):
        model_base.load_model(str(path), model_import_paths={"torch.nn.Linear"})


def test_local_v2_loader_warns_on_unexpected_parameters(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 2)
    path = tmp_path / "unexpected.pt"
    torch.save(
        _v2_data(
            "class: torch.nn.Linear\nkwargs:\n  in_features: 2\n  out_features: 2\n",
            state_dict={
                **model.state_dict(),
                "extra": torch.zeros(1),
            },
        ),
        path,
    )

    with pytest.warns(UserWarning, match=r"Unexpected model parameters.*extra"):
        loaded, _ = model_base.load_model(str(path), model_import_paths={"torch.nn.Linear"})

    assert isinstance(loaded, torch.nn.Linear)


def test_v2_loader_constructs_on_cpu_with_meta_default_device(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 2)
    path = tmp_path / "cpu-first.pt"
    torch.save(
        _v2_data(
            "class: torch.nn.Linear\nkwargs:\n  in_features: 2\n  out_features: 2\n",
            state_dict=model.state_dict(),
        ),
        path,
    )

    with torch.device("meta"):
        loaded, _ = model_base.load_model(str(path), model_import_paths={"torch.nn.Linear"})

    assert isinstance(loaded, torch.nn.Linear)
    assert loaded.weight.device.type == "cpu"


def test_v2_loader_reads_state_on_cpu_and_moves_once(monkeypatch: pytest.MonkeyPatch) -> None:
    class SpyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.to_devices: list[str] = []

        def to(self, *args: object, **kwargs: object) -> SpyModel:
            device = kwargs.get("device", args[0] if args else None)
            self.to_devices.append(str(device))
            return self

    model = SpyModel()
    data = _v2_data("class: torch.nn.Identity", state_dict={"weight": torch.zeros(1)})
    load = Mock(return_value=data)
    monkeypatch.setattr(model_base.torch, "load", load)
    monkeypatch.setattr(model_base, "build_module", Mock(return_value=model))
    monkeypatch.setattr(
        model_base,
        "validate_v2_artifact_with_policy",
        Mock(return_value=({"class": "torch.nn.Identity"}, {"weight": torch.zeros(1)})),
    )

    model_base._load_v2_model(
        "model.pt",
        "cuda",
        model_base._REGISTRY_IMPORT_POLICY,
    )

    assert load.call_args.kwargs["map_location"] == "cpu"
    assert model.to_devices == ["cuda"]


def test_v2_loader_preserves_float64_atomic_shifts(monkeypatch: pytest.MonkeyPatch) -> None:
    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.atomic_shift = AtomicShift("energy", "shifted")
            self.to_devices: list[str] = []

        def to(self, *args: object, **kwargs: object) -> Model:
            device = kwargs.get("device", args[0] if args else None)
            self.to_devices.append(str(device))
            return self

    model = Model()
    values = torch.zeros(64, 1, dtype=torch.float64)
    values[1, 0] = 1.0000000000000002
    values[2, 0] = 2.0000000000000004
    monkeypatch.setattr(
        model_base,
        "validate_v2_artifact_with_policy",
        Mock(return_value=({"class": "aimnet.models.AIMNet2"}, {"atomic_shift.shifts.weight": values})),
    )
    monkeypatch.setattr(model_base.torch, "load", Mock(return_value=_v2_data("class: aimnet.models.AIMNet2")))
    monkeypatch.setattr(model_base, "build_module", Mock(return_value=model))

    model_base._load_v2_model("model.pt", "cuda", model_base._REGISTRY_IMPORT_POLICY)

    assert model.atomic_shift.shifts.weight.dtype is torch.float64
    assert torch.equal(model.atomic_shift.shifts.weight.detach(), values)
    assert model.to_devices == ["cuda"]


def test_runtime_authorizer_receives_nested_import_roles() -> None:
    seen: list[tuple[str, str]] = []

    def authorize(path: str, role: str) -> None:
        seen.append((path, role))

    config.build_module(
        {
            "class": "aimnet.modules.Output",
            "kwargs": {
                "n_in": 2,
                "n_out": 1,
                "key_in": "aim",
                "key_out": "energy",
                "mlp": {"hidden": [2], "activation_fn": "torch.nn.GELU"},
            },
        },
        import_authorizer=authorize,
    )

    assert ("aimnet.modules.Output", "class") in seen
    assert ("torch.nn.GELU", "activation") in seen
    assert ("torch.nn.init.xavier_normal_", "initializer") in seen


def test_runtime_authorizer_resets_after_failure() -> None:
    def reject(path: str, role: str) -> None:
        raise ValueError(f"rejected {path} as {role}")

    with pytest.raises(ValueError, match="rejected"):
        config.build_module({"class": "torch.nn.Identity"}, import_authorizer=reject)

    assert config.get_module("torch.nn.Identity") is torch.nn.Identity


def test_runtime_authorizer_rejects_before_import(monkeypatch: pytest.MonkeyPatch) -> None:
    import_attempt = Mock(side_effect=AssertionError("import must not be reached"))
    monkeypatch.setattr(config, "import_module", import_attempt)

    def reject(path: str, role: str) -> None:
        raise ValueError(f"rejected {path} as {role}")

    with pytest.raises(ValueError, match="rejected"), config._import_authorization(reject):
        config.get_module("torch.nn.Identity")

    import_attempt.assert_not_called()


def test_successful_build_cleans_authorizer_context() -> None:
    seen: list[str] = []

    def authorize(path: str, role: str) -> None:
        seen.append(path)

    config.build_module({"class": "torch.nn.Identity"}, import_authorizer=authorize)

    assert seen == ["torch.nn.Identity"]
    assert config.get_module("torch.nn.Identity") is torch.nn.Identity


def test_nested_builds_restore_the_outer_policy() -> None:
    seen: list[str] = []

    def outer(path: str, role: str) -> None:
        seen.append("outer")

    def inner(path: str, role: str) -> None:
        seen.append("inner")

    with config._import_authorization(outer):
        config.build_module({"class": "torch.nn.Identity"})
        config.build_module({"class": "torch.nn.Identity"}, import_authorizer=inner)
        config.build_module({"class": "torch.nn.Identity"})

    assert seen == ["outer", "inner", "outer"]


def test_runtime_authorizers_are_isolated_between_threads() -> None:
    barrier = threading.Barrier(2)
    seen: list[str] = []
    errors: list[BaseException] = []

    def run(name: str) -> None:
        def authorize(path: str, role: str) -> None:
            barrier.wait()
            seen.append(name)

        try:
            config.build_module({"class": "torch.nn.Identity"}, import_authorizer=authorize)
        except BaseException as exc:  # pragma: no cover - assertion reports unexpected thread failures
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(name,)) for name in ("left", "right")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert sorted(seen) == ["left", "right"]


def test_runtime_authorizers_are_isolated_before_concurrent_lookup() -> None:
    barrier = threading.Barrier(2)
    seen: list[str] = []
    errors: list[BaseException] = []

    def run(name: str) -> None:
        def authorize(path: str, role: str) -> None:
            seen.append(name)

        try:
            with config._import_authorization(authorize):
                barrier.wait()
                config.get_module("torch.nn.Identity")
        except BaseException as exc:  # pragma: no cover - assertion reports unexpected thread failures
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(name,)) for name in ("left", "right")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert sorted(seen) == ["left", "right"]


def test_runtime_authorizers_are_isolated_between_async_tasks() -> None:
    seen: list[str] = []

    async def run(name: str) -> None:
        def authorize(path: str, role: str) -> None:
            seen.append(name)

        with config._import_authorization(authorize):
            await asyncio.sleep(0)
            config.get_module("torch.nn.Identity")

    async def main() -> None:
        await asyncio.gather(run("left"), run("right"))

    asyncio.run(main())

    assert sorted(seen) == ["left", "right"]


def test_runtime_authorizer_covers_future_symbol_lookups(monkeypatch: pytest.MonkeyPatch) -> None:
    original_get_init_module = config.get_init_module

    def future_constructor(
        name: str,
        args: list | None = None,
        kwargs: dict | None = None,
        *,
        role: str = "class",
    ) -> object:
        config.get_module("torch.nn.Linear", role="activation")
        return original_get_init_module(name, args=args, kwargs=kwargs, role=role)

    monkeypatch.setattr(config, "get_init_module", future_constructor)

    def reject(path: str, role: str) -> None:
        if path == "torch.nn.Linear":
            raise ValueError("future symbol rejected")

    with pytest.raises(ValueError, match="future symbol rejected"):
        config.build_module({"class": "torch.nn.Identity"}, import_authorizer=reject)


def test_hf_local_resolution_does_not_require_hub_extra(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(hf_hub, "_snapshot_download", None)

    assert hf_hub._resolve_repo(str(tmp_path), 0, None, None) == tmp_path


def test_hf_remote_resolution_requires_hub_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hf_hub, "_snapshot_download", None)

    with pytest.raises(ImportError, match=r"aimnet\[hf\]"):
        hf_hub._resolve_repo("org/repository", 0, None, None)


def test_hf_weight_loading_requires_safetensors_extra(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "ensemble_0.safetensors").write_bytes(b"placeholder")
    (tmp_path / "config.json").write_text(
        json.dumps({"model_yaml": "class: torch.nn.Identity", "cutoff": 5.0, "format_version": 2})
    )
    monkeypatch.setattr(hf_hub, "_load_safetensors_file", None)

    with pytest.raises(ImportError, match=r"aimnet\[hf\]"):
        hf_hub.load_from_hf_repo(
            str(tmp_path),
            model_import_paths={"torch.nn.Identity"},
        )


def test_direct_artifact_runtime_lookup_uses_authorizer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "runtime-boundary.pt"
    torch.save(_v2_data("class: aimnet.models.AIMNet2"), path)
    original_get_init_module = config.get_init_module

    def future_constructor(
        name: str,
        args: list | None = None,
        kwargs: dict | None = None,
        *,
        role: str = "class",
    ) -> object:
        config.get_module("torch.nn.Linear", role="activation")
        return original_get_init_module(name, args=args, kwargs=kwargs, role=role)

    monkeypatch.setattr(config, "get_init_module", future_constructor)

    with pytest.raises(ValueError, match="Untrusted import path"):
        model_base.load_model(str(path))


def test_hf_artifact_runtime_lookup_uses_authorizer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "ensemble_0.safetensors").write_bytes(b"placeholder")
    (tmp_path / "config.json").write_text(
        json.dumps({"model_yaml": "class: aimnet.models.AIMNet2", "cutoff": 5.0, "format_version": 2})
    )
    monkeypatch.setattr(hf_hub, "_load_safetensors_file", Mock(return_value={}))
    original_get_init_module = config.get_init_module

    def future_constructor(
        name: str,
        args: list | None = None,
        kwargs: dict | None = None,
        *,
        role: str = "class",
    ) -> object:
        config.get_module("torch.nn.Linear", role="activation")
        return original_get_init_module(name, args=args, kwargs=kwargs, role=role)

    monkeypatch.setattr(config, "get_init_module", future_constructor)

    with pytest.raises(ValueError, match="Untrusted import path"):
        hf_hub.load_from_hf_repo(str(tmp_path))


def test_hf_loader_reads_weights_on_cpu_and_moves_once_without_extra(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class SpyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.to_devices: list[str] = []

        def to(self, *args: object, **kwargs: object) -> SpyModel:
            device = kwargs.get("device", args[0] if args else None)
            self.to_devices.append(str(device))
            return self

    (tmp_path / "ensemble_0.safetensors").write_bytes(b"placeholder")
    (tmp_path / "config.json").write_text(
        json.dumps({"model_yaml": "class: aimnet.models.AIMNet2", "cutoff": 5.0, "format_version": 2})
    )
    load_file = Mock(return_value={})
    model = SpyModel()
    monkeypatch.setattr(hf_hub, "_load_safetensors_file", load_file)
    monkeypatch.setattr(model_base, "build_module", Mock(return_value=model))

    loaded, _ = hf_hub.load_from_hf_repo(str(tmp_path), device="cuda")

    assert loaded is model
    load_file.assert_called_once_with(str(tmp_path / "ensemble_0.safetensors"), device="cpu")
    assert model.to_devices == ["cuda"]


def test_hf_loader_preserves_float64_atomic_shifts_without_extra(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.outputs = torch.nn.Module()
            self.outputs.atomic_shift = AtomicShift("energy", "shifted")
            self.to_devices: list[str] = []

        def to(self, *args: object, **kwargs: object) -> Model:
            device = kwargs.get("device", args[0] if args else None)
            self.to_devices.append(str(device))
            return self

    values = torch.zeros(64, 1, dtype=torch.float64)
    values[1, 0] = 1.0000000000000002
    values[2, 0] = 2.0000000000000004
    (tmp_path / "ensemble_0.safetensors").write_bytes(b"placeholder")
    (tmp_path / "config.json").write_text(
        json.dumps({"model_yaml": "class: aimnet.models.AIMNet2", "cutoff": 5.0, "format_version": 2})
    )
    model = Model()
    monkeypatch.setattr(
        hf_hub,
        "_load_safetensors_file",
        Mock(
            return_value={
                "outputs.atomic_shift.shifts.weight": values,
            }
        ),
    )
    monkeypatch.setattr(model_base, "build_module", Mock(return_value=model))

    hf_hub.load_from_hf_repo(str(tmp_path), device="cuda")

    assert model.outputs.atomic_shift.shifts.weight.dtype is torch.float64
    assert torch.equal(model.outputs.atomic_shift.shifts.weight.detach(), values)
    assert model.to_devices == ["cuda"]


def test_hf_loader_fails_on_missing_key_without_extra(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "ensemble_0.safetensors").write_bytes(b"placeholder")
    (tmp_path / "config.json").write_text(
        json.dumps({
            "model_yaml": "class: torch.nn.Linear\nkwargs:\n  in_features: 2\n  out_features: 2\n",
            "cutoff": 5.0,
            "format_version": 2,
        })
    )
    model = torch.nn.Linear(2, 2)
    monkeypatch.setattr(hf_hub, "_load_safetensors_file", Mock(return_value={"weight": model.weight.detach().clone()}))

    with pytest.raises(RuntimeError, match=r"Missing model parameters.*bias"):
        hf_hub.load_from_hf_repo(str(tmp_path), model_import_paths={"torch.nn.Linear"})


def test_hf_loader_warns_on_complete_custom_unexpected_key_without_extra(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "ensemble_0.safetensors").write_bytes(b"placeholder")
    (tmp_path / "config.json").write_text(
        json.dumps({
            "model_yaml": "class: torch.nn.Linear\nkwargs:\n  in_features: 2\n  out_features: 2\n",
            "cutoff": 5.0,
            "format_version": 2,
        })
    )
    model = torch.nn.Linear(2, 2)
    monkeypatch.setattr(
        hf_hub,
        "_load_safetensors_file",
        Mock(return_value={**model.state_dict(), "extra": torch.zeros(1)}),
    )

    with pytest.warns(UserWarning, match=r"Unexpected model parameters.*extra"):
        hf_hub.load_from_hf_repo(str(tmp_path), model_import_paths={"torch.nn.Linear"})


def test_hf_registry_fallback_fails_on_unexpected_key_without_extra(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "ensemble_0.safetensors").write_bytes(b"placeholder")
    (tmp_path / "config.json").write_text(json.dumps({"member_names": ["aimnet2"]}))
    monkeypatch.setattr(
        hf_hub,
        "_fetch_pt_metadata_from_registry",
        Mock(
            return_value=(
                {"model_yaml": "class: aimnet.models.AIMNet2", "cutoff": 5.0, "format_version": 2},
                {"class": "aimnet.models.AIMNet2"},
            )
        ),
    )
    monkeypatch.setattr(hf_hub, "_load_safetensors_file", Mock(return_value={"extra": torch.zeros(1)}))
    monkeypatch.setattr(model_base, "build_module", Mock(return_value=torch.nn.Identity()))

    with pytest.raises(RuntimeError, match=r"Unexpected model parameters.*extra"):
        hf_hub.load_from_hf_repo(str(tmp_path))


@pytest.mark.parametrize(
    "path",
    [
        "aimnet.modules.D3TS",
        "aimnet.modules.lr.D3TS",
        "aimnet.modules.lr.DispParam",
    ],
)
def test_embedded_dispersion_modules_are_trusted(path: str) -> None:
    """First-party solvation artifacts reference these and cannot load without them.

    All are first-party ``nn.Module`` subclasses (or spellings of one) in
    ``aimnet/modules/lr.py``. When the artifact trust boundary was introduced
    they were absent from the default allowlist, so the solvation model
    failed to load with ``Untrusted import path for 'class'``. Downstream that
    surfaced only as a run aborting hours in, since nothing loads the
    solvation model until it is needed. ``aimnet.modules.lr.D3TS`` is the
    fully qualified spelling of the same class as ``aimnet.modules.D3TS``
    (the barrel re-export); the loader machinery that detects D3TS matches
    the "D3TS" substring regardless of spelling, so the exact-match allowlist
    must trust both.
    """
    assert path in ALLOWED_MODEL_IMPORT_PATHS


def test_trusted_dispersion_paths_resolve_to_real_modules() -> None:
    """An allowlist entry that does not resolve would trade a load failure for
    an import error, so membership alone is not enough to assert."""
    import importlib

    from torch import nn

    for path in ("aimnet.modules.D3TS", "aimnet.modules.lr.D3TS", "aimnet.modules.lr.DispParam"):
        module_name, class_name = path.rsplit(".", 1)
        obj = getattr(importlib.import_module(module_name), class_name)
        assert isinstance(obj, type) and issubclass(obj, nn.Module)


def test_d3ts_barrel_and_submodule_spellings_are_the_same_class() -> None:
    """Both allowlisted D3TS spellings must resolve to the identical object,
    not merely to two classes that happen to share a name."""
    from aimnet.modules import D3TS as barrel_d3ts
    from aimnet.modules.lr import D3TS as submodule_d3ts

    assert barrel_d3ts is submodule_d3ts


# --- Finding 1: ptfile forbidden in artifact model_yaml ---------------------


def test_ptfile_kwarg_is_forbidden_in_artifact_yaml() -> None:
    """DispParam.__init__ runs torch.load(ptfile, weights_only=True) on a
    YAML-supplied path; the walker must reject it wherever it appears."""
    with pytest.raises(ValueError, match="ptfile"):
        validate_model_yaml(
            "class: aimnet.modules.lr.DispParam\nkwargs:\n  ptfile: /etc/passwd\n",
        )


def test_ptfile_kwarg_is_forbidden_anywhere_in_the_tree() -> None:
    """The forbidden key is rejected regardless of which class it is nested under."""
    with pytest.raises(ValueError, match="ptfile"):
        validate_model_yaml(
            """
class: aimnet.models.AIMNet2
kwargs:
  nested:
    - ptfile: /etc/passwd
"""
        )


def test_dispparam_without_ptfile_passes_import_policy() -> None:
    validate_model_yaml(
        "class: aimnet.modules.lr.DispParam\nkwargs:\n  key_in: disp_param\n  key_out: disp_param\n",
    )


def test_ptfile_remains_forbidden_under_unsafe_import_mode() -> None:
    """Unsafe mode only skips path matching; forbidden-key checks stay active."""
    with pytest.raises(ValueError, match="ptfile"):
        validate_model_yaml(
            "class: aimnet.modules.lr.DispParam\nkwargs:\n  ptfile: /etc/passwd\n",
            model_import_mode="unsafe",
        )


# --- Positional `args` bypass of the constructor-kwarg guards ----------------
#
# `build_module` forwards `args` into `func(*args, **kwargs)`, but the guards
# above match keyword names only. `ptfile` is DispParam's third positional
# parameter and D3TS takes its damping parameters positionally, so without a
# blanket rejection of `args` both guards are evadable by respelling.


def test_positional_args_are_forbidden_in_artifact_yaml() -> None:
    with pytest.raises(ValueError, match="args"):
        validate_model_yaml(
            "class: aimnet.modules.lr.DispParam\nargs: [null, null, /etc/passwd]\n",
        )


def test_positional_args_are_forbidden_anywhere_in_the_tree() -> None:
    """Rejected regardless of nesting depth, matching the ptfile guard."""
    with pytest.raises(ValueError, match="args"):
        validate_model_yaml(
            """
class: aimnet.models.AIMNet2
kwargs:
  outputs:
    disp_param:
      class: aimnet.modules.lr.DispParam
      args: [null, null, /etc/passwd]
"""
        )


def test_positional_args_cannot_smuggle_d3ts_damping_parameters() -> None:
    """`a1=-1.0, a2=nan, s8=inf` positionally must fail like the kwarg spelling."""
    with pytest.raises(ValueError, match="args"):
        validate_model_yaml("class: aimnet.modules.D3TS\nargs: [-1.0, .nan, .inf]\n")


def test_positional_args_remain_forbidden_under_unsafe_import_mode() -> None:
    with pytest.raises(ValueError, match="args"):
        validate_model_yaml(
            "class: aimnet.modules.lr.DispParam\nargs: [null, null, /etc/passwd]\n",
            model_import_mode="unsafe",
        )


def test_kwargs_key_is_not_confused_with_args() -> None:
    """The guard must match the `args` key exactly, not the `kwargs` substring."""
    validate_model_yaml(
        "class: aimnet.modules.lr.DispParam\nkwargs:\n  key_in: disp_param\n  key_out: disp_param\n",
    )


# --- Finding 2: YAML<->metadata D3TS consistency -----------------------------

_D3TS_MODEL_YAML = """
class: aimnet.models.AIMNet2
kwargs:
  outputs:
    disp_param:
      class: aimnet.modules.lr.DispParam
      kwargs:
        key_in: disp_param
        key_out: disp_param
    d3ts:
      class: aimnet.modules.D3TS
      kwargs:
        a1: 0.55
        a2: 3.1
        s8: 1.5
        s6: 1.0
"""


def test_rejects_d3ts_in_yaml_not_declared_in_metadata() -> None:
    """D3TS embedded in model_yaml but has_embedded_d3ts=False must be rejected.

    Left unchecked, needs_dispersion could be set True alongside this and the
    calculator would silently double-count dispersion (embedded D3TS plus an
    external correction)."""
    with pytest.raises(ValueError, match="has_embedded_d3ts"):
        validate_v2_artifact(
            _v2_data(
                _D3TS_MODEL_YAML,
                has_embedded_lr=True,
                has_embedded_d3ts=False,
            )
        )


def test_rejects_declared_d3ts_absent_from_yaml() -> None:
    """has_embedded_d3ts=True with no D3TS in model_yaml must be rejected.

    Left unchecked, the calculator would trust an embedded-dispersion flag
    that no module actually backs, silently losing dispersion entirely."""
    with pytest.raises(ValueError, match="has_embedded_d3ts"):
        validate_v2_artifact(
            _v2_data(
                "class: aimnet.models.AIMNet2",
                has_embedded_lr=True,
                has_embedded_d3ts=True,
            )
        )


def test_truthful_embedded_d3ts_config_passes() -> None:
    model_config, _ = validate_v2_artifact(
        _v2_data(
            _D3TS_MODEL_YAML,
            needs_coulomb=False,
            needs_dispersion=False,
            coulomb_mode="none",
            has_embedded_lr=True,
            has_embedded_d3ts=True,
        )
    )
    assert model_config["kwargs"]["outputs"]["d3ts"]["class"] == "aimnet.modules.D3TS"


def test_registry_policy_admits_embedded_d3ts_and_disp_param_artifact_shape() -> None:
    """Finding 4 completeness fixture: the allowlist must admit the artifact
    shape this whole change exists for -- a DispParam module feeding a D3TS
    module -- validated end-to-end under REGISTRY_IMPORT_POLICY, the fixed
    default policy the registry loader uses."""
    model_config, state_dict = validate_registry_v2_artifact(
        _v2_data(
            _D3TS_MODEL_YAML,
            needs_coulomb=False,
            needs_dispersion=False,
            coulomb_mode="none",
            has_embedded_lr=True,
            has_embedded_d3ts=True,
        )
    )
    assert state_dict == {}
    outputs = model_config["kwargs"]["outputs"]
    assert outputs["disp_param"]["class"] == "aimnet.modules.lr.DispParam"
    assert outputs["d3ts"]["class"] == "aimnet.modules.D3TS"


# --- Finding 3: D3TS damping numerics ----------------------------------------


@pytest.mark.parametrize("d3ts_class", ["aimnet.modules.D3TS", "aimnet.modules.lr.D3TS"])
def test_d3ts_rejects_non_finite_damping_kwarg(d3ts_class: str) -> None:
    with pytest.raises(ValueError, match="s8"):
        validate_model_yaml(f"class: {d3ts_class}\nkwargs:\n  a1: 0.5\n  a2: 3.0\n  s8: .nan\n")


def test_d3ts_rejects_infinite_damping_kwarg() -> None:
    with pytest.raises(ValueError, match="a2"):
        validate_model_yaml("class: aimnet.modules.D3TS\nkwargs:\n  a1: 0.5\n  a2: .inf\n  s8: 1.0\n")


def test_d3ts_rejects_negative_damping_kwarg() -> None:
    with pytest.raises(ValueError, match="a1"):
        validate_model_yaml("class: aimnet.modules.D3TS\nkwargs:\n  a1: -1.0\n  a2: 3.0\n  s8: 1.0\n")


def test_d3ts_accepts_normal_damping_kwargs() -> None:
    validate_model_yaml("class: aimnet.modules.D3TS\nkwargs:\n  a1: 0.55\n  a2: 3.1\n  s8: 1.5\n  s6: 1.0\n")


def test_d3ts_accepts_absent_damping_kwargs() -> None:
    """Absent kwargs fall back to the class's own defaults and are not our concern."""
    validate_model_yaml("class: aimnet.modules.D3TS\nkwargs:\n  key_in: disp_param\n  key_out: energy\n")


def test_d3ts_damping_validation_applies_to_the_submodule_spelling_too() -> None:
    with pytest.raises(ValueError, match="s6"):
        validate_model_yaml("class: aimnet.modules.lr.D3TS\nkwargs:\n  s6: -1.0\n")
