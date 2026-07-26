"""Security and dispatch tests for serialized model artifacts."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from aimnet.models.artifact_validation import (
    ALLOWED_MODEL_IMPORT_PATHS,
    validate_model_yaml,
    validate_v2_artifact,
)


def _v2_data(model_yaml: str, **overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "model_yaml": model_yaml,
        "state_dict": {},
        "cutoff": 5.0,
        "format_version": 2,
    }
    data.update(overrides)
    return data


def test_default_extend_allows_official_and_torch_nn_paths() -> None:
    config = validate_model_yaml(
        """
class: torch.nn.Linear
kwargs:
  in_features: 2
  out_features: 2
activation_fn: torch.nn.ReLU
""",
    )
    assert config["class"] == "torch.nn.Linear"


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
        validate_model_yaml("class: aimnet.models.AIMNet2", model_import_mode="replace", model_import_paths={"tests.custom_models.*"})
    validate_model_yaml("class: torch.nn.Linear", model_import_mode="replace", model_import_paths={"torch.nn.*"})
    validate_model_yaml("class: aimnet.models.AIMNet2", model_import_mode="extend", model_import_paths=())


@pytest.mark.parametrize("mode", ["invalid", [], {}])
def test_invalid_mode_combinations(mode: object) -> None:
    with pytest.raises(ValueError, match="mode"):
        validate_model_yaml("class: aimnet.models.AIMNet2", model_import_mode=mode)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty"):
        validate_model_yaml("class: aimnet.models.AIMNet2", model_import_mode="replace")
    with pytest.raises(ValueError, match="unsafe"):
        validate_model_yaml("class: aimnet.models.AIMNet2", model_import_mode="unsafe", model_import_paths={"pkg.Class"})


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


def test_validate_v2_artifact_requires_complete_d3_params() -> None:
    with pytest.raises(ValueError, match="d3_params"):
        validate_v2_artifact(
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


def test_validate_v2_artifact_rejects_sr_coulomb_without_external_coulomb() -> None:
    with pytest.raises(ValueError, match="external Coulomb"):
        validate_v2_artifact(
            _v2_data(
                "class: aimnet.models.AIMNet2",
                needs_coulomb=False,
                coulomb_mode="sr_embedded",
                coulomb_sr_rc=4.6,
                coulomb_sr_envelope="exp",
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


def test_pt_torchscript_archive_does_not_route_to_legacy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
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
        str(path), model_import_mode="unsafe",
    )

    assert isinstance(model, torch.nn.Identity)


def test_local_pt_custom_import_paths_reach_construction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
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


def test_obsolete_policy_objects_are_not_exported() -> None:
    import aimnet.models

    assert not hasattr(aimnet.models, "Model" "ImportPolicy")
    assert not hasattr(aimnet.models, "custom_model_" "import_policy")


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
def test_registry_names_and_aliases_use_strict_loader(
    monkeypatch: pytest.MonkeyPatch, model_name: str
) -> None:
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
    assert "torch.nn.*" in ALLOWED_MODEL_IMPORT_PATHS
    assert all(path == "torch.nn.*" or "*" not in path for path in ALLOWED_MODEL_IMPORT_PATHS)
    with pytest.raises(AttributeError):
        ALLOWED_MODEL_IMPORT_PATHS.add("os.system")  # type: ignore[attr-defined]
