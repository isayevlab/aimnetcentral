"""Validation for serialized inference-model artifacts.

``ALLOWED_MODEL_IMPORT_PATHS`` is the default set of Python imports trusted in
model YAML. Registry loading uses this set unchanged; direct v2 artifacts may
extend, replace, or bypass path matching through the public import options.
"""

from __future__ import annotations

import keyword
import math
import os
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any, Literal

import yaml
from torch import Tensor

from aimnet.config import ImportRole

_DEFAULT_CLASS_IMPORT_PATHS = frozenset({
    "aimnet.models.AIMNet2",
    "aimnet.models.aimnet2.AIMNet2",
    "aimnet.modules.AtomicShift",
    "aimnet.modules.AtomicSum",
    "aimnet.modules.Dipole",
    "aimnet.modules.Output",
    "aimnet.modules.Quadrupole",
    "aimnet.modules.SRCoulomb",
    # Embedded-dispersion modules. Both are first-party nn.Modules in
    # aimnet/modules/lr.py and both are referenced by the shipped CPCM(water)
    # solvation artifact, which could not load without them.
    "aimnet.modules.D3TS",
    "aimnet.modules.lr.DispParam",
})
_DEFAULT_ACTIVATION_IMPORT_PATHS = frozenset({"torch.nn.GELU"})
_DEFAULT_INITIALIZER_IMPORT_PATHS = frozenset({"torch.nn.init.xavier_normal_"})
ALLOWED_MODEL_IMPORT_PATHS = frozenset({
    *_DEFAULT_CLASS_IMPORT_PATHS,
    *_DEFAULT_ACTIVATION_IMPORT_PATHS,
    *_DEFAULT_INITIALIZER_IMPORT_PATHS,
})

_MODEL_IMPORT_KEYS: dict[str, ImportRole] = {
    "class": "class",
    "activation_fn": "activation",
    "weight_init_fn": "initializer",
}
_ALWAYS_FORBIDDEN_IMPORT_KEYS = frozenset({"fn", "trainer", "evaluator"})
_RECOGNIZED_IMPORT_KEYS = frozenset(_MODEL_IMPORT_KEYS) | _ALWAYS_FORBIDDEN_IMPORT_KEYS


@dataclass(frozen=True)
class ModelImportPolicy:
    class_paths: frozenset[str]
    activation_paths: frozenset[str]
    initializer_paths: frozenset[str]
    unsafe: bool = False

    def require_allowed(self, path: str, role: ImportRole) -> None:
        """Reject a symbol that is not authorized for its construction role."""
        if self.unsafe:
            return
        allowed_paths = {
            "class": self.class_paths,
            "activation": self.activation_paths,
            "initializer": self.initializer_paths,
        }[role]
        if not any(_matches_import_pattern(path, pattern) for pattern in allowed_paths):
            raise ValueError(f"Untrusted import path for {role!r}: {path!r}.")


REGISTRY_IMPORT_POLICY = ModelImportPolicy(
    class_paths=_DEFAULT_CLASS_IMPORT_PATHS,
    activation_paths=_DEFAULT_ACTIVATION_IMPORT_PATHS,
    initializer_paths=_DEFAULT_INITIALIZER_IMPORT_PATHS,
)
_REGISTRY_IMPORT_POLICY = REGISTRY_IMPORT_POLICY


def is_legacy_jit_path(path: str) -> bool:
    """Return whether ``path`` selects the trusted legacy TorchScript loader."""
    return str(path).lower().endswith(".jpt")


def is_explicit_local_path(path: str) -> bool:
    """Return whether ``path`` unambiguously denotes a local filesystem path."""
    path = str(path)
    return os.path.isabs(path) or path.startswith("./") or path.startswith("../")


def uses_default_model_import_settings(
    paths: Collection[str] | None,
    mode: Literal["extend", "replace", "unsafe"],
) -> bool:
    """Return whether model imports use the unmodified default policy."""
    return paths is None and mode == "extend"


def _validate_import_pattern(path: object) -> str:
    if not isinstance(path, str):
        raise ValueError("Model import paths must be a collection of strings.")  # noqa: TRY004
    if not path or path != path.strip():
        raise ValueError(f"Invalid model import path: {path!r}.")

    is_namespace = path.endswith(".*")
    fixed_path = path[:-2] if is_namespace else path
    if "*" in fixed_path or "?" in path or "[" in path or "]" in path:
        raise ValueError(f"Invalid model import path: {path!r}.")
    segments = fixed_path.split(".")
    minimum_segments = 1 if is_namespace else 2
    if len(segments) < minimum_segments or any(not segment for segment in segments):
        raise ValueError(f"Invalid model import path: {path!r}.")
    if any(not segment.isidentifier() or keyword.iskeyword(segment) for segment in segments):
        raise ValueError(f"Invalid model import path: {path!r}.")
    if path.startswith("torch.") and path != "torch.nn.*" and not fixed_path.startswith("torch.nn."):
        raise ValueError(f"Invalid model import path: {path!r}.")
    return path


def _normalize_model_import_paths(paths: Collection[str]) -> frozenset[str]:
    if isinstance(paths, (str, bytes, Mapping)) or not isinstance(paths, Collection):
        raise ValueError("model_import_paths must be a collection of strings.")  # noqa: TRY004
    return frozenset(_validate_import_pattern(path) for path in paths)


def _matches_import_pattern(path: str, pattern: str) -> bool:
    if pattern.endswith(".*"):
        return path.startswith(pattern[:-1]) and path != pattern[:-2]
    return path == pattern


def resolve_model_import_policy(
    model_import_paths: Collection[str] | None,
    model_import_mode: Literal["extend", "replace", "unsafe"],
) -> ModelImportPolicy:
    if not isinstance(model_import_mode, str) or model_import_mode not in {"extend", "replace", "unsafe"}:
        raise ValueError(f"Invalid model_import_mode: {model_import_mode!r}.")
    if model_import_mode == "unsafe":
        if model_import_paths is not None:
            raise ValueError("model_import_paths cannot be used with unsafe model_import_mode.")
        return ModelImportPolicy(
            class_paths=frozenset(),
            activation_paths=frozenset(),
            initializer_paths=frozenset(),
            unsafe=True,
        )
    if model_import_mode == "replace":
        if model_import_paths is None:
            raise ValueError("replace model_import_mode requires a non-empty model_import_paths collection.")
        paths = _normalize_model_import_paths(model_import_paths)
        if not paths:
            raise ValueError("replace model_import_mode requires a non-empty model_import_paths collection.")
        return ModelImportPolicy(
            class_paths=paths,
            activation_paths=paths,
            initializer_paths=paths,
        )
    additions = frozenset() if model_import_paths is None else _normalize_model_import_paths(model_import_paths)
    return ModelImportPolicy(
        class_paths=_DEFAULT_CLASS_IMPORT_PATHS | additions,
        activation_paths=_DEFAULT_ACTIVATION_IMPORT_PATHS | additions,
        initializer_paths=_DEFAULT_INITIALIZER_IMPORT_PATHS | additions,
    )


def _walk_model_yaml(model_yaml: str, policy: ModelImportPolicy) -> dict[str, Any]:
    if not isinstance(model_yaml, str) or not model_yaml.strip():
        raise ValueError("model_yaml must be a nonempty string.")
    try:
        config = yaml.safe_load(model_yaml)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid model_yaml: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError("model_yaml root must be a mapping.")  # noqa: TRY004

    active: set[int] = set()
    visited: set[int] = set()

    def walk(value: object) -> None:
        if not isinstance(value, (dict, list)):
            return
        value_id = id(value)
        if value_id in active:
            raise ValueError("model_yaml contains a recursive alias cycle.")
        if value_id in visited:
            return
        active.add(value_id)
        visited.add(value_id)
        if isinstance(value, dict):
            for key, child in value.items():
                if key in _RECOGNIZED_IMPORT_KEYS:
                    if key in _ALWAYS_FORBIDDEN_IMPORT_KEYS:
                        raise ValueError(f"Import key {key!r} is forbidden in model artifacts.")
                    if not isinstance(child, str):
                        raise ValueError(f"Import key {key!r} must contain a string path.")
                    policy.require_allowed(child, _MODEL_IMPORT_KEYS[key])
                walk(child)
        else:
            for child in value:
                walk(child)
        active.remove(value_id)

    walk(config)
    return config


def validate_model_yaml(
    model_yaml: str,
    *,
    model_import_paths: Collection[str] | None = None,
    model_import_mode: Literal["extend", "replace", "unsafe"] = "extend",
) -> dict[str, Any]:
    """Parse model YAML safely and validate its runtime import paths.

    ``model_import_paths`` contains exact dotted paths or namespaces ending in
    ``.*``. ``extend`` adds them to the default trusted paths. ``replace``
    requires a nonempty collection and trusts only those paths. ``unsafe``
    cannot be combined with paths and skips only path matching; safe YAML,
    cycle, type, and forbidden-key checks remain active. Use ``unsafe`` only
    for trusted artifacts because downstream construction may execute imported
    code.
    """
    return _walk_model_yaml(model_yaml, resolve_model_import_policy(model_import_paths, model_import_mode))


def _validate_registry_model_yaml(model_yaml: str) -> dict[str, Any]:
    return _walk_model_yaml(model_yaml, REGISTRY_IMPORT_POLICY)


def validate_v2_artifact_with_policy(
    data: object,
    policy: ModelImportPolicy,
    *,
    validation: Literal["structural", "canonical"] = "structural",
) -> tuple[dict[str, Any], Mapping[str, Tensor]]:
    if not isinstance(data, dict):
        raise ValueError(f"v2 artifact must be a dictionary, got {type(data).__name__}.")  # noqa: TRY004

    model_yaml = data.get("model_yaml")
    if not isinstance(model_yaml, str) or not model_yaml.strip():
        raise ValueError("v2 artifact field 'model_yaml' must be a nonempty string.")
    try:
        model_config = _walk_model_yaml(model_yaml, policy)
    except ValueError as exc:
        raise ValueError(f"Invalid v2 artifact field 'model_yaml': {exc}") from exc
    if validation not in {"structural", "canonical"}:
        raise ValueError(f"Unsupported v2 artifact validation mode: {validation!r}.")
    validate_model_metadata(
        data,
        require_cutoff=True,
        require_structural_consistency=True,
        require_cross_field_consistency=validation == "canonical",
    )

    state_dict = data.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("v2 artifact field 'state_dict' must be a mapping.")  # noqa: TRY004
    for key, value in state_dict.items():
        if not isinstance(key, str):
            raise ValueError("v2 artifact state_dict keys must be strings.")  # noqa: TRY004
        if not isinstance(value, Tensor):
            raise ValueError(f"v2 artifact state_dict value for {key!r} must be a tensor.")  # noqa: TRY004

    format_version = data.get("format_version", 2)
    if type(format_version) is not int or format_version != 2:
        raise ValueError("v2 artifact field 'format_version' must be integer 2.")

    return model_config, state_dict


def validate_v2_artifact(
    data: object,
    *,
    model_import_paths: Collection[str] | None = None,
    model_import_mode: Literal["extend", "replace", "unsafe"] = "extend",
) -> tuple[dict[str, Any], Mapping[str, Tensor]]:
    """Validate a v2 artifact and return its parsed config and state dict.

    Import-path handling matches :func:`validate_model_yaml`. This function
    also validates the v2 envelope, metadata, and tensor-only state dict.
    """
    policy = resolve_model_import_policy(model_import_paths, model_import_mode)
    return validate_v2_artifact_with_policy(data, policy)


def validate_registry_v2_artifact(data: object) -> tuple[dict[str, Any], Mapping[str, Tensor]]:
    return validate_v2_artifact_with_policy(data, REGISTRY_IMPORT_POLICY, validation="canonical")


_validate_registry_v2_artifact = validate_registry_v2_artifact


def validate_model_metadata(
    metadata: Mapping[str, Any],
    *,
    require_cutoff: bool = False,
    require_structural_consistency: bool = False,
    require_cross_field_consistency: bool = False,
) -> None:
    """Validate scalar metadata consumed by the calculator.

    Parameters
    ----------
    metadata
        Metadata mapping to validate.
    require_cutoff
        Require a finite, positive ``cutoff`` field.
    require_cross_field_consistency
        Enforce relationships between Coulomb, dispersion, embedded-module,
        and external-module flags.
    """
    if require_cutoff and "cutoff" not in metadata:
        raise ValueError("model metadata requires a 'cutoff' field.")
    if "cutoff" in metadata:
        cutoff = metadata["cutoff"]
        if isinstance(cutoff, bool) or not isinstance(cutoff, Real) or not math.isfinite(float(cutoff)) or cutoff <= 0:
            raise ValueError("model metadata field 'cutoff' must be a finite positive real number.")
    if "format_version" in metadata and (
        type(metadata["format_version"]) is not int or metadata["format_version"] not in {1, 2}
    ):
        raise ValueError("model metadata field 'format_version' must be integer 1 or 2.")

    for key in ("needs_coulomb", "needs_dispersion", "has_embedded_lr", "has_embedded_d3ts"):
        if key in metadata and type(metadata[key]) is not bool:
            raise ValueError(f"model metadata field {key!r} must be a bool.")
    if (
        "supports_charged_systems" in metadata
        and metadata["supports_charged_systems"] is not None
        and type(metadata["supports_charged_systems"]) is not bool
    ):
        raise ValueError("model metadata field 'supports_charged_systems' must be a bool or null.")

    if "coulomb_mode" in metadata and metadata["coulomb_mode"] not in {"none", "sr_embedded", "full_embedded"}:
        raise ValueError("model metadata field 'coulomb_mode' has an unsupported value.")
    if "coulomb_sr_rc" in metadata and metadata["coulomb_sr_rc"] is not None:
        rc = metadata["coulomb_sr_rc"]
        if isinstance(rc, bool) or not isinstance(rc, Real) or not math.isfinite(float(rc)) or rc <= 0:
            raise ValueError("model metadata field 'coulomb_sr_rc' must be a finite positive real number.")
    if (
        "coulomb_sr_envelope" in metadata
        and metadata["coulomb_sr_envelope"] is not None
        and metadata["coulomb_sr_envelope"] not in {"exp", "cosine"}
    ):
        raise ValueError("model metadata field 'coulomb_sr_envelope' has an unsupported value.")

    d3_params = metadata.get("d3_params")
    if "d3_params" in metadata and metadata["d3_params"] is not None:
        if not isinstance(d3_params, Mapping):
            raise ValueError("model metadata field 'd3_params' must be a mapping or null.")
        for key in ("s6", "s8", "a1", "a2"):
            if key in d3_params:
                value = d3_params[key]
                if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
                    raise ValueError(f"d3_params[{key!r}] must be a finite real number.")

    if "implemented_species" in metadata:
        species = metadata["implemented_species"]
        if not isinstance(species, list) or any(type(value) is not int or value <= 0 for value in species):
            raise ValueError("model metadata field 'implemented_species' must be a list of positive integers.")
    if "family" in metadata and metadata["family"] is not None and not isinstance(metadata["family"], str):
        raise ValueError("model metadata field 'family' must be a string or null.")

    if require_structural_consistency or require_cross_field_consistency:
        coulomb_mode = metadata.get("coulomb_mode", "none")
        has_embedded_lr = metadata.get("has_embedded_lr", False)
        if coulomb_mode == "sr_embedded" and (
            metadata.get("coulomb_sr_rc") is None or metadata.get("coulomb_sr_envelope") is None
        ):
            raise ValueError("sr_embedded Coulomb metadata requires cutoff and envelope fields.")
        if coulomb_mode == "sr_embedded" and not has_embedded_lr:
            raise ValueError("sr_embedded Coulomb metadata requires embedded LR metadata.")
        if (
            coulomb_mode == "sr_embedded"
            and metadata.get("cutoff") is not None
            and metadata.get("coulomb_sr_rc") is not None
            and metadata["coulomb_sr_rc"] > metadata["cutoff"]
        ):
            raise ValueError("coulomb_sr_rc cannot exceed model cutoff.")
        if coulomb_mode == "full_embedded" and not has_embedded_lr:
            raise ValueError("full_embedded Coulomb metadata requires embedded LR metadata.")
        if metadata.get("has_embedded_d3ts", False) and not has_embedded_lr:
            raise ValueError("embedded D3TS metadata requires embedded LR metadata.")

    if require_cross_field_consistency:
        needs_coulomb = metadata.get("needs_coulomb", False)
        needs_dispersion = metadata.get("needs_dispersion", False)
        coulomb_mode = metadata.get("coulomb_mode", "none")
        if coulomb_mode == "sr_embedded" and not needs_coulomb:
            raise ValueError("sr_embedded Coulomb metadata requires external Coulomb.")
        if needs_coulomb and coulomb_mode == "full_embedded":
            raise ValueError("full_embedded Coulomb metadata cannot request external Coulomb.")
        if needs_dispersion:
            if d3_params is None:
                raise ValueError("needs_dispersion metadata requires d3_params.")
            missing_d3 = {"s8", "a1", "a2"} - set(d3_params)
            if missing_d3:
                raise ValueError(f"needs_dispersion metadata is missing d3_params: {sorted(missing_d3)}.")
            if metadata.get("has_embedded_d3ts", False):
                raise ValueError("needs_dispersion cannot be combined with embedded D3TS.")


def validate_runtime_model_metadata(
    metadata: Mapping[str, Any],
    *,
    needs_coulomb: bool,
    needs_dispersion: bool,
) -> None:
    """Validate metadata after calculator flags have resolved runtime behavior."""
    effective = dict(metadata)
    effective["needs_coulomb"] = needs_coulomb
    effective["needs_dispersion"] = needs_dispersion
    if "format_version" in metadata:
        is_legacy_runtime = type(effective.get("format_version")) is int and effective["format_version"] == 1
        validate_model_metadata(
            effective,
            require_cutoff=not is_legacy_runtime,
            require_structural_consistency=not is_legacy_runtime,
        )
    # Raw nn.Module metadata predates the artifact schema and may expose only
    # operation-specific fields. It remains exempt from schema and structural
    # requirements, but effective runtime combinations must still be safe.
    if needs_coulomb and effective.get("coulomb_mode") == "full_embedded":
        raise ValueError("full_embedded Coulomb metadata cannot request external Coulomb.")
    if needs_dispersion:
        d3_params = effective.get("d3_params")
        if not isinstance(d3_params, Mapping):
            raise ValueError("needs_dispersion metadata requires d3_params.")
        missing_d3 = {"s8", "a1", "a2"} - set(d3_params)
        if missing_d3:
            raise ValueError(f"needs_dispersion metadata is missing d3_params: {sorted(missing_d3)}.")
        if effective.get("has_embedded_d3ts", False):
            raise ValueError("needs_dispersion cannot be combined with embedded D3TS.")
