"""Hugging Face Hub integration for AIMNet2 models.

Enables loading models from HF repos containing safetensors weights + config.json.
Supports both local directories and HF repo IDs (e.g. "isayevlab/aimnet2-wb97m-d3").

Security: model_yaml in config.json is validated before construction. Direct
repositories may customize the import set; registry fallback always uses the
shared immutable set.
"""

from __future__ import annotations

import json
import re
import warnings
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn

try:
    from huggingface_hub import snapshot_download as _snapshot_download
except ImportError:  # pragma: no cover - exercised through optional-dependency tests
    _snapshot_download = None

try:
    from safetensors.torch import load_file as _load_safetensors_file
except ImportError:  # pragma: no cover - exercised through optional-dependency tests
    _load_safetensors_file = None

from aimnet.calculators.model_registry import get_family_policy, get_registry_model_path
from aimnet.models.artifact_validation import (
    REGISTRY_IMPORT_POLICY,
    resolve_model_import_policy,
    uses_default_model_import_settings,
    validate_model_metadata,
    validate_model_yaml,
    validate_registry_v2_artifact,
)
from aimnet.models.base import ModelMetadata, assemble_v2_model

_HF_ROUTING_CONFIG_KEYS = frozenset({
    "config_schema_version",
    "family_name",
    "member_names",
    "ensemble_size",
})
_ARTIFACT_METADATA_KEYS = frozenset({
    "format_version",
    "model_yaml",
    "cutoff",
    "needs_coulomb",
    "needs_dispersion",
    "coulomb_mode",
    "coulomb_sr_rc",
    "coulomb_sr_envelope",
    "d3_params",
    "has_embedded_lr",
    "has_embedded_d3ts",
    "implemented_species",
    "family",
    "supports_charged_systems",
})


def is_hf_repo_id(model: str) -> bool:
    """Check if a string looks like a HF repo ID (org/name format).

    Uses strict pattern: exactly one slash, alphanumeric+hyphen segments.
    """
    parts = model.split("/")
    if len(parts) != 2:
        return False
    org, name = parts
    pattern = re.compile(r"^[a-zA-Z0-9._-]+$")
    if not (pattern.match(org) and pattern.match(name)):
        return False
    return not Path(model).exists()


def _extract_sr_coulomb_from_config(config: Mapping[str, object]) -> tuple[float | None, str | None]:
    """Extract an unambiguous SRCoulomb pair from parsed model config.

    Duplicate identical definitions are treated as one pair. Distinct pairs
    are rejected because no single metadata value can describe the model.
    """
    pairs = _find_srcoulomb_params(config)
    if len(pairs) > 1:
        raise ValueError(f"ambiguous SRCoulomb definitions contain distinct parameter pairs: {sorted(pairs)!r}.")
    return next(iter(pairs), (None, None))


def _find_srcoulomb_params(obj: object) -> set[tuple[float, str]]:
    """Recursively collect complete SRCoulomb ``(rc, envelope)`` pairs."""
    pairs: set[tuple[float, str]] = set()
    if isinstance(obj, dict):
        class_name = obj.get("class")
        if isinstance(class_name, str) and class_name.endswith("SRCoulomb"):
            kwargs = obj.get("kwargs", {})
            if isinstance(kwargs, Mapping):
                rc = kwargs.get("rc")
                envelope = kwargs.get("envelope")
                if rc is not None and envelope is not None:
                    if not isinstance(envelope, str):
                        raise ValueError("SRCoulomb model_yaml field 'coulomb_sr_envelope' must be a supported string.")
                    validate_model_metadata({
                        "coulomb_sr_rc": rc,
                        "coulomb_sr_envelope": envelope,
                    })
                    pairs.add((float(rc), envelope))
        for value in obj.values():
            pairs.update(_find_srcoulomb_params(value))
    elif isinstance(obj, list):
        for item in obj:
            pairs.update(_find_srcoulomb_params(item))
    return pairs


def _derive_sr_coulomb_metadata(config: dict, model_config: Mapping[str, object]) -> None:
    """Validate and fill short-range Coulomb metadata from complete model YAML."""
    coulomb_sr_rc, coulomb_sr_envelope = _extract_sr_coulomb_from_config(model_config)
    explicit_rc = config.get("coulomb_sr_rc")
    explicit_envelope = config.get("coulomb_sr_envelope")
    discovered = coulomb_sr_rc is not None and coulomb_sr_envelope is not None

    if discovered and explicit_rc is not None and float(explicit_rc) != coulomb_sr_rc:
        raise ValueError(
            "config.json field 'coulomb_sr_rc' conflicts with the SRCoulomb value discovered in model_yaml."
        )
    if discovered and explicit_envelope is not None and explicit_envelope != coulomb_sr_envelope:
        raise ValueError(
            "config.json field 'coulomb_sr_envelope' conflicts with the SRCoulomb value discovered in model_yaml."
        )

    needs_sr_pair = config.get("coulomb_mode", "none") == "sr_embedded"
    if needs_sr_pair and (explicit_rc is None or explicit_envelope is None) and not discovered:
        raise ValueError(
            "sr_embedded metadata with an omitted Coulomb field requires exactly one distinct complete "
            "SRCoulomb parameter pair in model_yaml."
        )

    if explicit_rc is None and discovered:
        config["coulomb_sr_rc"] = coulomb_sr_rc
    if explicit_envelope is None and discovered:
        config["coulomb_sr_envelope"] = coulomb_sr_envelope


def _validate_ensemble_member(ensemble_member: object) -> int:
    if type(ensemble_member) is not int or ensemble_member < 0:
        raise ValueError("ensemble_member must be a non-boolean integer greater than or equal to zero.")
    return ensemble_member


def _validated_member_names(config: Mapping[str, Any], ensemble_member: int) -> list[str] | None:
    if "member_names" not in config:
        return None
    member_names = config["member_names"]
    if (
        not isinstance(member_names, list)
        or not member_names
        or any(not isinstance(name, str) for name in member_names)
    ):
        raise ValueError("config.json field 'member_names' must be a nonempty list of strings.")
    if ensemble_member >= len(member_names):
        raise ValueError(
            f"ensemble_member {ensemble_member} is out of range for config.json 'member_names' "
            f"with {len(member_names)} entries."
        )
    return member_names


def _complete_model_metadata(config: Mapping[str, Any]) -> ModelMetadata:
    format_version = config.get("format_version", 2)
    if type(format_version) is not int or format_version != 2:
        raise ValueError("HF model metadata field 'format_version' must be integer 2.")
    return {
        "format_version": format_version,
        "cutoff": config["cutoff"],
        "needs_coulomb": config.get("needs_coulomb", False),
        "needs_dispersion": config.get("needs_dispersion", False),
        "coulomb_mode": config.get("coulomb_mode", "none"),
        "coulomb_sr_rc": config.get("coulomb_sr_rc"),
        "coulomb_sr_envelope": config.get("coulomb_sr_envelope"),
        "d3_params": config.get("d3_params"),
        "has_embedded_lr": config.get("has_embedded_lr", False),
        "implemented_species": config.get("implemented_species", []),
        "family": config.get("family"),
        "supports_charged_systems": config.get("supports_charged_systems"),
        "has_embedded_d3ts": config.get("has_embedded_d3ts", False),
    }


def _validate_registry_fallback_config(
    family_config: Mapping[str, Any],
    registry_metadata: Mapping[str, Any],
) -> ModelMetadata:
    unsupported = set(family_config) - _HF_ROUTING_CONFIG_KEYS - _ARTIFACT_METADATA_KEYS
    if unsupported:
        fields = ", ".join(repr(field) for field in sorted(unsupported))
        raise ValueError(
            f"Registry HF fallback config fields {fields} are not permitted; "
            "family config may contain only routing fields and matching artifact metadata."
        )

    metadata = _complete_model_metadata(registry_metadata)
    authoritative: dict[str, Any] = {"model_yaml": registry_metadata["model_yaml"], **metadata}
    for key in _ARTIFACT_METADATA_KEYS & family_config.keys():
        if family_config[key] != authoritative[key]:
            raise ValueError(
                f"Registry HF fallback config field {key!r} conflicts with authoritative registry metadata."
            )
    return metadata


def _fetch_pt_metadata_from_registry(
    config: dict,
    repo_id_or_path: str,
    ensemble_member: int,
) -> tuple[dict, dict[str, object]]:
    """Fetch full ``.pt`` metadata from the model registry as a fallback.

    Used when the HF repo's config.json (family-level schema v1) was uploaded
    without fields like model_yaml, d3_params, coulomb_sr_rc, etc. The member
    name is looked up from the config's member_names list, then the registry
    artifact is loaded to extract all metadata.

    Returns the full .pt metadata dict (everything except state_dict) and the
    already validated parsed model configuration.
    """
    ensemble_member = _validate_ensemble_member(ensemble_member)
    member_names = _validated_member_names(config, ensemble_member)
    if member_names is not None:
        member_name = member_names[ensemble_member]
    else:
        # Derive from family_name or repo slug via the registry's family policy
        # (repo slugs carry the "aimnet2-" prefix, family tags don't).
        family_name = config.get("family_name") or Path(repo_id_or_path).name
        policy = get_family_policy(family_name)
        if not policy.members:
            policy = get_family_policy(family_name.removeprefix("aimnet2-"))
        candidates = policy.members
        if not candidates:
            raise ValueError(
                f"config.json in '{repo_id_or_path}' has no 'model_yaml' field and "
                "no 'member_names' list to look up a fallback. "
                "Please re-upload the repo with a config.json that includes 'model_yaml'."
            )
        if ensemble_member >= len(candidates):
            raise ValueError(
                f"ensemble_member {ensemble_member} is out of range for family {family_name!r} "
                f"with {len(candidates)} registry members."
            )
        member_name = candidates[ensemble_member]

    warnings.warn(
        f"config.json in '{repo_id_or_path}' is missing fields (model_yaml, d3_params, etc.). "
        f"Falling back to the model registry for member '{member_name}'. "
        "Re-upload the HF repo with a complete config.json to avoid this.",
        UserWarning,
        stacklevel=5,
    )

    pt_path = get_registry_model_path(member_name)
    data = torch.load(pt_path, map_location="cpu", weights_only=True)
    try:
        model_config, _ = validate_registry_v2_artifact(data)
    except ValueError as exc:
        raise ValueError(f"Invalid registry .pt file for '{member_name}': {exc}") from exc
    # Return everything except state_dict
    return ({k: v for k, v in data.items() if k != "state_dict"}, model_config)


def load_from_hf_repo(
    repo_id_or_path: str,
    ensemble_member: int = 0,
    device: str = "cpu",
    revision: str | None = None,
    token: str | None = None,
    *,
    model_import_paths: Collection[str] | None = None,
    model_import_mode: Literal["extend", "replace", "unsafe"] = "extend",
) -> tuple[nn.Module, ModelMetadata]:
    """Load an AIMNet2 model from a Hugging Face repo or HF-format directory.

    Parameters
    ----------
    repo_id_or_path : str
        Repository ID, such as ``"isayevlab/aimnet2-wb97m-d3"``, or a local
        directory containing ``config.json`` and safetensors weights.
    ensemble_member : int
        Zero-based ensemble member to load.
    device : str
        Device on which to load the model.
    revision : str, optional
        Repository revision, branch, or tag.
    token : str, optional
        Hugging Face access token for private repositories.
    model_import_paths : Collection[str] | None, optional
        Trusted imports for a complete repository containing ``model_yaml``.
    model_import_mode : {"extend", "replace", "unsafe"}, optional
        How to combine trusted imports; see
        :func:`aimnet.models.base.load_model`. Registry fallback accepts only
        ``model_import_paths=None`` and ``model_import_mode="extend"``.

    Returns
    -------
    model : nn.Module
        The loaded model with weights.
    metadata : ModelMetadata
        Model metadata dictionary.
    """
    ensemble_member = _validate_ensemble_member(ensemble_member)
    policy = resolve_model_import_policy(model_import_paths, model_import_mode)
    customized = not uses_default_model_import_settings(model_import_paths, model_import_mode)
    local_dir = _resolve_repo(repo_id_or_path, ensemble_member, revision, token, include_weights=False)

    # Load config.json
    config_path = local_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {local_dir}")
    config = json.loads(config_path.read_text())
    if not isinstance(config, Mapping):
        raise TypeError("config.json root must be a mapping.")
    config = dict(config)
    validate_model_metadata(config)
    _validated_member_names(config, ensemble_member)

    # Validate model_yaml imports before build_module().
    # Family-level configs (config_schema_version=1 uploaded to HF) may not
    # include model_yaml or other per-member fields. Fall back to loading them
    # from the registry artifact.
    _pt_meta: Mapping[str, Any] | None = None
    model_config: dict[str, object]
    model_yaml = config.get("model_yaml")
    if model_yaml is None:
        if customized:
            raise ValueError("Custom import settings are forbidden for registry HF fallback.")
        _pt_meta, model_config = _fetch_pt_metadata_from_registry(config, repo_id_or_path, ensemble_member)
        metadata = _validate_registry_fallback_config(config, _pt_meta)
        validate_model_metadata(
            metadata,
            require_cutoff=True,
            require_structural_consistency=True,
            require_cross_field_consistency=True,
        )
        construction_policy = REGISTRY_IMPORT_POLICY
        unexpected: Literal["warn", "error"] = "error"
    else:
        model_config = validate_model_yaml(
            model_yaml,
            model_import_paths=model_import_paths,
            model_import_mode=model_import_mode,
        )
        _derive_sr_coulomb_metadata(config, model_config)
        validate_model_metadata(config, require_cutoff=True)
        metadata = _complete_model_metadata(config)
        validate_model_metadata(
            metadata,
            require_cutoff=True,
            require_structural_consistency=True,
        )
        construction_policy = policy
        unexpected = "warn"

    # Resolve and read weights only after final metadata passes source-specific
    # structural or canonical validation.
    snapshot_revision = _snapshot_revision(local_dir, revision)
    local_dir = _resolve_repo(
        repo_id_or_path,
        ensemble_member,
        snapshot_revision,
        token,
        include_weights=True,
    )

    st_name = f"ensemble_{ensemble_member}.safetensors"
    st_path = local_dir / st_name
    if not st_path.exists():
        raise FileNotFoundError(f"{st_name} not found in {local_dir}")
    if _load_safetensors_file is None:
        raise ImportError(
            'Loading Hugging Face weights requires the "hf" extra. Install with: pip install "aimnet[hf]"'
        )
    state_dict = _load_safetensors_file(str(st_path), device="cpu")

    model = assemble_v2_model(
        model_config,
        state_dict,
        metadata,
        policy=construction_policy,
        device=device,
        source=str(st_path),
        unexpected=unexpected,
    )
    attached_metadata: ModelMetadata = model.__dict__["_metadata"]
    return model, attached_metadata


def _snapshot_revision(local_dir: Path, requested_revision: str | None) -> str | None:
    """Return the immutable commit encoded by a standard HF snapshot path.

    Local directories and test doubles that do not use the Hub cache layout
    retain the caller's requested revision.
    """
    commit = local_dir.name
    if local_dir.parent.name == "snapshots" and re.fullmatch(r"[0-9a-fA-F]{40,64}", commit):
        return commit
    return requested_revision


def _resolve_repo(
    repo_id_or_path: str,
    ensemble_member: int,
    revision: str | None,
    token: str | None,
    *,
    include_weights: bool = True,
) -> Path:
    """Resolve a HF repo ID to a local directory (downloading if needed).

    Uses snapshot_download (documented API) instead of hf_hub_download.
    """
    local = Path(repo_id_or_path)
    if local.is_dir():
        return local

    allow_patterns = ["config.json"]
    if include_weights:
        allow_patterns.append(f"ensemble_{ensemble_member}.safetensors")
    if _snapshot_download is None:
        raise ImportError(
            'Loading Hugging Face repositories requires the "hf" extra. Install with: pip install "aimnet[hf]"'
        )
    local_dir = _snapshot_download(
        repo_id=repo_id_or_path,
        allow_patterns=allow_patterns,
        revision=revision,
        token=token,
    )

    return Path(local_dir)
