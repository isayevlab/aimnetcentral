"""Hugging Face Hub integration for AIMNet2 models.

Enables loading models from HF repos containing safetensors weights + config.json.
Supports both local directories and HF repo IDs (e.g. "isayevlab/aimnet2-wb97m-d3").

Security: model_yaml in config.json is validated before construction. Direct
repositories may customize the import set; registry fallback always uses the
shared immutable set.
"""

from __future__ import annotations

import copy
import json
import re
import warnings
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Literal

import torch
from torch import nn

from aimnet.calculators.model_registry import get_family_policy, get_registry_model_path
from aimnet.config import build_module
from aimnet.models.artifact_validation import (
    _resolve_user_import_policy,
    _validate_registry_v2_artifact,
    validate_model_metadata,
    validate_model_yaml,
)
from aimnet.models.base import ModelMetadata


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
    """Extract coulomb_sr_rc and coulomb_sr_envelope from parsed model config.

    Looks for an SRCoulomb module definition in the parsed model config and
    returns its rc and envelope kwargs. Returns (None, None) if not found.
    """
    return _find_srcoulomb_params(config)


def _find_srcoulomb_params(obj) -> tuple[float | None, str | None]:
    """Recursively search a config dict for SRCoulomb kwargs."""
    if isinstance(obj, dict):
        if obj.get("class", "").endswith("SRCoulomb"):
            kwargs = obj.get("kwargs", {})
            rc = kwargs.get("rc")
            envelope = kwargs.get("envelope")
            return (float(rc) if rc is not None else None, envelope)
        for v in obj.values():
            result = _find_srcoulomb_params(v)
            if result != (None, None):
                return result
    elif isinstance(obj, list):
        for item in obj:
            result = _find_srcoulomb_params(item)
            if result != (None, None):
                return result
    return (None, None)


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
    member_names = config.get("member_names")
    if member_names and ensemble_member < len(member_names):
        member_name = member_names[ensemble_member]
    else:
        # Best-effort: derive from family_name or repo slug via the registry's
        # family policy (repo slugs carry the "aimnet2-" prefix, family tags don't).
        family_name = config.get("family_name") or Path(repo_id_or_path).name
        member_name = None
        policy = get_family_policy(family_name)
        if not policy.members:
            policy = get_family_policy(family_name.removeprefix("aimnet2-"))
        candidates = policy.members
        if candidates:
            member_name = candidates[ensemble_member] if ensemble_member < len(candidates) else candidates[0]
        if member_name is None:
            raise ValueError(
                f"config.json in '{repo_id_or_path}' has no 'model_yaml' field and "
                "no 'member_names' list to look up a fallback. "
                "Please re-upload the repo with a config.json that includes 'model_yaml'."
            )

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
        model_config, _ = _validate_registry_v2_artifact(data)
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
    _resolve_user_import_policy(model_import_paths, model_import_mode)
    customized = model_import_paths is not None or model_import_mode != "extend"
    local_dir = _resolve_repo(repo_id_or_path, ensemble_member, revision, token, include_weights=False)

    # Load config.json
    config_path = local_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {local_dir}")
    config = json.loads(config_path.read_text())
    if not isinstance(config, Mapping):
        raise TypeError("config.json root must be a mapping.")
    validate_model_metadata(config)

    # Validate model_yaml imports before build_module().
    # Family-level configs (config_schema_version=1 uploaded to HF) may not
    # include model_yaml or other per-member fields. Fall back to loading them
    # from the registry artifact.
    _pt_meta: dict | None = None
    model_config: dict[str, object]
    model_yaml = config.get("model_yaml")
    if model_yaml is None:
        if customized:
            raise ValueError("Custom import settings are forbidden for registry HF fallback.")
        _pt_meta, model_config = _fetch_pt_metadata_from_registry(config, repo_id_or_path, ensemble_member)
        config = {**_pt_meta, **config}
        model_yaml = config["model_yaml"]
        metadata_config = config
    else:
        metadata_config = config
        model_config = validate_model_yaml(
            model_yaml,
            model_import_paths=model_import_paths,
            model_import_mode=model_import_mode,
        )
    validate_model_metadata(
        metadata_config,
        require_cutoff=True,
        require_cross_field_consistency=True,
    )

    # Fetch remote weights only after config, model YAML, and metadata pass.
    local_dir = _resolve_repo(repo_id_or_path, ensemble_member, revision, token, include_weights=True)

    # Load safetensors only after config and model YAML validation.
    from safetensors.torch import load_file

    st_name = f"ensemble_{ensemble_member}.safetensors"
    st_path = local_dir / st_name
    if not st_path.exists():
        raise FileNotFoundError(f"{st_name} not found in {local_dir}")
    state_dict = load_file(str(st_path), device=device)

    # Rebuild model from config's model_yaml
    model = build_module(copy.deepcopy(model_config), allow_file_references=False)
    if not isinstance(model, nn.Module):
        raise TypeError("Built model configuration did not produce an nn.Module.")

    # Load state dict with key validation (not silent strict=False)
    from aimnet.models.utils import validate_state_dict_keys

    load_result = model.load_state_dict(state_dict, strict=False)
    real_missing, real_unexpected = validate_state_dict_keys(load_result.missing_keys, load_result.unexpected_keys)
    if real_missing:
        raise RuntimeError(f"Missing keys in safetensors file: {real_missing}")
    if real_unexpected:
        warnings.warn(f"Unexpected keys in safetensors file: {real_unexpected}", stacklevel=2)

    model = model.to(device)

    # Fix float64 atomic shifts: load_state_dict copies float64 safetensors
    # data into float32 buffers, truncating precision. We must:
    # 1) Convert the buffer to float64
    # 2) Re-copy the original float64 data from safetensors
    if hasattr(model, "outputs") and hasattr(model.outputs, "atomic_shift"):
        shift_key = "outputs.atomic_shift.shifts.weight"
        model.outputs.atomic_shift.shifts = model.outputs.atomic_shift.shifts.double()
        if shift_key in state_dict:
            model.outputs.atomic_shift.shifts.weight.data.copy_(state_dict[shift_key].to(device))

    # For fields not present in the flat family-level config.json (coulomb_sr_rc,
    # coulomb_sr_envelope, d3_params, has_embedded_lr) fall back first to
    # _pt_meta (already loaded above), then to parsing model_yaml.
    def _cfg(key, default=None):
        """Get a config value, falling back to _pt_meta, then default."""
        val = config.get(key)
        if val is None and _pt_meta is not None:
            val = _pt_meta.get(key)
        if val is None:
            val = default
        return val

    coulomb_sr_rc = _cfg("coulomb_sr_rc")
    coulomb_sr_envelope = _cfg("coulomb_sr_envelope")
    # If still None, try extracting from model_yaml (SRCoulomb module kwargs)
    if coulomb_sr_rc is None or coulomb_sr_envelope is None:
        _sr_rc, _sr_env = _extract_sr_coulomb_from_config(model_config)
        if coulomb_sr_rc is None:
            coulomb_sr_rc = _sr_rc
        if coulomb_sr_envelope is None:
            coulomb_sr_envelope = _sr_env

    # Build metadata
    metadata: ModelMetadata = {
        "format_version": _cfg("format_version", 2),
        "cutoff": config["cutoff"],
        "needs_coulomb": _cfg("needs_coulomb", False),
        "needs_dispersion": _cfg("needs_dispersion", False),
        "coulomb_mode": _cfg("coulomb_mode", "none"),
        "coulomb_sr_rc": coulomb_sr_rc,
        "coulomb_sr_envelope": coulomb_sr_envelope,
        "d3_params": _cfg("d3_params"),
        "has_embedded_lr": _cfg("has_embedded_lr", False),
        "implemented_species": _cfg("implemented_species", []),
        "family": _cfg("family"),
        "supports_charged_systems": _cfg("supports_charged_systems"),
        "has_embedded_d3ts": _cfg("has_embedded_d3ts", False),
    }
    validate_model_metadata(metadata, require_cutoff=True, require_cross_field_consistency=True)

    model._metadata = metadata
    return model, metadata


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

    from huggingface_hub import snapshot_download

    allow_patterns = ["config.json"]
    if include_weights:
        allow_patterns.append(f"ensemble_{ensemble_member}.safetensors")
    local_dir = snapshot_download(
        repo_id=repo_id_or_path,
        allow_patterns=allow_patterns,
        revision=revision,
        token=token,
    )

    return Path(local_dir)
