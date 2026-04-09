"""Hugging Face Hub integration for AIMNet2 models.

Enables loading models from HF repos containing safetensors weights + config.json.
Supports both local directories and HF repo IDs (e.g. "isayevlab/aimnet2-wb97m-d3").

Security: model_yaml in config.json is validated against an allowlist of aimnet
class names before calling build_module() to prevent arbitrary code execution.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from torch import nn

from aimnet.config import build_module
from aimnet.models.base import ModelMetadata

# Allowlist of class prefixes permitted in model_yaml.
_ALLOWED_CLASS_PREFIXES = ("aimnet.",)


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


def _validate_model_yaml(model_yaml: str) -> None:
    """Validate that all class references in model_yaml are in the allowlist."""
    import yaml

    config = yaml.safe_load(model_yaml)
    _check_classes_recursive(config)


def _check_classes_recursive(obj) -> None:
    """Recursively check 'class' keys in nested config dict."""
    if isinstance(obj, dict):
        if "class" in obj:
            cls_name = obj["class"]
            if not any(cls_name.startswith(prefix) for prefix in _ALLOWED_CLASS_PREFIXES):
                raise ValueError(
                    f"Untrusted class in model config: '{cls_name}'. "
                    f"Only classes starting with {_ALLOWED_CLASS_PREFIXES} are allowed."
                )
        for v in obj.values():
            _check_classes_recursive(v)
    elif isinstance(obj, list):
        for item in obj:
            _check_classes_recursive(item)


def _extract_sr_coulomb_from_yaml(model_yaml: str) -> tuple[float | None, str | None]:
    """Extract coulomb_sr_rc and coulomb_sr_envelope from model_yaml.

    Looks for an SRCoulomb module definition in the model config YAML and
    returns its rc and envelope kwargs. Returns (None, None) if not found.
    """
    import yaml

    config = yaml.safe_load(model_yaml)
    # Walk the config tree looking for SRCoulomb class entries
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
) -> dict:
    """Fetch the full .pt metadata dict from the GCS model registry as a fallback.

    Used when the HF repo's config.json (family-level schema v1) was uploaded
    without fields like model_yaml, d3_params, coulomb_sr_rc, etc. The member
    name is looked up from the config's member_names list, then the GCS .pt
    file is loaded to extract all metadata.

    Returns the full .pt data dict (everything except state_dict).
    """
    import warnings

    member_names = config.get("member_names")
    if member_names and ensemble_member < len(member_names):
        member_name = member_names[ensemble_member]
    else:
        # Best-effort: derive from family_name or repo slug
        family_name = config.get("family_name") or Path(repo_id_or_path).name
        member_name = None
        try:
            from aimnet.calculators.model_registry import load_model_registry

            registry = load_model_registry()
            family_slug = family_name.replace("-", "_")
            all_models = list(registry.get("models", {}).keys())
            candidates = [k for k in all_models if family_slug in k]
            if candidates:
                member_name = candidates[ensemble_member] if ensemble_member < len(candidates) else candidates[0]
        except (ImportError, KeyError, IndexError):
            pass
        if member_name is None:
            raise ValueError(
                f"config.json in '{repo_id_or_path}' has no 'model_yaml' field and "
                "no 'member_names' list to look up a fallback. "
                "Please re-upload the repo with a config.json that includes 'model_yaml'."
            )

    warnings.warn(
        f"config.json in '{repo_id_or_path}' is missing fields (model_yaml, d3_params, etc.). "
        f"Falling back to GCS model registry for member '{member_name}'. "
        "Re-upload the HF repo with a complete config.json to avoid this.",
        UserWarning,
        stacklevel=5,
    )

    from aimnet.calculators.model_registry import get_model_path

    pt_path = get_model_path(member_name)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", ".*weights_only.*")
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
    if "model_yaml" not in data:
        raise ValueError(f"GCS .pt file for '{member_name}' also lacks 'model_yaml'. Cannot reconstruct model.")
    # Return everything except state_dict
    return {k: v for k, v in data.items() if k != "state_dict"}


def load_from_hf_repo(
    repo_id_or_path: str,
    ensemble_member: int = 0,
    device: str = "cpu",
    revision: str | None = None,
    token: str | None = None,
) -> tuple[nn.Module, ModelMetadata]:
    """Load an AIMNet2 model from a Hugging Face repo or local directory.

    Parameters
    ----------
    repo_id_or_path : str
        Either a HF repo ID ("isayevlab/aimnet2-wb97m-d3") or local directory path.
    ensemble_member : int
        Which ensemble member to load (0-3). Default: 0.
    device : str
        Device to load model on. Default: "cpu".
    revision : str, optional
        HF repo revision/branch.
    token : str, optional
        HF API token for private repos.

    Returns
    -------
    model : nn.Module
        The loaded model with weights.
    metadata : ModelMetadata
        Model metadata dictionary.
    """
    import copy
    import warnings

    import yaml

    local_dir = _resolve_repo(repo_id_or_path, ensemble_member, revision, token)

    # Load config.json
    config_path = local_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {local_dir}")
    config = json.loads(config_path.read_text())

    # Load safetensors weights
    from safetensors.torch import load_file

    st_name = f"ensemble_{ensemble_member}.safetensors"
    st_path = local_dir / st_name
    if not st_path.exists():
        raise FileNotFoundError(f"{st_name} not found in {local_dir}")
    state_dict = load_file(str(st_path), device=device)

    # Validate model_yaml against allowlist before build_module()
    # Family-level configs (config_schema_version=1 uploaded to HF) may not
    # include model_yaml or other per-member fields. Fall back to loading them
    # from the GCS .pt file via the model registry.
    _pt_meta: dict | None = None
    model_yaml = config.get("model_yaml")
    if model_yaml is None:
        _pt_meta = _fetch_pt_metadata_from_registry(config, repo_id_or_path, ensemble_member)
        model_yaml = _pt_meta["model_yaml"]
    _validate_model_yaml(model_yaml)

    # Rebuild model from config's model_yaml
    model_config = yaml.safe_load(model_yaml)
    model = build_module(copy.deepcopy(model_config))

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
        _sr_rc, _sr_env = _extract_sr_coulomb_from_yaml(model_yaml)
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
    }

    model._metadata = metadata
    return model, metadata


def _resolve_repo(
    repo_id_or_path: str,
    ensemble_member: int,
    revision: str | None,
    token: str | None,
) -> Path:
    """Resolve a HF repo ID to a local directory (downloading if needed).

    Uses snapshot_download (documented API) instead of hf_hub_download.
    """
    local = Path(repo_id_or_path)
    if local.is_dir():
        return local

    from huggingface_hub import snapshot_download

    local_dir = snapshot_download(
        repo_id=repo_id_or_path,
        allow_patterns=["config.json", f"ensemble_{ensemble_member}.safetensors"],
        revision=revision,
        token=token,
    )

    return Path(local_dir)
