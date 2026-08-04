"""Model-source resolution for :class:`~aimnet.calculators.AIMNet2Calculator`.

Owns the dispatch between registry names/aliases, Hugging Face repo ids,
local HF-style directories, plain file paths, and raw ``nn.Module`` objects,
plus the family-policy reconciliation applied to the resolved metadata.
"""

import os
import re
from collections.abc import Collection, Mapping
from typing import Any, Literal, cast

from torch import nn

from aimnet.models.artifact_validation import (
    is_explicit_local_path,
    is_legacy_jit_path,
    resolve_model_import_policy,
    uses_default_model_import_settings,
)
from aimnet.models.base import load_model, load_registry_model

from .model_registry import (
    get_family_policy,
    get_model_path,
    get_registry_model_family,
    try_resolve_registry_model_name,
)

# Inline org/name pattern — exactly one slash, both segments alphanumeric+._-
# This avoids importing optional HF deps for ordinary file paths containing slashes.
_HF_ID_RE = re.compile(r"^[a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+$")
_load_registry_model = load_registry_model


def _apply_family_defaults(metadata: Mapping[str, Any], registry_family: str | None) -> dict[str, Any]:
    """Apply calculator-side compatibility defaults for released model families."""
    metadata = dict(metadata)
    if registry_family is not None:
        metadata_family = metadata.get("family")
        if metadata_family is None:
            metadata["family"] = registry_family
        elif metadata_family != registry_family:
            raise ValueError(
                f"Registry family '{registry_family}' does not match model metadata family "
                f"'{metadata_family}'. Refusing to load ambiguous energy scale."
            )

    policy = get_family_policy(metadata.get("family"))

    if policy.supports_charged_systems is not None:
        supports_charged = metadata.get("supports_charged_systems")
        if supports_charged is None:
            metadata["supports_charged_systems"] = policy.supports_charged_systems
        elif supports_charged is not policy.supports_charged_systems:
            raise ValueError(
                f"aimnet2-{policy.family} models must declare "
                f"supports_charged_systems={policy.supports_charged_systems}."
            )

    if policy.posthoc_d3_params is not None and not metadata.get("has_embedded_d3ts", False):
        metadata["needs_dispersion"] = True
        if metadata.get("d3_params") is None:
            metadata["d3_params"] = dict(policy.posthoc_d3_params)

    return metadata


def resolve_model(
    model: str | nn.Module,
    *,
    device: str,
    ensemble_member: int = 0,
    revision: str | None = None,
    token: str | None = None,
    model_import_paths: Collection[str] | None = None,
    model_import_mode: Literal["extend", "replace", "unsafe"] = "extend",
) -> tuple[nn.Module, Mapping[str, Any] | None, float]:
    """Resolve a model source and move the resulting module to ``device``.

    Parameters
    ----------
    model
        Registry name or alias, local model path, Hugging Face repository or
        local HF-format directory, or an existing module.
    device
        Device on which to place the model.
    ensemble_member
        Zero-based member selected from a Hugging Face ensemble.
    revision
        Hugging Face repository revision, branch, or tag.
    token
        Hugging Face access token for private repositories.
    model_import_paths, model_import_mode
        Import settings for a direct local v2 artifact or a complete Hugging
        Face repository. Path syntax and modes match
        :func:`aimnet.models.base.load_model`.

    Returns
    -------
    tuple
        ``(module, metadata, cutoff)`` with family defaults applied. Registry
        names and aliases, registry HF fallback, raw modules, and ``.jpt``
        files accept only ``model_import_paths=None`` and
        ``model_import_mode="extend"``.
    """
    resolve_model_import_policy(model_import_paths, model_import_mode)
    customized = not uses_default_model_import_settings(model_import_paths, model_import_mode)
    metadata: Mapping[str, Any] | None = None
    registry_family: str | None = None
    if isinstance(model, str):
        if is_legacy_jit_path(model) and customized:
            raise ValueError("Import settings are not supported for .jpt sources.")
        explicit_local = is_explicit_local_path(model)
        registry_name = None if explicit_local else try_resolve_registry_model_name(model)
        if registry_name is not None:
            if customized:
                raise ValueError("Custom import settings are forbidden for registry models.")
            registry_family = get_registry_model_family(registry_name)
            p = get_model_path(registry_name)
            module, metadata = _load_registry_model(p, device=device)
            cutoff = metadata["cutoff"]
        else:
            _is_hf_dir = os.path.isdir(model)
            if explicit_local and not _is_hf_dir:
                module, metadata = load_model(
                    model,
                    device=device,
                    model_import_paths=model_import_paths,
                    model_import_mode=model_import_mode,
                )
                cutoff = metadata["cutoff"]
            elif (not explicit_local and bool(_HF_ID_RE.match(model))) or _is_hf_dir:
                # Check for HF repo ID or local HF-style directory.
                # (lazy import to keep optional HF dependencies optional)
                try:
                    from aimnet.calculators.hf_hub import is_hf_repo_id, load_from_hf_repo
                except ImportError:
                    raise ImportError(
                        f"Loading from HF repo '{model}' requires optional dependencies. "
                        "Install with: pip install aimnet[hf]"
                    ) from None
                if is_hf_repo_id(model) or _is_hf_dir:
                    module, metadata = load_from_hf_repo(
                        model,
                        ensemble_member=ensemble_member,
                        device=device,
                        revision=revision,
                        token=token,
                        model_import_paths=model_import_paths,
                        model_import_mode=model_import_mode,
                    )
                    cutoff = metadata["cutoff"]
                else:
                    module, metadata = load_model(
                        model,
                        device=device,
                        model_import_paths=model_import_paths,
                        model_import_mode=model_import_mode,
                    )
                    cutoff = metadata["cutoff"]
            else:
                p = get_model_path(model)
                module, metadata = load_model(
                    p,
                    device=device,
                    model_import_paths=model_import_paths,
                    model_import_mode=model_import_mode,
                )
                cutoff = metadata["cutoff"]
    elif isinstance(model, nn.Module):
        if customized:
            raise ValueError("Import settings are not supported for raw nn.Module sources.")
        module = model.to(device)
        cutoff = getattr(module, "cutoff", 5.0)
        metadata = cast(Mapping[str, Any] | None, getattr(module, "metadata", None))
        if metadata is None:
            metadata = cast(Mapping[str, Any] | None, getattr(module, "_metadata", None))
    else:
        raise TypeError("Invalid model type/name.")

    if metadata is not None:
        metadata = _apply_family_defaults(metadata, registry_family)
        module._metadata = metadata  # type: ignore[assignment]

    return module, metadata, cutoff
