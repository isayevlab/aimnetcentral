"""Model-source resolution for :class:`~aimnet.calculators.AIMNet2Calculator`.

Owns the dispatch between registry names/aliases, Hugging Face repo ids,
local HF-style directories, plain file paths, and raw ``nn.Module`` objects,
plus the family-policy reconciliation applied to the resolved metadata.
"""

import os
import re
from collections.abc import Mapping
from typing import Any, cast

from torch import nn

from aimnet.models.base import load_model

from .model_registry import get_family_policy, get_model_path, get_registry_model_family

# Inline org/name pattern — exactly one slash, both segments alphanumeric+._-
# This avoids importing optional HF deps for ordinary file paths containing slashes.
_HF_ID_RE = re.compile(r"^[a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+$")


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
) -> tuple[nn.Module, Mapping[str, Any] | None, float]:
    """Resolve a model spec to ``(module_on_device, metadata, cutoff)``.

    ``metadata`` has family-policy defaults applied and, when not ``None``,
    is also attached to the returned module as ``_metadata``.
    """
    metadata: Mapping[str, Any] | None = None
    registry_family: str | None = None
    if isinstance(model, str):
        # Check for HF repo ID or local HF-style directory
        # (lazy import to keep safetensors/huggingface_hub optional)
        _is_hf_dir = os.path.isdir(model)
        _looks_like_hf = bool(_HF_ID_RE.match(model))
        if _looks_like_hf or _is_hf_dir:
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
                )
                cutoff = metadata["cutoff"]
            else:
                # _looks_like_hf matched but it's a local file path — fall through
                if not os.path.isfile(model):
                    registry_family = get_registry_model_family(model)
                p = get_model_path(model)
                module, metadata = load_model(p, device=device)
                cutoff = metadata["cutoff"]
        else:
            if not os.path.isfile(model):
                registry_family = get_registry_model_family(model)
            p = get_model_path(model)
            module, metadata = load_model(p, device=device)
            cutoff = metadata["cutoff"]
    elif isinstance(model, nn.Module):
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
