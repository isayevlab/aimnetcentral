from __future__ import annotations

import contextlib
from collections.abc import Collection
from pathlib import Path
from typing import ClassVar, Final, Literal, NotRequired, TypedDict

import torch
from torch import Tensor, nn

from aimnet import nbops
from aimnet.config import build_module
from aimnet.models.artifact_validation import (
    _REGISTRY_IMPORT_POLICY,
    ModelImportPolicy,
    resolve_model_import_policy,
    validate_v2_artifact_with_policy,
)
from aimnet.models.utils import (
    convert_atomic_shifts_to_float64,
    extract_d3_params,
    extract_species,
    has_externalizable_dftd3,
    load_state_dict_checked,
)


class ModelMetadata(TypedDict):
    """Metadata returned by load_model().

    This TypedDict documents the structure of the metadata dictionary.
    """

    format_version: int  # 1 = legacy .jpt, 2 = new .pt
    cutoff: float  # Model cutoff radius

    # Action flags - what calculator should add externally
    needs_coulomb: bool  # Add external Coulomb?
    needs_dispersion: bool  # Add external DFTD3?

    # Coulomb mode descriptor - what's in the model
    # "sr_embedded": Model has SRCoulomb, add FULL externally
    # "full_embedded": Full Coulomb in model (legacy JIT)
    # "none": No Coulomb anywhere
    coulomb_mode: str
    coulomb_sr_rc: NotRequired[float | None]  # Only if coulomb_mode="sr_embedded"
    coulomb_sr_envelope: NotRequired[str | None]  # "exp" | "cosine", only if sr_embedded

    # Dispersion parameters (optional)
    d3_params: NotRequired[dict | None]  # {s8, a1, a2, s6} if needs_dispersion=True
    has_embedded_lr: NotRequired[bool]  # True if model has embedded LR (legacy or D3TS)

    implemented_species: list[int]  # Supported atomic numbers

    family: NotRequired[str | None]  # e.g. "rxn"; None for legacy/families that don't declare
    supports_charged_systems: NotRequired[bool | None]  # False for rxn; None for legacy
    has_embedded_d3ts: NotRequired[bool]  # True when D3TS module is embedded (distinct from has_embedded_lr,
    # which conflates D3TS with SRCoulomb — see _has_embedded_dispersion)


def load_legacy_jit(path: str, device: str = "cpu") -> tuple[torch.jit.ScriptModule, ModelMetadata]:
    """Load a legacy TorchScript model from a trusted ``.jpt`` source.

    TorchScript is format-specific but is not a sandbox. Only load ``.jpt``
    files from sources whose code and provenance the caller trusts.
    """
    model = torch.jit.load(path, map_location=device)
    legacy_metadata: ModelMetadata = {
        "format_version": 1,
        "cutoff": float(model.cutoff),
        "needs_coulomb": False,
        "needs_dispersion": False,
        "coulomb_mode": "full_embedded",
        "d3_params": extract_d3_params(model) if has_externalizable_dftd3(model) else None,
        "implemented_species": extract_species(model),
    }

    with contextlib.suppress(AttributeError, RuntimeError):
        model._metadata = legacy_metadata  # type: ignore[attr-defined]

    return model, legacy_metadata


def load_model(
    path: str,
    device: str = "cpu",
    *,
    model_import_paths: Collection[str] | None = None,
    model_import_mode: Literal["extend", "replace", "unsafe"] = "extend",
) -> tuple[nn.Module, ModelMetadata]:
    """Load a v2 model or explicitly routed legacy ``.jpt`` model.

    Files ending in ``.jpt`` (case-insensitive) are loaded with
    :func:`torch.jit.load` and therefore must come from a trusted source.
    Every other suffix is loaded exactly once with restricted
    ``torch.load(weights_only=True)``.

    Parameters
    ----------
    path : str
        Path to a v2 ``.pt`` or trusted legacy ``.jpt`` model.
    device : str
        Device on which to load the model.
    model_import_paths : Collection[str] | None
        Python imports trusted when loading a v2 file. Entries are exact dotted
        paths or namespaces ending in ``.*``, such as
        ``{"my_package.models.CustomModel", "my_package.layers.*"}``.
    model_import_mode : {"extend", "replace", "unsafe"}
        ``extend`` adds ``model_import_paths`` to the default trusted paths;
        ``replace`` requires a nonempty collection and uses only those paths.
        ``unsafe`` cannot be combined with paths and permits arbitrary imported
        constructors. No mode relaxes restricted deserialization or artifact
        validation.

    Returns
    -------
    model : nn.Module
        The loaded model with weights.
    metadata : ModelMetadata
        Validated model metadata.

    Use ``unsafe`` only for locally trusted artifacts. Legacy ``.jpt`` files
    accept only ``model_import_paths=None`` and ``model_import_mode="extend"``.
    """
    policy = resolve_model_import_policy(model_import_paths, model_import_mode)
    if Path(path).suffix.lower() == ".jpt":
        if model_import_paths is not None or model_import_mode != "extend":
            raise ValueError("Import settings are not supported for .jpt sources.")
        return load_legacy_jit(path, device)
    return _load_v2_model(path, device, policy)


def _load_v2_model(
    path: str,
    device: str,
    policy: ModelImportPolicy,
    *,
    unexpected: Literal["warn", "error"] = "warn",
) -> tuple[nn.Module, ModelMetadata]:
    data = torch.load(path, map_location="cpu", weights_only=True)
    model_config, state_dict = validate_v2_artifact_with_policy(data, policy)
    with torch.device("cpu"):
        model = build_module(
            model_config,
            allow_file_references=False,
            import_authorizer=policy.require_allowed,
        )
    if not isinstance(model, nn.Module):
        raise TypeError("Built model configuration did not produce an nn.Module.")

    # Atomic shifts store SAE/reference-energy values and may be float64 in
    # the file. Cast before load_state_dict so copy_ does not truncate them
    # into the default float32 embedding.
    convert_atomic_shifts_to_float64(model)

    load_state_dict_checked(
        model,
        state_dict,
        source=path,
        unexpected=unexpected,
    )

    model = model.to(device)
    metadata: ModelMetadata = {
        "format_version": data.get("format_version", 2),
        "cutoff": data["cutoff"],
        "needs_coulomb": data.get("needs_coulomb", False),
        "needs_dispersion": data.get("needs_dispersion", False),
        "coulomb_mode": data.get("coulomb_mode", "none"),
        "coulomb_sr_rc": data.get("coulomb_sr_rc"),
        "coulomb_sr_envelope": data.get("coulomb_sr_envelope"),
        "d3_params": data.get("d3_params"),
        "has_embedded_lr": data.get("has_embedded_lr", False),
        "implemented_species": data.get("implemented_species", []),
        "family": data.get("family"),
        "supports_charged_systems": data.get("supports_charged_systems"),
        "has_embedded_d3ts": data.get("has_embedded_d3ts", False),
    }
    model._metadata = metadata  # type: ignore[assignment]
    return model, metadata


def _load_registry_model(path: str, device: str = "cpu") -> tuple[nn.Module, ModelMetadata]:
    """Load a registry artifact with its immutable import policy."""
    return _load_v2_model(path, device, _REGISTRY_IMPORT_POLICY, unexpected="error")


class AIMNet2Base(nn.Module):
    """Base class for AIMNet2 models. Implements pre-processing data:
    converting to right dtype and device, setting nb mode, calculating masks.
    """

    __default_dtype = torch.get_default_dtype()

    _required_keys: Final = ["coord", "numbers", "charge"]
    _required_keys_dtype: Final = [__default_dtype, torch.int64, __default_dtype]
    _optional_keys: Final = [
        "mult",
        "nbmat",
        "nbmat_lr",
        "mol_idx",
        "shifts",
        "shifts_lr",
        "cell",
        "nbmat_dftd3",
        "shifts_dftd3",
        "cutoff_dftd3",
        "nbmat_coulomb",
        "shifts_coulomb",
        "cutoff_coulomb",
        "pbc",
    ]
    _optional_keys_dtype: Final = [
        __default_dtype,  # mult
        torch.int32,  # nbmat
        torch.int32,  # nbmat_lr
        torch.int64,  # mol_idx
        __default_dtype,  # shifts
        __default_dtype,  # shifts_lr
        __default_dtype,  # cell
        torch.int32,  # nbmat_dftd3
        __default_dtype,  # shifts_dftd3
        __default_dtype,  # cutoff_dftd3
        torch.int32,  # nbmat_coulomb
        __default_dtype,  # shifts_coulomb
        __default_dtype,  # cutoff_coulomb
        torch.bool,  # pbc
    ]
    __constants__: ClassVar = ["_required_keys", "_required_keys_dtype", "_optional_keys", "_optional_keys_dtype"]
    # TypedDict not supported in TorchScript; exclude from serialization
    __jit_unused_properties__: ClassVar = ["metadata"]

    def __init__(self):
        super().__init__()
        # Use object.__setattr__ to avoid TorchScript tracing this attribute
        object.__setattr__(self, "_metadata", None)

    @property
    def metadata(self) -> ModelMetadata | None:
        """Return model metadata if available."""
        return getattr(self, "_metadata", None)

    def _prepare_dtype(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        for k, d in zip(self._required_keys, self._required_keys_dtype, strict=False):
            assert k in data, f"Key {k} is required"
            data[k] = data[k].to(d)
        for k, d in zip(self._optional_keys, self._optional_keys_dtype, strict=False):
            if k in data:
                data[k] = data[k].to(d)
        return data

    def prepare_input(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        """Common operations for input preparation."""
        data = self._prepare_dtype(data)
        data = nbops.set_nb_mode(data)
        data = nbops.calc_masks(data)

        assert data["charge"].ndim == 1, "Charge should be 1D tensor."
        if "mult" in data:
            assert data["mult"].ndim == 1, "Mult should be 1D tensor."
        return data
