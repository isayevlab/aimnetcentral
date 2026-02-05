from __future__ import annotations

import contextlib
import warnings
from typing import ClassVar, Final, NotRequired, TypedDict

import torch
from torch import Tensor, nn

from aimnet import nbops
from aimnet.config import build_module
from aimnet.models.utils import (
    extract_d3_params,
    extract_species,
    has_externalizable_dftd3,
    validate_state_dict_keys,
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

    implemented_species: list[int]  # Supported atomic numbers


def load_model(path: str, device: str = "cpu") -> tuple[nn.Module, ModelMetadata]:
    """Load model from file, supporting both new and legacy formats.

    Automatically detects format:
    - New format: state dict with embedded YAML config and metadata
    - Legacy format: JIT-compiled TorchScript model

    Parameters
    ----------
    path : str
        Path to the model file (.pt or .jpt).
    device : str
        Device to load the model on. Default is "cpu".

    Returns
    -------
    model : nn.Module
        The loaded model with weights.
    metadata : ModelMetadata
        Dictionary containing model metadata. See ModelMetadata TypedDict for fields.

    Notes
    -----
    For legacy JIT models (format_version=1), `needs_coulomb` and `needs_dispersion`
    are False because LR modules are already embedded in the TorchScript model.
    """
    import yaml

    # torch.load auto-detects TorchScript and dispatches to torch.jit.load
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", ".*looks like a TorchScript archive.*")
        data = torch.load(path, map_location=device, weights_only=False)

    # Check result type to determine format
    if isinstance(data, dict) and "model_yaml" in data:
        # New state dict format
        model_config = yaml.safe_load(data["model_yaml"])
        model = build_module(model_config)

        # Use strict=False because modules may differ between formats
        load_result = model.load_state_dict(data["state_dict"], strict=False)

        # Check for unexpected missing/extra keys
        real_missing, real_unexpected = validate_state_dict_keys(load_result.missing_keys, load_result.unexpected_keys)
        if real_missing or real_unexpected:
            msg_parts = []
            if real_missing:
                msg_parts.append(f"Missing keys: {real_missing}")
            if real_unexpected:
                msg_parts.append(f"Unexpected keys: {real_unexpected}")
            warnings.warn(f"State dict mismatch during model loading. {'; '.join(msg_parts)}", stacklevel=2)

        model = model.to(device)

        # Preserve float64 precision for atomic shifts (SAE values) after device transfer
        if hasattr(model, "outputs") and hasattr(model.outputs, "atomic_shift"):
            model.outputs.atomic_shift.shifts = model.outputs.atomic_shift.shifts.double()

        metadata: ModelMetadata = {
            "format_version": data.get("format_version", 2),  # Default 2 for early v2 files without version
            "cutoff": data["cutoff"],
            "needs_coulomb": data.get("needs_coulomb", False),
            "needs_dispersion": data.get("needs_dispersion", False),
            "coulomb_mode": data.get("coulomb_mode", "none"),
            "coulomb_sr_rc": data.get("coulomb_sr_rc"),
            "coulomb_sr_envelope": data.get("coulomb_sr_envelope"),
            "d3_params": data.get("d3_params"),
            "has_embedded_lr": data.get("has_embedded_lr", False),
            "implemented_species": data.get("implemented_species", []),
        }

        # Attach metadata to model for easy access
        model._metadata = metadata

        return model, metadata

    elif isinstance(data, torch.jit.ScriptModule):
        # Legacy JIT format - LR modules are already embedded in the TorchScript model
        model = data
        metadata: ModelMetadata = {
            "format_version": 1,  # Legacy .jpt format is v1
            "cutoff": float(model.cutoff),
            # Legacy models have LR modules embedded - don't add external ones
            "needs_coulomb": False,
            "needs_dispersion": False,
            "coulomb_mode": "full_embedded",
            # No coulomb_sr_rc/envelope for legacy (full Coulomb is embedded)
            "d3_params": extract_d3_params(model) if has_externalizable_dftd3(model) else None,
            "implemented_species": extract_species(model),
        }

        # Attempt metadata assignment; silently fails for JIT models
        with contextlib.suppress(AttributeError, RuntimeError):
            model._metadata = metadata  # type: ignore[attr-defined]

        return model, metadata

    else:
        raise ValueError(f"Unknown model format: {type(data)}")


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
    ]
    _optional_keys_dtype: Final = [
        __default_dtype,  # mult
        torch.int64,  # nbmat
        torch.int64,  # nbmat_lr
        torch.int64,  # mol_idx
        __default_dtype,  # shifts
        __default_dtype,  # shifts_lr
        __default_dtype,  # cell
        torch.int64,  # nbmat_dftd3
        __default_dtype,  # shifts_dftd3
        __default_dtype,  # cutoff_dftd3
        torch.int64,  # nbmat_coulomb
        __default_dtype,  # shifts_coulomb
        __default_dtype,  # cutoff_coulomb
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
