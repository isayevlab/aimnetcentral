"""Utility functions for model inspection and metadata extraction.

This module provides helper functions for:
- Recursive module traversal
- Extracting attributes from JIT-compiled models
- Detecting embedded Coulomb and dispersion modules
- Extracting D3 parameters and implemented species
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import torch
from torch import nn


def named_children_rec(module: nn.Module) -> Iterator[tuple[str, nn.Module]]:
    """Recursively yield (name, child) for all descendants.

    Parameters
    ----------
    module : nn.Module
        The module to traverse.

    Yields
    ------
    tuple[str, nn.Module]
        Pairs of (name, child_module) for all descendants.
    """
    if isinstance(module, nn.Module):
        for name, child in module.named_children():
            yield name, child
            yield from named_children_rec(child)


def get_jit_attr(module: nn.Module, attr: str, default: float) -> float:
    """Extract attribute from JIT module, handling TorchScript constants.

    JIT models store scalar attributes as TorchScript constants which may
    need special handling to extract as Python floats.

    Parameters
    ----------
    module : nn.Module
        The module to extract the attribute from.
    attr : str
        The attribute name.
    default : float
        Default value if attribute is not found.

    Returns
    -------
    float
        The attribute value as a float.
    """
    val = None

    # Try direct attribute access first
    with contextlib.suppress(Exception):
        val = getattr(module, attr, None)

    # If that failed, try __getattr__ for TorchScript modules
    if val is None:
        with contextlib.suppress(AttributeError, RuntimeError):
            val = module.__getattr__(attr)

    # If still None, return default
    if val is None:
        return default

    # Convert tensor/number to float
    if hasattr(val, "item"):
        return float(val.item())
    elif hasattr(val, "__float__") or isinstance(val, (int, float)):
        return float(val)

    return default


def has_dispersion(model: nn.Module) -> bool:
    """Check if model has any dispersion module embedded (DFTD3, D3BJ, or D3TS).

    .. deprecated::
        Use ``model.metadata`` instead. This function iterates through model
        children which is slow and unreliable for JIT models.

    All dispersion modules need nbmat_lr for neighbor calculations,
    regardless of whether they use tabulated (DFTD3/D3BJ) or learned (D3TS) parameters.

    Parameters
    ----------
    model : nn.Module
        The model to check.

    Returns
    -------
    bool
        True if any dispersion module is found.
    """
    import warnings

    warnings.warn(
        "has_dispersion() is deprecated. Use model._metadata instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return any(name in ("dftd3", "d3bj", "d3ts") for name, _ in named_children_rec(model))


def has_externalizable_dftd3(model: nn.Module) -> bool:
    """Check if model has DFTD3/D3BJ that can be externalized.

    D3TS uses learned parameters from the NN and must stay embedded.
    Only DFTD3/D3BJ with tabulated parameters can be externalized.

    Parameters
    ----------
    model : nn.Module
        The model to check.

    Returns
    -------
    bool
        True if DFTD3 or D3BJ module is found.
    """
    return any(name in ("dftd3", "d3bj") for name, _ in named_children_rec(model))


def has_d3ts(model: nn.Module) -> bool:
    """Check if model has D3TS module (learned dispersion parameters).

    D3TS uses learned parameters from the NN and must stay embedded.

    Parameters
    ----------
    model : nn.Module
        The model to check.

    Returns
    -------
    bool
        True if D3TS module is found.
    """
    return any(name == "d3ts" for name, _ in named_children_rec(model))


def has_lrcoulomb(model: nn.Module) -> bool:
    """Check if model has LRCoulomb module embedded.

    Parameters
    ----------
    model : nn.Module
        The model to check.

    Returns
    -------
    bool
        True if LRCoulomb module is found.
    """
    return any(name == "lrcoulomb" for name, _ in named_children_rec(model))


def iter_lrcoulomb_mods(model: nn.Module) -> Iterator[nn.Module]:
    """Iterate over all LRCoulomb modules in the model.

    .. deprecated::
        Use ``model.metadata`` instead. This function iterates through model
        children which is slow and unreliable for JIT models.

    Parameters
    ----------
    model : nn.Module
        The model to search.

    Yields
    ------
    nn.Module
        Each LRCoulomb module found.
    """
    import warnings

    warnings.warn(
        "iter_lrcoulomb_mods() is deprecated. Use model._metadata instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    for name, module in named_children_rec(model):
        if name == "lrcoulomb":
            yield module


def extract_d3_params(model: nn.Module) -> dict[str, float] | None:
    """Extract D3 parameters from model's DFTD3/D3BJ module.

    Only extracts from DFTD3/D3BJ (tabulated params), not D3TS (learned params).
    Handles TorchScript constants which may be stored differently than
    regular Python attributes.

    Parameters
    ----------
    model : nn.Module
        The model to extract D3 parameters from.

    Returns
    -------
    dict or None
        Dictionary with s6, s8, a1, a2 parameters, or None if not found.
    """
    for name, module in named_children_rec(model):
        if name in ("dftd3", "d3bj"):  # NOT d3ts - it uses learned params
            return {
                "s8": get_jit_attr(module, "s8", 0.0),
                "a1": get_jit_attr(module, "a1", 0.0),
                "a2": get_jit_attr(module, "a2", 0.0),
                "s6": get_jit_attr(module, "s6", 1.0),
            }
    return None


def extract_coulomb_rc(model: nn.Module) -> float:
    """Extract Coulomb cutoff (rc) from model's LRCoulomb module.

    Parameters
    ----------
    model : nn.Module
        The model to extract the cutoff from.

    Returns
    -------
    float
        The Coulomb short-range cutoff value.

    Raises
    ------
    KeyError
        If LRCoulomb module is not found or rc attribute is missing.
    """
    for name, module in named_children_rec(model):
        if name == "lrcoulomb":
            rc = getattr(module, "rc", None)
            if rc is not None:
                return float(rc.item()) if hasattr(rc, "item") else float(rc)
            raise KeyError("LRCoulomb module found but 'rc' attribute is missing")
    raise KeyError("No LRCoulomb module found in model")


def extract_species(model: nn.Module) -> list[int]:
    """Extract implemented species from model's afv.weight (non-NaN entries).

    Checks afv.weight for non-NaN entries to determine which elements are implemented.

    Parameters
    ----------
    model : nn.Module
        The model to extract species from.

    Returns
    -------
    list[int]
        Sorted list of atomic numbers that are implemented in the model.
    """
    sd = model.state_dict()
    afv_weight = sd.get("afv.weight")
    if afv_weight is not None:
        species = []
        for i in range(1, afv_weight.shape[0]):
            # Element is implemented if its row is not all NaN
            if not torch.isnan(afv_weight[i]).all():
                species.append(i)
        return species
    return []


def has_d3ts_in_config(config: dict) -> bool:
    """Check if YAML config contains D3TS module.

    Parameters
    ----------
    config : dict
        Model YAML configuration dictionary.

    Returns
    -------
    bool
        True if D3TS is in the outputs section.
    """
    outputs = config.get("kwargs", {}).get("outputs", {})
    return "d3ts" in outputs


def has_dftd3_in_config(config: dict) -> bool:
    """Check if YAML config contains DFTD3 or D3BJ module.

    Parameters
    ----------
    config : dict
        Model YAML configuration dictionary.

    Returns
    -------
    bool
        True if DFTD3 or D3BJ is in the outputs section.
    """
    outputs = config.get("kwargs", {}).get("outputs", {})
    return "dftd3" in outputs or "d3bj" in outputs


# --- State dict key validation ---


def validate_state_dict_keys(
    missing_keys: list[str],
    unexpected_keys: list[str],
) -> tuple[list[str], list[str]]:
    """Filter out expected missing/unexpected keys during format migration.

    During v1→v2 model conversion, certain keys are expected to be missing
    (SRCoulomb added) or unexpected (LRCoulomb/DFTD3 removed). This function
    filters those out and returns only keys that indicate actual problems.

    Parameters
    ----------
    missing_keys : list[str]
        Keys missing from the state dict.
    unexpected_keys : list[str]
        Keys in the state dict that weren't expected.

    Returns
    -------
    tuple[list[str], list[str]]
        (real_missing, real_unexpected) - keys that indicate actual problems.
    """
    # Prefixes for keys that are expected to be missing/unexpected
    EXPECTED_MISSING_PREFIXES = ("outputs.srcoulomb.",)
    EXPECTED_UNEXPECTED_PREFIXES = (
        "outputs.lrcoulomb.",
        "outputs.dftd3.",
        "outputs.d3bj.",
    )

    def is_expected_missing(k: str) -> bool:
        return k.startswith(EXPECTED_MISSING_PREFIXES)

    def is_expected_unexpected(k: str) -> bool:
        return k.startswith(EXPECTED_UNEXPECTED_PREFIXES)

    real_missing = [k for k in missing_keys if not is_expected_missing(k)]
    real_unexpected = [k for k in unexpected_keys if not is_expected_unexpected(k)]
    return real_missing, real_unexpected


# --- YAML config manipulation ---


def strip_lr_modules_from_yaml(
    config: dict,
    source: dict | nn.Module,
) -> tuple[dict, str, bool, dict[str, float] | None, float | None, str]:
    """Remove LRCoulomb and DFTD3 from YAML config, add SRCoulomb.

    This is the unified function for both export (from state dict) and
    convert (from JIT model) paths.

    Parameters
    ----------
    config : dict
        Model YAML configuration dictionary.
    source : dict | nn.Module
        Either a state dict (for export path) or a JIT model (for convert path).
        Used to extract metadata like Coulomb rc and D3 params.

    Returns
    -------
    tuple
        (config, coulomb_mode, needs_dispersion, d3_params, coulomb_sr_rc, coulomb_sr_envelope):
        - config: Modified config with LR modules removed and SRCoulomb added
        - coulomb_mode: "sr_embedded" if LRCoulomb was present, else "none"
        - needs_dispersion: True if DFTD3/D3BJ was present
        - d3_params: D3 parameters dict or None
        - coulomb_sr_rc: Short-range Coulomb cutoff or None
        - coulomb_sr_envelope: Envelope function ("exp" or "cosine")

    Raises
    ------
    ValueError
        If model has both D3TS and DFTD3/D3BJ (double dispersion).
        If LRCoulomb is present but rc cannot be determined.

    Notes
    -----
    SRCoulomb is added to outputs only when LRCoulomb was present in the
    original config. This ensures proper energy accounting when the
    calculator adds external LRCoulomb.
    """
    import copy

    config = copy.deepcopy(config)
    outputs = config.get("kwargs", {}).get("outputs", {})

    # Determine source type
    is_jit_model = isinstance(source, nn.Module)

    # --- Detect Coulomb ---
    if is_jit_model:
        has_coulomb = has_lrcoulomb(source)
        coulomb_sr_rc = extract_coulomb_rc(source) if has_coulomb else None
        # Legacy models always used exp envelope
        coulomb_sr_envelope = "exp"
    else:
        # State dict path - check YAML config first, then state dict
        has_coulomb_in_sd = any(k.startswith("outputs.lrcoulomb") for k in source)
        if "lrcoulomb" in outputs:
            has_coulomb = True
            lrc_config = outputs["lrcoulomb"]
            lrc_kwargs = lrc_config.get("kwargs", {})
            rc_value = lrc_kwargs.get("rc")
            coulomb_sr_rc = float(rc_value) if rc_value is not None else None
            coulomb_sr_envelope = lrc_kwargs.get("envelope", "exp")
        elif has_coulomb_in_sd:
            has_coulomb = True
            rc_key = "outputs.lrcoulomb.rc"
            coulomb_sr_rc = float(source[rc_key].item()) if rc_key in source else None
            coulomb_sr_envelope = "exp"  # Cannot extract from state dict
        else:
            has_coulomb = False
            coulomb_sr_rc = None
            coulomb_sr_envelope = "exp"

    # Validate: if Coulomb is needed, rc must be determinable
    if has_coulomb and coulomb_sr_rc is None:
        raise ValueError(
            "Model requires Coulomb but 'rc' could not be determined from YAML config or source. "
            "Please specify 'rc' explicitly in the LRCoulomb config kwargs."
        )

    # --- Detect Dispersion ---
    if is_jit_model:
        # Check if model has dftd3/d3bj modules
        has_d3_module = any(name in ("dftd3", "d3bj") for name, _ in named_children_rec(source))

        # Check YAML to determine if it's D3TS (not externalizable)
        # D3TS uses NN-predicted C6/alpha and must stay embedded
        is_d3ts = False
        if has_d3_module:
            for key in ["dftd3", "d3bj"]:
                if key in outputs:
                    d3_class = outputs[key].get("class", "")
                    if "D3TS" in d3_class:
                        is_d3ts = True
                        break

        # Only externalize if NOT D3TS (DFTD3/D3BJ with tabulated params can be externalized)
        needs_dispersion = has_d3_module and not is_d3ts

        if needs_dispersion:
            # Try to extract from JIT model first
            d3_params = extract_d3_params(source)
            # If extraction failed or returned zeros, try YAML config
            if d3_params is None or (
                d3_params.get("s8") == 0.0 and d3_params.get("a1") == 0.0 and d3_params.get("a2") == 0.0
            ):
                for key in ["dftd3", "d3bj"]:
                    if key in outputs:
                        d3_config = outputs[key]
                        d3_kwargs = d3_config.get("kwargs", {})
                        d3_params = {
                            "s8": d3_kwargs.get("s8", 0.0),
                            "a1": d3_kwargs.get("a1", 0.0),
                            "a2": d3_kwargs.get("a2", 0.0),
                            "s6": d3_kwargs.get("s6", 1.0),
                        }
                        break
        else:
            d3_params = None
    else:
        # State dict path - check YAML config
        needs_dispersion = False
        d3_params = None
        for key in ["dftd3", "d3bj"]:
            if key in outputs:
                d3_config = outputs[key]
                # Check if it's D3TS (must stay embedded, not externalizable)
                module_class = d3_config.get("class", "")
                if "D3TS" in module_class:
                    # D3TS uses NN-predicted C6/alpha, must stay embedded
                    needs_dispersion = False
                    d3_params = None
                    break
                # DFTD3/D3BJ with tabulated params can be externalized
                needs_dispersion = True
                d3_kwargs = d3_config.get("kwargs", {})
                d3_params = {
                    "s8": d3_kwargs.get("s8", 0.0),
                    "a1": d3_kwargs.get("a1", 0.0),
                    "a2": d3_kwargs.get("a2", 0.0),
                    "s6": d3_kwargs.get("s6", 1.0),
                }
                break

    # Validate: D3TS + DFTD3/D3BJ is invalid (would cause double dispersion)
    has_d3ts_model = has_d3ts(source) if is_jit_model else False
    if needs_dispersion and (has_d3ts_model or has_d3ts_in_config(config)):
        raise ValueError(
            "Model has both D3TS (learned) and DFTD3/D3BJ (tabulated) dispersion. "
            "D3TS uses learned parameters and must stay embedded, while DFTD3/D3BJ "
            "would be externalized. This configuration leads to double dispersion "
            "correction. Remove either D3TS or DFTD3/D3BJ from the model."
        )

    # --- Rebuild outputs dict ---
    new_outputs = {}
    for key, value in outputs.items():
        if key == "lrcoulomb":
            pass  # Will be added externally by calculator
        elif key in ["dftd3", "d3bj"]:
            # Check if it's D3TS (must stay embedded)
            module_class = value.get("class", "")
            if "D3TS" in module_class:
                new_outputs[key] = value  # Keep D3TS embedded
            else:
                pass  # Remove DFTD3/D3BJ for externalization
        else:
            new_outputs[key] = value

    # Strip ptfile from DispParam configs (buffer is in state dict)
    for _key, value in new_outputs.items():
        if isinstance(value, dict):
            module_class = value.get("class", "")
            if "DispParam" in module_class:
                kwargs = value.get("kwargs", {})
                if "ptfile" in kwargs:
                    kwargs.pop("ptfile")

    # Add SRCoulomb if LRCoulomb was present
    if has_coulomb:
        new_outputs["srcoulomb"] = {
            "class": "aimnet.modules.SRCoulomb",
            "kwargs": {
                "rc": coulomb_sr_rc,
                "key_in": "charges",
                "key_out": "energy",
                "envelope": coulomb_sr_envelope,
            },
        }

    config["kwargs"]["outputs"] = new_outputs
    coulomb_mode = "sr_embedded" if has_coulomb else "none"

    return (
        config,
        coulomb_mode,
        needs_dispersion,
        d3_params,
        coulomb_sr_rc,
        coulomb_sr_envelope if coulomb_sr_envelope else "exp",
    )


# --- Model loading ---


def load_v1_model(
    jpt_path: str,
    yaml_config_path: str,
    output_path: str | None = None,
    verbose: bool = True,
) -> tuple[nn.Module, dict]:
    """Load legacy JIT model (v1) and convert to v2 format.

    This is the primary entry point for loading legacy models.

    Parameters
    ----------
    jpt_path : str
        Path to the input JIT-compiled model file (.jpt).
    yaml_config_path : str
        Path to the model YAML configuration file.
    output_path : str, optional
        If provided, save the converted model to this path.
    verbose : bool
        Whether to print progress messages.

    Returns
    -------
    model : nn.Module
        The loaded model in v2 format.
    metadata : dict
        Model metadata dictionary with keys:
        - format_version: 2
        - cutoff: float
        - needs_coulomb: bool
        - needs_dispersion: bool
        - coulomb_mode: str
        - coulomb_sr_rc: float | None
        - coulomb_sr_envelope: str | None
        - d3_params: dict | None
        - implemented_species: list[int]

    Example
    -------
    >>> from aimnet.models.utils import load_v1_model
    >>> model, metadata = load_v1_model("model.jpt", "config.yaml")
    >>> print(metadata["format_version"])  # 2

    Warnings
    --------
    UserWarning
        If D3 parameter extraction produces zero values.
    """
    import copy

    import torch
    import yaml

    from aimnet.config import build_module

    # Load YAML config
    with open(yaml_config_path, encoding="utf-8") as f:
        model_config = yaml.safe_load(f)

    # Load JIT model
    if verbose:
        print(f"Loading JIT model from {jpt_path}")
    jit_model = torch.jit.load(jpt_path, map_location="cpu")

    # Extract metadata from JIT
    cutoff = float(jit_model.cutoff)
    implemented_species = extract_species(jit_model)

    # Strip LR modules from YAML and add SRCoulomb
    core_config, coulomb_mode, needs_dispersion, d3_params, coulomb_sr_rc, coulomb_sr_envelope = (
        strip_lr_modules_from_yaml(model_config, jit_model)
    )

    # Inform user about dispersion handling
    if verbose:
        if needs_dispersion:
            # External dispersion (DFTD3/D3BJ with tabulated params)
            if d3_params is None:
                print("WARNING: Model has DFTD3 module but D3 params extraction failed!")
            elif d3_params.get("s8") == 0.0 and d3_params.get("a1") == 0.0 and d3_params.get("a2") == 0.0:
                print("WARNING: D3 params appear to be all zeros - extraction may have failed!")
                print(f"  Extracted: {d3_params}")
            else:
                print(
                    f"  D3 parameters: s6={d3_params['s6']}, s8={d3_params['s8']}, "
                    f"a1={d3_params['a1']}, a2={d3_params['a2']}"
                )
        else:
            # Check if D3TS is embedded
            outputs = model_config.get("kwargs", {}).get("outputs", {})
            has_d3ts = any("D3TS" in outputs.get(k, {}).get("class", "") for k in ["dftd3", "d3bj", "d3ts"])
            if has_d3ts:
                print("  D3TS dispersion kept embedded (uses NN-predicted C6/alpha)")

    # Detect if model has any embedded LR modules that need nbmat_lr
    outputs = model_config.get("kwargs", {}).get("outputs", {})
    has_embedded_lr = False

    # Check for embedded D3TS
    has_d3ts = any("D3TS" in outputs.get(k, {}).get("class", "") for k in ["dftd3", "d3bj", "d3ts"])
    if has_d3ts:
        has_embedded_lr = True

    # Check for embedded SRCoulomb (model had LRCoulomb before conversion)
    if coulomb_mode == "sr_embedded":
        has_embedded_lr = True

    # Convert config to YAML string
    core_yaml_str = yaml.dump(core_config, default_flow_style=False, sort_keys=False)

    # Build model from modified config
    if verbose:
        print("Building model from YAML config...")
    core_model = build_module(copy.deepcopy(core_config))

    # Load weights from JIT model
    jit_sd = jit_model.state_dict()
    load_result = core_model.load_state_dict(jit_sd, strict=False)

    # Validate keys
    real_missing, real_unexpected = validate_state_dict_keys(load_result.missing_keys, load_result.unexpected_keys)
    if real_missing:
        print(f"WARNING: Unexpected missing keys: {real_missing}")
    if real_unexpected:
        print(f"WARNING: Unexpected extra keys: {real_unexpected}")
    if not real_missing and not real_unexpected and verbose:
        print("Loaded weights successfully")

    # Convert atomic_shift to float64 to preserve SAE precision
    if hasattr(core_model, "outputs") and hasattr(core_model.outputs, "atomic_shift"):
        core_model.outputs.atomic_shift.double()
        atomic_shift_key = "outputs.atomic_shift.shifts.weight"
        if atomic_shift_key in jit_sd:
            core_model.outputs.atomic_shift.shifts.weight.data.copy_(jit_sd[atomic_shift_key])
            if verbose:
                print("  Atomic shift converted to float64")

    core_model.eval()

    # Create metadata
    needs_coulomb = coulomb_mode == "sr_embedded"
    metadata = {
        "format_version": 2,
        "model_yaml": core_yaml_str,
        "cutoff": cutoff,
        "needs_coulomb": needs_coulomb,
        "needs_dispersion": needs_dispersion,
        "coulomb_mode": coulomb_mode,
        "coulomb_sr_rc": coulomb_sr_rc if needs_coulomb else None,
        "coulomb_sr_envelope": coulomb_sr_envelope if needs_coulomb else None,
        "d3_params": d3_params if needs_dispersion else None,
        "has_embedded_lr": has_embedded_lr,
        "implemented_species": implemented_species,
    }

    # Save if output path provided
    if output_path is not None:
        save_data = {**metadata, "state_dict": core_model.state_dict()}
        torch.save(save_data, output_path)
        if verbose:
            print(f"\nSaved model to {output_path}")
            print(f"  cutoff: {cutoff:.3f}")
            print(f"  needs_coulomb: {needs_coulomb}")
            print(f"  needs_dispersion: {needs_dispersion}")
            print(f"  has_embedded_lr: {has_embedded_lr}")

    return core_model, metadata
