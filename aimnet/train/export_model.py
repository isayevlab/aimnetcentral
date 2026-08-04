#!/usr/bin/env python3
"""Export trained model to distributable state dict format.

This script creates a self-contained .pt file from training artifacts:
- Raw PyTorch weights (.pt)
- Self-atomic energies (.sae)
- Model YAML configuration

The output file contains:
- model_yaml: Core model config (without LRCoulomb/DFTD3, with SRCoulomb if needed)
- cutoff: Model cutoff
- needs_coulomb: Whether calculator should add external Coulomb
- needs_dispersion: Whether calculator should add external DFTD3
- coulomb_mode: "sr_embedded" | "none" (describes what's in the model)
- coulomb_sr_rc: Coulomb short-range cutoff (optional, if coulomb_mode="sr_embedded")
- coulomb_sr_envelope: Envelope function ("exp" or "cosine", optional)
- d3_params: D3 parameters {s8, a1, a2, s6} (optional, if needs_dispersion=True)
- has_embedded_lr: Whether model has embedded LR modules (D3TS, SRCoulomb) needing nbmat_lr
- has_embedded_d3ts: Whether the model contains embedded learned D3TS dispersion
- implemented_species: Parametrized atomic numbers
- state_dict: Model weights with SAE baked into atomic_shift (float64)
"""

import copy
import os
import stat
import tempfile
from collections.abc import Collection, Mapping
from pathlib import Path

import click
import torch
import yaml
from torch import nn

from aimnet.config import build_module, load_yaml
from aimnet.models.artifact_validation import (
    resolve_model_import_policy,
    validate_model_yaml,
    validate_v2_artifact_with_policy,
)
from aimnet.models.utils import strip_lr_modules_from_yaml, validate_state_dict_keys


def load_sae(sae_file: str) -> dict[int, float]:
    """Load SAE file (YAML-like format: atomic_number: energy)."""
    sae = load_yaml(sae_file)
    if not isinstance(sae, dict):
        raise TypeError("SAE file must contain a dictionary.")
    return {int(k): float(v) for k, v in sae.items()}


def bake_sae_into_model(model: nn.Module, sae: dict[int, float]) -> nn.Module:
    """Add SAE values to atomic_shift.shifts.weight (converted to float64)."""
    # Disable gradients before in-place operation
    for p in model.parameters():
        p.requires_grad_(False)
    model.outputs.atomic_shift.double()  # type: ignore
    for k, v in sae.items():
        model.outputs.atomic_shift.shifts.weight[k] += v  # type: ignore
    return model


def extract_cutoff(model: nn.Module) -> float:
    """Extract cutoff from model's AEV module."""
    return float(model.aev.rc_s.item())  # type: ignore


def get_implemented_species(sae: dict[int, float]) -> list[int]:
    """Get list of implemented species from SAE."""
    return sorted(sae.keys())


def mask_not_implemented_species(model: nn.Module, species: list[int]) -> nn.Module:
    """Set NaN for species not in the SAE."""
    weight = model.afv.weight  # type: ignore
    for i in range(1, weight.shape[0]):  # type: ignore
        if i not in species:
            weight[i, :] = torch.nan  # type: ignore
    return model


def _save_artifact_atomically(artifact: dict[str, object], output: str | Path) -> None:
    """Save an artifact without replacing an existing destination on failure."""
    destination = Path(output)
    destination_mode = stat.S_IMODE(destination.stat().st_mode) if destination.exists() else None
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)

    try:
        if destination_mode is not None:
            if hasattr(os, "fchmod"):
                os.fchmod(fd, destination_mode)
            else:
                temporary.chmod(destination_mode)
        stream = os.fdopen(fd, "wb")
        fd = None
        with stream:
            torch.save(artifact, stream)
        os.replace(temporary, destination)
    except BaseException:
        if fd is not None:
            os.close(fd)
        temporary.unlink(missing_ok=True)
        raise


@click.command()
@click.argument("weights", type=click.Path(exists=True))
@click.argument("output", type=str)
@click.option("--model", "-m", type=click.Path(exists=True), required=True, help="Path to model definition YAML file")
@click.option("--sae", "-s", type=click.Path(exists=True), required=True, help="Path to the SAE YAML file")
@click.option(
    "--needs-coulomb/--no-coulomb",
    default=None,
    help="Override Coulomb detection. --no-coulomb is invalid for detected sr_embedded Coulomb.",
)
@click.option(
    "--needs-dispersion/--no-dispersion",
    default=None,
    help="Override dispersion detection. Enabled dispersion requires complete D3 parameters.",
)
@click.option(
    "--model-import-path",
    "model_import_paths",
    multiple=True,
    help="Explicitly trust an exact constructor path or namespace for this local export.",
)
def export_model(
    weights: str,
    output: str,
    model: str,
    sae: str,
    needs_coulomb: bool | None,
    needs_dispersion: bool | None,
    model_import_paths: Collection[str] = (),
):
    """Export trained model to distributable state dict format.

    weights: Path to the raw PyTorch weights file (.pt).
    output: Path to the output .pt file.

    Example:
        aimnet export weights.pt model.pt --model config.yaml --sae model.sae
    """
    # Load model YAML
    print(f"Loading config from {model}")
    with open(model, encoding="utf-8") as f:
        model_config = yaml.safe_load(f)

    # Load SAE
    print(f"Loading SAE from {sae}")
    sae_dict = load_sae(sae)
    implemented_species = get_implemented_species(sae_dict)

    # Load source state dict
    print(f"Loading weights from {weights}")
    source_sd = torch.load(weights, map_location="cpu", weights_only=True)

    # Strip LR modules and detect flags
    core_config, coulomb_mode, needs_dispersion_auto, d3_params, coulomb_sr_rc, coulomb_sr_envelope, disp_ptfile = (
        strip_lr_modules_from_yaml(model_config, source_sd)
    )

    # Serialize YAML BEFORE building module (build_module mutates the dict)
    core_yaml_str = yaml.dump(core_config, default_flow_style=False, sort_keys=False)
    explicit_import_paths = tuple(model_import_paths) or None
    validated_core_config = validate_model_yaml(
        core_yaml_str,
        model_import_paths=explicit_import_paths,
    )
    import_policy = resolve_model_import_policy(explicit_import_paths, "extend")

    # Resolve CLI overrides without permitting an invalid exported runtime.
    auto_needs_coulomb = coulomb_mode == "sr_embedded"
    auto_needs_dispersion = needs_dispersion_auto
    final_needs_coulomb = needs_coulomb if needs_coulomb is not None else auto_needs_coulomb
    final_needs_dispersion = needs_dispersion if needs_dispersion is not None else auto_needs_dispersion
    if coulomb_mode == "sr_embedded" and not final_needs_coulomb:
        raise click.ClickException(
            "--no-coulomb cannot be used with detected sr_embedded Coulomb: "
            "the exported model requires external Coulomb to restore the full interaction."
        )
    if final_needs_dispersion:
        missing_d3 = {"s8", "a1", "a2"}
        if isinstance(d3_params, Mapping):
            missing_d3 -= set(d3_params)
        if missing_d3:
            missing = ", ".join(sorted(missing_d3))
            raise click.ClickException(
                "Cannot enable dispersion without complete D3 parameters "
                f"(missing: {missing}). Provide s8, a1, and a2 or use --no-dispersion."
            )

    # Build model from modified config
    print("Building model...")
    core_model = build_module(
        copy.deepcopy(validated_core_config),
        allow_file_references=False,
        import_authorizer=import_policy.require_allowed,
    )
    if not isinstance(core_model, nn.Module):
        raise TypeError("Built module is not an nn.Module")

    # Load weights with strict=False (modules may differ)
    load_result = core_model.load_state_dict(source_sd, strict=False)

    # Check for unexpected missing/extra keys
    real_missing, real_unexpected = validate_state_dict_keys(load_result.missing_keys, load_result.unexpected_keys)
    if real_missing:
        print(f"WARNING: Unexpected missing keys: {real_missing}")
    if real_unexpected:
        print(f"WARNING: Unexpected extra keys in source: {real_unexpected}")
    if not real_missing and not real_unexpected:
        print("Loaded weights successfully")

    # Load dispersion parameters from ptfile and inject into model
    # (raw training weights don't contain disp_param0 buffer)
    if disp_ptfile is not None:
        disp_params = torch.load(disp_ptfile, map_location="cpu", weights_only=True)
        for _name, module in core_model.named_modules():
            if hasattr(module, "disp_param0"):
                # Resize buffer if needed (ptfile may have different shape than placeholder)
                if module.disp_param0.shape != disp_params.shape:
                    module.disp_param0 = torch.zeros_like(disp_params)
                module.disp_param0.copy_(disp_params)
                print(f"Loaded disp_param0 from {disp_ptfile}")
                break

    # Bake SAE into atomic_shift (float64)
    print("Baking SAE into atomic_shift...")
    core_model = bake_sae_into_model(core_model, sae_dict)

    # Mask not-implemented species
    core_model = mask_not_implemented_species(core_model, implemented_species)

    # Extract cutoff
    cutoff = extract_cutoff(core_model)

    # Set model to eval mode
    core_model.eval()

    # Detect if model has any embedded LR modules that need nbmat_lr
    outputs = model_config.get("kwargs", {}).get("outputs", {})
    has_embedded_lr = False

    # Check for embedded D3TS (uses NN-predicted C6/alpha, must stay embedded)
    has_d3ts = any("D3TS" in outputs.get(k, {}).get("class", "") for k in ["dftd3", "d3bj", "d3ts"])
    if has_d3ts:
        has_embedded_lr = True

    # Check for embedded SRCoulomb (model had LRCoulomb before conversion)
    if coulomb_mode == "sr_embedded":
        has_embedded_lr = True

    # Create new format dict
    new_format = {
        "format_version": 2,  # v2 = new .pt format (v1 = legacy .jpt)
        "model_yaml": core_yaml_str,
        "cutoff": cutoff,
        "needs_coulomb": final_needs_coulomb,
        "needs_dispersion": final_needs_dispersion,
        "coulomb_mode": coulomb_mode,
        "coulomb_sr_rc": coulomb_sr_rc if coulomb_mode == "sr_embedded" else None,
        "coulomb_sr_envelope": coulomb_sr_envelope if coulomb_mode == "sr_embedded" else None,
        "d3_params": d3_params if final_needs_dispersion else None,
        "has_embedded_lr": has_embedded_lr,
        "has_embedded_d3ts": has_d3ts,
        "implemented_species": implemented_species,
        "state_dict": core_model.state_dict(),
    }

    validate_v2_artifact_with_policy(new_format, import_policy, validation="canonical")
    _save_artifact_atomically(new_format, output)
    print(f"\nSaved model to {output}")
    print(f"  cutoff: {cutoff}")
    print(f"  needs_coulomb: {final_needs_coulomb}")
    print(f"  needs_dispersion: {final_needs_dispersion}")
    print(f"  coulomb_mode: {coulomb_mode}")
    if final_needs_coulomb:
        print(f"  coulomb_sr_rc: {coulomb_sr_rc}")
        print(f"  coulomb_sr_envelope: {coulomb_sr_envelope}")
    if final_needs_dispersion:
        print(f"  d3_params: {d3_params}")
    print(f"  has_embedded_lr: {has_embedded_lr}")
    print(f"  has_embedded_d3ts: {has_d3ts}")
    print(f"  implemented_species: {implemented_species}")


if __name__ == "__main__":
    export_model()
