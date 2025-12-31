import os

import click
import torch
from torch import nn

from aimnet.config import build_module, load_yaml


def set_eval(model: nn.Module) -> torch.nn.Module:
    for p in model.parameters():
        p.requires_grad_(False)
    return model.eval()


def add_cutoff(model: nn.Module, cutoff: float | None = 5.0, cutoff_lr: float | None = float("inf")) -> nn.Module:
    if cutoff is None:
        cutoff = max(v.item() for k, v in model.state_dict().items() if k.endswith("aev.rc_s"))
    model.cutoff = cutoff  # type: ignore[assignment]
    if cutoff_lr is not None:
        model.cutoff_lr = cutoff_lr  # type: ignore[assignment]
    return model


def add_sae_to_shifts(model: nn.Module, sae_file: str) -> nn.Module:
    sae = load_yaml(sae_file)
    if not isinstance(sae, dict):
        raise TypeError("SAE file must contain a dictionary.")
    model.outputs.atomic_shift.double()  # type: ignore
    for k, v in sae.items():
        model.outputs.atomic_shift.shifts.weight[k] += v  # type: ignore
    return model


def mask_not_implemented_species(model: nn.Module, species: list[int]) -> nn.Module:
    weight = model.afv.weight  # type: ignore
    for i in range(1, weight.shape[0]):  # type: ignore
        if i not in species:
            weight[i, :] = torch.nan  # type: ignore
    return model


_default_aimnet2_config = os.path.join(os.path.dirname(__file__), "..", "models", "aimnet2.yaml")


@click.command(short_help="Compile PyTorch model to TorchScript.")
@click.argument("pt", type=str)  # , help='Path to the input PyTorch weights file.')
@click.argument("jpt", type=str)  # , help='Path to the output TorchScript file.')
@click.option("--model", type=str, default=_default_aimnet2_config, help="Path to model definition YAML file")
@click.option("--sae", type=str, default=None, help="Path to the energy shift YAML file.")
@click.option("--species", type=str, default=None, help="Comma-separated list of parametrized atomic numbers.")
@click.option("--no-lr", is_flag=True, help="Do not add LR cutoff for model")
def jitcompile(model: str, pt: str, jpt: str, sae=None, species=None, no_lr=False):  # type: ignore
    """Build model from YAML config, load weight from PT file and write JIT-compiled JPT file.
    Plus some modifications to work with aimnet2calc.
    """
    model: nn.Module = build_module(model)  # type: ignore[annotation-unchecked]
    model = set_eval(model)
    cutoff_lr = None if no_lr else float("inf")
    model = add_cutoff(model, cutoff_lr=cutoff_lr)
    sd = torch.load(pt, map_location="cpu", weights_only=True)
    print(model.load_state_dict(sd, strict=False))
    if sae:
        model = add_sae_to_shifts(model, sae)
    numbers = None
    if species:
        numbers = list(map(int, species.split(",")))
    elif sae:
        numbers = list(load_yaml(sae).keys())  # type: ignore[union-attr]
    if numbers:
        model = mask_not_implemented_species(model, numbers)  # type: ignore[call-arg]
        model.register_buffer("implemented_species", torch.tensor(numbers, dtype=torch.int64))
    model_jit = torch.jit.script(model)
    model_jit.save(jpt)


if __name__ == "__main__":
    jitcompile()
