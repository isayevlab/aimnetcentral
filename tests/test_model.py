import os

import ase.io
import numpy as np
import pytest
import torch
from torch import nn

from aimnet.calculators.model_registry import get_model_path
from aimnet.config import build_module
from aimnet.modules.core import Forces

aimnet2_def = os.path.join(os.path.dirname(__file__), "..", "aimnet", "models", "aimnet2.yaml")
aimnet2_d3_def = os.path.join(os.path.dirname(__file__), "..", "aimnet", "models", "aimnet2_dftd3_wb97m.yaml")
model_defs = [aimnet2_d3_def]


def build_model(model_def):
    assert os.path.exists(model_def), f"Model definition file not found: {model_def}."
    model = build_module(model_def)
    assert isinstance(model, nn.Module), "The model is not an instance of AIMNet2."
    return model


def jit_compile(model):
    return torch.jit.script(model)


@pytest.mark.parametrize("model_def", model_defs)
def test_model_from_yaml(model_def):
    build_model(model_def)


@pytest.mark.parametrize("model_def", model_defs)
def test_model_compile(model_def):
    model = build_model(model_def)
    jit_compile(model)


def test_aimnet2():
    model = build_model(aimnet2_d3_def)
    model.outputs.atomic_shift.shifts.double()
    model_from_zoo = torch.jit.load(get_model_path("aimnet2"))
    model.load_state_dict(model_from_zoo.state_dict(), strict=False)
    model = Forces(model)
    atoms = ase.io.read(os.path.join(os.path.dirname(__file__), "data", "caffeine.xyz"))
    ref_e = atoms.get_total_energy()  # type: ignore
    ref_f = atoms.get_forces()  # type: ignore
    ref_q = atoms.get_charges()  # type: ignore
    _in = {
        "coord": torch.as_tensor(atoms.get_positions()).unsqueeze(0),  # type: ignore
        "numbers": torch.as_tensor(atoms.get_atomic_numbers()).unsqueeze(0),  # type: ignore
        "charge": torch.tensor([0.0]),
    }
    _out = model(_in)
    e = _out["energy"].item()
    f = _out["forces"].squeeze(0).detach().cpu().numpy()
    q = _out["charges"].squeeze(0).detach().cpu().numpy()

    np.testing.assert_allclose(e, ref_e, atol=1e-5)
    np.testing.assert_allclose(f, ref_f, atol=1e-4)
    np.testing.assert_allclose(q, ref_q, atol=1e-3)
