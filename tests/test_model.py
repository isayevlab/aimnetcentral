import os

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


@pytest.mark.ase
def test_aimnet2():
    pytest.importorskip("ase", reason="ASE not installed. Install with: pip install aimnet[ase]")
    import ase.io

    model = build_model(aimnet2_d3_def)
    model.outputs.atomic_shift.shifts.double()
    model_from_zoo = torch.jit.load(get_model_path("aimnet2"))
    model.load_state_dict(model_from_zoo.state_dict(), strict=False)
    model = Forces(model)
    atoms = ase.io.read(os.path.join(os.path.dirname(__file__), "data", "caffeine.xyz"), format="extxyz")
    ref_e = atoms.get_potential_energy()
    ref_f = atoms.get_forces()
    ref_q = atoms.get_charges()
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


class TestTorchScript:
    """Tests for TorchScript compilation."""

    @pytest.mark.parametrize("model_def", model_defs)
    def test_torchscript_fresh_model(self, model_def):
        """Compile freshly built model with torch.jit.script."""
        model = build_model(model_def)
        scripted = torch.jit.script(model)
        assert scripted is not None

    @pytest.mark.parametrize("model_def", model_defs)
    def test_torchscript_inference_matches(self, model_def):
        """Verify scripted model output matches eager mode."""
        model = build_model(model_def)
        scripted = torch.jit.script(model)

        # Create simple input
        _in = {
            "coord": torch.tensor([[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]]),
            "numbers": torch.tensor([[8, 1, 1]]),
            "charge": torch.tensor([0.0]),
        }

        # Run both models
        with torch.no_grad():
            eager_out = model(_in.copy())
            scripted_out = scripted(_in.copy())

        # Results should match
        np.testing.assert_allclose(
            eager_out["energy"].numpy(),
            scripted_out["energy"].numpy(),
            atol=1e-5,
        )

    def test_torchscript_save_load(self):
        """Test that scripted model can be saved and loaded."""
        import tempfile

        model = build_model(aimnet2_d3_def)
        scripted = torch.jit.script(model)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=True) as f:
            torch.jit.save(scripted, f.name)
            loaded = torch.jit.load(f.name)

        # Create simple input
        _in = {
            "coord": torch.tensor([[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]]),
            "numbers": torch.tensor([[8, 1, 1]]),
            "charge": torch.tensor([0.0]),
        }

        with torch.no_grad():
            original_out = scripted(_in.copy())
            loaded_out = loaded(_in.copy())

        np.testing.assert_allclose(
            original_out["energy"].numpy(),
            loaded_out["energy"].numpy(),
            atol=1e-6,
        )
