import os

import numpy as np
import pytest
import torch
from conftest import add_dftd3_keys, temp_file
from torch import nn

from aimnet.calculators.model_registry import get_model_path
from aimnet.config import build_module
from aimnet.models.base import load_model
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
    """Test building model from YAML and loading weights from zoo model.

    The YAML config (aimnet2_dftd3_wb97m.yaml) builds a model with embedded
    D3TS dispersion and SRCoulomb. We load weights from the zoo model and
    verify inference matches reference values.

    Note: We do NOT apply external modules here because the YAML-built model
    already has D3TS and SRCoulomb embedded in its architecture.
    """
    pytest.importorskip("ase", reason="ASE not installed. Install with: pip install aimnet[ase]")
    import ase.io

    model = build_model(aimnet2_d3_def)
    model.outputs.atomic_shift.shifts.double()

    # Use load_model() instead of torch.jit.load() for v2 format support
    model_from_zoo, _ = load_model(get_model_path("aimnet2"), device="cpu")
    model.load_state_dict(model_from_zoo.state_dict(), strict=False)
    model = Forces(model)

    # Load caffeine with stored reference values (energy, forces, charges in extxyz)
    atoms = ase.io.read(os.path.join(os.path.dirname(__file__), "data", "caffeine.xyz"), format="extxyz")
    ref_e = atoms.get_potential_energy()
    ref_f = atoms.get_forces()
    # ASE versions parse charge key differently: "initial_charges" or "charge"
    ref_q = atoms.arrays.get("initial_charges", atoms.arrays.get("charge"))

    _in = {
        "coord": torch.as_tensor(atoms.get_positions()).unsqueeze(0),  # type: ignore
        "numbers": torch.as_tensor(atoms.get_atomic_numbers()).unsqueeze(0),  # type: ignore
        "charge": torch.tensor([0.0]),
    }
    _in = add_dftd3_keys(_in)
    _out = model(_in)

    e = _out["energy"].item()
    f = _out["forces"].squeeze(0).detach().cpu().numpy()
    q = _out["charges"].squeeze(0).detach().cpu().numpy()

    # Compare against stored reference values from caffeine.xyz
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
        _in = add_dftd3_keys(_in)

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
        model = build_model(aimnet2_d3_def)
        scripted = torch.jit.script(model)

        with temp_file(suffix=".pt") as path:
            torch.jit.save(scripted, str(path))
            loaded = torch.jit.load(str(path))

        # Create simple input
        _in = {
            "coord": torch.tensor([[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]]),
            "numbers": torch.tensor([[8, 1, 1]]),
            "charge": torch.tensor([0.0]),
        }
        _in = add_dftd3_keys(_in)

        with torch.no_grad():
            original_out = scripted(_in.copy())
            loaded_out = loaded(_in.copy())

        np.testing.assert_allclose(
            original_out["energy"].numpy(),
            loaded_out["energy"].numpy(),
            atol=1e-6,
        )


class TestFromFile:
    """Tests for load_model() model loading."""

    def test_from_file_returns_module(self):
        """Test that loading model returns nn.Module."""
        p = get_model_path("aimnet2")
        model, _metadata = load_model(p, device="cpu")

        # Should return an nn.Module
        assert isinstance(model, nn.Module)

    def test_from_file_metadata(self):
        """Test that model returns correct metadata."""
        p = get_model_path("aimnet2")
        _, metadata = load_model(p, device="cpu")

        # Check metadata keys exist
        assert "format_version" in metadata
        assert "cutoff" in metadata
        assert "needs_coulomb" in metadata
        assert "needs_dispersion" in metadata
        assert "coulomb_mode" in metadata
        assert "d3_params" in metadata
        assert "implemented_species" in metadata

        # V2 models are format version 2
        assert metadata["format_version"] == 2

        # V2 models have external modules
        assert metadata["needs_coulomb"] is True
        assert metadata["needs_dispersion"] is True

        # V2 models have SR Coulomb embedded, need external LR
        assert metadata["coulomb_mode"] == "sr_embedded"

        # Cutoff should be extracted from model
        assert metadata["cutoff"] == 5.0

    def test_from_file_legacy_jit_extracts_d3_params(self):
        """Test that D3 parameters are extracted from legacy JIT model."""
        p = get_model_path("aimnet2")
        _, metadata = load_model(p, device="cpu")

        # Should have D3 params (model has embedded DFTD3)
        assert metadata["d3_params"] is not None
        d3 = metadata["d3_params"]
        assert "s6" in d3
        assert "s8" in d3
        assert "a1" in d3
        assert "a2" in d3

        # wB97M-D3 parameters
        assert d3["s6"] == 1.0
        assert abs(d3["s8"] - 0.3908) < 0.001
        assert abs(d3["a1"] - 0.566) < 0.001
        assert abs(d3["a2"] - 3.128) < 0.001

    def test_from_file_legacy_jit_extracts_species(self):
        """Test that implemented species are extracted from legacy JIT model."""
        p = get_model_path("aimnet2")
        _, metadata = load_model(p, device="cpu")

        species = metadata["implemented_species"]
        assert isinstance(species, list)
        assert len(species) > 0
        # AIMNet2 supports H, C, N, O, etc.
        assert 1 in species  # H
        assert 6 in species  # C
        assert 7 in species  # N
        assert 8 in species  # O

    def test_from_file_model_inference(self):
        """Test that load_model returns working model with correct metadata.

        Verifies that:
        - load_model() correctly loads the model and returns proper metadata
        - The model is callable and produces valid output
        - Metadata correctly indicates external modules are needed
        """
        p = get_model_path("aimnet2")
        model, metadata = load_model(p, device="cpu")

        # Verify model is callable and produces output
        _in = {
            "coord": torch.tensor([[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0]]]),
            "numbers": torch.tensor([[8, 1]]),
            "charge": torch.tensor([0.0]),
        }
        _in = add_dftd3_keys(_in)

        with torch.no_grad():
            _out = model(_in)

        # Verify output contains expected keys
        assert "energy" in _out
        assert "charges" in _out
        assert torch.isfinite(_out["energy"]).all()

        # Verify metadata indicates correct external modules for v2 format
        assert metadata["needs_coulomb"] is True
        assert metadata["needs_dispersion"] is True
        assert metadata["coulomb_mode"] == "sr_embedded"

    def test_from_file_invalid_format_raises(self):
        """Test that invalid file format raises ValueError."""
        # Create a file with invalid format (just a tensor, not a model)
        with temp_file(suffix=".pt") as path:
            torch.save({"invalid": "data"}, str(path))
            with pytest.raises(ValueError, match="Unknown model format"):
                load_model(str(path))


class TestNewFormat:
    """Tests for new format model loading and external module attachment."""

    @pytest.fixture
    def new_format_model(self, tmp_path):
        """Create a minimal new-format model file for testing."""
        import yaml

        # Build model from YAML config
        model = build_model(aimnet2_d3_def)

        # Get state dict and modify for new format (strip LR modules would happen in export)
        state_dict = model.state_dict()

        # Create minimal config without LRCoulomb/DFTD3 (simulating exported model)
        with open(aimnet2_d3_def, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Serialize config as YAML string
        model_yaml = yaml.dump(config, default_flow_style=False, sort_keys=False)

        # Create new format file with needs_coulomb=True, needs_dispersion=True
        new_format = {
            "format_version": 2,  # v2 = new .pt format
            "model_yaml": model_yaml,
            "cutoff": 5.0,
            "needs_coulomb": True,
            "needs_dispersion": True,
            "coulomb_mode": "sr_embedded",
            "coulomb_sr_rc": 4.6,
            "coulomb_sr_envelope": "exp",
            "d3_params": {"s6": 1.0, "s8": 0.3908, "a1": 0.566, "a2": 3.128},
            "implemented_species": [1, 6, 7, 8, 9, 16, 17],
            "state_dict": state_dict,
        }

        model_path = tmp_path / "test_model.pt"
        torch.save(new_format, model_path)
        return str(model_path)

    def test_from_file_new_format_loads(self, new_format_model):
        """Test that new format model loads correctly."""
        model, _metadata = load_model(new_format_model, device="cpu")

        # Should return an nn.Module (not ScriptModule)
        assert isinstance(model, nn.Module)
        assert not isinstance(model, torch.jit.ScriptModule)

    def test_from_file_new_format_metadata(self, new_format_model):
        """Test that new format model returns correct metadata."""
        _, metadata = load_model(new_format_model, device="cpu")

        # Check metadata keys
        assert metadata["format_version"] == 2  # New format is v2
        assert metadata["cutoff"] == 5.0
        assert metadata["needs_coulomb"] is True
        assert metadata["needs_dispersion"] is True
        assert metadata["coulomb_mode"] == "sr_embedded"
        assert metadata["coulomb_sr_rc"] == 4.6
        assert metadata["coulomb_sr_envelope"] == "exp"
        assert metadata["d3_params"] is not None
        assert metadata["implemented_species"] == [1, 6, 7, 8, 9, 16, 17]

    def test_from_file_new_format_d3_params(self, new_format_model):
        """Test that D3 parameters are correctly returned in metadata."""
        _, metadata = load_model(new_format_model, device="cpu")

        d3 = metadata["d3_params"]
        assert d3["s6"] == 1.0
        assert abs(d3["s8"] - 0.3908) < 0.001
        assert abs(d3["a1"] - 0.566) < 0.001
        assert abs(d3["a2"] - 3.128) < 0.001

    def test_calculator_attaches_external_coulomb(self, new_format_model):
        """Test that calculator attaches external LRCoulomb for new format models."""
        from aimnet.calculators import AIMNet2Calculator

        calc = AIMNet2Calculator(new_format_model)

        # Should have external Coulomb attached
        assert calc.external_coulomb is not None
        assert calc.external_coulomb.method == "simple"
        assert calc.external_coulomb.subtract_sr is False

    def test_calculator_attaches_external_dftd3(self, new_format_model):
        """Test that calculator attaches external DFTD3 for new format models."""
        from aimnet.calculators import AIMNet2Calculator

        calc = AIMNet2Calculator(new_format_model)

        # Should have external DFTD3 attached
        assert calc.external_dftd3 is not None
        assert abs(calc.external_dftd3.s8 - 0.3908) < 0.001

    def test_set_dftd3_cutoff_method(self, new_format_model):
        """Test set_dftd3_cutoff() method on calculator."""
        from aimnet.calculators import AIMNet2Calculator

        calc = AIMNet2Calculator(new_format_model)

        # Default cutoff should be 15.0
        assert calc.external_dftd3.smoothing_off == 15.0

        # Change cutoff with 10% smoothing width
        calc.set_dftd3_cutoff(20.0, 0.1)

        # Verify change
        assert calc.external_dftd3.smoothing_off == 20.0
        assert calc.external_dftd3.smoothing_on == 18.0  # 20.0 * (1 - 0.1)

    @pytest.fixture
    def new_format_no_coulomb(self, tmp_path):
        """Create new-format model without Coulomb (needs_coulomb=False)."""
        import yaml

        model = build_model(aimnet2_d3_def)
        state_dict = model.state_dict()

        with open(aimnet2_d3_def, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        model_yaml = yaml.dump(config, default_flow_style=False, sort_keys=False)

        # No external Coulomb/DFTD3
        new_format = {
            "format_version": 2,  # v2 = new .pt format
            "model_yaml": model_yaml,
            "cutoff": 5.0,
            "needs_coulomb": False,
            "needs_dispersion": False,
            "coulomb_mode": "none",
            "coulomb_sr_rc": None,
            "coulomb_sr_envelope": None,
            "d3_params": None,
            "implemented_species": [1, 6, 7, 8],
            "state_dict": state_dict,
        }

        model_path = tmp_path / "test_model_no_coulomb.pt"
        torch.save(new_format, model_path)
        return str(model_path)

    def test_calculator_no_external_modules_when_not_needed(self, new_format_no_coulomb):
        """Test that calculator doesn't attach external modules when not needed."""
        from aimnet.calculators import AIMNet2Calculator

        calc = AIMNet2Calculator(new_format_no_coulomb)

        # Should NOT have external modules
        assert calc.external_coulomb is None
        assert calc.external_dftd3 is None

    @pytest.mark.ase
    def test_new_format_inference(self):
        """Test that new format model can run inference via calculator.

        This test uses a legacy model loaded via from_file to verify
        the loading and inference pipeline works.
        """
        pytest.importorskip("ase")
        from aimnet.calculators import AIMNet2Calculator

        # Use the standard aimnet2 model which is known to work
        calc = AIMNet2Calculator("aimnet2")

        # Simple water molecule input
        data = {
            "coord": torch.tensor([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]),
            "numbers": torch.tensor([8, 1, 1]),
            "charge": torch.tensor([0.0]),
        }

        # Should be able to run inference
        result = calc.eval(data)

        # Should have energy output
        assert "energy" in result
        assert result["energy"].shape == (1,)

    def test_export_load_roundtrip(self, tmp_path):
        """Test export -> load -> inference produces valid results.

        This test exercises the full pipeline:
        1. Build model from YAML, strip LR modules
        2. Build model from stripped config and load weights
        3. Save as v2 format and reload
        4. Run inference and verify energy is finite
        """
        import copy

        import yaml

        from aimnet.config import build_module
        from aimnet.models.utils import strip_lr_modules_from_yaml, validate_state_dict_keys

        # 1. Build original model and get state dict
        original_model = build_model(aimnet2_d3_def)
        original_sd = original_model.state_dict()

        # Load YAML config
        with open(aimnet2_d3_def, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # 2. Strip LR modules (simulating export)
        core_config, coulomb_mode, needs_dispersion, d3_params, coulomb_sr_rc, coulomb_sr_envelope, _disp_ptfile = (
            strip_lr_modules_from_yaml(config, original_sd)
        )
        core_yaml_str = yaml.dump(core_config, default_flow_style=False, sort_keys=False)

        # Build model from stripped config and load weights (like export does)
        core_model = build_module(copy.deepcopy(core_config))
        load_result = core_model.load_state_dict(original_sd, strict=False)

        # Validate keys
        real_missing, _real_unexpected = validate_state_dict_keys(load_result.missing_keys, load_result.unexpected_keys)
        # Expected: SRCoulomb missing, LRCoulomb/DFTD3 unexpected
        assert len(real_missing) == 0, f"Unexpected missing keys: {real_missing}"

        # Get state dict from the properly loaded model
        core_sd = core_model.state_dict()

        # Create v2 format file
        needs_coulomb = coulomb_mode == "sr_embedded"
        v2_format = {
            "format_version": 2,
            "model_yaml": core_yaml_str,
            "cutoff": 5.0,
            "needs_coulomb": needs_coulomb,
            "needs_dispersion": needs_dispersion,
            "coulomb_mode": coulomb_mode,
            "coulomb_sr_rc": coulomb_sr_rc if needs_coulomb else None,
            "coulomb_sr_envelope": coulomb_sr_envelope if needs_coulomb else None,
            "d3_params": d3_params if needs_dispersion else None,
            "implemented_species": [1, 6, 7, 8],
            "state_dict": core_sd,  # Use state dict from core model
        }

        model_path = tmp_path / "roundtrip_model.pt"
        torch.save(v2_format, model_path)

        # 3. Load with load_model
        loaded_model, metadata = load_model(str(model_path), device="cpu")

        # Verify metadata
        assert metadata["format_version"] == 2
        assert metadata["cutoff"] == 5.0
        assert metadata["needs_coulomb"] == needs_coulomb
        assert metadata["needs_dispersion"] == needs_dispersion

        # 4. Run inference
        _in = {
            "coord": torch.tensor([[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]]),
            "numbers": torch.tensor([[8, 1, 1]]),
            "charge": torch.tensor([0.0]),
        }
        _in = add_dftd3_keys(_in)

        with torch.no_grad():
            _out = loaded_model(_in)

        # Verify energy is finite
        assert "energy" in _out
        assert torch.isfinite(_out["energy"]).all(), "Energy should be finite"

    def test_strip_lr_modules_returns_disp_ptfile(self, tmp_path):
        """Test strip_lr_modules_from_yaml returns disp_ptfile path."""
        from aimnet.models.utils import strip_lr_modules_from_yaml

        # Create a dummy ptfile
        ptfile = tmp_path / "disp_params.pt"
        disp_params = torch.rand(64, 2)
        torch.save(disp_params, ptfile)

        # Create config with DispParam referencing ptfile
        config = {
            "class": "aimnet.models.AIMNet2",
            "kwargs": {
                "outputs": {
                    "c6_alpha": {
                        "class": "aimnet.modules.DispParam",
                        "kwargs": {"ptfile": str(ptfile)},
                    }
                }
            },
        }

        # Call strip_lr_modules_from_yaml with empty state dict
        result = strip_lr_modules_from_yaml(config, {})

        # Check disp_ptfile is returned (last element of tuple)
        assert result[-1] == str(ptfile)

        # Verify ptfile was stripped from config
        stripped_config = result[0]
        c6_alpha_kwargs = stripped_config["kwargs"]["outputs"]["c6_alpha"]["kwargs"]
        assert "ptfile" not in c6_alpha_kwargs

    def test_disp_param_shape_mismatch_handling(self, tmp_path):
        """Test that disp_param0 shape mismatch is handled correctly."""
        from aimnet.modules.lr import DispParam

        # Create DispParam with default (87, 2) placeholder
        module = DispParam()
        assert module.disp_param0.shape == (87, 2)

        # Create ptfile with different shape
        ptfile = tmp_path / "disp_params.pt"
        disp_params = torch.rand(64, 2)
        torch.save(disp_params, ptfile)

        # Load and inject (simulating what export_model does)
        loaded_params = torch.load(ptfile, map_location="cpu", weights_only=True)
        if module.disp_param0.shape != loaded_params.shape:
            module.disp_param0 = torch.zeros_like(loaded_params)
        module.disp_param0.copy_(loaded_params)

        # Verify shape is now (64, 2)
        assert module.disp_param0.shape == (64, 2)
        assert torch.allclose(module.disp_param0, disp_params)
