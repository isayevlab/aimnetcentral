"""Tests for torch.compile with CUDA graphs support."""

import os

import numpy as np
import pytest
import torch

from aimnet.calculators import AIMNet2Calculator

file = os.path.join(os.path.dirname(__file__), "data", "caffeine.xyz")


def load_mol(filepath):
    """Load molecule from XYZ file."""
    pytest.importorskip("ase", reason="ASE not installed. Install with: pip install aimnet[ase]")
    import ase.io

    atoms = ase.io.read(filepath)
    data = {
        "coord": atoms.get_positions(),  # type: ignore
        "numbers": atoms.get_atomic_numbers(),  # type: ignore
        "charge": 0.0,
    }
    return data


@pytest.mark.gpu
@pytest.mark.ase
def test_compile_mode_requires_cuda():
    """Test that compile_mode raises error without CUDA."""
    if torch.cuda.is_available():
        pytest.skip("CUDA is available, skipping non-CUDA test")
    with pytest.raises(ValueError, match="compile_mode requires CUDA"):
        AIMNet2Calculator("aimnet2", compile_mode=True)


@pytest.mark.gpu
@pytest.mark.ase
def test_compile_mode_requires_model_name():
    """Test that compile_mode requires model name (str), not nn.Module."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    # Load a regular model first
    calc = AIMNet2Calculator("aimnet2")
    # Try to use compile_mode with the model instance
    with pytest.raises(ValueError, match="compile_mode requires model name"):
        AIMNet2Calculator(calc.model, compile_mode=True)


@pytest.mark.gpu
@pytest.mark.ase
def test_compile_mode_basic():
    """Test compile mode produces results."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    calc = AIMNet2Calculator("aimnet2", compile_mode=True)
    data = load_mol(file)
    res = calc(data)
    assert "energy" in res
    assert "charges" in res


@pytest.mark.gpu
@pytest.mark.ase
def test_compile_mode_energy_consistency():
    """Test compile mode produces same energy as normal mode (within tolerance)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    data = load_mol(file)

    # Normal mode
    calc_normal = AIMNet2Calculator("aimnet2")
    res_normal = calc_normal(data)

    # Compile mode
    calc_compile = AIMNet2Calculator("aimnet2", compile_mode=True)
    res_compile = calc_compile(data)

    # Results should be very close (float32 tolerance)
    np.testing.assert_allclose(
        res_normal["energy"].cpu().numpy(),
        res_compile["energy"].cpu().numpy(),
        rtol=1e-4,
        atol=1e-5,
    )


@pytest.mark.gpu
@pytest.mark.ase
def test_compile_mode_forces():
    """Test compile mode with forces calculation."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    calc = AIMNet2Calculator("aimnet2", compile_mode=True)
    data = load_mol(file)
    res = calc(data, forces=True)
    assert "energy" in res
    assert "forces" in res
    assert res["forces"].shape[0] == 24  # caffeine has 24 atoms


@pytest.mark.gpu
@pytest.mark.ase
def test_compile_mode_forces_consistency():
    """Test compile mode produces same forces as normal mode."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    data = load_mol(file)

    # Normal mode
    calc_normal = AIMNet2Calculator("aimnet2")
    res_normal = calc_normal(data, forces=True)

    # Compile mode
    calc_compile = AIMNet2Calculator("aimnet2", compile_mode=True)
    res_compile = calc_compile(data, forces=True)

    # Forces should be close
    np.testing.assert_allclose(
        res_normal["forces"].cpu().numpy(),
        res_compile["forces"].cpu().numpy(),
        rtol=1e-3,
        atol=1e-4,
    )


@pytest.mark.gpu
@pytest.mark.ase
def test_compile_mode_multiple_calls():
    """Test CUDA graph reuse across multiple calls."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    calc = AIMNet2Calculator("aimnet2", compile_mode=True)
    data = load_mol(file)

    # First call (warm-up / compilation)
    res1 = calc(data, forces=True)

    # Subsequent calls should use cached CUDA graph
    res2 = calc(data, forces=True)
    res3 = calc(data, forces=True)

    # All results should be identical
    np.testing.assert_array_equal(
        res1["energy"].cpu().numpy(),
        res2["energy"].cpu().numpy(),
    )
    np.testing.assert_array_equal(
        res2["energy"].cpu().numpy(),
        res3["energy"].cpu().numpy(),
    )


@pytest.mark.gpu
@pytest.mark.ase
@pytest.mark.parametrize("model", ["aimnet2", "aimnet2_b973c"])
def test_compile_mode_different_models(model):
    """Test compile mode with different models."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    calc = AIMNet2Calculator(model, compile_mode=True)
    data = load_mol(file)
    res = calc(data)
    assert "energy" in res
    assert "charges" in res


@pytest.mark.gpu
@pytest.mark.ase
def test_compile_mode_pbc_not_supported():
    """Test that PBC raises error in compile mode."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    calc = AIMNet2Calculator("aimnet2", compile_mode=True)
    data = load_mol(file)
    data["cell"] = np.eye(3) * 10.0  # Add cell for PBC

    with pytest.raises(NotImplementedError, match="PBC is not supported in compile mode"):
        calc(data)


@pytest.mark.gpu
@pytest.mark.ase
def test_ase_calculator_compile_mode():
    """Test ASE calculator with compile mode."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    pytest.importorskip("ase", reason="ASE not installed")
    from ase.io import read

    from aimnet.calculators import AIMNet2ASE

    atoms = read(file)
    atoms.calc = AIMNet2ASE("aimnet2", compile_mode=True)

    e = atoms.get_potential_energy()
    assert isinstance(e, float)

    f = atoms.get_forces()
    assert f.shape == (24, 3)

    q = atoms.get_charges()
    assert q.shape == (24,)
