"""GPU-specific tests for AIMNet2 calculator.

All tests in this module require CUDA and are marked with @pytest.mark.gpu.
Run with: pytest -m gpu
"""

import os

import pytest
import torch

from aimnet.calculators import AIMNet2Calculator

# Skip entire module if CUDA is not available
pytestmark = pytest.mark.gpu

if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

file = os.path.join(os.path.dirname(__file__), "data", "caffeine.xyz")


def load_mol(filepath):
    """Helper to load molecule from xyz file."""
    pytest.importorskip("ase", reason="ASE not installed")
    import ase.io

    atoms = ase.io.read(filepath)
    data = {
        "coord": atoms.get_positions(),
        "numbers": atoms.get_atomic_numbers(),
        "charge": 0.0,
    }
    return data


def create_cpu_calculator():
    """Create a CPU-only calculator instance for comparison tests.

    Uses __new__ to bypass __init__ which auto-selects CUDA when available.
    This approach directly initializes attributes to force CPU execution.
    """
    from aimnet.calculators.model_registry import get_model_path

    calc = AIMNet2Calculator.__new__(AIMNet2Calculator)
    calc.device = "cpu"

    p = get_model_path("aimnet2")
    calc.model = torch.jit.load(p, map_location="cpu")
    calc.cutoff = calc.model.cutoff
    calc.lr = hasattr(calc.model, "cutoff_lr")
    calc.cutoff_lr = getattr(calc.model, "cutoff_lr", float("inf")) if calc.lr else None
    calc.max_density = 0.2
    calc.nb_threshold = 0
    calc._batch = None
    calc._max_mol_size = 0
    calc._saved_for_grad = {}
    calc._coulomb_method = "simple"

    return calc


class TestGPUBasics:
    """Basic GPU functionality tests."""

    @pytest.mark.ase
    def test_model_on_cuda(self):
        """Test that model is loaded on CUDA device."""
        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)
        assert calc.device == "cuda"

    @pytest.mark.ase
    def test_inference_on_cuda(self):
        """Test basic inference on CUDA."""
        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)
        data = load_mol(file)
        res = calc(data)

        assert "energy" in res
        assert res["energy"].device.type == "cuda"

    @pytest.mark.ase
    def test_forces_on_cuda(self):
        """Test force calculation on CUDA."""
        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)
        data = load_mol(file)
        res = calc(data, forces=True)

        assert "forces" in res
        assert res["forces"].device.type == "cuda"


class TestGPUvsCPUConsistency:
    """Tests verifying GPU and CPU produce consistent results."""

    @pytest.mark.ase
    def test_energy_consistency(self):
        """Test that GPU and CPU produce the same energy."""
        # GPU calculation
        calc_gpu = AIMNet2Calculator("aimnet2", nb_threshold=0)
        data = load_mol(file)
        res_gpu = calc_gpu(data)
        energy_gpu = res_gpu["energy"].cpu()

        # CPU calculation
        calc_cpu = create_cpu_calculator()
        res_cpu = calc_cpu(data)
        energy_cpu = res_cpu["energy"]

        # Energies should match closely
        assert torch.allclose(energy_gpu, energy_cpu, rtol=1e-5, atol=1e-6)

    @pytest.mark.ase
    def test_forces_consistency(self):
        """Test that GPU and CPU produce the same forces."""
        # GPU calculation
        calc_gpu = AIMNet2Calculator("aimnet2", nb_threshold=0)
        data = load_mol(file)
        res_gpu = calc_gpu(data, forces=True)
        forces_gpu = res_gpu["forces"].cpu()

        # CPU calculation
        calc_cpu = create_cpu_calculator()
        res_cpu = calc_cpu(data, forces=True)
        forces_cpu = res_cpu["forces"]

        # Forces should match closely
        assert torch.allclose(forces_gpu, forces_cpu, rtol=1e-4, atol=1e-5)


class TestCUDANeighborList:
    """Tests for CUDA neighbor list computation."""

    def test_nbmat_cuda_vs_cpu(self):
        """Test that CUDA and CPU neighbor lists produce same results."""
        from aimnet.calculators.nbmat import calc_nbmat

        torch.manual_seed(42)
        N = 50
        coord_cpu = torch.rand((N, 3)) * 5
        coord_gpu = coord_cpu.cuda()

        cutoff = (3.0, None)
        maxnb = (100, None)

        # CPU computation
        nbmat_cpu, _, _, _ = calc_nbmat(coord_cpu, cutoff, maxnb)

        # GPU computation
        nbmat_gpu, _, _, _ = calc_nbmat(coord_gpu, cutoff, maxnb)

        # Move to same device for comparison
        nbmat_gpu_cpu = nbmat_gpu.cpu()

        # Shapes should match
        assert nbmat_cpu.shape == nbmat_gpu_cpu.shape

        # Content might differ in order but should have same neighbors
        # Check that for each atom, the set of neighbors is the same
        for i in range(N):
            nb_cpu = set(nbmat_cpu[i][nbmat_cpu[i] < N].tolist())
            nb_gpu = set(nbmat_gpu_cpu[i][nbmat_gpu_cpu[i] < N].tolist())
            assert nb_cpu == nb_gpu, f"Neighbors differ for atom {i}"


class TestGPUBatching:
    """Tests for GPU-specific batching behavior."""

    @pytest.mark.ase
    def test_large_batch_on_gpu(self):
        """Test large batch processing on GPU."""
        calc = AIMNet2Calculator("aimnet2", nb_threshold=500)  # Force batched mode
        data = load_mol(file)

        # Create batch of 10 copies
        import numpy as np

        batch_coord = np.stack([data["coord"]] * 10)
        batch_numbers = np.stack([data["numbers"]] * 10)

        batch_data = {
            "coord": torch.tensor(batch_coord, dtype=torch.float32),
            "numbers": torch.tensor(batch_numbers),
            "charge": torch.zeros(10),
        }

        res = calc(batch_data)
        assert res["energy"].shape == (10,)

        # All energies should be the same (same molecule)
        assert torch.allclose(res["energy"], res["energy"][0].expand(10), rtol=1e-5)

    @pytest.mark.ase
    def test_nb_threshold_behavior(self):
        """Test that nb_threshold controls batching behavior."""
        data = load_mol(file)
        n_atoms = len(data["numbers"])

        # With high threshold, should use batched mode
        calc_batch = AIMNet2Calculator("aimnet2", nb_threshold=n_atoms + 100)
        # With low threshold, should use flattened mode
        calc_flat = AIMNet2Calculator("aimnet2", nb_threshold=0)

        # Both should give same results
        res_batch = calc_batch(data)
        res_flat = calc_flat(data)

        assert torch.allclose(res_batch["energy"], res_flat["energy"], rtol=1e-5)


class TestGPUMemory:
    """Tests for GPU memory management."""

    @pytest.mark.ase
    def test_memory_cleanup_after_inference(self):
        """Test that GPU memory is properly cleaned up after inference."""
        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)
        data = load_mol(file)

        # Run inference multiple times
        for _ in range(5):
            res = calc(data, forces=True)
            del res

        # Force garbage collection
        import gc

        gc.collect()
        torch.cuda.empty_cache()

        # Get memory stats
        memory_allocated = torch.cuda.memory_allocated()
        memory_reserved = torch.cuda.memory_reserved()

        # Memory should be bounded (not growing indefinitely)
        # This is a basic sanity check
        assert memory_allocated < 2e9  # Less than 2GB allocated
        assert memory_reserved < 4e9  # Less than 4GB reserved

    @pytest.mark.ase
    def test_no_memory_leak_in_forces(self):
        """Test that force calculation doesn't leak memory."""
        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)
        data = load_mol(file)

        # Warm up
        _ = calc(data, forces=True)
        torch.cuda.synchronize()

        # Record baseline memory
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()

        # Run many iterations
        for _ in range(10):
            res = calc(data, forces=True)
            del res

        torch.cuda.synchronize()
        final = torch.cuda.memory_allocated()

        # Memory growth should be minimal
        growth = final - baseline
        assert growth < 100e6  # Less than 100MB growth


class TestGPUErrorHandling:
    """Tests for GPU-specific error handling."""

    @pytest.mark.ase
    def test_handles_cuda_oom_gracefully(self):
        """Test behavior when approaching CUDA OOM (skipped if not enough GPU memory)."""
        # This test intentionally tries to use a lot of memory
        # Skip if GPU has less than 4GB free
        free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
        if free_memory < 4e9:
            pytest.skip("Not enough GPU memory for OOM test")

        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)

        # Try to process a very large "molecule" (this might OOM on smaller GPUs)
        try:
            large_data = {
                "coord": torch.rand((5000, 3), dtype=torch.float32),
                "numbers": torch.ones(5000, dtype=torch.long) * 6,  # All carbon
                "charge": torch.tensor([0.0]),
            }
            _ = calc(large_data)
        except RuntimeError as e:
            # Should get a clear CUDA OOM error
            assert "CUDA" in str(e) or "out of memory" in str(e).lower()
