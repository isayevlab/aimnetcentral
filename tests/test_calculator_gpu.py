"""GPU-specific tests for AIMNet2 calculator.

All tests in this module require CUDA and are marked with @pytest.mark.gpu.
Run with: pytest -m gpu
"""

import pytest
import torch
from conftest import CAFFEINE_FILE, load_mol

from aimnet.calculators import AIMNet2Calculator

# Skip entire module if CUDA is not available
pytestmark = pytest.mark.gpu

if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)


class TestGPUBasics:
    """Basic GPU functionality tests."""

    @pytest.mark.ase
    def test_model_on_cuda(self):
        """Test that model is loaded on CUDA device."""
        calc = AIMNet2Calculator("aimnet2", device="cuda", nb_threshold=0)
        assert calc.device == "cuda"

    @pytest.mark.ase
    def test_inference_on_cuda(self):
        """Test basic inference on CUDA."""
        calc = AIMNet2Calculator("aimnet2", device="cuda", nb_threshold=0)
        data = load_mol(CAFFEINE_FILE)
        res = calc(data)

        assert "energy" in res
        assert res["energy"].device.type == "cuda"

    @pytest.mark.ase
    def test_forces_on_cuda(self):
        """Test force calculation on CUDA."""
        calc = AIMNet2Calculator("aimnet2", device="cuda", nb_threshold=0)
        data = load_mol(CAFFEINE_FILE)
        res = calc(data, forces=True)

        assert "forces" in res
        assert res["forces"].device.type == "cuda"


class TestGPUvsCPUConsistency:
    """Tests verifying GPU and CPU produce consistent results."""

    @pytest.mark.ase
    def test_energy_consistency(self):
        """Test that GPU and CPU produce the same energy."""
        # GPU calculation
        calc_gpu = AIMNet2Calculator("aimnet2", device="cuda", nb_threshold=0)
        data = load_mol(CAFFEINE_FILE)
        res_gpu = calc_gpu(data)
        energy_gpu = res_gpu["energy"].cpu()

        # CPU calculation
        calc_cpu = AIMNet2Calculator("aimnet2", device="cpu", nb_threshold=0)
        res_cpu = calc_cpu(data)
        energy_cpu = res_cpu["energy"]

        # Energies should match closely
        assert torch.allclose(energy_gpu, energy_cpu, rtol=1e-5, atol=1e-6)

    @pytest.mark.ase
    def test_forces_consistency(self):
        """Test that GPU and CPU produce the same forces."""
        # GPU calculation
        calc_gpu = AIMNet2Calculator("aimnet2", device="cuda", nb_threshold=0)
        data = load_mol(CAFFEINE_FILE)
        res_gpu = calc_gpu(data, forces=True)
        forces_gpu = res_gpu["forces"].cpu()

        # CPU calculation
        calc_cpu = AIMNet2Calculator("aimnet2", device="cpu", nb_threshold=0)
        res_cpu = calc_cpu(data, forces=True)
        forces_cpu = res_cpu["forces"]

        # Forces should match closely
        assert torch.allclose(forces_gpu, forces_cpu, rtol=1e-4, atol=1e-5)


class TestCUDANeighborList:
    """Tests for CUDA neighbor list computation."""

    def test_nbmat_cuda_vs_cpu(self):
        """Test that CUDA and CPU neighbor lists produce same results."""
        from nvalchemiops.neighborlist import neighbor_list

        torch.manual_seed(42)
        N = 50
        coord_cpu = torch.rand((N, 3)) * 5
        coord_gpu = coord_cpu.cuda()

        cutoff = 3.0
        max_neighbors = 100

        # CPU computation
        nbmat_cpu, _num_nb_cpu = neighbor_list(
            positions=coord_cpu,
            cutoff=cutoff,
            max_neighbors=max_neighbors,
            half_fill=False,
            fill_value=N,
        )

        # GPU computation
        nbmat_gpu, _num_nb_gpu = neighbor_list(
            positions=coord_gpu,
            cutoff=cutoff,
            max_neighbors=max_neighbors,
            half_fill=False,
            fill_value=N,
        )

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
        calc = AIMNet2Calculator("aimnet2", device="cuda", nb_threshold=500)  # Force batched mode
        data = load_mol(CAFFEINE_FILE)

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
        data = load_mol(CAFFEINE_FILE)
        n_atoms = len(data["numbers"])

        # With high threshold, should use batched mode
        calc_batch = AIMNet2Calculator("aimnet2", device="cuda", nb_threshold=n_atoms + 100)
        # With low threshold, should use flattened mode
        calc_flat = AIMNet2Calculator("aimnet2", device="cuda", nb_threshold=0)

        # Both should give same results
        res_batch = calc_batch(data)
        res_flat = calc_flat(data)

        assert torch.allclose(res_batch["energy"], res_flat["energy"], rtol=1e-5)


class TestGPUMemory:
    """Tests for GPU memory management."""

    @pytest.mark.ase
    def test_memory_cleanup_after_inference(self):
        """Test that GPU memory is properly cleaned up after inference."""
        calc = AIMNet2Calculator("aimnet2", device="cuda", nb_threshold=0)
        data = load_mol(CAFFEINE_FILE)

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
        calc = AIMNet2Calculator("aimnet2", device="cuda", nb_threshold=0)
        data = load_mol(CAFFEINE_FILE)

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


class TestCPUGPUConsistency:
    """Cross-validation tests between CPU and GPU implementations."""

    @pytest.mark.ase
    def test_energy_cpu_vs_gpu_single_molecule(self):
        """Verify energy is identical on CPU and GPU."""
        data = load_mol(CAFFEINE_FILE)

        # CPU calculation
        calc_cpu = AIMNet2Calculator("aimnet2", device="cpu", nb_threshold=0)
        res_cpu = calc_cpu(data)
        e_cpu = res_cpu["energy"].item()

        # GPU calculation
        calc_gpu = AIMNet2Calculator("aimnet2", device="cuda", nb_threshold=0)
        res_gpu = calc_gpu(data)
        e_gpu = res_gpu["energy"].cpu().item()

        assert abs(e_cpu - e_gpu) < 1e-5, f"Energy mismatch: CPU={e_cpu}, GPU={e_gpu}"

    @pytest.mark.ase
    def test_forces_cpu_vs_gpu_single_molecule(self):
        """Verify forces are identical on CPU and GPU."""
        data = load_mol(CAFFEINE_FILE)

        # CPU calculation
        calc_cpu = AIMNet2Calculator("aimnet2", device="cpu", nb_threshold=0)
        res_cpu = calc_cpu(data, forces=True)
        f_cpu = res_cpu["forces"].detach().numpy()

        # GPU calculation
        calc_gpu = AIMNet2Calculator("aimnet2", device="cuda", nb_threshold=0)
        res_gpu = calc_gpu(data, forces=True)
        f_gpu = res_gpu["forces"].cpu().detach().numpy()

        import numpy as np

        np.testing.assert_allclose(f_cpu, f_gpu, atol=1e-5, rtol=1e-4)

    @pytest.mark.ase
    def test_energy_cpu_vs_gpu_batch(self):
        """Verify batched inference gives consistent results on CPU and GPU."""
        # Create batch of molecules
        data = {
            "coord": torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [0.96, 0.0, 0.0],
                    [-0.24, 0.93, 0.0],
                    [10.0, 0.0, 0.0],
                    [10.96, 0.0, 0.0],
                    [9.76, 0.93, 0.0],
                ],
                dtype=torch.float32,
            ),
            "numbers": torch.tensor([8, 1, 1, 8, 1, 1]),
            "mol_idx": torch.tensor([0, 0, 0, 1, 1, 1]),
            "charge": torch.tensor([0.0, 0.0]),
        }

        # CPU calculation
        calc_cpu = AIMNet2Calculator("aimnet2", device="cpu", nb_threshold=0)
        res_cpu = calc_cpu(data)
        e_cpu = res_cpu["energy"].detach().numpy()

        # GPU calculation
        calc_gpu = AIMNet2Calculator("aimnet2", device="cuda", nb_threshold=0)
        res_gpu = calc_gpu(data)
        e_gpu = res_gpu["energy"].cpu().detach().numpy()

        import numpy as np

        np.testing.assert_allclose(e_cpu, e_gpu, atol=1e-5)

    def test_neighborlist_cpu_vs_gpu_with_pbc(self):
        """Verify PBC neighborlist and shifts match between devices."""
        from nvalchemiops.neighborlist import neighbor_list

        torch.manual_seed(123)
        N = 20
        coord_cpu = torch.rand((N, 3)) * 8.0
        coord_gpu = coord_cpu.cuda()

        # Define a cubic cell
        cell_cpu = torch.eye(3, dtype=torch.float32) * 10.0
        cell_gpu = cell_cpu.cuda()
        pbc = torch.tensor([True, True, True])

        cutoff = 4.0
        max_neighbors = 50

        # CPU computation with PBC
        nbmat_cpu, _num_nb_cpu, shifts_cpu = neighbor_list(
            positions=coord_cpu,
            cutoff=cutoff,
            cell=cell_cpu.unsqueeze(0),
            pbc=pbc.unsqueeze(0),
            max_neighbors=max_neighbors,
            half_fill=False,
            fill_value=N,
        )

        # GPU computation with PBC
        nbmat_gpu, _num_nb_gpu, shifts_gpu = neighbor_list(
            positions=coord_gpu,
            cutoff=cutoff,
            cell=cell_gpu.unsqueeze(0),
            pbc=pbc.cuda().unsqueeze(0),
            max_neighbors=max_neighbors,
            half_fill=False,
            fill_value=N,
        )

        # Move to CPU for comparison
        nbmat_gpu_cpu = nbmat_gpu.cpu()
        shifts_gpu_cpu = shifts_gpu.cpu()

        # Shapes should match
        assert nbmat_cpu.shape == nbmat_gpu_cpu.shape
        assert shifts_cpu.shape == shifts_gpu_cpu.shape

        # Check neighbors match for each atom
        for i in range(N):
            nb_cpu = set(nbmat_cpu[i][nbmat_cpu[i] < N].tolist())
            nb_gpu = set(nbmat_gpu_cpu[i][nbmat_gpu_cpu[i] < N].tolist())
            assert nb_cpu == nb_gpu, f"PBC neighbors differ for atom {i}"
