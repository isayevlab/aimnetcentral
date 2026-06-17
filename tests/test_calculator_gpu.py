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


class _CountingNeighborList:
    def __init__(self, wrapped):
        object.__setattr__(self, "wrapped", wrapped)
        object.__setattr__(self, "calls", 0)

    def __getattr__(self, name):
        return getattr(self.wrapped, name)

    def __setattr__(self, name, value):
        if name in {"wrapped", "calls"}:
            object.__setattr__(self, name, value)
        else:
            setattr(self.wrapped, name, value)

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.wrapped(*args, **kwargs)


class _CountingExternalModule:
    def __init__(self, wrapped):
        object.__setattr__(self, "wrapped", wrapped)
        object.__setattr__(self, "calls", 0)

    def __getattr__(self, name):
        return getattr(self.wrapped, name)

    def __setattr__(self, name, value):
        if name in {"wrapped", "calls"}:
            object.__setattr__(self, name, value)
        else:
            setattr(self.wrapped, name, value)

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.wrapped(*args, **kwargs)


def _wrap_neighbor_lists(calc):
    for attr in ("_nblist", "_nblist_lr", "_nblist_coulomb", "_nblist_dftd3"):
        nblist = getattr(calc, attr)
        if nblist is not None:
            setattr(calc, attr, _CountingNeighborList(nblist))


def _neighbor_list_calls(calc):
    return sum(
        getattr(getattr(calc, attr), "calls", 0)
        for attr in ("_nblist", "_nblist_lr", "_nblist_coulomb", "_nblist_dftd3")
    )


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
        from nvalchemiops.torch.neighbors import neighbor_list

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

    @staticmethod
    def _water_data():
        return {
            "coord": torch.tensor(
                [[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.2390, 0.9270, 0.0]],
                dtype=torch.float32,
            ),
            "numbers": torch.tensor([8, 1, 1]),
            "charge": torch.tensor(0.0),
        }

    @classmethod
    def _resident_water_cluster(cls, copies=4):
        water = cls._water_data()
        coords = []
        for i in range(copies):
            coords.append(water["coord"] + torch.tensor([4.0 * i, 0.0, 0.0], dtype=torch.float32))
        return {
            "coord": torch.cat(coords, dim=0).cuda(),
            "numbers": water["numbers"].repeat(copies).cuda(),
            "charge": torch.tensor(0.0, device="cuda"),
        }

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

    @pytest.mark.ase
    def test_static_cache_reuses_neighbor_matrices_for_same_resident_tensors(self):
        """Opt-in static cache avoids rebuilding neighbor matrices on repeated CUDA tensors."""
        data = self._resident_water_cluster()
        calc = AIMNet2Calculator("aimnet2", device="cuda", nb_threshold=0, cache_static=True)
        _wrap_neighbor_lists(calc)

        first = calc(data, forces=True)
        calls_after_first = _neighbor_list_calls(calc)
        assert calls_after_first > 0

        second = calc(data, forces=True)
        calls_after_second = _neighbor_list_calls(calc)
        assert calls_after_second == calls_after_first
        torch.testing.assert_close(first["energy"], second["energy"], rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(first["forces"], second["forces"], rtol=1e-6, atol=1e-6)

        data["coord"][0, 0] += 0.01
        calc(data, forces=True)
        assert _neighbor_list_calls(calc) > calls_after_second

    @pytest.mark.ase
    def test_static_cache_is_disabled_by_default(self):
        """Default calculator behavior rebuilds neighbor matrices every call."""
        data = self._resident_water_cluster()
        calc = AIMNet2Calculator("aimnet2", device="cuda", nb_threshold=0)
        _wrap_neighbor_lists(calc)

        calc(data, forces=True)
        calls_after_first = _neighbor_list_calls(calc)
        assert calls_after_first > 0
        calc(data, forces=True)
        assert _neighbor_list_calls(calc) > calls_after_first

    @pytest.mark.ase
    def test_static_cache_reuses_dftd3_terms_for_same_resident_tensors(self):
        """Opt-in static cache avoids rerunning external DFTD3 on repeated CUDA tensors."""
        data = self._resident_water_cluster()
        calc = AIMNet2Calculator("aimnet2", device="cuda", nb_threshold=0, cache_static=True)
        assert calc.external_dftd3 is not None
        calc.external_dftd3 = _CountingExternalModule(calc.external_dftd3)

        first = calc(data, forces=True)
        assert calc.external_dftd3.calls == 1

        second = calc(data, forces=True)
        assert calc.external_dftd3.calls == 1
        torch.testing.assert_close(first["energy"], second["energy"], rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(first["forces"], second["forces"], rtol=1e-6, atol=1e-6)

        data["coord"][0, 0] += 0.01
        calc(data, forces=True)
        assert calc.external_dftd3.calls == 2

    @pytest.mark.ase
    def test_static_dftd3_cache_is_disabled_by_default(self):
        """Default calculator behavior reruns external DFTD3 every call."""
        data = self._resident_water_cluster()
        calc = AIMNet2Calculator("aimnet2", device="cuda", nb_threshold=0)
        assert calc.external_dftd3 is not None
        calc.external_dftd3 = _CountingExternalModule(calc.external_dftd3)

        calc(data, forces=True)
        assert calc.external_dftd3.calls == 1
        calc(data, forces=True)
        assert calc.external_dftd3.calls == 2

    @pytest.mark.ase
    def test_static_cache_reuses_geometry_work_across_charge_changes(self):
        """Same resident geometry can reuse cached static work while charge-dependent outputs recompute."""
        data_neutral = self._resident_water_cluster()
        data_charged = dict(data_neutral)
        data_charged["charge"] = torch.tensor(1.0, device="cuda")

        calc = AIMNet2Calculator("aimnet2", device="cuda", nb_threshold=0, cache_static=True)
        assert calc.external_dftd3 is not None
        _wrap_neighbor_lists(calc)
        calc.external_dftd3 = _CountingExternalModule(calc.external_dftd3)

        neutral = calc(data_neutral, forces=True)
        neighbor_calls = _neighbor_list_calls(calc)
        dftd3_calls = calc.external_dftd3.calls
        assert neighbor_calls > 0
        assert dftd3_calls == 1

        charged = calc(data_charged, forces=True)
        assert _neighbor_list_calls(calc) == neighbor_calls
        assert calc.external_dftd3.calls == dftd3_calls
        assert torch.isfinite(charged["energy"]).all()
        assert torch.isfinite(charged["charges"]).all()
        assert torch.isfinite(charged["forces"]).all()
        assert not torch.allclose(neutral["charges"], charged["charges"], rtol=1e-5, atol=1e-6)


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
        from nvalchemiops.torch.neighbors import neighbor_list

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


class TestVectorizedHessian:
    """End-to-end Hessian via vmap-based paths through the kernel.

    These tests exercise the vmap rules on aimnet::conv_sv_2d_sp_bwd and
    aimnet::conv_sv_2d_sp_bwd_bwd by calling torch.func.vmap directly on a vjp
    closure built from the calculator's energy graph. AIMNet2Calculator's own
    Hessian path is not invoked.
    """

    def _water_inputs(self, device):
        coords = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
            device=device,
        )
        nums = torch.tensor([[8, 1, 1]], device=device)
        charge = torch.tensor([0.0], device=device)
        return coords, nums, charge

    def test_func_vmap_hessian_matches_internal(self):
        """torch.func.vmap-based Hessian matches the calculator's loop hessian.

        torch.func.hessian is not used directly here because it pushes vmap through
        the entire forward pass, which is blocked by an upstream non-vmap-compatible
        op.  Instead we use the equivalent explicit form: compute dE/dr with
        create_graph=True, then vmap over autograd.grad for the second derivative.
        This exercises exactly the same vmap rules on aimnet::conv_sv_2d_sp_bwd and
        aimnet::conv_sv_2d_sp_bwd_bwd as torch.func.hessian would, but confines the
        vmap to the backward graph.
        """
        device = torch.device("cuda")
        calc = AIMNet2Calculator("aimnet2", device=device, nb_threshold=0)
        coords, nums, charge = self._water_inputs(device)

        H_internal = calc(
            {"coord": coords.unsqueeze(0).clone(), "numbers": nums, "charge": charge},
            hessian=True,
        )["hessian"]
        assert H_internal.shape == (3, 3, 3, 3)

        def energy_fn(x):
            out = calc({"coord": x.unsqueeze(0), "numbers": nums, "charge": charge}, forces=False)
            return out["energy"][0]

        coord_req = coords.clone().requires_grad_(True)
        dEdx_flat = torch.autograd.grad(energy_fn(coord_req), coord_req, create_graph=True)[0].flatten()

        n = dEdx_flat.numel()
        eye = torch.eye(n, device=device, dtype=dEdx_flat.dtype)

        def vjp(go):
            return torch.autograd.grad(
                dEdx_flat,
                coord_req,
                grad_outputs=go,
                retain_graph=True,
                allow_unused=True,
            )[0]

        H_func = torch.func.vmap(vjp, 0)(eye).reshape(3, 3, 3, 3)
        assert H_func.shape == (3, 3, 3, 3)
        assert torch.isfinite(H_func).all()
        assert (H_internal - H_func).abs().max().item() < 5e-3
