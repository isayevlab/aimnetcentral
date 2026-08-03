"""CPU and GPU coverage for full-3D periodic global mode 2."""

import pytest
import torch

from aimnet import nbops
from aimnet.modules.lr import DFTD3, LRCoulomb, _mode2_backend_inputs


def _periodic_mode2_data(device: torch.device, batch_size: int = 2) -> dict[str, torch.Tensor]:
    B, N, M = batch_size, 5, 8
    coord = torch.zeros((B, N, 3), device=device)
    coord[:, 1, 0] = 1.0
    coord[:, 2, 1] = 1.1
    coord[:, 3, 2] = 1.2
    if B > 1:
        coord[1, :4] += torch.tensor([0.2, 0.3, 0.4], device=device)
    numbers = torch.tensor([[8, 1, 1, 1, 0]] * B, device=device)
    charges = torch.tensor([[0.4, -0.1, -0.1, -0.1, 0.0]] * B, device=device)
    nbmat = torch.full((B, N, M), B * N, dtype=torch.int32, device=device)
    for b in range(B):
        for i in range(N - 1):
            targets = [b * N + j for j in range(N - 1) if j != i]
            nbmat[b, i, : len(targets)] = torch.tensor(targets, device=device)
    shifts = torch.zeros((B, N, M, 3), device=device)
    shifts[:, 0, 0, 0] = 1
    shifts[:, 1, 0, 0] = -1
    data = {
        "coord": coord,
        "numbers": numbers,
        "charges": charges,
        "nbmat": nbmat,
        "nbmat_lr": nbmat,
        "nbmat_coulomb": nbmat,
        "nbmat_dftd3": nbmat,
        "shifts": shifts,
        "shifts_lr": shifts,
        "shifts_coulomb": shifts,
        "shifts_dftd3": shifts,
        "cell": torch.eye(3, device=device).expand(B, -1, -1) * 12.0,
        "pbc": torch.ones((B, 3), dtype=torch.bool, device=device),
    }
    return nbops.calc_masks(nbops.set_nb_mode(data))


def _single_mode2_periodic_data(data: dict[str, torch.Tensor], batch_index: int) -> dict[str, torch.Tensor]:
    """Extract one global mode-2 system for independent execution."""
    B, N, _M = data["nbmat"].shape
    source_sentinel = B * N
    sentinel = N
    local_nbmat = torch.where(
        data["nbmat"][batch_index] == source_sentinel,
        torch.full_like(data["nbmat"][batch_index], sentinel),
        data["nbmat"][batch_index] - batch_index * N,
    ).to(torch.int32)
    mode1 = {
        "coord": data["coord"][batch_index : batch_index + 1],
        "numbers": data["numbers"][batch_index : batch_index + 1],
        "charges": data["charges"][batch_index : batch_index + 1],
        "cell": data["cell"][batch_index : batch_index + 1],
        "pbc": data["pbc"][batch_index : batch_index + 1],
    }
    for suffix in ("", "_lr", "_coulomb", "_dftd3"):
        mode1[f"nbmat{suffix}"] = local_nbmat.unsqueeze(0)
        mode1[f"shifts{suffix}"] = data[f"shifts{suffix}"][batch_index : batch_index + 1]
    return nbops.calc_masks(nbops.set_nb_mode(mode1))


def _module(backend: str):
    if backend == "dftd3":
        return DFTD3(s8=0.3908, a1=0.5660, a2=3.1280)
    return LRCoulomb(method=backend, subtract_sr=False, ewald_accuracy=1e-5)


@pytest.mark.parametrize("backend", ["dsf", "dftd3", "ewald", "pme"])
def test_global_mode2_cpu_periodic_observables(backend: str):
    data = _periodic_mode2_data(torch.device("cpu"))
    module = _module(backend)
    result = module(data)
    energy_key = "energy" if backend == "dftd3" else "e_h"
    assert result[energy_key].shape == (2,)
    assert torch.isfinite(result[energy_key]).all()


@pytest.mark.parametrize("backend", ["dsf", "dftd3", "ewald", "pme"])
def test_global_mode2_cpu_periodic_forces_and_stress(backend: str):
    data = _periodic_mode2_data(torch.device("cpu"))
    if backend == "ewald":
        data = {
            **data,
            "coord": data["coord"].detach().requires_grad_(True),
            "cell": data["cell"].detach().requires_grad_(True),
        }
        result = _module(backend)(data)
        forces = -torch.autograd.grad(result["e_h"].sum(), data["coord"], retain_graph=True)[0]
        virial = torch.autograd.grad(result["e_h"].sum(), data["cell"])[0]
        terms = None
    else:
        result, terms = _module(backend)(data, compute_forces=True, compute_virial=True)
        forces = terms.forces if terms is not None else None
        virial = terms.virial if terms is not None else None
    energy_key = "energy" if backend == "dftd3" else "e_h"
    assert torch.isfinite(result[energy_key]).all()
    assert forces is not None and virial is not None
    assert forces.shape == data["coord"].shape
    assert virial.shape[-2:] == (3, 3)
    assert torch.isfinite(forces).all()
    assert torch.isfinite(virial).all()


@pytest.mark.parametrize("backend", ["dsf", "dftd3", "ewald", "pme"])
def test_global_mode2_periodic_matches_single_system(backend: str):
    data = _periodic_mode2_data(torch.device("cpu"))
    module = _module(backend)
    batch_result, batch_terms = module(data, compute_forces=True, compute_virial=True)
    energy_key = "energy" if backend == "dftd3" else "e_h"
    for batch_index in range(data["coord"].shape[0]):
        single_data = _single_mode2_periodic_data(data, batch_index)
        single_result, single_terms = module(single_data, compute_forces=True, compute_virial=True)
        torch.testing.assert_close(
            batch_result[energy_key][batch_index],
            single_result[energy_key].reshape(-1)[0],
            atol=2e-5,
            rtol=2e-4,
        )
        if backend == "ewald":
            batch_forces, batch_virial = _ewald_forces_and_virial(data)
            single_forces, single_virial = _ewald_forces_and_virial(single_data)
            torch.testing.assert_close(
                batch_forces[batch_index],
                single_forces[0],
                atol=2e-5,
                rtol=2e-4,
            )
            torch.testing.assert_close(
                batch_virial[batch_index],
                single_virial[0],
                atol=2e-5,
                rtol=2e-4,
            )
        else:
            assert batch_terms is not None and single_terms is not None
            torch.testing.assert_close(
                batch_terms.forces[batch_index],
                single_terms.forces[0],
                atol=2e-5,
                rtol=2e-4,
            )
            torch.testing.assert_close(
                batch_terms.virial[batch_index],
                single_terms.virial[0],
                atol=2e-5,
                rtol=2e-4,
            )


def _ewald_forces_and_virial(data: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiate the Ewald energy for its autograd-only observables."""
    sample = {key: value.clone() for key, value in data.items()}
    sample["coord"] = sample["coord"].detach().requires_grad_(True)
    sample["cell"] = sample["cell"].detach().requires_grad_(True)
    energy = _module("ewald")(sample)["e_h"].sum()
    grad_coord, grad_cell = torch.autograd.grad(energy, (sample["coord"], sample["cell"]))
    return -grad_coord, grad_cell


@pytest.mark.parametrize("backend", ["dsf", "dftd3", "ewald", "pme"])
def test_global_mode2_periodic_hessian_diagonal_matches_independent(backend: str):
    data = _periodic_mode2_data(torch.device("cpu"))
    energy_key = "energy" if backend == "dftd3" else "e_h"
    epsilon = 1e-3

    def energy_at(sample: dict[str, torch.Tensor], index: int) -> torch.Tensor:
        fresh = {key: value.clone() for key, value in sample.items()}
        fresh.pop("e_h", None)
        fresh.pop("energy", None)
        return _module(backend)(fresh)[energy_key].reshape(-1)[index]

    for batch_index in range(data["coord"].shape[0]):
        single_data = _single_mode2_periodic_data(data, batch_index)
        batch_energy = energy_at(data, batch_index)
        single_energy = energy_at(single_data, 0)
        plus_batch = {key: value.clone() for key, value in data.items()}
        minus_batch = {key: value.clone() for key, value in data.items()}
        plus_batch["coord"][batch_index, 0, 0] += epsilon
        minus_batch["coord"][batch_index, 0, 0] -= epsilon
        plus_single = {key: value.clone() for key, value in single_data.items()}
        minus_single = {key: value.clone() for key, value in single_data.items()}
        plus_single["coord"][0, 0, 0] += epsilon
        minus_single["coord"][0, 0, 0] -= epsilon
        batch_hessian = (
            energy_at(plus_batch, batch_index) - 2 * batch_energy + energy_at(minus_batch, batch_index)
        ) / epsilon**2
        single_hessian = (energy_at(plus_single, 0) - 2 * single_energy + energy_at(minus_single, 0)) / epsilon**2
        torch.testing.assert_close(batch_hessian, single_hessian, atol=5e-3, rtol=5e-3)


def test_global_mode2_backend_views_share_storage():
    data = _periodic_mode2_data(torch.device("cpu"))
    inputs = _mode2_backend_inputs(data, "_lr")
    assert inputs.coord._base is not None
    assert inputs.neighbor_matrix._base is not None
    assert inputs.shifts is not None and inputs.shifts._base is not None
    assert inputs.coord.storage().data_ptr() == data["coord"].storage().data_ptr()
    assert inputs.neighbor_matrix.storage().data_ptr() == data["_nbmat_kernel_lr"].storage().data_ptr()
    assert inputs.shifts.storage().data_ptr() == data["shifts_lr"].storage().data_ptr()


@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["dsf", "dftd3", "ewald", "pme"])
def test_global_mode2_periodic_backends_accept_dummy_rows(backend: str):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    data = _periodic_mode2_data(torch.device("cuda"))
    module = _module(backend).cuda()
    if backend == "dftd3":
        inputs = module._prepare_dftd3_inputs(data)
        assert inputs.coord_flat.shape[0] == data["coord"].numel() // 3
        assert inputs.neighbor_matrix.shape[0] == data["coord"].numel() // 3
        assert inputs.numbers_flat[-1] == 0
    else:
        inputs = module._dsf_inputs(data, "_lr") if backend == "dsf" else _mode2_backend_inputs(data, "_coulomb")
        if backend == "dsf":
            coord_flat, _charges, _batch_idx, neighbor_matrix, _cell, _shifts, fill_value, _num_systems = inputs
        else:
            coord_flat, neighbor_matrix, _shifts, _batch_idx, fill_value, _num_systems, _cell = inputs
        assert coord_flat.shape[0] == data["coord"].numel() // 3
        assert neighbor_matrix.shape[0] == data["coord"].numel() // 3
    if backend == "dftd3":
        fill_value = inputs.fill_value
    assert int(fill_value) == data["coord"].shape[0] * data["coord"].shape[1]


@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["dsf", "dftd3", "ewald", "pme"])
def test_global_mode2_periodic_backends_cross_batch_isolation(backend: str):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    data = _periodic_mode2_data(torch.device("cuda"))
    module = _module(backend).cuda()
    key = "energy" if backend == "dftd3" else "e_h"
    baseline = module({**data})[key].detach().clone()
    mutated = {name: value.clone() if isinstance(value, torch.Tensor) else value for name, value in data.items()}
    mutated["coord"][0, 1, 0] += 0.4
    changed = module(mutated)[key].detach()
    assert torch.allclose(changed[1], baseline[1], atol=1e-6, rtol=1e-6)
    assert not torch.allclose(changed[0], baseline[0], atol=1e-6, rtol=1e-6)


@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["dsf", "dftd3", "ewald", "pme"])
def test_global_mode2_gpu_periodic_observables(backend: str):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    data = _periodic_mode2_data(torch.device("cuda"))
    result = _module(backend).cuda()(data)
    key = "energy" if backend == "dftd3" else "e_h"
    assert torch.isfinite(result[key]).all()
