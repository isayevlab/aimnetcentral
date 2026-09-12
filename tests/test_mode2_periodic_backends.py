"""CPU and GPU coverage for full-3D periodic global mode 2."""

import warnings

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
    if backend in ("ewald", "pme"):
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
        if backend in ("ewald", "pme"):
            batch_forces, batch_virial = _energy_graph_forces_and_virial(backend, data)
            single_forces, single_virial = _energy_graph_forces_and_virial(backend, single_data)
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


def _energy_graph_forces_and_virial(backend: str, data: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiate an energy-graph backend (Ewald/PME) for its autograd-only observables."""
    sample = {key: value.clone() for key, value in data.items()}
    sample["coord"] = sample["coord"].detach().requires_grad_(True)
    sample["cell"] = sample["cell"].detach().requires_grad_(True)
    energy = _module(backend)(sample)["e_h"].sum()
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


# ---------------------------------------------------------------------------
# Mode 1 (flat) vs mode 2 (padded) parity for the energy-graph backends.
#
# nvalchemiops estimates the Ewald splitting parameter (and the k-space cutoff
# or PME mesh) from the per-system atom count of the ``batch_idx`` it receives.
# Mode 2 hands the kernel every padded row, so unless the parameters are
# estimated from the real atoms the energy of a system depends on how much
# padding its batch carries.  These tests pin the mode-2 result to the flat
# mode-1 result and to itself across padding widths.
# ---------------------------------------------------------------------------


def _cation_in_box(device: torch.device):
    """A +1 cation in a 12 A cubic cell.

    The 15 A neighbor list used with it is shorter than the Kolafa-Perram
    real-space cutoff (~20 A at accuracy 1e-6): fine for parity between
    layouts that share the list, not a converged absolute reference.
    """
    coord = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.1, 0.0], [-0.3, 0.9, 0.2], [0.2, -0.4, 1.0]],
        dtype=torch.float64,
        device=device,
    )
    numbers = torch.tensor([7, 1, 1, 1], device=device)
    charges = torch.tensor([0.4, 0.2, 0.2, 0.2], dtype=torch.float64, device=device)
    cell = torch.eye(3, dtype=torch.float64, device=device) * 12.0
    return coord, numbers, charges, cell


def _periodic_neighbors(coord: torch.Tensor, cell: torch.Tensor, cutoff: float = 15.0):
    from aimnet.calculators.neighbors import AdaptiveNeighborList

    pbc = torch.ones((1, 3), dtype=torch.bool, device=coord.device)
    nbmat, _num, shifts = AdaptiveNeighborList(cutoff=cutoff)(coord, cell.unsqueeze(0), pbc)
    return nbmat.to(torch.int64), shifts.to(torch.float64)


def _flat_periodic_inputs(coord, numbers, charges, cell, nbmat, shifts) -> dict[str, torch.Tensor]:
    """Mode 1: one trailing padding row, single molecule."""
    N, M = nbmat.shape
    data = {
        "coord": torch.cat([coord, coord.new_zeros(1, 3)]),
        "numbers": torch.cat([numbers, numbers.new_zeros(1)]),
        "charges": torch.cat([charges, charges.new_zeros(1)]),
        "mol_idx": torch.zeros(N + 1, dtype=torch.long, device=coord.device),
        "cell": cell,
        "pbc": torch.ones(3, dtype=torch.bool, device=coord.device),
        "nbmat": torch.cat([nbmat, torch.full((1, M), N, dtype=nbmat.dtype, device=coord.device)]),
        "shifts": torch.cat([shifts, shifts.new_zeros(1, M, 3)]),
    }
    data["nbmat_coulomb"] = data["nbmat"]
    data["shifts_coulomb"] = data["shifts"]
    return nbops.calc_masks(nbops.set_nb_mode(data))


def _padded_mode2_inputs(coord, numbers, charges, cell, nbmat, shifts, pads: int) -> dict[str, torch.Tensor]:
    """Mode 2, B=1, with ``pads`` dummy rows appended to the real atoms."""
    N, M = nbmat.shape
    sentinel = N + pads
    real = nbmat < N
    nb = torch.full((1, sentinel, M), sentinel, dtype=torch.int32, device=coord.device)
    nb[0, :N] = torch.where(real, nbmat, torch.full_like(nbmat, sentinel)).to(torch.int32)
    sh = torch.zeros((1, sentinel, M, 3), dtype=torch.float64, device=coord.device)
    sh[0, :N] = torch.where(real.unsqueeze(-1), shifts, torch.zeros_like(shifts))
    data = {
        "coord": torch.cat([coord, coord.new_zeros(pads, 3)]).unsqueeze(0),
        "numbers": torch.cat([numbers, numbers.new_zeros(pads)]).unsqueeze(0),
        "charges": torch.cat([charges, charges.new_zeros(pads)]).unsqueeze(0),
        "cell": cell.unsqueeze(0),
        "pbc": torch.ones((1, 3), dtype=torch.bool, device=coord.device),
        "nbmat": nb,
        "shifts": sh,
    }
    data["nbmat_coulomb"] = nb
    data["shifts_coulomb"] = sh
    return nbops.calc_masks(nbops.set_nb_mode(data))


def _energy_forces_cell_grad(module, data: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Energy, real-atom forces, and the cell gradient (the strain/stress input)."""
    sample = dict(data)
    sample["coord"] = sample["coord"].detach().clone().requires_grad_(True)
    sample["cell"] = sample["cell"].detach().clone().requires_grad_(True)
    energy = module(sample)["e_h"].sum()
    grad_coord, grad_cell = torch.autograd.grad(energy, (sample["coord"], sample["cell"]))
    n_real = int((sample["numbers"] != 0).sum())
    return energy.detach(), -grad_coord.reshape(-1, 3)[:n_real], grad_cell.reshape(-1, 3, 3)


@pytest.mark.parametrize("backend", ["ewald", "pme"])
def test_global_mode2_matches_flat_mode1(backend: str):
    """A padded mode-2 system reproduces the flat mode-1 energy, forces, and cell gradient."""
    device = torch.device("cpu")
    coord, numbers, charges, cell = _cation_in_box(device)
    nbmat, shifts = _periodic_neighbors(coord, cell)
    module = LRCoulomb(method=backend, subtract_sr=False, ewald_accuracy=1e-6)
    flat = _energy_forces_cell_grad(module, _flat_periodic_inputs(coord, numbers, charges, cell, nbmat, shifts))
    mode2 = _energy_forces_cell_grad(module, _padded_mode2_inputs(coord, numbers, charges, cell, nbmat, shifts, pads=1))
    for got, expected in zip(mode2, flat, strict=True):
        torch.testing.assert_close(got, expected, atol=1e-9, rtol=0.0)


@pytest.mark.parametrize("backend", ["ewald", "pme"])
def test_global_mode2_energy_independent_of_padding_width(backend: str):
    """Adding dummy rows to a system must not change its energy, forces, or cell gradient."""
    device = torch.device("cpu")
    coord, numbers, charges, cell = _cation_in_box(device)
    nbmat, shifts = _periodic_neighbors(coord, cell)
    module = LRCoulomb(method=backend, subtract_sr=False, ewald_accuracy=1e-6)
    pads_1 = _energy_forces_cell_grad(module, _padded_mode2_inputs(coord, numbers, charges, cell, nbmat, shifts, 1))
    pads_8 = _energy_forces_cell_grad(module, _padded_mode2_inputs(coord, numbers, charges, cell, nbmat, shifts, 8))
    for got, expected in zip(pads_8, pads_1, strict=True):
        torch.testing.assert_close(got, expected, atol=1e-9, rtol=0.0)


@pytest.mark.parametrize("backend", ["dsf", "dftd3"])
def test_global_mode2_explicit_terms_independent_of_padding_width(backend: str):
    """DSF and DFT-D3 energies, explicit forces, and virials ignore the padding width."""
    device = torch.device("cpu")
    coord, numbers, charges, cell = _cation_in_box(device)
    nbmat, shifts = _periodic_neighbors(coord, cell)
    module = _module(backend).double()
    key = "energy" if backend == "dftd3" else "e_h"
    results = []
    for pads in (1, 8):
        data = _padded_mode2_inputs(coord, numbers, charges, cell, nbmat, shifts, pads)
        for suffix in ("_lr", "_dftd3"):
            data[f"nbmat{suffix}"] = data["nbmat"]
            data[f"shifts{suffix}"] = data["shifts"]
        data = nbops.calc_masks(nbops.set_nb_mode(data))
        result, terms = module(data, compute_forces=True, compute_virial=True)
        results.append((result[key].detach(), terms.forces.reshape(-1, 3)[:4], terms.virial.reshape(-1, 3, 3)))
    for got, expected in zip(results[1], results[0], strict=True):
        torch.testing.assert_close(got, expected, atol=1e-9, rtol=0.0)


@pytest.mark.parametrize("backend", ["ewald", "pme"])
def test_global_mode2_empty_system_contributes_zero(backend: str):
    """A system of dummy rows only has zero energy and does not change the others.

    With a zero real-atom count the kernel's own estimate degenerates
    (Ewald: alpha = 0, PME: a zero in the batch median), so the parameters
    are estimated over the non-empty systems only.
    """
    device = torch.device("cpu")
    coord, numbers, charges, cell = _cation_in_box(device)
    nbmat, shifts = _periodic_neighbors(coord, cell)
    module = LRCoulomb(method=backend, subtract_sr=False, ewald_accuracy=1e-6)
    alone = _energy_forces_cell_grad(module, _padded_mode2_inputs(coord, numbers, charges, cell, nbmat, shifts, 1))

    single = _padded_mode2_inputs(coord, numbers, charges, cell, nbmat, shifts, 1)
    Np, M = single["nbmat"].shape[1:]
    sentinel = 2 * Np
    nb = torch.full((2, Np, M), sentinel, dtype=torch.int32, device=device)
    nb[0] = torch.where(single["nbmat"][0] == Np, torch.full_like(single["nbmat"][0], sentinel), single["nbmat"][0])
    sh = torch.zeros((2, Np, M, 3), dtype=torch.float64, device=device)
    sh[0] = single["shifts"][0]
    data = {
        "coord": torch.cat([single["coord"], torch.zeros_like(single["coord"])]),
        "numbers": torch.cat([single["numbers"], torch.zeros_like(single["numbers"])]),
        "charges": torch.cat([single["charges"], torch.zeros_like(single["charges"])]),
        "cell": cell.unsqueeze(0).expand(2, -1, -1).contiguous(),
        "pbc": torch.ones((2, 3), dtype=torch.bool, device=device),
        "nbmat": nb,
        "shifts": sh,
        "nbmat_coulomb": nb,
        "shifts_coulomb": sh,
    }
    sample = nbops.calc_masks(nbops.set_nb_mode(data))
    sample["coord"] = sample["coord"].detach().clone().requires_grad_(True)
    energies = module(sample)["e_h"]
    (grad,) = torch.autograd.grad(energies.sum(), sample["coord"])
    assert torch.isfinite(energies).all()
    assert energies[1].item() == 0.0
    torch.testing.assert_close(energies[0], alone[0], atol=1e-9, rtol=0.0)
    torch.testing.assert_close(-grad[0, :4], alone[1], atol=1e-9, rtol=0.0)
    assert torch.equal(grad[1], torch.zeros_like(grad[1]))


@pytest.mark.parametrize("backend", ["ewald", "pme"])
def test_global_mode2_warns_when_cutoff_coulomb_is_short(backend: str):
    """A supplied ``cutoff_coulomb`` below the estimated real-space cutoff warns."""
    device = torch.device("cpu")
    coord, numbers, charges, cell = _cation_in_box(device)
    nbmat, shifts = _periodic_neighbors(coord, cell)
    module = LRCoulomb(method=backend, subtract_sr=False, ewald_accuracy=1e-6)
    data = _padded_mode2_inputs(coord, numbers, charges, cell, nbmat, shifts, 1)
    data["cutoff_coulomb"] = torch.tensor([4.6], dtype=torch.float64)
    with pytest.warns(RuntimeWarning, match="real-space"):
        module(dict(data))
    data["cutoff_coulomb"] = torch.tensor([100.0], dtype=torch.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        module(dict(data))


@pytest.mark.parametrize("backend", ["ewald", "pme"])
def test_global_mode2_mixed_size_batch_matches_flat_batch(backend: str):
    """Two systems of different size in one mode-2 batch match the flat mode-1 batch.

    The reference is the flat two-molecule batch (``mol_idx`` 0/1 plus one
    padding row), i.e. what the calculator's flat path hands the kernel, so
    the batch-shared PME parameters are the same on both sides.
    """
    device = torch.device("cpu")
    coord_a, numbers_a, charges_a, cell = _cation_in_box(device)
    coord_b, numbers_b, charges_b = coord_a[:3] + 0.3, numbers_a[:3], charges_a[:3] + 0.1
    nb_a, sh_a = _periodic_neighbors(coord_a, cell)
    nb_b, sh_b = _periodic_neighbors(coord_b, cell)
    module = LRCoulomb(method=backend, subtract_sr=False, ewald_accuracy=1e-6)
    M = max(nb_a.shape[1], nb_b.shape[1])
    cells = cell.unsqueeze(0).expand(2, -1, -1).contiguous()

    # Flat mode 1: rows [a0..a3, b0..b2, pad], fill value 7, mol_idx 0/1.
    n_flat = 7
    flat_nbmat = torch.full((n_flat + 1, M), n_flat, dtype=torch.int64, device=device)
    flat_shifts = torch.zeros((n_flat + 1, M, 3), dtype=torch.float64, device=device)
    for offset, (nb, sh, n_real) in ((0, (nb_a, sh_a, 4)), (4, (nb_b, sh_b, 3))):
        real = nb < n_real
        flat_nbmat[offset : offset + n_real, : nb.shape[1]] = torch.where(
            real, nb + offset, torch.full_like(nb, n_flat)
        )
        flat_shifts[offset : offset + n_real, : nb.shape[1]] = torch.where(real.unsqueeze(-1), sh, torch.zeros_like(sh))
    flat = {
        "coord": torch.cat([coord_a, coord_b, coord_a.new_zeros(1, 3)]),
        "numbers": torch.cat([numbers_a, numbers_b, numbers_a.new_zeros(1)]),
        "charges": torch.cat([charges_a, charges_b, charges_a.new_zeros(1)]),
        "mol_idx": torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], device=device),
        "cell": cells,
        "pbc": torch.ones((2, 3), dtype=torch.bool, device=device),
        "nbmat": flat_nbmat,
        "shifts": flat_shifts,
        "nbmat_coulomb": flat_nbmat,
        "shifts_coulomb": flat_shifts,
    }
    flat_sample = nbops.calc_masks(nbops.set_nb_mode(flat))
    flat_sample["coord"] = flat_sample["coord"].detach().clone().requires_grad_(True)
    e_flat = module(flat_sample)["e_h"]
    (g_flat,) = torch.autograd.grad(e_flat.sum(), flat_sample["coord"])
    e_flat, f_flat = e_flat.detach(), -g_flat[:n_flat]

    # Mode 2: one (2, 6, M) batch, system a gets 2 dummies and b gets 3.
    Np = 6
    sentinel = 2 * Np
    nbmat = torch.full((2, Np, M), sentinel, dtype=torch.int32, device=device)
    shifts = torch.zeros((2, Np, M, 3), dtype=torch.float64, device=device)
    for b, (nb, sh, n_real) in enumerate(((nb_a, sh_a, 4), (nb_b, sh_b, 3))):
        real = nb < n_real
        nbmat[b, :n_real, : nb.shape[1]] = torch.where(real, nb + b * Np, torch.full_like(nb, sentinel)).to(torch.int32)
        shifts[b, :n_real, : nb.shape[1]] = torch.where(real.unsqueeze(-1), sh, torch.zeros_like(sh))
    coord = torch.zeros((2, Np, 3), dtype=torch.float64, device=device)
    coord[0, :4], coord[1, :3] = coord_a, coord_b
    numbers = torch.zeros((2, Np), dtype=torch.long, device=device)
    numbers[0, :4], numbers[1, :3] = numbers_a, numbers_b
    charges = torch.zeros((2, Np), dtype=torch.float64, device=device)
    charges[0, :4], charges[1, :3] = charges_a, charges_b
    data = {
        "coord": coord,
        "numbers": numbers,
        "charges": charges,
        "cell": cells,
        "pbc": torch.ones((2, 3), dtype=torch.bool, device=device),
        "nbmat": nbmat,
        "shifts": shifts,
        "nbmat_coulomb": nbmat,
        "shifts_coulomb": shifts,
    }
    sample = nbops.calc_masks(nbops.set_nb_mode(data))
    sample["coord"] = sample["coord"].detach().clone().requires_grad_(True)
    energies = module(sample)["e_h"]
    (grad,) = torch.autograd.grad(energies.sum(), sample["coord"])
    torch.testing.assert_close(energies, e_flat, atol=1e-9, rtol=0.0)
    torch.testing.assert_close(-grad[0, :4], f_flat[:4], atol=1e-9, rtol=0.0)
    torch.testing.assert_close(-grad[1, :3], f_flat[4:7], atol=1e-9, rtol=0.0)


@pytest.mark.parametrize("backend", ["dsf", "dftd3", "ewald", "pme"])
@pytest.mark.parametrize("cell_shape", [(3, 3), (1, 3, 3)])
def test_global_mode2_shared_cell_broadcasts_to_every_system(backend: str, cell_shape: tuple[int, ...]):
    """A single shared cell must apply to every system, not only to system 0.

    A shared cell must reproduce the per-system ``(B, 3, 3)`` result exactly.
    The two systems of ``_periodic_mode2_data`` are translation-equivalent, so
    their energies must also agree (to float32 rounding, and for PME to the
    mesh-interpolation error, which is not translation-invariant).
    """
    data = _periodic_mode2_data(torch.device("cpu"))
    key = "energy" if backend == "dftd3" else "e_h"
    explicit = backend in ("dsf", "dftd3")
    kwargs = {"compute_forces": True, "compute_virial": True} if explicit else {}
    reference = _module(backend)(dict(data), **kwargs)
    shared = dict(data)
    shared["cell"] = data["cell"][0].reshape(cell_shape).clone()
    result = _module(backend)(shared, **kwargs)
    if explicit:
        (reference, reference_terms), (result, result_terms) = reference, result
        torch.testing.assert_close(result_terms.forces, reference_terms.forces)
        torch.testing.assert_close(result_terms.virial, reference_terms.virial)
    reference, result = reference[key].detach(), result[key].detach()
    torch.testing.assert_close(result, reference)
    torch.testing.assert_close(result[0], result[1], atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("backend", ["dsf", "dftd3", "ewald", "pme"])
def test_global_mode2_rejects_cell_batch_mismatch(backend: str):
    data = _periodic_mode2_data(torch.device("cpu"))
    data["cell"] = data["cell"][0].expand(3, -1, -1).clone()
    with pytest.raises(ValueError, match="cell"):
        _module(backend)(data)
