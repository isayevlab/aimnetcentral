"""Shared primitive dispatch and public reusable-neighbor API contracts."""

from types import SimpleNamespace

import pytest
import torch

from aimnet import nbops, ops
from aimnet.calculators.neighbor_buffers import ReusableNeighborMatrix
from aimnet.modules.lr import DFTD3, LRCoulomb


def _dense():
    data = {
        "coord": torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]]),
        "numbers": torch.tensor([[1, 1, 1]]),
        "charge": torch.tensor([0.0]),
        "charges": torch.tensor([[0.2, -0.1, -0.1]]),
    }
    nbops.set_nb_mode(data)
    nbops.calc_masks(data)
    data["d_ij"], _ = ops.calc_distances(data)
    return data


@pytest.mark.parametrize("flags", [{"hessian": True}, {"training_derivatives": True}, {"compute_virial": True}])
def test_higher_derivatives_do_not_enter_first_order_streaming(monkeypatch, flags):
    module = LRCoulomb(subtract_sr=False)

    def forbidden(*args):
        pytest.fail("higher derivative request selected the first-order kernel")

    monkeypatch.setattr(module, "_coul_simple_streaming", forbidden)
    data = _dense()
    data["_inference_kernels"] = torch.tensor(True)
    data["_inference_atoms_per_system"] = torch.tensor(3)
    result = module(data, **flags)
    output = result[0] if isinstance(result, tuple) else result
    assert torch.isfinite(output["e_h"]).all()


def test_cpu_inference_optin_preserves_reference():
    module = LRCoulomb(subtract_sr=False)
    data = _dense()
    expected = module(dict(data))["e_h"]
    data["_inference_kernels"] = torch.tensor(True)
    data["_inference_atoms_per_system"] = torch.tensor(3)
    assert module._coul_simple_streaming(data) is None
    torch.testing.assert_close(module(data)["e_h"], expected, rtol=0, atol=0)


def test_public_neighbor_buffers_are_reused_and_rebuilt():
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    batch_idx, batch_ptr = torch.zeros(3, dtype=torch.int32), torch.tensor([0, 3], dtype=torch.int32)
    buffers = ReusableNeighborMatrix(1.5, batch_idx, batch_ptr, 2, 3)
    matrix, counts = buffers.fill(positions)
    assert matrix.data_ptr() == buffers.matrix.data_ptr()
    assert counts.tolist() == [1, 1, 0]
    assert matrix.tolist() == [[1, 3], [0, 3], [3, 3]]
    positions[2, 0] = 1.2
    matrix2, counts2 = buffers.fill(positions)
    assert matrix2.data_ptr() == matrix.data_ptr()
    assert counts2.data_ptr() == counts.data_ptr()
    assert counts2.tolist() == [2, 2, 2]
    # External metadata mutations cannot change the stored layout.
    batch_idx.fill_(99)
    assert buffers.batch_idx.tolist() == [0, 0, 0]


def test_public_neighbor_overflow_is_observable_without_host_read_in_fill():
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
    buffers = ReusableNeighborMatrix(1.5, torch.zeros(3, dtype=torch.int32), torch.tensor([0, 3]), 1, 3)
    matrix, counts = buffers.fill(positions)
    assert matrix.shape == (3, 1)
    assert (counts > buffers.capacity).all()


def test_dense_d3_uses_prepared_neighbor_storage():
    module = DFTD3(s8=1.0, a1=0.4, a2=4.0)
    data = _dense()
    buffers = ReusableNeighborMatrix(1.5, torch.zeros(3, dtype=torch.int32), torch.tensor([0, 3]), 2, 3)
    matrix, _ = buffers.fill(data["coord"].flatten(0, 1))
    data["_dftd3_neighbor_matrix"] = matrix
    data["_dftd3_batch_idx"] = buffers.batch_idx
    inputs = module._prepare_dftd3_inputs(data)
    assert inputs.neighbor_matrix.data_ptr() == matrix.data_ptr()
    assert inputs.num_systems == 1
    assert inputs.fill_value == 3
    assert inputs.batch_idx.data_ptr() == buffers.batch_idx.data_ptr()


def test_prepared_d3_matches_reference_cutoff_after_geometry_changes():
    module = DFTD3(s8=1.0, a1=0.4, a2=4.0, cutoff=2.5)
    data = _dense()
    data["numbers"].fill_(6)
    buffers = ReusableNeighborMatrix(2.5, torch.zeros(3, dtype=torch.int32), torch.tensor([0, 3]), 2, 3)
    for distance in (2.0, 5.0, 2.0):
        data["coord"][0, 2, 1] = distance
        regular, regular_terms = module(dict(data), compute_forces=True)
        matrix, _ = buffers.fill(data["coord"].flatten(0, 1))
        prepared, prepared_terms = module(
            {**data, "_dftd3_neighbor_matrix": matrix, "_dftd3_batch_idx": buffers.batch_idx},
            compute_forces=True,
        )
        torch.testing.assert_close(prepared["energy"], regular["energy"], rtol=0, atol=0)
        torch.testing.assert_close(prepared_terms.forces, regular_terms.forces, rtol=0, atol=0)


@pytest.mark.parametrize("has_d3,streaming", [(False, False), (False, True), (True, False), (True, True)])
def test_all_pairs_are_omitted_only_without_a_consumer(has_d3, streaming):
    from aimnet.calculators import AIMNet2Calculator

    calls = []

    def neighbors(*, positions, **kwargs):
        calls.append(kwargs.get("fill_value"))
        matrix = torch.tensor([[1, 2], [0, 2], [0, 1]])
        return matrix, torch.tensor([2, 2, 2]), None

    class LRNeighbors:
        def __call__(self, **kwargs):
            return neighbors(**kwargs)

    calc = SimpleNamespace(
        _max_mol_size=3,
        _nblist=neighbors,
        _nblist_lr=LRNeighbors(),
        _coulomb_cutoff=float("inf"),
        _coulomb_method="simple",
        external_coulomb=LRCoulomb(subtract_sr=False),
        external_dftd3=object() if has_d3 else None,
        _detect_embedded_lr_modules=lambda: False,
        _has_embedded_dispersion=lambda: False,
    )
    data = {"coord": torch.zeros((3, 3)), "mol_idx": torch.zeros(3, dtype=torch.long)}
    result = AIMNet2Calculator.make_nbmat(calc, data, streaming_coulomb=streaming)
    skip = streaming and not has_d3
    assert len(calls) == (1 if skip else 2)
    assert ("nbmat_lr" in result) is not skip


@pytest.mark.gpu
@pytest.mark.weights
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph integration")
@pytest.mark.parametrize("sparse", [False, True])
def test_graph_d3_neighbors_preserve_reference_coordination_semantics(sparse):
    from aimnet.calculators import AIMNet2Calculator

    threshold = 0 if sparse else 120
    regular = AIMNet2Calculator("aimnet2-2025", cuda_graph=False, nb_threshold=threshold)
    graph = AIMNet2Calculator("aimnet2-2025", cuda_graph=True, nb_threshold=threshold)
    for calc in (regular, graph):
        # Small cutoff exposes the difference between a dense bounded D3
        # list and the sparse all-pairs CN list shared with simple Coulomb.
        calc.external_dftd3.set_smoothing(2.5)
    data = {
        "coord": torch.tensor([[[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [3.4, 0.0, 0.0], [4.8, 0.0, 0.0]]], device="cuda"),
        "numbers": torch.full((1, 4), 6, device="cuda", dtype=torch.long),
        "charge": torch.zeros(1, device="cuda"),
    }
    try:
        for offset in (0.0, 0.0, 0.0, 5.0, 0.1):
            data["coord"][0, 3, 0] = 4.8 + offset
            reference = regular(data, forces=True)
            candidate = graph(data, forces=True)
            torch.testing.assert_close(candidate["energy"], reference["energy"], rtol=0, atol=5e-5)
            torch.testing.assert_close(candidate["forces"], reference["forces"], rtol=2e-5, atol=1e-4)
        assert graph._cuda_graph is not None
        assert len(graph._cuda_graph._engines) > 0
    finally:
        if graph._cuda_graph is not None:
            graph._cuda_graph.clear()


@pytest.mark.gpu
@pytest.mark.weights
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph integration")
def test_d3_only_sparse_recaptures_when_unified_neighbor_cutoff_changes():
    from aimnet.calculators import AIMNet2Calculator

    regular = AIMNet2Calculator("aimnet2-2025", cuda_graph=False, nb_threshold=0)
    graph = AIMNet2Calculator("aimnet2-2025", cuda_graph=True, nb_threshold=0)
    for calc in (regular, graph):
        calc.external_coulomb = None
        calc.external_dftd3.set_smoothing(2.5)
        calc.set_lr_cutoff(3.0)
    data = {
        "coord": torch.tensor([[[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [3.4, 0.0, 0.0], [4.8, 0.0, 0.0]]], device="cuda"),
        "numbers": torch.full((1, 4), 6, device="cuda", dtype=torch.long),
        "charge": torch.zeros(1, device="cuda"),
    }
    try:
        for _ in range(3):
            graph(data, forces=True)
        cache = graph._cuda_graph
        assert cache is not None and cache._engines
        captures = cache.captures
        for calc in (regular, graph):
            calc.set_lr_cutoff(4.5)
            assert calc.external_dftd3.smoothing_off == 2.5
        for _ in range(3):
            actual, expected = graph(data, forces=True), regular(data, forces=True)
            torch.testing.assert_close(actual["energy"], expected["energy"], rtol=0, atol=5e-5)
            torch.testing.assert_close(actual["forces"], expected["forces"], rtol=2e-5, atol=1e-4)
        assert cache.captures > captures
    finally:
        graph._cuda_graph.clear()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA streaming primitive")
@pytest.mark.parametrize("packed", [False, True])
def test_shared_streaming_matches_reference_energy_coordinate_and_charge_gradients(packed):
    torch.manual_seed(1729)
    B, N = 2, 7
    coord = torch.rand((B, N, 3), device="cuda") * 3
    charges = torch.randn((B, N), device="cuda") * 0.1
    numbers = torch.full((B, N), 6, dtype=torch.long, device="cuda")
    if packed:
        coord = torch.cat((coord.flatten(0, 1), coord.new_zeros((1, 3))))
        charges = torch.cat((charges.flatten(), charges.new_zeros(1)))
        numbers = torch.cat((numbers.flatten(), numbers.new_zeros(1)))
    results = []
    for optimized in (False, True):
        data = {
            "coord": coord.detach().clone().requires_grad_(),
            "charges": charges.detach().clone().requires_grad_(),
            "numbers": numbers,
            "charge": torch.zeros(B, device="cuda"),
        }
        if packed:
            rows = torch.arange(N, device="cuda").expand(N, -1)
            rows = rows[~torch.eye(N, device="cuda", dtype=torch.bool)].view(N, N - 1)
            nbmat = torch.cat(tuple(rows + b * N for b in range(B)))
            nbmat = torch.cat((nbmat, torch.full((1, N - 1), B * N, device="cuda")))
            data["nbmat"] = data["nbmat_coulomb"] = nbmat
            data["mol_idx"] = torch.cat((
                torch.arange(B, device="cuda").repeat_interleave(N),
                numbers.new_tensor([B - 1]),
            ))
        nbops.set_nb_mode(data)
        nbops.calc_masks(data)
        data["d_ij"], _ = ops.calc_distances(data)
        data["_inference_kernels"] = torch.tensor(optimized)
        data["_inference_atoms_per_system"] = torch.tensor(N)
        module = LRCoulomb(subtract_sr=False).to("cuda")
        energy = module(data)["e_h"]
        gx, gq = torch.autograd.grad((energy * energy.new_tensor([0.7, -0.3])).sum(), (data["coord"], data["charges"]))
        results.append((energy.detach(), gx, gq))
    for reference, candidate in zip(*results, strict=True):
        torch.testing.assert_close(candidate, reference, rtol=5e-5, atol=3e-6)


@pytest.mark.parametrize("capacity", [0, -1, 1.5])
def test_invalid_neighbor_capacity(capacity):
    with pytest.raises(ValueError, match="capacity"):
        ReusableNeighborMatrix(1.5, torch.zeros(3, dtype=torch.int32), torch.tensor([0, 3]), capacity, 3)
