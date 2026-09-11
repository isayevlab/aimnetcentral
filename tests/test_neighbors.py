"""Tests for the adaptive neighbor-list buffer sizing."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from aimnet.calculators.neighbors import AdaptiveNeighborList


def _true_max_neighbors(positions: torch.Tensor, cutoff: float) -> int:
    """Brute-force neighbor count, excluding self."""
    d = torch.cdist(positions, positions)
    within = (d < cutoff) & ~torch.eye(len(positions), dtype=torch.bool, device=d.device)
    return int(within.sum(-1).max().item())


def _cluster(n: int, spacing: float, device: str) -> torch.Tensor:
    """Simple cubic lattice of `n` atoms; smaller spacing -> denser."""
    side = int(np.ceil(n ** (1 / 3)))
    grid = np.stack(np.meshgrid(*(np.arange(side),) * 3, indexing="ij"), -1).reshape(-1, 3)[:n]
    return torch.tensor(grid * spacing, dtype=torch.float32, device=device)


def _single_system(positions: torch.Tensor) -> torch.Tensor:
    """batch_idx for one system; the batch_naive backend requires it."""
    return torch.zeros(len(positions), dtype=torch.int32, device=positions.device)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_buffer_grows_back_after_shrinking_on_a_sparse_system(device: str) -> None:
    """A sparse system must not permanently cap the buffer for later dense ones.

    nvalchemiops reports the true neighbor count but silently truncates the
    matrix to `max_neighbors` columns, so an undersized buffer drops neighbors
    from the energy instead of raising. Combined with the shrink heuristic that
    made a calculator's results depend on what it had evaluated before.
    """
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    cutoff = 5.0
    nblist = AdaptiveNeighborList(cutoff=cutoff)

    dense = _cluster(64, spacing=1.6, device=device)
    expected = _true_max_neighbors(dense, cutoff)

    # Drive the buffer down with a sparse system, then rebuild the dense one.
    sparse = _cluster(8, spacing=4.9, device=device)
    nblist(sparse, batch_idx=_single_system(sparse))
    shrunk = nblist.max_neighbors
    nbmat, num_neighbors, _ = nblist(dense, batch_idx=_single_system(dense))

    assert shrunk < expected, "precondition: the sparse system must shrink the buffer below what dense needs"
    assert int(num_neighbors.max().item()) == expected
    assert nbmat.shape[1] >= expected, "neighbor matrix was silently truncated"
    assert nblist.max_neighbors > expected, "buffer must regrow with headroom, not to exactly the count"


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_neighbor_matrix_is_history_independent(device: str) -> None:
    """The same geometry must give the same neighbor list on a reused instance."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    cutoff = 5.0
    dense = _cluster(64, spacing=1.6, device=device)

    fresh = AdaptiveNeighborList(cutoff=cutoff)
    reference, _, _ = fresh(dense, batch_idx=_single_system(dense))

    reused = AdaptiveNeighborList(cutoff=cutoff)
    reused(dense, batch_idx=_single_system(dense))
    reused(sparse := _cluster(8, spacing=4.9, device=device), batch_idx=_single_system(sparse))
    after, _, _ = reused(dense, batch_idx=_single_system(dense))

    assert after.shape == reference.shape
    assert torch.equal(after.sort(dim=-1).values, reference.sort(dim=-1).values)


def _true_max_neighbors_pbc(positions: torch.Tensor, cell_len: float, cutoff: float) -> int:
    """Brute-force count over the +-1 periodic images of a cubic cell (valid for cutoff < cell_len)."""
    assert cutoff < cell_len
    n = len(positions)
    count = torch.zeros(n, dtype=torch.long, device=positions.device)
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            for k in (-1, 0, 1):
                shift = torch.tensor([i, j, k], dtype=positions.dtype, device=positions.device) * cell_len
                within = torch.cdist(positions, positions + shift) < cutoff
                if i == j == k == 0:
                    within &= ~torch.eye(n, dtype=torch.bool, device=positions.device)
                count += within.sum(-1)
    return int(count.max().item())


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_fresh_periodic_instance_is_not_truncated_by_the_initial_estimate(device: str) -> None:
    """A dense periodic cell exceeds the initial density estimate on the very first call.

    No history is needed: a fresh instance at a 5 A cutoff starts at 112
    columns and this 64-atom, 6.4 A cell needs more, so before the grow branch
    the first neighbor list was already silently truncated. This also covers
    the PBC branch, which the motivating +79.7 eV case ran through.
    """
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    cutoff, cell_len = 5.0, 6.4
    positions = _cluster(64, spacing=cell_len / 4, device=device)
    cell = torch.eye(3, device=device).unsqueeze(0) * cell_len
    pbc = torch.ones(1, 3, dtype=torch.bool, device=device)
    expected = _true_max_neighbors_pbc(positions, cell_len, cutoff)

    nblist = AdaptiveNeighborList(cutoff=cutoff)
    assert nblist.max_neighbors < expected, "precondition: the initial estimate must undersize this cell"
    nbmat, num_neighbors, shifts = nblist(positions, cell=cell, pbc=pbc, batch_idx=_single_system(positions))

    assert int(num_neighbors.max().item()) == expected
    assert nbmat.shape[1] >= expected, "periodic neighbor matrix was silently truncated on a fresh instance"
    assert shifts is not None and shifts.shape[:2] == nbmat.shape
