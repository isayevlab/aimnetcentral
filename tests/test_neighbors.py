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
