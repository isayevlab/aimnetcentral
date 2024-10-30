import pytest
import torch

from aimnet.calculators.nbmat import TooManyNeighborsError, calc_nbmat


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def random_coord():
    coord = torch.rand((100, 3), device=get_device())
    dmat = torch.cdist(coord, coord)
    dmat[torch.eye(100, dtype=torch.bool, device=get_device())] = 0
    w = dmat > 0
    dmat_f = dmat[w]
    q = torch.tensor([0.5, 0.8], device=get_device())
    cutoff1, cutoff2 = torch.quantile(dmat_f, q).unbind()
    maxnnb1 = (w & (dmat < cutoff1)).sum(-1).max().item()
    maxnnb2 = (w & (dmat < cutoff2)).sum(-1).max().item()
    return coord, dmat, cutoff1.item(), cutoff2.item(), maxnnb1, maxnnb2


def random_coord_pbc():
    coord = torch.rand((100, 3), device=get_device())
    cell = torch.eye(3, device=get_device())
    return coord, cell


def test_calc_nbmat():
    coord, dmat, cutoff1, _, maxnb1, _ = random_coord()
    N = coord.shape[0]
    nbmat1, _, _, _ = calc_nbmat(coord, (cutoff1, None), (coord.shape[0], None))  # type: ignore

    assert nbmat1.shape[1] <= maxnb1

    diag_mask = torch.eye(N, dtype=torch.bool, device=get_device())
    eps = 1.0e-5
    nbmat1_w = torch.where(nbmat1 < N)

    # assert all distances from nbmat are less than cutoff1+eps
    assert dmat[(nbmat1_w[0], nbmat1[nbmat1_w])].max().item() < cutoff1 + eps

    # assert that all distances in not in nbmat1 are greater than cutoff1 - eps
    mask = torch.ones_like(dmat, dtype=torch.bool)
    mask[diag_mask] = False
    mask[(nbmat1_w[0], nbmat1[nbmat1_w])] = False
    assert dmat[mask].min().item() > cutoff1 - eps


def test_calc_nbmat_dual():
    coord, dmat, cutoff1, cutoff2, maxnb1, maxnb2 = random_coord()
    N = coord.shape[0]
    nbmat1, nbmat2, _, _ = calc_nbmat(coord, (cutoff1, cutoff2), (coord.shape[0], maxnb2))  # type: ignore
    assert nbmat1.shape[1] <= maxnb1
    assert nbmat2.shape[1] <= maxnb2  # type: ignore

    diag_mask = torch.eye(N, dtype=torch.bool, device=get_device())
    eps = 1.0e-5
    nbmat1_w = torch.where(nbmat1 < N)
    nbmat2_w = torch.where(nbmat2 < N)  # type: ignore

    assert dmat[(nbmat1_w[0], nbmat1[nbmat1_w])].max().item() < cutoff1 + eps
    assert dmat[(nbmat2_w[0], nbmat2[nbmat2_w])].max().item() < cutoff2 + eps  # type: ignore

    mask = torch.ones_like(dmat, dtype=torch.bool)
    mask[diag_mask] = False
    mask[(nbmat1_w[0], nbmat1[nbmat1_w])] = False
    assert dmat[mask].min().item() > cutoff1 - eps

    mask = torch.ones_like(dmat, dtype=torch.bool)
    mask[diag_mask] = False
    mask[(nbmat2_w[0], nbmat2[nbmat2_w])] = False  # type: ignore
    assert dmat[mask].min().item() > cutoff2 - eps


def test_nbmat_maxnb():
    coord, _, cutoff1, cutoff2, maxnb1, maxnb2 = random_coord()
    assert maxnb1 > 2
    assert maxnb2 > 2
    with pytest.raises(TooManyNeighborsError):
        calc_nbmat(coord, (cutoff1, None), (maxnb1 - 1, None))  # type: ignore
    with pytest.raises(TooManyNeighborsError):
        calc_nbmat(coord, (cutoff1, cutoff2), (maxnb1 - 1, maxnb2))  # type: ignore
    with pytest.raises(TooManyNeighborsError):
        calc_nbmat(coord, (cutoff1, cutoff2), (maxnb1, maxnb2 - 1))  # type: ignore


def test_nbmat_pbc():
    coord, cell = random_coord_pbc()
    N = coord.shape[0]
    maxnb1 = 400
    while True:
        try:
            nbmat1, _, shifts1, _ = calc_nbmat(coord, (1.0, None), (maxnb1, None), cell=cell)  # type: ignore
            break
        except TooManyNeighborsError:
            maxnb1 = int(maxnb1 * 1.2)

    coord_p = torch.nn.functional.pad(coord, (0, 0, 0, 1), value=0.0)
    coord_i = coord_p.unsqueeze(1)
    coord_j = coord_p[nbmat1] + shifts1.to(cell.dtype) @ cell  # type: ignore
    dist = torch.linalg.norm(coord_i - coord_j, dim=-1)
    mask = nbmat1 < N
    assert dist[mask].max().item() < 1.0 + 1.0e-5
    assert (shifts1.reshape(-1, 3).sum(0) == 0).all()  # type: ignore


def test_nbmat_dual_pbc():
    coord, cell = random_coord_pbc()
    N = coord.shape[0]
    maxnb1 = 400
    maxnb2 = 3000
    while True:
        try:
            nbmat1, nbmat2, shifts1, shifts2 = calc_nbmat(coord, (1.0, 2.0), (maxnb1, maxnb2), cell=cell)  # type: ignore
            break
        except TooManyNeighborsError:
            maxnb1 = int(maxnb1 * 1.2)
            maxnb2 = int(maxnb2 * 1.2)

    coord_p = torch.nn.functional.pad(coord, (0, 0, 0, 1), value=0.0)
    coord_i = coord_p.unsqueeze(1)
    coord_j = coord_p[nbmat1] + shifts1.to(cell.dtype) @ cell  # type: ignore
    dist = torch.linalg.norm(coord_i - coord_j, dim=-1)
    mask = nbmat1 < N
    assert dist[mask].max().item() < 1.0 + 1.0e-5
    assert (shifts1.reshape(-1, 3).sum(0) == 0).all()  # type: ignore

    coord_j = coord_p[nbmat2] + shifts2.to(cell.dtype) @ cell  # type: ignore
    dist = torch.linalg.norm(coord_i - coord_j, dim=-1)
    mask = nbmat2 < N  # type: ignore
    assert dist[mask].max().item() < 2.0 + 1.0e-5
    assert (shifts2.reshape(-1, 3).sum(0) == 0).all()  # type: ignore
