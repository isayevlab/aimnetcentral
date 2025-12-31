import torch
from torch import Tensor

from .nb_kernel_cpu import _expand_shifts


class TooManyNeighborsError(Exception):
    pass


# Always import CPU kernels
from .nb_kernel_cpu import _nbmat_cpu, _nbmat_dual_cpu, _nbmat_dual_pbc_cpu, _nbmat_pbc_cpu

if torch.cuda.is_available():
    import numba.cuda

    if not numba.cuda.is_available():
        raise ImportError("PyTorch CUDA is available, but Numba CUDA is not available.")
    _numba_cuda_available = True
    from .nb_kernel_cuda import _nbmat_cuda, _nbmat_dual_cuda, _nbmat_pbc_cuda, _nbmat_pbc_dual_cuda
else:
    _numba_cuda_available = False


def calc_nbmat(
    coord: Tensor,
    cutoffs: tuple[float, float | None],
    maxnb: tuple[int, int | None],
    cell: Tensor | None = None,
    mol_idx: Tensor | None = None,
):
    device = coord.device
    N = coord.shape[0]

    _pbc = cell is not None
    if _pbc and mol_idx is not None and mol_idx[-1] > 0:
        raise ValueError("Multiple molecules are not supported with PBC.")

    if mol_idx is None:
        mol_idx = torch.zeros(N, dtype=torch.long, device=device)
        mol_end_idx = torch.tensor([N], dtype=torch.long, device=device)
    else:
        _, mol_size = torch.unique(mol_idx, return_counts=True)
        mol_end_idx = mol_size.cumsum(0)

    _cuda = device.type == "cuda" and _numba_cuda_available

    # Select appropriate kernels based on device
    if _cuda:
        _kernel_nbmat = _nbmat_cuda
        _kernel_nbmat_dual = _nbmat_dual_cuda
        _kernel_nbmat_pbc = _nbmat_pbc_cuda
        _kernel_nbmat_pbc_dual = _nbmat_pbc_dual_cuda
    else:
        _kernel_nbmat = _nbmat_cpu
        _kernel_nbmat_dual = _nbmat_dual_cpu
        _kernel_nbmat_pbc = _nbmat_pbc_cpu
        _kernel_nbmat_pbc_dual = _nbmat_dual_pbc_cpu

    _dual_cutoff = cutoffs[1] is not None
    if _dual_cutoff and maxnb[1] is None:
        raise ValueError("maxnb[1] must be specified for dual cutoff.")

    nnb1 = torch.zeros(N, dtype=torch.long, device=device)
    nbmat1 = torch.full((N + 1, maxnb[0]), N, dtype=torch.long, device=device)

    if _dual_cutoff:
        nnb2 = torch.zeros(N, dtype=torch.long, device=device)
        nbmat2 = torch.full((N + 1, maxnb[1]), N, dtype=torch.long, device=device)  # type: ignore

    if _pbc:
        cell_inv = torch.inverse(cell)  # type: ignore[arg-type]
        cutoff = max(cutoffs) if _dual_cutoff else cutoffs[0]  # type: ignore
        nshift = torch.ceil(cutoff * cell_inv.norm(dim=-1)).to(torch.long).cpu().numpy()
        shifts = _expand_shifts(nshift)
        S = shifts.shape[0]
        shifts = torch.from_numpy(shifts).to(device)
        shifts1 = torch.zeros(N + 1, maxnb[0], 3, dtype=torch.long, device=device)
        if _dual_cutoff:
            shifts2 = torch.zeros(N + 1, maxnb[1], 3, dtype=torch.long, device=device)  # type: ignore
    else:
        S = 1

    # convert tensors and launch the kernel
    if _cuda:
        _coord = numba.cuda.as_cuda_array(coord)
        _mol_idx = numba.cuda.as_cuda_array(mol_idx)
        _mol_end_idx = numba.cuda.as_cuda_array(mol_end_idx)
        _nnb1 = numba.cuda.as_cuda_array(nnb1)
        _nbmat1 = numba.cuda.as_cuda_array(nbmat1)
        if _dual_cutoff:
            _nnb2 = numba.cuda.as_cuda_array(nnb2)
            _nbmat2 = numba.cuda.as_cuda_array(nbmat2)
        if _pbc:
            _cell = numba.cuda.as_cuda_array(cell)
            _shifts = numba.cuda.as_cuda_array(shifts)
            _shifts1 = numba.cuda.as_cuda_array(shifts1)
            if _dual_cutoff:
                _shifts2 = numba.cuda.as_cuda_array(shifts2)
        threads_per_block = 32
        blocks_per_grid = (N * S + (threads_per_block - 1)) // threads_per_block

        if _pbc:
            if _dual_cutoff:
                _kernel_nbmat_pbc_dual[blocks_per_grid, threads_per_block](  # type: ignore
                    _coord,
                    _cell,
                    cutoffs[0] ** 2,
                    cutoffs[1] ** 2,  # type: ignore
                    _shifts,
                    _nnb1,
                    _nnb2,
                    _nbmat1,
                    _nbmat2,
                    _shifts1,
                    _shifts2,
                )
            else:
                _kernel_nbmat_pbc[blocks_per_grid, threads_per_block](  # type: ignore
                    _coord,
                    _cell,
                    cutoffs[0] ** 2,
                    _shifts,
                    _nnb1,
                    _nbmat1,
                    _shifts1,
                )
        else:
            if _dual_cutoff:
                _kernel_nbmat_dual[blocks_per_grid, threads_per_block](  # type: ignore
                    _coord,
                    cutoffs[0] ** 2,
                    cutoffs[1] ** 2,  # type: ignore
                    _mol_idx,
                    _mol_end_idx,
                    _nbmat1,
                    _nbmat2,
                    _nnb1,
                    _nnb2,
                )
            else:
                _kernel_nbmat[blocks_per_grid, threads_per_block](  # type: ignore
                    _coord,
                    cutoffs[0] ** 2,
                    _mol_idx,
                    _mol_end_idx,
                    _nbmat1,
                    _nnb1,
                )

    else:
        _coord = coord.numpy()
        _mol_idx = mol_idx.numpy()
        _mol_end_idx = mol_end_idx.numpy()
        _nnb1 = nnb1.numpy()
        _nbmat1 = nbmat1.numpy()
        if _dual_cutoff:
            _nnb2 = nnb2.numpy()
            _nbmat2 = nbmat2.numpy()
        if _pbc:
            _cell = cell.numpy()  # type: ignore[union-attr]
            _shifts = shifts.numpy()

        if _pbc:
            _shifts1 = shifts1.numpy()
            if _dual_cutoff:
                _shifts2 = shifts2.numpy()
                _kernel_nbmat_pbc_dual(
                    _coord,
                    _cell,
                    cutoffs[0] ** 2,
                    cutoffs[1] ** 2,  # type: ignore
                    _shifts,
                    _nnb1,
                    _nnb2,
                    _nbmat1,
                    _nbmat2,
                    _shifts1,  # type: ignore
                    _shifts2,  # type: ignore
                )  # type: ignore
            else:
                _kernel_nbmat_pbc(_coord, _cell, cutoffs[0] ** 2, _shifts, _nnb1, _nbmat1, _shifts1)  # type: ignore
        else:
            if _dual_cutoff:
                _kernel_nbmat_dual(
                    _coord,
                    cutoffs[0] ** 2,
                    cutoffs[1] ** 2,  # type: ignore
                    _mol_idx,
                    _mol_end_idx,
                    _nbmat1,
                    _nbmat2,
                    _nnb1,
                    _nnb2,
                )  # type: ignore
            else:
                _kernel_nbmat(_coord, cutoffs[0] ** 2, _mol_idx, _mol_end_idx, _nbmat1, _nnb1)

    if not _pbc:
        shifts1 = None  # type: ignore[assignment]
        shifts2 = None  # type: ignore[assignment]

    nnb1_max = nnb1.max().item()
    if nnb1_max > maxnb[0]:
        raise TooManyNeighborsError(f"maxnb is too small: {nnb1_max=}, {maxnb=}")
    nbmat1 = nbmat1[:, :nnb1_max]  # type: ignore
    if _pbc:
        shifts1 = shifts1[:, :nnb1_max]  # type: ignore
    if _dual_cutoff:
        nnb2_max = nnb2.max().item()
        if nnb2_max > maxnb[1]:  # type: ignore
            raise TooManyNeighborsError(f"maxnb is too small: {nnb1_max=}, {nnb2_max=}, {maxnb=}")
        nbmat2 = nbmat2[:, :nnb2_max]
        if _pbc:
            shifts2 = shifts2[:, :nnb2_max]  # type: ignore
    else:
        nbmat2 = None
        if _pbc:
            shifts2 = None
    return nbmat1, nbmat2, shifts1, shifts2
