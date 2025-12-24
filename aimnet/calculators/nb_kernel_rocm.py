import numba
import numpy as np


@numba.njit(cache=True, parallel=False, fastmath=True)
def _nbmat_dual_rocm(
    coord: np.ndarray,  # float, (N, 3)
    cutoff1_squared: float,
    cutoff2_squared: float,
    mol_idx: np.ndarray,  # int, (N,)
    mol_end_idx: np.ndarray,  # int, (M,)
    nbmat1: np.ndarray,  # int, (N, maxnb1)
    nbmat2: np.ndarray,  # int, (N, maxnb2)
    nnb1: np.ndarray,  # int, zeros, (N,)
    nnb2: np.ndarray,  # int, zeros, (N,)
):
    maxnb1 = nbmat1.shape[1]
    maxnb2 = nbmat2.shape[1]
    N = coord.shape[0]
    for i in range(N):
        c_i = coord[i]
        _mol_idx = mol_idx[i]
        _j_start = i + 1
        _j_end = mol_end_idx[_mol_idx]
        for j in range(_j_start, _j_end):
            diff = c_i - coord[j]
            dx, dy, dz = diff[0], diff[1], diff[2]
            dist2 = dx * dx + dy * dy + dz * dz
            if dist2 < cutoff1_squared:
                pos = nnb1[i]
                nnb1[i] += 1
                if pos < maxnb1:
                    nbmat1[i, pos] = j

            if dist2 < cutoff2_squared:
                pos = nnb2[i]
                nnb2[i] += 1
                if pos < maxnb2:
                    nbmat2[i, pos] = j
    _expand_nb(nnb1, nbmat1)
    _expand_nb(nnb2, nbmat2)


@numba.njit(cache=True, parallel=False, fastmath=True)
def _nbmat_rocm(
    coord: np.ndarray,  # float, (N, 3)
    cutoff1_squared: float,
    mol_idx: np.ndarray,  # int, (N,)
    mol_end_idx: np.ndarray,  # int, (M,)
    nbmat1: np.ndarray,  # int, (N, maxnb1)
    nnb1: np.ndarray,  # int, zeros, (N,)
):
    maxnb1 = nbmat1.shape[1]
    N = coord.shape[0]
    for i in range(N):
        c_i = coord[i]
        _mol_idx = mol_idx[i]
        _j_start = i + 1
        _j_end = mol_end_idx[_mol_idx]
        for j in range(_j_start, _j_end):
            diff = c_i - coord[j]
            dx, dy, dz = diff[0], diff[1], diff[2]
            dist2 = dx * dx + dy * dy + dz * dz
            if dist2 < cutoff1_squared:
                pos = nnb1[i]
                nnb1[i] += 1
                if pos < maxnb1:
                    nbmat1[i, pos] = j
    _expand_nb(nnb1, nbmat1)


@numba.njit(cache=True, inline="always")
def _expand_nb(nnb, nbmat):
    nnb_copy = nnb.copy()
    N = nnb.shape[0]
    for i in range(N):
        for m in range(nnb_copy[i]):
            if m >= nbmat.shape[1]:
                continue
            j = nbmat[i, m]
            if j < N:
                pos = nnb[j]
                nnb[j] += 1
                if pos < nbmat.shape[1]:
                    nbmat[j, pos] = i


@numba.njit(cache=True, inline="always")
def _expand_nb_pbc(nnb, nbmat, shifts):
    nnb_copy = nnb.copy()
    N = nnb.shape[0]
    for i in range(N):
        for m in range(nnb_copy[i]):
            if m >= nbmat.shape[1]:
                continue
            j = nbmat[i, m]
            if j < N:
                pos = nnb[j]
                nnb[j] += 1
                if pos < nbmat.shape[1]:
                    nbmat[j, pos] = i
                    shift = shifts[i, m]
                    shifts[j, pos] = -shift


@numba.njit(cache=True)
def _expand_shifts(nshift):
    tot_shifts = (nshift[0] + 1) * (2 * nshift[1] + 1) * (2 * nshift[2] + 1)
    shifts = np.zeros((tot_shifts, 3), dtype=np.float32)
    i = 0
    for k1 in range(-nshift[0], nshift[0] + 1):
        for k2 in range(-nshift[1], nshift[1] + 1):
            for k3 in range(-nshift[2], nshift[2] + 1):
                if k1 > 0 or (k1 == 0 and k2 > 0) or (k1 == 0 and k2 == 0 and k3 >= 0):
                    shifts[i, 0] = k1
                    shifts[i, 1] = k2
                    shifts[i, 2] = k3
                    i += 1
    shifts = shifts[:i]
    return shifts


@numba.njit(cache=True, parallel=False, fastmath=True)
def shift_coords(coord, cell, shifts):
    N = coord.shape[0]
    S = shifts.shape[0]
    # pre-compute shifted coords
    coord_shifted = np.empty((N, S, 3), dtype=coord.dtype)
    for i in range(N):
        for s in range(S):
            shift = shifts[s]
            c_x = coord[i, 0] + shift[0] * cell[0, 0] + shift[1] * cell[1, 0] + shift[2] * cell[2, 0]
            c_y = coord[i, 1] + shift[0] * cell[0, 1] + shift[1] * cell[1, 1] + shift[2] * cell[2, 1]
            c_z = coord[i, 2] + shift[0] * cell[0, 2] + shift[1] * cell[1, 2] + shift[2] * cell[2, 2]
            coord_shifted[i, s] = c_x, c_y, c_z
    return coord_shifted


@numba.njit(cache=True, parallel=False, fastmath=True)
def _nbmat_pbc_rocm(
    coord: np.ndarray,  # float, (N, 3)
    cell: np.ndarray,  # float, (3, 3)
    cutoff1_squared: float,
    shifts: np.ndarray,  # float, (S, 3)
    nnb1: np.ndarray,  # int, zeros, (N,)
    nbmat1: np.ndarray,  # int, (N, M)
    shifts1: np.ndarray,  # int, (N, M, 3)
):
    maxnb1 = nbmat1.shape[1]
    N = coord.shape[0]
    S = shifts.shape[0]

    coord_shifted = shift_coords(coord, cell, shifts)

    for i in range(N):
        c_i = coord[i]
        for s in range(S):
            shift = shifts[s]
            zero_shift = shift[0] == 0 and shift[1] == 0 and shift[2] == 0
            _j_end = i if zero_shift else N
            for j in range(_j_end):
                c_j = coord_shifted[j, s]
                dx = c_i[0] - c_j[0]
                dy = c_i[1] - c_j[1]
                dz = c_i[2] - c_j[2]
                r2 = dx * dx + dy * dy + dz * dz
                if r2 < cutoff1_squared:
                    pos = nnb1[i]
                    nnb1[i] += 1
                    if pos < maxnb1:
                        nbmat1[i, pos] = j
                        shifts1[i, pos] = shift
    _expand_nb_pbc(nnb1, nbmat1, shifts1)


@numba.njit(cache=True, parallel=False, fastmath=True)
def _nbmat_dual_pbc_rocm(
    coord: np.ndarray,  # float, (N, 3)
    cell: np.ndarray,  # float, (3, 3)
    cutoff1_squared: float,
    cutoff2_squared: float,
    shifts: np.ndarray,  # float, (S, 3)
    nnb1: np.ndarray,  # int, zeros, (N,)
    nnb2: np.ndarray,  # int, zeros, (N,)
    nbmat1: np.ndarray,  # int, (N, M)
    nbmat2: np.ndarray,  # int, (N, M)
    shifts1: np.ndarray,  # int, (N, M, 3)
    shifts2: np.ndarray,  # int, (N, M, 3)
):
    maxnb1 = nbmat1.shape[1]
    maxnb2 = nbmat2.shape[1]
    N = coord.shape[0]
    S = shifts.shape[0]

    coord_shifted = shift_coords(coord, cell, shifts)

    for i in range(N):
        c_i = coord[i]
        for s in range(S):
            shift = shifts[s]
            zero_shift = shift[0] == 0 and shift[1] == 0 and shift[2] == 0
            _j_end = i if zero_shift else N
            for j in range(_j_end):
                c_j = coord_shifted[j, s]
                dx = c_i[0] - c_j[0]
                dy = c_i[1] - c_j[1]
                dz = c_i[2] - c_j[2]
                r2 = dx * dx + dy * dy + dz * dz
                if r2 < cutoff1_squared:
                    pos = nnb1[i]
                    nnb1[i] += 1
                    if pos < maxnb1:
                        nbmat1[i, pos] = j
                        shifts1[i, pos] = shift
                if r2 < cutoff2_squared:
                    pos = nnb2[i]
                    nnb2[i] += 1
                    if pos < maxnb2:
                        nbmat2[i, pos] = j
                        shifts2[i, pos] = shift

    _expand_nb_pbc(nnb1, nbmat1, shifts1)
    _expand_nb_pbc(nnb2, nbmat2, shifts2)
