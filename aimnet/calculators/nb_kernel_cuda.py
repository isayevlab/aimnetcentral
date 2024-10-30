# type: ignore
import numba
import numba.cuda
from numba.core import config

config.CUDA_LOW_OCCUPANCY_WARNINGS = 0


@numba.cuda.jit(fastmath=True, cache=True)
def _nbmat_dual_cuda(coord, cutoff1_squared, cutoff2_squared, mol_idx, mol_end_idx, nbmat1, nbmat2, nnb1, nnb2):
    N = coord.shape[0]
    i = numba.cuda.grid(1)

    if i >= N:
        return

    c0 = coord[i, 0]
    c1 = coord[i, 1]
    c2 = coord[i, 2]

    maxnb1 = nbmat1.shape[1]
    maxnb2 = nbmat2.shape[1]

    _mol_idx = mol_idx[i]
    _j_start = i + 1
    _j_end = mol_end_idx[_mol_idx]

    for j in range(_j_start, _j_end):
        d0 = c0 - coord[j, 0]
        d1 = c1 - coord[j, 1]
        d2 = c2 - coord[j, 2]
        dist_squared = d0 * d0 + d1 * d1 + d2 * d2
        if dist_squared < cutoff1_squared:
            pos = numba.cuda.atomic.add(nnb1, i, 1)
            if pos < maxnb1:
                nbmat1[i, pos] = j
            pos = numba.cuda.atomic.add(nnb1, j, 1)
            if pos < maxnb1:
                nbmat1[j, pos] = i
        if dist_squared < cutoff2_squared:
            pos = numba.cuda.atomic.add(nnb2, i, 1)
            if pos < maxnb2:
                nbmat2[i, pos] = j
            pos = numba.cuda.atomic.add(nnb2, j, 1)
            if pos < maxnb2:
                nbmat2[j, pos] = i


@numba.cuda.jit(fastmath=True, cache=True)
def _nbmat_cuda(coord, cutoff1_squared, mol_idx, mol_end_idx, nbmat1, nnb1):
    N = coord.shape[0]
    i = numba.cuda.grid(1)

    if i >= N:
        return

    c0 = coord[i, 0]
    c1 = coord[i, 1]
    c2 = coord[i, 2]

    maxnb1 = nbmat1.shape[1]

    _mol_idx = mol_idx[i]
    _j_start = i + 1
    _j_end = mol_end_idx[_mol_idx]

    for j in range(_j_start, _j_end):
        d0 = c0 - coord[j, 0]
        d1 = c1 - coord[j, 1]
        d2 = c2 - coord[j, 2]
        dist_squared = d0 * d0 + d1 * d1 + d2 * d2
        if dist_squared < cutoff1_squared:
            pos = numba.cuda.atomic.add(nnb1, i, 1)
            if pos < maxnb1:
                nbmat1[i, pos] = j
            pos = numba.cuda.atomic.add(nnb1, j, 1)
            if pos < maxnb1:
                nbmat1[j, pos] = i


@numba.cuda.jit(cache=True, fastmath=True)
def _nbmat_pbc_dual_cuda(
    coord,  # N, 3
    cell,  # 3, 3
    cutoff1_squared: float,
    cutoff2_squared: float,
    shifts,  # S, 3
    nnb1,  # N
    nnb2,  # N
    nbmat1,  # N, M
    nbmat2,  # N, K
    shifts1,  # N, M, 3
    shifts2,  # N, K, 3
):
    idx = numba.cuda.grid(1)

    _n = coord.shape[0]
    _s = shifts.shape[0]

    shift_idx, atom_idx = idx // _n, idx % _n
    if shift_idx >= _s:
        return

    maxnb1 = nbmat1.shape[1]
    maxnb2 = nbmat2.shape[1]

    shift_x = shifts[shift_idx, 0]
    shift_y = shifts[shift_idx, 1]
    shift_z = shifts[shift_idx, 2]

    zero_shift = shift_x == 0 and shift_y == 0 and shift_z == 0

    shift_x = numba.float32(shift_x)
    shift_y = numba.float32(shift_y)
    shift_z = numba.float32(shift_z)

    coord_shifted_x = coord[atom_idx, 0] + shift_x * cell[0, 0] + shift_y * cell[1, 0] + shift_z * cell[2, 0]
    coord_shifted_y = coord[atom_idx, 1] + shift_x * cell[0, 1] + shift_y * cell[1, 1] + shift_z * cell[2, 1]
    coord_shifted_z = coord[atom_idx, 2] + shift_x * cell[0, 2] + shift_y * cell[1, 2] + shift_z * cell[2, 2]

    for i in range(_n):
        if zero_shift and i >= atom_idx:
            continue

        dx = coord_shifted_x - coord[i, 0]
        dy = coord_shifted_y - coord[i, 1]
        dz = coord_shifted_z - coord[i, 2]

        r2 = dx * dx + dy * dy + dz * dz

        if r2 < cutoff1_squared:
            pos = numba.cuda.atomic.add(nnb1, i, 1)
            if pos < maxnb1:
                nbmat1[i, pos] = atom_idx
                shifts1[i, pos, 0] = shift_x
                shifts1[i, pos, 1] = shift_y
                shifts1[i, pos, 2] = shift_z
            pos = numba.cuda.atomic.add(nnb1, atom_idx, 1)
            if pos < maxnb1:
                nbmat1[atom_idx, pos] = i
                shifts1[atom_idx, pos, 0] = -shift_x
                shifts1[atom_idx, pos, 1] = -shift_y
                shifts1[atom_idx, pos, 2] = -shift_z

        if r2 < cutoff2_squared:
            pos = numba.cuda.atomic.add(nnb2, i, 1)
            if pos < maxnb2:
                nbmat2[i, pos] = atom_idx
                shifts2[i, pos, 0] = shift_x
                shifts2[i, pos, 1] = shift_y
                shifts2[i, pos, 2] = shift_z
            pos = numba.cuda.atomic.add(nnb2, atom_idx, 1)
            if pos < maxnb2:
                nbmat2[atom_idx, pos] = i
                shifts2[atom_idx, pos, 0] = -shift_x
                shifts2[atom_idx, pos, 1] = -shift_y
                shifts2[atom_idx, pos, 2] = -shift_z


@numba.cuda.jit(cache=True, fastmath=True)
def _nbmat_pbc_cuda(
    coord,  # N, 3
    cell,  # 3, 3
    cutoff1_squared: float,
    shifts,  # S, 3
    nnb1,  # N
    nbmat1,  # N, M
    shifts1,  # N, M, 3
):
    idx = numba.cuda.grid(1)

    _n = coord.shape[0]
    _s = shifts.shape[0]

    shift_idx, atom_idx = idx // _n, idx % _n
    if shift_idx >= _s:
        return

    maxnb1 = nbmat1.shape[1]

    shift_x = shifts[shift_idx, 0]
    shift_y = shifts[shift_idx, 1]
    shift_z = shifts[shift_idx, 2]

    zero_shift = shift_x == 0 and shift_y == 0 and shift_z == 0

    shift_x = numba.float32(shift_x)
    shift_y = numba.float32(shift_y)
    shift_z = numba.float32(shift_z)

    coord_shifted_x = coord[atom_idx, 0] + shift_x * cell[0, 0] + shift_y * cell[1, 0] + shift_z * cell[2, 0]
    coord_shifted_y = coord[atom_idx, 1] + shift_x * cell[0, 1] + shift_y * cell[1, 1] + shift_z * cell[2, 1]
    coord_shifted_z = coord[atom_idx, 2] + shift_x * cell[0, 2] + shift_y * cell[1, 2] + shift_z * cell[2, 2]

    for i in range(_n):
        if zero_shift and i >= atom_idx:
            continue

        dx = coord_shifted_x - coord[i, 0]
        dy = coord_shifted_y - coord[i, 1]
        dz = coord_shifted_z - coord[i, 2]

        r2 = dx * dx + dy * dy + dz * dz

        if r2 < cutoff1_squared:
            pos = numba.cuda.atomic.add(nnb1, i, 1)
            if pos < maxnb1:
                nbmat1[i, pos] = atom_idx
                shifts1[i, pos, 0] = shift_x
                shifts1[i, pos, 1] = shift_y
                shifts1[i, pos, 2] = shift_z
            pos = numba.cuda.atomic.add(nnb1, atom_idx, 1)
            if pos < maxnb1:
                nbmat1[atom_idx, pos] = i
                shifts1[atom_idx, pos, 0] = -shift_x
                shifts1[atom_idx, pos, 1] = -shift_y
                shifts1[atom_idx, pos, 2] = -shift_z
