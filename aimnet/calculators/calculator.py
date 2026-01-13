import warnings
from typing import Any, ClassVar, Literal

import torch
from nvalchemiops.neighborlist import neighbor_list
from nvalchemiops.neighborlist.neighbor_utils import NeighborOverflowError
from torch import Tensor, nn

from .model_registry import get_model_path


class AIMNet2Calculator:
    """Genegic AIMNet2 calculator
    A helper class to load AIMNet2 models and perform inference.
    """

    keys_in: ClassVar[dict[str, torch.dtype]] = {"coord": torch.float, "numbers": torch.int, "charge": torch.float}
    keys_in_optional: ClassVar[dict[str, torch.dtype]] = {
        "mult": torch.float,
        "mol_idx": torch.int,
        "nbmat": torch.int,
        "nbmat_lr": torch.int,
        "nb_pad_mask": torch.bool,
        "nb_pad_mask_lr": torch.bool,
        "shifts": torch.float,
        "shifts_lr": torch.float,
        "cell": torch.float,
    }
    keys_out: ClassVar[list[str]] = ["energy", "charges", "forces", "hessian", "stress"]
    atom_feature_keys: ClassVar[list[str]] = ["coord", "numbers", "charges", "forces"]

    def __init__(self, model: str | nn.Module = "aimnet2", nb_threshold: int = 320):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(model, str):
            p = get_model_path(model)
            self.model = torch.jit.load(p, map_location=self.device)
        elif isinstance(model, nn.Module):
            self.model = model.to(self.device)
        else:
            raise TypeError("Invalid model type/name.")

        self.cutoff = self.model.cutoff
        self.lr = hasattr(self.model, "cutoff_lr")
        self.cutoff_lr = getattr(self.model, "cutoff_lr", float("inf")) if self.lr else None
        self.max_density = 0.2
        self.nb_threshold = nb_threshold

        # indicator if input was flattened
        self._batch = None
        self._max_mol_size: int = 0
        # placeholder for tensors that require grad
        self._saved_for_grad = {}
        # set flag of current Coulomb model
        coul_methods = {getattr(mod, "method", None) for mod in iter_lrcoulomb_mods(self.model)}
        if len(coul_methods) > 1:
            raise ValueError("Multiple Coulomb modules found.")
        if len(coul_methods):
            self._coulomb_method = coul_methods.pop()
        else:
            self._coulomb_method = None

    def __call__(self, *args, **kwargs):
        return self.eval(*args, **kwargs)

    def set_lrcoulomb_method(
        self, method: Literal["simple", "dsf", "ewald"], cutoff: float = 15.0, dsf_alpha: float = 0.2
    ):
        if method not in ("simple", "dsf", "ewald"):
            raise ValueError(f"Invalid method: {method}")
        for mod in iter_lrcoulomb_mods(self.model):
            mod.method = method  # type: ignore
            if method == "simple":
                self.cutoff_lr = float("inf")
            elif method == "dsf":
                self.cutoff_lr = cutoff
                mod.dsf_alpha = dsf_alpha  # type: ignore
                mod.dsf_rc = cutoff  # type: ignore
            elif method == "ewald":
                # current implementation of Ewald does not use nb mat
                self.cutoff_lr = cutoff
        self._coulomb_method = method

    def eval(self, data: dict[str, Any], forces=False, stress=False, hessian=False) -> dict[str, Tensor]:
        data = self.prepare_input(data)
        if hessian and "mol_idx" in data and data["mol_idx"][-1] > 0:
            raise NotImplementedError("Hessian calculation is not supported for multiple molecules")
        data = self.set_grad_tensors(data, forces=forces, stress=stress, hessian=hessian)
        with torch.jit.optimized_execution(False):  # type: ignore
            data = self.model(data)
        data = self.get_derivatives(data, forces=forces, stress=stress, hessian=hessian)
        data = self.process_output(data)
        return data

    def prepare_input(self, data: dict[str, Any]) -> dict[str, Tensor]:
        data = self.to_input_tensors(data)
        data = self.mol_flatten(data)
        if data.get("cell") is not None:
            if data["mol_idx"][-1] > 0:
                raise NotImplementedError("PBC with multiple molecules is not implemented yet.")
            if self._coulomb_method == "simple":
                warnings.warn("Switching to DSF Coulomb for PBC", stacklevel=1)
                self.set_lrcoulomb_method("dsf")
        if data["coord"].ndim == 2:
            data = self.make_nbmat(data)
            data = self.pad_input(data)
        return data

    def process_output(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        if data["coord"].ndim == 2:
            data = self.unpad_output(data)
        data = self.mol_unflatten(data)
        data = self.keep_only(data)
        return data

    def to_input_tensors(self, data: dict[str, Any]) -> dict[str, Tensor]:
        ret = {}
        for k in self.keys_in:
            if k not in data:
                raise KeyError(f"Missing key {k} in the input data")
            # always detach !!
            ret[k] = torch.as_tensor(data[k], device=self.device, dtype=self.keys_in[k]).detach()
        for k in self.keys_in_optional:
            if k in data and data[k] is not None:
                ret[k] = torch.as_tensor(data[k], device=self.device, dtype=self.keys_in_optional[k]).detach()
        # convert any scalar tensors to shape (1,) tensors
        for k, v in ret.items():
            if v.ndim == 0:
                ret[k] = v.unsqueeze(0)
        return ret

    def mol_flatten(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        """Flatten the input data for multiple molecules.
        Will not flatten for batched input and molecule size below threshold.
        """
        ndim = data["coord"].ndim
        if ndim == 2:
            # single molecule or already flattened
            self._batch = None
            if "mol_idx" not in data:
                data["mol_idx"] = torch.zeros(data["coord"].shape[0], dtype=torch.long, device=self.device)
                self._max_mol_size = data["coord"].shape[0]
            elif data["mol_idx"][-1] == 0:
                self._max_mol_size = len(data["mol_idx"])
            else:
                self._max_mol_size = data["mol_idx"].unique(return_counts=True)[1].max().item()

        elif ndim == 3:
            # batched input
            B, N = data["coord"].shape[:2]
            if self.nb_threshold < N or self.device == "cpu":
                self._batch = B
                data["mol_idx"] = torch.repeat_interleave(
                    torch.arange(0, B, device=self.device), torch.full((B,), N, device=self.device)
                )
                for k, v in data.items():
                    if k in self.atom_feature_keys:
                        data[k] = v.flatten(0, 1)
            else:
                self._batch = None
            self._max_mol_size = N
        return data

    def mol_unflatten(self, data: dict[str, Tensor], batch=None) -> dict[str, Tensor]:
        batch = batch if batch is not None else self._batch
        if batch is not None:
            for k, v in data.items():
                if k in self.atom_feature_keys:
                    data[k] = v.view(batch, -1, *v.shape[1:])
        return data

    def make_nbmat(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        assert self._max_mol_size > 0, "Molecule size is not set"

        # Prepare batch_idx from mol_idx
        mol_idx = data.get("mol_idx")

        if "cell" in data and data["cell"] is not None:
            data["coord"] = move_coord_to_cell(data["coord"], data["cell"], mol_idx)
            cell = data["cell"]
        else:
            cell = None

        N = data["coord"].shape[0]
        _pbc = cell is not None
        batch_idx = mol_idx.to(torch.int32) if mol_idx is not None else None

        # Prepare cell and pbc tensors for nvalchemiops
        if _pbc:
            if cell.ndim == 2:
                cell_batched = cell.unsqueeze(0)  # (1, 3, 3)
            else:
                cell_batched = cell  # (num_systems, 3, 3)
            num_systems = cell_batched.shape[0]
            pbc = torch.tensor([[True, True, True]] * num_systems, dtype=torch.bool, device=cell.device)
        else:
            cell_batched = None
            pbc = None

        while True:
            try:
                maxnb1 = calc_max_nb(self.cutoff, self.max_density)
                if cell is None:
                    maxnb1 = min(maxnb1, self._max_mol_size)

                # Determine if we need dual cutoff
                # When cutoff_lr is finite, use dual cutoff mode
                # When cutoff_lr is infinite, use the short-range cutoff for LR too (for simple Coulomb)
                _use_dual_cutoff = self.lr and self.cutoff_lr is not None and self.cutoff_lr < float("inf")

                if _use_dual_cutoff:
                    maxnb2 = calc_max_nb(self.cutoff_lr, self.max_density)
                    if cell is None:
                        maxnb2 = min(maxnb2, self._max_mol_size)

                # Call nvalchemiops neighbor_list directly
                if _use_dual_cutoff:
                    # Dual cutoff case (finite cutoff_lr)
                    # Note: use max_neighbors1/max_neighbors2 for dual cutoff functions
                    if _pbc:
                        result = neighbor_list(
                            positions=data["coord"],
                            cutoff=self.cutoff,
                            cutoff2=self.cutoff_lr,
                            cell=cell_batched,
                            pbc=pbc,
                            batch_idx=batch_idx,
                            max_neighbors1=maxnb1,
                            max_neighbors2=maxnb2,
                            half_fill=False,
                            fill_value=N,
                        )
                        nbmat1, num_nb1, shifts1, nbmat2, num_nb2, shifts2 = result
                    else:
                        result = neighbor_list(
                            positions=data["coord"],
                            cutoff=self.cutoff,
                            cutoff2=self.cutoff_lr,
                            batch_idx=batch_idx,
                            max_neighbors1=maxnb1,
                            max_neighbors2=maxnb2,
                            half_fill=False,
                            fill_value=N,
                        )
                        nbmat1, num_nb1, nbmat2, num_nb2 = result
                        shifts1 = None
                        shifts2 = None
                else:
                    # Single cutoff case
                    if _pbc:
                        result = neighbor_list(
                            positions=data["coord"],
                            cutoff=self.cutoff,
                            cell=cell_batched,
                            pbc=pbc,
                            batch_idx=batch_idx,
                            max_neighbors=maxnb1,
                            half_fill=False,
                            fill_value=N,
                        )
                        nbmat1, num_nb1, shifts1 = result
                    else:
                        result = neighbor_list(
                            positions=data["coord"],
                            cutoff=self.cutoff,
                            batch_idx=batch_idx,
                            max_neighbors=maxnb1,
                            half_fill=False,
                            fill_value=N,
                        )
                        nbmat1, num_nb1 = result
                        shifts1 = None
                    # For simple Coulomb method (cutoff_lr = inf), reuse nbmat1 as nbmat_lr
                    nbmat2 = None
                    num_nb2 = None
                    shifts2 = None

                # Process results: trim and add padding row
                nbmat1, shifts1 = process_neighbor_result(nbmat1, num_nb1, shifts1, maxnb1, N)
                if nbmat2 is not None:
                    nbmat2, shifts2 = process_neighbor_result(nbmat2, num_nb2, shifts2, maxnb2, N)

                break
            except NeighborOverflowError:
                self.max_density *= 1.2
                assert self.max_density <= 4, "Something went wrong in nbmat calculation"

        data["nbmat"] = nbmat1
        if self.lr:
            if nbmat2 is not None:
                data["nbmat_lr"] = nbmat2
            else:
                # For simple Coulomb method (cutoff_lr = inf), reuse nbmat1 as nbmat_lr
                # All short-range neighbors are within infinite long-range cutoff
                data["nbmat_lr"] = nbmat1
        if cell is not None:
            assert shifts1 is not None
            data["shifts"] = shifts1
            if self.lr:
                if shifts2 is not None:
                    data["shifts_lr"] = shifts2
                else:
                    # Reuse shifts1 for the infinite cutoff case
                    data["shifts_lr"] = shifts1
        return data

    def pad_input(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        N = data["nbmat"].shape[0]
        data["mol_idx"] = maybe_pad_dim0(data["mol_idx"], N, value=data["mol_idx"][-1].item())
        for k in ("coord", "numbers"):
            if k in data:
                data[k] = maybe_pad_dim0(data[k], N)
        return data

    def unpad_output(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        N = data["nbmat"].shape[0] - 1
        for k, v in data.items():
            if k in self.atom_feature_keys:
                data[k] = maybe_unpad_dim0(v, N)
        return data

    def set_grad_tensors(self, data: dict[str, Tensor], forces=False, stress=False, hessian=False) -> dict[str, Tensor]:
        self._saved_for_grad = {}
        if forces or hessian:
            data["coord"].requires_grad_(True)
            self._saved_for_grad["coord"] = data["coord"]
        if stress:
            assert "cell" in data and data["cell"] is not None, "Stress calculation requires cell"
            scaling = torch.eye(3, requires_grad=True, dtype=data["cell"].dtype, device=data["cell"].device)
            data["coord"] = data["coord"] @ scaling
            data["cell"] = data["cell"] @ scaling
            self._saved_for_grad["scaling"] = scaling
        return data

    def keep_only(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        ret = {}
        for k, v in data.items():
            if k in self.keys_out or (k.endswith("_std") and k[:-4] in self.keys_out):
                ret[k] = v
        return ret

    def get_derivatives(self, data: dict[str, Tensor], forces=False, stress=False, hessian=False) -> dict[str, Tensor]:
        training = getattr(self.model, "training", False)
        _create_graph = hessian or training
        x = []
        if hessian:
            forces = True
        if forces and ("forces" not in data or (_create_graph and not data["forces"].requires_grad)):
            forces = True
            x.append(self._saved_for_grad["coord"])
        if stress:
            x.append(self._saved_for_grad["scaling"])
        if x:
            tot_energy = data["energy"].sum()
            deriv = torch.autograd.grad(tot_energy, x, create_graph=_create_graph)
            if forces:
                data["forces"] = -deriv[0]
            if stress:
                dedc = deriv[0] if not forces else deriv[1]
                cell = data["cell"].detach()
                if cell.ndim == 2:
                    # Single cell (3, 3)
                    volume = cell.det().abs()
                else:
                    # Batched cells (B, 3, 3) - compute volume for each cell
                    volume = torch.linalg.det(cell).abs().unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
                data["stress"] = dedc / volume
        if hessian:
            data["hessian"] = self.calculate_hessian(data["forces"], self._saved_for_grad["coord"])
        return data

    @staticmethod
    def calculate_hessian(forces: Tensor, coord: Tensor) -> Tensor:
        # here forces have shape (N, 3) and coord has shape (N+1, 3)
        # return hessian with shape (N, 3, N, 3)
        hessian = -torch.stack([
            torch.autograd.grad(_f, coord, retain_graph=True)[0] for _f in forces.flatten().unbind()
        ]).view(-1, 3, coord.shape[0], 3)[:-1, :, :-1, :]
        return hessian


def process_neighbor_result(
    nbmat: Tensor,
    num_neighbors: Tensor,
    shifts: Tensor | None,
    maxnb: int,
    N: int,
) -> tuple[Tensor, Tensor | None]:
    """Trim neighbor matrix to actual max neighbors and add padding row.

    Args:
        nbmat: Neighbor matrix from nvalchemiops, shape (N, max_neighbors)
        num_neighbors: Number of neighbors per atom, shape (N,)
        shifts: Shift vectors for PBC or None, shape (N, max_neighbors, 3)
        maxnb: Maximum allowed neighbors
        N: Number of atoms (used as fill value for padding row)

    Returns:
        Tuple of (nbmat, shifts) with padding row added, shapes (N+1, nnb_max) and (N+1, nnb_max, 3)
    """
    nnb_max = num_neighbors.max().item()
    if nnb_max > maxnb:
        raise NeighborOverflowError(maxnb, nnb_max)

    # Trim to actual max neighbors
    nbmat = nbmat[:, :nnb_max]
    if shifts is not None:
        shifts = shifts[:, :nnb_max]

    # Add padding row
    device = nbmat.device
    dtype = nbmat.dtype
    padding_row = torch.full((1, nnb_max), N, dtype=dtype, device=device)
    nbmat = torch.cat([nbmat, padding_row], dim=0)

    if shifts is not None:
        shifts_padding = torch.zeros((1, nnb_max, 3), dtype=shifts.dtype, device=device)
        shifts = torch.cat([shifts, shifts_padding], dim=0)

    return nbmat, shifts


def maybe_pad_dim0(a: Tensor, N: int, value=0.0) -> Tensor:
    _shape_diff = N - a.shape[0]
    assert _shape_diff == 0 or _shape_diff == 1, "Invalid shape"
    if _shape_diff == 1:
        a = pad_dim0(a, value=value)
    return a


def pad_dim0(a: Tensor, value=0.0) -> Tensor:
    shapes = [0] * ((a.ndim - 1) * 2) + [0, 1]
    a = torch.nn.functional.pad(a, shapes, mode="constant", value=value)
    return a


def maybe_unpad_dim0(a: Tensor, N: int) -> Tensor:
    _shape_diff = a.shape[0] - N
    assert _shape_diff == 0 or _shape_diff == 1, "Invalid shape"
    if _shape_diff == 1:
        a = a[:-1]
    return a


def move_coord_to_cell(coord: Tensor, cell: Tensor, mol_idx: Tensor | None = None) -> Tensor:
    """Move coordinates into the periodic cell.

    Args:
        coord: Coordinates tensor, shape (N, 3) or (B, N, 3)
        cell: Cell tensor, shape (3, 3) or (B, 3, 3)
        mol_idx: Molecule index for each atom, shape (N,), required for batched cells with flat coords

    Returns:
        Coordinates wrapped into the cell
    """
    if cell.ndim == 2:
        # Single cell (3, 3)
        cell_inv = torch.linalg.inv(cell)
        coord_f = coord @ cell_inv
        coord_f = coord_f % 1
        return coord_f @ cell
    else:
        # Batched cells (B, 3, 3)
        if coord.ndim == 3:
            # Batched coords (B, N, 3) with batched cells (B, 3, 3)
            cell_inv = torch.linalg.inv(cell)  # (B, 3, 3)
            coord_f = torch.bmm(coord, cell_inv)  # (B, N, 3)
            coord_f = coord_f % 1
            return torch.bmm(coord_f, cell)
        else:
            # Flat coords (N_total, 3) with batched cells (B, 3, 3) - need mol_idx
            assert mol_idx is not None, "mol_idx required for batched cells with flat coordinates"
            cell_inv = torch.linalg.inv(cell)  # (B, 3, 3)
            # Get cell and cell_inv for each atom
            atom_cell = cell[mol_idx]  # (N_total, 3, 3)
            atom_cell_inv = cell_inv[mol_idx]  # (N_total, 3, 3)
            coord_f = torch.bmm(coord.unsqueeze(1), atom_cell_inv).squeeze(1)  # (N_total, 3)
            coord_f = coord_f % 1
            return torch.bmm(coord_f.unsqueeze(1), atom_cell).squeeze(1)


def _named_children_rec(module):
    if isinstance(module, torch.nn.Module):
        for name, child in module.named_children():
            yield name, child
            yield from _named_children_rec(child)


def iter_lrcoulomb_mods(model):
    for name, module in _named_children_rec(model):
        if name == "lrcoulomb":
            yield module


def calc_max_nb(cutoff: float, density: float = 0.2) -> int | float:
    return int(density * 4 / 3 * 3.14159 * cutoff**3) if cutoff < float("inf") else float("inf")
