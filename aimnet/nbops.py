import torch
from torch import Tensor


def set_nb_mode(data: dict[str, Tensor]) -> dict[str, Tensor]:
    """Logic to guess and set the neighbor model."""
    if "nbmat" in data:
        if data["nbmat"].ndim == 2:
            data["_nb_mode"] = torch.tensor(1)
        elif data["nbmat"].ndim == 3:
            data["_nb_mode"] = torch.tensor(2)
        else:
            raise ValueError(f"Invalid neighbor matrix shape: {data['nbmat'].shape}")
    else:
        data["_nb_mode"] = torch.tensor(0)
    return data


def get_nb_mode(data: dict[str, Tensor]) -> int:
    """Get the neighbor model."""
    return int(data["_nb_mode"].item())


def calc_masks(data: dict[str, Tensor]) -> dict[str, Tensor]:
    """Calculate neighbor masks"""
    nb_mode = get_nb_mode(data)
    if nb_mode == 0:
        data["mask_i"] = data["numbers"] == 0
        data["mask_ij"] = torch.eye(
            data["numbers"].shape[1], device=data["numbers"].device, dtype=torch.bool
        ).unsqueeze(0)
        if data["mask_i"].any():
            data["_input_padded"] = torch.tensor(True)
            data["_natom"] = data["mask_i"].logical_not().sum(-1)
            data["mol_sizes"] = (~data["mask_i"]).sum(-1)
            data["mask_ij"] = data["mask_ij"] | (data["mask_i"].unsqueeze(-2) + data["mask_i"].unsqueeze(-1))
        else:
            data["_input_padded"] = torch.tensor(False)
            data["_natom"] = torch.tensor(data["numbers"].shape[1], device=data["numbers"].device)
            data["mol_sizes"] = torch.tensor(data["numbers"].shape[1], device=data["numbers"].device)
        data["mask_ij_lr"] = data["mask_ij"]
    elif nb_mode == 1:
        # padding must be the last atom
        data["mask_i"] = torch.zeros(data["numbers"].shape[0], device=data["numbers"].device, dtype=torch.bool)
        data["mask_i"][-1] = True
        # Track processed arrays by their data pointer to avoid redundant mask calculations
        processed: dict[int, str] = {}  # data_ptr -> mask_suffix
        for suffix in ("", "_lr", "_coulomb", "_dftd3"):
            nbmat_key = f"nbmat{suffix}"
            if nbmat_key in data:
                if not torch.jit.is_scripting():
                    # data_ptr() not supported in TorchScript
                    ptr = data[nbmat_key].data_ptr()
                    if ptr in processed:
                        # Same array - reuse existing mask
                        data[f"mask_ij{suffix}"] = data[f"mask_ij{processed[ptr]}"]
                        continue
                    processed[ptr] = suffix
                data[f"mask_ij{suffix}"] = data[nbmat_key] == data["numbers"].shape[0] - 1
        data["_input_padded"] = torch.tensor(True)
        data["mol_sizes"] = torch.bincount(data["mol_idx"])
        # last atom is padding
        data["mol_sizes"][-1] -= 1
    elif nb_mode == 2:
        data["mask_i"] = data["numbers"] == 0
        w = torch.where(data["mask_i"])
        pad_idx = w[0] * data["numbers"].shape[1] + w[1]
        # Track processed arrays by their data pointer to avoid redundant mask calculations
        processed: dict[int, str] = {}  # data_ptr -> mask_suffix
        for suffix in ("", "_lr", "_coulomb", "_dftd3"):
            nbmat_key = f"nbmat{suffix}"
            if nbmat_key in data:
                if not torch.jit.is_scripting():
                    # data_ptr() not supported in TorchScript
                    ptr = data[nbmat_key].data_ptr()
                    if ptr in processed:
                        # Same array - reuse existing mask
                        data[f"mask_ij{suffix}"] = data[f"mask_ij{processed[ptr]}"]
                        continue
                    processed[ptr] = suffix
                data[f"mask_ij{suffix}"] = torch.isin(data[nbmat_key], pad_idx)
        data["_input_padded"] = torch.tensor(True)
        data["mol_sizes"] = (~data["mask_i"]).sum(-1)
    else:
        raise ValueError(f"Invalid neighbor mode: {nb_mode}")

    return data


def mask_ij_(
    x: Tensor,
    data: dict[str, Tensor],
    mask_value: float = 0.0,
    inplace: bool = True,
    suffix: str = "",
) -> Tensor:
    mask = data[f"mask_ij{suffix}"]
    for _i in range(x.ndim - mask.ndim):
        mask = mask.unsqueeze(-1)
    if inplace:
        x.masked_fill_(mask, mask_value)
    else:
        x = x.masked_fill(mask, mask_value)
    return x


def mask_i_(x: Tensor, data: dict[str, Tensor], mask_value: float = 0.0, inplace: bool = True) -> Tensor:
    nb_mode = get_nb_mode(data)
    if nb_mode == 0:
        if data["_input_padded"].item():
            mask = data["mask_i"]
            for _i in range(x.ndim - mask.ndim):
                mask = mask.unsqueeze(-1)
            if inplace:
                x.masked_fill_(mask, mask_value)
            else:
                x = x.masked_fill(mask, mask_value)
    elif nb_mode == 1:
        if inplace:
            x[-1] = mask_value
        else:
            x = torch.cat([x[:-1], torch.zeros_like(x[:1])], dim=0)
    elif nb_mode == 2:
        if inplace:
            x[:, -1] = mask_value
        else:
            x = torch.cat([x[:, :-1], torch.zeros_like(x[:, :1])], dim=1)
    else:
        raise ValueError(f"Invalid neighbor mode: {nb_mode}")
    return x


def resolve_suffix(data: dict[str, Tensor], suffixes: list[str]) -> str:
    """Try suffixes in order, return first found, raise if none exist.

    This function makes fallback behavior explicit by requiring a list
    of acceptable suffixes. Each module controls which neighbor lists
    are acceptable for its operations.

    For nb_mode=0 (no neighbor matrix), returns empty string since
    neighbor lists are not used in that mode.

    Parameters
    ----------
    data : dict
        Data dictionary containing neighbor matrices.
    suffixes : list[str]
        List of suffixes to try in priority order (e.g., ["_dftd3", "_lr"]).
        Empty string "" can be included for fallback to base nbmat.

    Returns
    -------
    str
        The first suffix that has a corresponding nbmat{suffix} in data.

    Raises
    ------
    KeyError
        If none of the suffixes have corresponding neighbor matrices.
    """
    # In nb_mode=0, there are no neighbor matrices - suffix is unused
    nb_mode = get_nb_mode(data)
    if nb_mode == 0:
        return ""

    for suffix in suffixes:
        if f"nbmat{suffix}" in data:
            return suffix

    raise KeyError(f"No neighbor matrix found for any suffix in {suffixes}")


def get_ij(x: Tensor, data: dict[str, Tensor], suffix: str = "") -> tuple[Tensor, Tensor]:
    nb_mode = get_nb_mode(data)
    if nb_mode == 0:
        x_i = x.unsqueeze(2)
        x_j = x.unsqueeze(1)
    elif nb_mode == 1:
        x_i = x.unsqueeze(1)
        idx = data[f"nbmat{suffix}"]
        x_j = torch.index_select(x, 0, idx.flatten()).unflatten(0, idx.shape)
    elif nb_mode == 2:
        x_i = x.unsqueeze(2)
        idx = data[f"nbmat{suffix}"]
        x_j = torch.index_select(x.flatten(0, 1), 0, idx.flatten()).unflatten(0, idx.shape)
    else:
        raise ValueError(f"Invalid neighbor mode: {nb_mode}")
    return x_i, x_j


def get_i(x: Tensor, data: dict[str, Tensor]) -> Tensor:
    """Get the i-component of pairwise expansion without computing j.

    This is an optimized version of get_ij when only x_i is needed,
    avoiding the expensive index_select operation for x_j.

    Parameters
    ----------
    x : Tensor
        Input tensor to expand.
    data : dict[str, Tensor]
        Data dictionary containing neighbor mode information.

    Returns
    -------
    Tensor
        The i-component with appropriate unsqueeze for the neighbor mode.
    """
    nb_mode = get_nb_mode(data)
    if nb_mode == 0:
        return x.unsqueeze(2)
    elif nb_mode == 1:
        return x.unsqueeze(1)
    elif nb_mode == 2:
        return x.unsqueeze(2)
    else:
        raise ValueError(f"Invalid neighbor mode: {nb_mode}")


def mol_sum(x: Tensor, data: dict[str, Tensor]) -> Tensor:
    nb_mode = get_nb_mode(data)
    if nb_mode in (0, 2):
        res = x.sum(dim=1)
    elif nb_mode == 1:
        assert x.ndim in (
            1,
            2,
        ), "Invalid tensor shape for mol_sum, ndim should be 1 or 2"
        idx = data["mol_idx"]
        # assuming mol_idx is sorted, replace with max if not
        out_size = int(idx[-1].item()) + 1

        if x.ndim == 1:
            res = torch.zeros(out_size, device=x.device, dtype=x.dtype)
        else:
            idx = idx.unsqueeze(-1).expand(-1, x.shape[1])
            res = torch.zeros(out_size, x.shape[1], device=x.device, dtype=x.dtype)
        res.scatter_add_(0, idx, x)
    else:
        raise ValueError(f"Invalid neighbor mode: {nb_mode}")
    return res
