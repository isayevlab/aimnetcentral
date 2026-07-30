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


def infer_nb_mode(data: dict[str, Tensor]) -> int:
    """Derive the neighbor mode from tensor metadata alone.

    This is the definition `set_nb_mode` writes into `_nb_mode`, factored out
    so it can be evaluated without reading a tensor value. Ranks are static
    metadata, so this costs nothing and -- unlike `Tensor.item()` -- does not
    force a graph break under `torch.compile`.

    The `numbers` fallback covers data dicts assembled by hand for the packed
    layout without a neighbor matrix (`mol_sum` only needs `mol_idx`): packed
    inputs carry a flat 1D `numbers`, dense ones carry (B, N).
    """
    if "nbmat" in data:
        ndim = data["nbmat"].ndim
        if ndim == 2:
            return 1
        if ndim == 3:
            return 2
        raise ValueError("Invalid neighbor matrix shape")
    if "numbers" in data and data["numbers"].ndim == 1:
        return 1
    return 0


def get_nb_mode(data: dict[str, Tensor]) -> int:
    """Get the neighbor model.

    Eager reads the `_nb_mode` tensor, which is the serialized contract and
    stays the single source of truth there. Under `torch.compile` the same
    value is recovered from tensor metadata instead: `Tensor.item()` is a
    graph break, and this function is called ~27 times per forward, which on
    its own accounted for most of the breaks in a compiled AIMNet2 forward.

    The two agree by construction for anything that went through
    `set_nb_mode`, which every model forward calls in `prepare_input` before
    any consumer runs.
    """
    if torch.compiler.is_compiling():
        return infer_nb_mode(data)
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
        # Track processed arrays by their data pointer to avoid redundant mask
        # calculations. `Tensor.data_ptr()` is not traceable: under
        # torch.compile the resulting dict key is unhashable, and dynamo
        # responds by skipping the whole enclosing frame rather than breaking
        # the graph. Recomputing these masks -- one elementwise compare each --
        # is much cheaper than losing the compiled forward, so the dedup is
        # eager-only.
        dedup = not torch.compiler.is_compiling()
        processed: dict[int, str] = {}  # data_ptr -> mask_suffix
        for suffix in ("", "_lr", "_coulomb", "_dftd3"):
            nbmat_key = f"nbmat{suffix}"
            if nbmat_key in data:
                if dedup:
                    ptr = data[nbmat_key].data_ptr()
                    if ptr in processed:
                        data[f"mask_ij{suffix}"] = data[f"mask_ij{processed[ptr]}"]
                        continue
                    processed[ptr] = suffix
                data[f"mask_ij{suffix}"] = data[nbmat_key] == data["numbers"].shape[0] - 1
        data["_input_padded"] = torch.tensor(True)
        data["mol_sizes"] = torch.bincount(data["mol_idx"])
        # last atom is padding
        data["mol_sizes"][-1] -= 1
        # cache number of molecules as a CPU tensor (same pattern as _nb_mode),
        # so mol_sum does not need a device-to-host sync on every call. Not
        # cached under torch.compile: materializing bincount's data-dependent
        # shape as a tensor inside the traced graph trips inductor codegen,
        # and compiled mol_sum does not read the cache.
        if not torch.compiler.is_compiling():
            data["_num_mol"] = torch.tensor(data["mol_sizes"].shape[0])
    elif nb_mode == 2:
        data["mask_i"] = data["numbers"] == 0
        # Same eager-only dedup as mode 1: see the note there on data_ptr().
        dedup_nb2 = not torch.compiler.is_compiling()
        processed_nb2: dict[int, str] = {}  # data_ptr -> mask_suffix
        for suffix in ("", "_lr", "_coulomb", "_dftd3"):
            nbmat_key = f"nbmat{suffix}"
            if nbmat_key in data:
                if dedup_nb2:
                    ptr = data[nbmat_key].data_ptr()
                    if ptr in processed_nb2:
                        data[f"mask_ij{suffix}"] = data[f"mask_ij{processed_nb2[ptr]}"]
                        continue
                    processed_nb2[ptr] = suffix
                data[f"mask_ij{suffix}"] = _calc_mask_ij_mode2(data[nbmat_key], data["mask_i"])
        data["_input_padded"] = torch.tensor(True)
        data["mol_sizes"] = (~data["mask_i"]).sum(-1)
    else:
        raise ValueError(f"Invalid neighbor mode: {nb_mode}")

    return data


def _calc_mask_ij_mode2(nbmat: Tensor, mask_i: Tensor) -> Tensor:
    """Mask padded neighbor entries for batched neighbor matrices.

    Historically mode-2 callers have used both local per-system indices
    ``0..N-1`` and flattened global indices ``b*N + i``. Treat both as valid
    input conventions for masking so padded atoms are excluded consistently
    before downstream code canonicalizes its own neighbor representation.
    """
    _, N = mask_i.shape
    local_idx = torch.arange(N, device=nbmat.device).view(1, 1, N)
    local_pad = (nbmat.unsqueeze(-1) == local_idx) & mask_i.to(device=nbmat.device).unsqueeze(1).unsqueeze(1)

    global_pad_idx = torch.where(mask_i.to(device=nbmat.device).flatten())[0]
    if global_pad_idx.numel():
        global_pad = torch.isin(nbmat, global_pad_idx)
    else:
        global_pad = torch.zeros_like(nbmat, dtype=torch.bool)

    center_pad = mask_i.to(device=nbmat.device).unsqueeze(-1)
    return center_pad | local_pad.any(dim=-1) | global_pad


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


def is_input_padded(data: dict[str, Tensor]) -> bool:
    """Whether the input carries padding atoms that must be masked out.

    Packed (mode 1) and batched-neighbor (mode 2) layouts always reserve
    padding, so the answer is structural. Only the dense mode-0 layout makes
    it depend on the data, and reading that bool costs a `Tensor.item()`.

    Under `torch.compile` that read is a graph break, so this reports True
    unconditionally instead. That is exact for modes 1 and 2, and for mode 0
    it only means the mask is applied when it happens to be all-false --
    `masked_fill` with an all-false mask is the identity, on values and on
    gradients alike, so the result is unchanged either way.
    """
    if torch.compiler.is_compiling():
        return True
    return bool(data["_input_padded"].item())


def mask_i_(x: Tensor, data: dict[str, Tensor], mask_value: float = 0.0, inplace: bool = True) -> Tensor:
    nb_mode = get_nb_mode(data)
    if nb_mode == 0:
        if is_input_padded(data):
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
        mask = data["mask_i"]
        for _i in range(x.ndim - mask.ndim):
            mask = mask.unsqueeze(-1)
        if inplace:
            x.masked_fill_(mask, mask_value)
        else:
            x = x.masked_fill(mask, mask_value)
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
        if torch.compiler.is_compiling() and "charge" in data and x.device.type != "cpu":
            # `charge` carries one entry per molecule and is a genuine model
            # input, so its length is static shape metadata: reading it costs
            # no device sync and no graph break.
            #
            # Deliberately NOT `data["mol_sizes"].shape[0]`, even though it
            # is the same number: mol_sizes comes out of `torch.bincount`, so
            # its length is a data-dependent (unbacked) symbol and sizing an
            # allocation from it hands inductor a shape it cannot reason about.
            #
            # CPU is excluded on purpose: it keeps the .item() graph break
            # below. Through torch 2.10, inductor's CPU scheduler fuses the
            # atomic_add scatters of the PBC distance backward with a
            # dependent pointwise, and CppScheduling.try_loop_split then dies
            # on the fused group with `AssertionError: expected_var_ranges ==
            # extra_indexing_ranges` (a degenerate loop split). The fusion is
            # outlawed upstream by pytorch/pytorch#172301, first released in
            # torch 2.11. The break costs nothing on CPU -- .item() has no
            # device sync there -- and restores the graph partitioning that
            # avoids the fused group. Drop this exclusion when the supported
            # torch floor reaches 2.11.
            out_size = data["charge"].shape[0]
        elif torch.compiler.is_compiling():
            # data dict assembled without `charge`: dynamo handles the .item()
            # graph break, while routing it through a cached scalar tensor
            # trips inductor codegen
            # assuming mol_idx is sorted, replace with max if not
            out_size = int(idx[-1].item()) + 1
        else:
            # number of molecules is cached by calc_masks as a CPU tensor, so
            # reading it does not sync the GPU; compute and cache it here for
            # data dicts that were not built through calc_masks
            if "_num_mol" not in data:
                # assuming mol_idx is sorted, replace with max if not
                data["_num_mol"] = torch.tensor(int(idx[-1].item()) + 1)
            out_size = int(data["_num_mol"].item())

        if torch.compiler.is_compiling() and out_size == 1:
            # A single molecule makes the scatter degenerate into a plain sum
            # over atoms. Spell it that way under torch.compile: inductor
            # (2.9.1+cu128) miscompiles `scatter_add_` into a size-1 leading
            # dim when the result is gathered from later in the same graph --
            # it fuses the degenerate reduction into the consumer and returns
            # garbage, silently, with no error. Verified standalone: the same
            # pattern is correct for out_size >= 2.
            # Compile-only so eager stays bit-for-bit unchanged; mol_idx is all
            # zeros whenever out_size is 1, so the two agree exactly up to
            # summation order.
            res = x.sum(dim=0, keepdim=True)
        else:
            if x.ndim == 1:
                res = torch.zeros(out_size, device=x.device, dtype=x.dtype)
            else:
                idx = idx.unsqueeze(-1).expand(-1, x.shape[1])
                res = torch.zeros(out_size, x.shape[1], device=x.device, dtype=x.dtype)
            res.scatter_add_(0, idx, x)
    else:
        raise ValueError(f"Invalid neighbor mode: {nb_mode}")
    return res
