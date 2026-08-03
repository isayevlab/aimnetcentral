import torch
from torch import Tensor

NBMAT_SUFFIXES = ("", "_lr", "_coulomb", "_dftd3")
_SIGNED_INTEGER_DTYPES = {torch.int8, torch.int16, torch.int32, torch.int64}


def _mode2_check(condition: Tensor, message: str) -> None:
    """Raise on CPU or queue a device-side assertion on CUDA."""
    if condition.device.type == "cuda":
        torch._assert_async(condition, message)
    elif not condition.item():
        raise ValueError(message)


def normalize_mode2_periodic_geometry(data: dict[str, Tensor], *, B: int) -> dict[str, Tensor]:
    """Normalize full-3D periodic mode-2 geometry in place."""
    cell = data.get("cell")
    pbc = data.get("pbc")
    shift_keys = [f"shifts{suffix}" for suffix in NBMAT_SUFFIXES]
    has_shifts = any(data.get(key) is not None for key in shift_keys)
    if cell is None:
        if pbc is not None:
            raise ValueError("pbc requires cell for mode-2 input.")
        if has_shifts:
            raise ValueError("shifts require cell for mode-2 input.")
        return data

    if not isinstance(cell, Tensor):
        cell = torch.as_tensor(cell)
    if B == 1:
        if cell.ndim == 2 and cell.shape == (3, 3):
            cell = cell.unsqueeze(0)
        elif cell.ndim != 3 or cell.shape != (1, 3, 3):
            raise ValueError("cell must have shape (3, 3) or (1, 3, 3) for B=1.")
    elif cell.ndim != 3 or cell.shape != (B, 3, 3):
        raise ValueError("cell must have shape (B, 3, 3) for batched mode-2 input.")
    data["cell"] = cell

    if pbc is None:
        pbc = torch.ones((B, 3), dtype=torch.bool, device=cell.device)
    else:
        pbc = torch.as_tensor(pbc, dtype=torch.bool, device=cell.device)
        if pbc.ndim == 1 and pbc.shape == (3,):
            pbc = pbc.unsqueeze(0).expand(B, -1)
        elif pbc.ndim != 2 or pbc.shape != (B, 3):
            raise ValueError("pbc must have shape (3,) or (B, 3).")
    _mode2_check(pbc.all(), "mode-2 periodic input requires full-3D pbc.")
    data["pbc"] = pbc
    return data


def validate_neighbor_suffix_layout(data: dict[str, Tensor]) -> None:
    """Require every supplied neighbor suffix to match the primary representation."""
    primary = data.get("nbmat")
    present = [
        (suffix, data.get(f"nbmat{suffix}"), data.get(f"shifts{suffix}"))
        for suffix in NBMAT_SUFFIXES
        if data.get(f"nbmat{suffix}") is not None or data.get(f"shifts{suffix}") is not None
    ]
    if not present:
        return
    if not isinstance(primary, Tensor):
        for suffix, neighbor, _shifts in present:
            if neighbor is None:
                raise ValueError(f"shifts{suffix} requires matching nbmat{suffix}.")
            if neighbor.ndim == 3:
                raise ValueError("3D suffixed neighbor matrices require a primary nbmat.")
        return
    if primary.ndim not in (2, 3):
        raise ValueError("nbmat must be 2D or 3D when suffixed neighbor matrices are supplied.")
    prefix = primary.shape[:2] if primary.ndim == 3 else primary.shape[:1]
    for suffix, neighbor, shifts in present:
        key = f"nbmat{suffix}"
        if neighbor is None:
            raise ValueError(f"{f'shifts{suffix}'} requires matching {key}.")
        if not isinstance(neighbor, Tensor) or neighbor.ndim != primary.ndim or neighbor.shape[: len(prefix)] != prefix:
            raise ValueError(f"{key} must match nbmat rank and leading shape {prefix}.")
        if shifts is not None and not isinstance(shifts, Tensor):
            raise ValueError(f"shifts{suffix} must be a tensor.")


def _validate_mode2_view_layout(value: Tensor, name: str) -> None:
    """Require flattening the batch and atom dimensions to be a view."""
    if value.ndim >= 2 and value.shape[0] and value.shape[1] and value.stride(0) != value.stride(1) * value.shape[1]:
        raise ValueError(f"{name} must be flattenable across (B, N) without a copy.")


def validate_mode2_nbmat_raw(data: dict[str, Tensor], *, suffix: str) -> None:
    """Validate one raw mode-2 neighbor matrix before dtype narrowing."""
    nbmat_key = f"nbmat{suffix}"
    shifts_key = f"shifts{suffix}"
    nbmat = data.get(nbmat_key)
    shifts = data.get(shifts_key)
    if nbmat is None:
        if shifts is not None:
            raise ValueError(f"{shifts_key} requires matching {nbmat_key}.")
        raise ValueError(f"{nbmat_key} is required for mode-2 validation.")
    if not isinstance(nbmat, Tensor):
        raise ValueError(f"{nbmat_key} must be a tensor.")  # noqa: TRY004
    if nbmat.ndim != 3:
        raise ValueError(f"{nbmat_key} must have shape (B, N, M).")
    if nbmat.dtype not in _SIGNED_INTEGER_DTYPES:
        raise ValueError(f"{nbmat_key} must use a signed integer dtype.")

    coord = data.get("coord")
    numbers = data.get("numbers")
    if not isinstance(coord, Tensor) or not isinstance(numbers, Tensor):
        raise ValueError("coord and numbers are required for mode-2 validation.")  # noqa: TRY004
    B, N, _M = nbmat.shape
    if coord.shape[:2] != (B, N) or numbers.shape != (B, N):
        raise ValueError(f"{nbmat_key} must match coord and numbers shape prefix (B, N).")
    for name, value in (("coord", coord), ("numbers", numbers), (nbmat_key, nbmat)):
        _validate_mode2_view_layout(value, name)
        if value.device != nbmat.device:
            raise ValueError(f"{name} and {nbmat_key} must be on the same device.")

    total_atoms = B * N
    if total_atoms > torch.iinfo(torch.int32).max:
        raise ValueError(f"{nbmat_key} B*N must fit in int32.")

    cell = data.get("cell")
    pbc = data.get("pbc")
    if cell is None:
        if pbc is not None:
            raise ValueError("pbc requires cell for mode-2 input.")
        if shifts is not None:
            raise ValueError(f"{shifts_key} requires cell for mode-2 input.")
    else:
        if not isinstance(cell, Tensor) or cell.device != nbmat.device:
            raise ValueError("cell and mode-2 neighbor tensors must be on the same device.")
        if cell.ndim != 3 or cell.shape != (B, 3, 3):
            raise ValueError("cell must be normalized to shape (B, 3, 3).")
        if pbc is None:
            raise ValueError("pbc must be normalized when cell is present.")
        if pbc.device != nbmat.device or pbc.shape != (B, 3):
            raise ValueError("pbc must be normalized to shape (B, 3).")
        _mode2_check(pbc.all(), "mode-2 periodic input requires full-3D pbc.")
        if shifts is None:
            raise ValueError(f"{shifts_key} is required when cell is present.")
        if shifts.shape != (*nbmat.shape, 3):
            raise ValueError(f"{shifts_key} must match {nbmat_key} shape plus a final dimension of 3.")
        if shifts.device != nbmat.device:
            raise ValueError(f"{shifts_key} and {nbmat_key} must be on the same device.")
        _validate_mode2_view_layout(shifts, shifts_key)
        if (
            shifts.dtype == torch.bool
            or shifts.dtype.is_complex
            or not (shifts.dtype in _SIGNED_INTEGER_DTYPES or shifts.dtype.is_floating_point)
        ):
            raise ValueError(f"{shifts_key} must be an integer or floating-point tensor.")
        if shifts.dtype.is_floating_point:
            _mode2_check(torch.isfinite(shifts).all(), f"{shifts_key} must be finite.")
            _mode2_check((shifts == shifts.round()).all(), f"{shifts_key} must be integral-valued.")
        _mode2_check(
            ((shifts >= -(2**31)) & (shifts < 2**31)).all(),
            f"{shifts_key} values must fit in int32.",
        )

    sentinel = total_atoms
    is_sentinel = nbmat == sentinel
    _mode2_check(
        ((nbmat >= 0) & (nbmat <= sentinel)).all(),
        f"{nbmat_key} contains an index outside [0, B*N].",
    )
    starts = torch.arange(B, device=nbmat.device, dtype=nbmat.dtype).view(B, 1, 1) * N
    in_batch = is_sentinel | ((nbmat >= starts) & (nbmat < starts + N))
    _mode2_check(in_batch.all(), f"{nbmat_key} contains an index outside its batch interval.")

    mask_i = numbers == 0
    _mode2_check(mask_i[..., -1].all(), "numbers must reserve the final atom as the final dummy.")
    _mode2_check(
        ~(mask_i[..., :-1] & ~mask_i[..., 1:]).any(),
        "numbers padding must be a contiguous tail.",
    )
    safe_idx = nbmat.clamp(0, sentinel - 1)
    padded_neighbor = numbers.flatten().index_select(0, safe_idx.flatten()).view_as(nbmat) == 0
    excluded = is_sentinel | padded_neighbor
    _mode2_check(
        ~(excluded[..., :-1] & ~excluded[..., 1:]).any(),
        f"{nbmat_key} must have a packed sentinel/padded-neighbor tail.",
    )
    _mode2_check(
        ~(mask_i.unsqueeze(-1) & ~is_sentinel).any(),
        f"{nbmat_key} padded center rows must contain only the sentinel.",
    )
    if shifts is not None:
        center_slots = mask_i.unsqueeze(-1).unsqueeze(-1).expand_as(shifts)
        neighbor_slots = excluded.unsqueeze(-1).expand_as(shifts) & ~center_slots
        _mode2_check(
            (shifts.eq(0) | ~neighbor_slots).all(),
            f"{shifts_key} must be zero for sentinel and padded-neighbor slots.",
        )
        _mode2_check(
            (shifts.eq(0) | ~center_slots).all(),
            f"{shifts_key} must be zero for padded center rows.",
        )


def _prepare_mode2_neighbor_tensors(data: dict[str, Tensor]) -> None:
    """Create mode-2 masks and safe gather indices from validated int32 inputs."""
    nbmat = data["nbmat"]
    B, N, _M = nbmat.shape
    sentinel = B * N
    dedup = not torch.compiler.is_compiling()
    previous: list[tuple[Tensor, Tensor, Tensor, Tensor]] = []
    mask_i = data["mask_i"]
    for suffix in NBMAT_SUFFIXES:
        key = f"nbmat{suffix}"
        if key not in data:
            continue
        current = data[key]
        reused = False
        if dedup:
            for source, mask_ij, gather, kernel in previous:
                if (
                    current is source
                    and current.shape == source.shape
                    and current.stride() == source.stride()
                    and current.storage_offset() == source.storage_offset()
                ):
                    data[f"mask_ij{suffix}"] = mask_ij
                    data[f"_nbmat_gather{suffix}"] = gather
                    data[f"_nbmat_kernel{suffix}"] = kernel
                    reused = True
                    break
        if reused:
            continue
        is_sentinel = current == sentinel
        safe_idx = torch.where(is_sentinel, torch.zeros_like(current), current)
        padded_neighbor = mask_i.flatten().index_select(0, safe_idx.flatten()).view_as(current)
        center_pad = mask_i.unsqueeze(-1)
        mask_ij = center_pad | is_sentinel | padded_neighbor
        gather = safe_idx.masked_fill(mask_ij, 0)
        kernel = current.masked_fill(mask_ij, sentinel)
        data[f"mask_ij{suffix}"] = mask_ij
        data[f"_nbmat_gather{suffix}"] = gather
        data[f"_nbmat_kernel{suffix}"] = kernel
        if dedup:
            previous.append((current, mask_ij, gather, kernel))


def convert_mode2_local_to_global(nbmat_local: Tensor, *, padding_mask: Tensor) -> Tensor:
    """Convert a legacy local matrix to packed global int32 indices.

    ``nbmat_local`` and ``padding_mask`` must be ``(B, N, M)`` tensors on the
    same device. The matrix must use a signed integer dtype, the mask must be
    boolean, exclusions must form a tail in every row, and the final center
    row must be fully excluded because ``N - 1`` is the required dummy atom.
    Unmasked values are local indices in ``[0, N - 1)``. The returned tensor
    is a new contiguous int32 matrix whose masked entries are the global
    sentinel ``B * N``. This helper does not reorder neighbors or shifts;
    producers owning aligned shifts must repack interleaved exclusions.

    See ``docs/calculator.md#batched-sparse-neighbor-matrices-mode-2`` for the
    complete public mode-2 contract and migration guidance.
    """
    if nbmat_local.ndim != 3 or padding_mask.shape != nbmat_local.shape:
        raise ValueError("nbmat_local and padding_mask must have identical shape (B, N, M).")
    if nbmat_local.device != padding_mask.device:
        raise ValueError("nbmat_local and padding_mask must be on the same device.")
    if nbmat_local.dtype not in _SIGNED_INTEGER_DTYPES:
        raise ValueError("nbmat_local must use a signed integer dtype.")
    if padding_mask.dtype != torch.bool:
        raise ValueError("padding_mask must be boolean.")
    B, N, _M = nbmat_local.shape
    sentinel = B * N
    if sentinel > torch.iinfo(torch.int32).max:
        raise ValueError("B*N must fit in int32.")
    _mode2_check(
        ~((padding_mask[..., :-1]) & ~padding_mask[..., 1:]).any(),
        "padding_mask must be tail-packed.",
    )
    _mode2_check(
        padding_mask[:, -1, :].all(),
        "padding_mask must exclude the final dummy center row.",
    )
    _mode2_check(
        (~padding_mask & ((nbmat_local < 0) | (nbmat_local >= N - 1))).logical_not().all(),
        "unmasked local indices are outside the local index range [0, N-1).",
    )
    local_int32 = nbmat_local.to(torch.int32)
    offsets = torch.arange(B, device=nbmat_local.device, dtype=torch.int32).view(B, 1, 1) * N
    result = torch.where(padding_mask, torch.full_like(local_int32, sentinel), local_int32 + offsets)
    return result.contiguous()


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
        _prepare_mode2_neighbor_tensors(data)
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
        idx = data[f"_nbmat_gather{suffix}"]
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
