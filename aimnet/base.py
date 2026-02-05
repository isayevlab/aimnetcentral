from typing import ClassVar, Final

import torch
from torch import Tensor, nn

from aimnet import nbops


class AIMNet2Base(nn.Module):
    """Base class for AIMNet2 models. Implements pre-processing data:
    converting to right dtype and device, setting nb mode, calculating masks.
    """

    _required_keys: Final = ["coord", "numbers", "charge"]
    _required_keys_dtype: Final = [torch.float32, torch.int64, torch.float32]
    _optional_keys: Final = [
        "mult",
        "nbmat",
        "nbmat_lr",
        "mol_idx",
        "shifts",
        "shifts_lr",
        "cell",
    ]
    _optional_keys_dtype: Final = [
        torch.float32,  # mult
        torch.int64,  # nbmat
        torch.int64,  # nbmat_lr
        torch.int64,  # mol_idx
        torch.float32,  # shifts
        torch.float32,  # shifts_lr
        torch.float32,  # cell
    ]
    __constants__: ClassVar = ["_required_keys", "_required_keys_dtype", "_optional_keys", "_optional_keys_dtype"]

    def __init__(self):
        super().__init__()

    def _prepare_dtype(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        for k, d in zip(self._required_keys, self._required_keys_dtype, strict=False):
            assert k in data, f"Key {k} is required"
            data[k] = data[k].to(d)
        for k, d in zip(self._optional_keys, self._optional_keys_dtype, strict=False):
            if k in data:
                data[k] = data[k].to(d)
        return data

    def prepare_input(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        """Some sommon operations"""
        data = self._prepare_dtype(data)
        data = nbops.set_nb_mode(data)
        data = nbops.calc_masks(data)

        assert data["charge"].ndim == 1, "Charge should be 1D tensor."
        if "mult" in data:
            assert data["mult"].ndim == 1, "Mult should be 1D tensor."
        return data
