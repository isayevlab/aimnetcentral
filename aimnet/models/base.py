from typing import ClassVar, Dict, Final

import torch
from torch import Tensor, nn

from aimnet import nbops


class AIMNet2Base(nn.Module):  # pylint: disable=abstract-method
    """Base class for AIMNet2 models. Implements pre-processing data:
    converting to right dtype and device, setting nb mode, calculating masks.
    """

    __default_dtype = torch.get_default_dtype()

    _required_keys: Final = ["coord", "numbers", "charge"]
    _required_keys_dtype: Final = [__default_dtype, torch.int64, __default_dtype]
    _optional_keys: Final = ["mult", "nbmat", "nbmat_lr", "mol_idx", "shifts", "shifts_lr", "cell"]
    _optional_keys_dtype: Final = [
        __default_dtype,
        torch.int64,
        torch.int64,
        torch.int64,
        __default_dtype,
        __default_dtype,
        __default_dtype,
    ]
    __constants__: ClassVar = ["_required_keys", "_required_keys_dtype", "_optional_keys", "_optional_keys_dtype"]

    def __init__(self):
        super().__init__()

        # flags for enableing compile and cudagraphs
        self.compile_cudagraphs = False

    def _prepare_dtype(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        for k, d in zip(self._required_keys, self._required_keys_dtype):
            assert k in data, f"Key {k} is required"
            data[k] = data[k].to(d)
        for k, d in zip(self._optional_keys, self._optional_keys_dtype):
            if k in data:
                data[k] = data[k].to(d)
        return data

    def prepare_input(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Some sommon operations"""
        data = self._prepare_dtype(data)

        if not self.compile_cudagraphs:
            data = nbops.set_nb_mode(data)
            data = nbops.calc_masks(data)
        else:

            
            # check here we have one molecule
            assert data["coord"].shape[0] == 1
            data = nbops.calc_masks_compile_cudagraphs(data, nb_mode=0)

        assert data["charge"].ndim == 1, "Charge should be 1D tensor."
        if "mult" in data:
            assert data["mult"].ndim == 1, "Mult should be 1D tensor."
        return data
