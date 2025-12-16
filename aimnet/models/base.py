from typing import ClassVar, Final

import torch
from torch import Tensor, nn

from aimnet import nbops


class AIMNet2Base(nn.Module):
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
        # Compile mode attributes
        self._compile_mode: bool = False
        self._compile_nb_mode: int = -1  # -1 = dynamic, 0/1/2 = fixed

    def enable_compile_mode(self, nb_mode: int = 0) -> None:
        """Enable compile mode with fixed nb_mode for CUDA graphs compatibility.

        This sets up the model to use compile-time constant control flow,
        avoiding .item() calls that break CUDA graph capture.

        Args:
            nb_mode: Fixed neighbor mode (0=dense, 1=sparse, 2=batched).
                     Currently only 0 is supported.
        """
        if nb_mode not in (0, 1, 2):
            raise ValueError(f"nb_mode must be 0, 1, or 2, got {nb_mode}")
        if nb_mode != 0:
            raise NotImplementedError(f"Compile mode only supports nb_mode=0 currently, got {nb_mode}")

        self._compile_mode = True
        self._compile_nb_mode = nb_mode

        # Propagate to all submodules
        for module in self.modules():
            if hasattr(module, "_compile_mode"):
                module._compile_mode = True
                module._compile_nb_mode = nb_mode

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

        if self._compile_mode:
            # In compile mode, use fixed nb_mode - no data-dependent branching
            data["_nb_mode"] = torch.tensor(self._compile_nb_mode)
            data = nbops.calc_masks_fixed_nb_mode(data, self._compile_nb_mode)
        else:
            # Dynamic mode - detect nb_mode from data
            data = nbops.set_nb_mode(data)
            data = nbops.calc_masks(data)

        assert data["charge"].ndim == 1, "Charge should be 1D tensor."
        if "mult" in data:
            assert data["mult"].ndim == 1, "Mult should be 1D tensor."
        return data
