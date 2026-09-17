from functools import partial
from typing import Any

import torch
from torch import Tensor

from aimnet.config import get_module


class MTLoss:
    """Multi-target loss function with fixed weights.

    This class allows for the combination of multiple loss functions, each with a specified weight.
    The weights are normalized to sum to 1. The loss functions are applied to the predictions and
    true values, and the weighted sum of the losses is computed.

    Loss functions definition must contain keys:
        name (str): The name of the loss function.
        fn (str): The loss function (e.g. `aimnet2.train.loss.mse_loss_fn`).
        weight (float): The weight of the loss function.
        kwargs (Dict): Optional, additional keyword arguments for the loss function.

    Methods:
        __call__(y_pred, y_true):
            Computes the weighted sum of the losses from the individual loss functions.
            Args:
                y_pred (Dict[str, Tensor]): Predicted values.
                y_true (Dict[str, Tensor]): True values.
            Returns:
                Dict[str, Tensor]: total loss under key 'loss' and values for individual components.
    """

    def __init__(self, components: dict[str, Any]):
        w_sum = sum(c["weight"] for c in components.values())
        self.components = {}
        for name, c in components.items():
            kwargs = c.get("kwargs", {})
            fn = partial(get_module(c["fn"]), **kwargs)
            self.components[name] = (fn, c["weight"] / w_sum)

    def __call__(self, y_pred: dict[str, Tensor], y_true: dict[str, Tensor]) -> dict[str, Tensor]:
        loss = {}
        for name, (fn, w) in self.components.items():
            _l = fn(y_pred=y_pred, y_true=y_true)
            loss[name] = _l * w
        # special name for the total loss
        loss["loss"] = sum(loss.values())
        return loss


def mse_loss_fn(y_pred: dict[str, Tensor], y_true: dict[str, Tensor], key_pred: str, key_true: str) -> Tensor:
    """General MSE loss function"""
    x = y_true[key_true]
    y = y_pred[key_pred]
    loss = torch.nn.functional.mse_loss(x, y)
    return loss


def peratom_loss_fn(y_pred: dict[str, Tensor], y_true: dict[str, Tensor], key_pred: str, key_true: str) -> Tensor:
    """MSE for per-atom targets, excluding rows whose atomic number is zero."""
    x = y_true[key_true]
    y = y_pred[key_pred]

    numbers = y_pred.get("numbers")
    if not isinstance(numbers, Tensor) or x.ndim < numbers.ndim or x.shape[: numbers.ndim] != numbers.shape:
        raise ValueError("peratom_loss_fn requires atom-aligned numbers, prediction, and target tensors.")
    real_atoms = numbers != 0
    return torch.nn.functional.mse_loss(x[real_atoms], y[real_atoms])


def energy_loss_fn(
    y_pred: dict[str, Tensor], y_true: dict[str, Tensor], key_pred: str = "energy", key_true: str = "energy"
) -> Tensor:
    """MSE loss normalized by the number of atoms."""
    x = y_true[key_true]
    y = y_pred[key_pred]
    s = y_pred["_natom"] ** 0.5
    loss = ((x - y).pow(2) / s).mean() if y_pred["_natom"].numel() > 1 else torch.nn.functional.mse_loss(x, y) / s
    return loss
