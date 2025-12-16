import logging
from collections import defaultdict

import ignite.distributed as idist
import numpy as np
import torch
from ignite.exceptions import NotComputableError
from ignite.metrics import Metric
from ignite.metrics.metric import reinit__is_reduced
from torch import Tensor


def regression_stats(pred: Tensor, true: Tensor) -> dict[str, Tensor]:
    diff = true - pred
    diff2 = diff.pow(2)
    mae = diff.abs().mean(-1)
    rmse = diff2.mean(-1).sqrt()
    true_mean = true.mean()
    tot = (true - true_mean).pow(2).to(torch.double).sum()
    res = diff2.to(torch.double).sum(-1)
    r2 = 1 - (res / tot)
    return {"mae": mae, "rmse": rmse, "r2": r2}


def cat_flatten(y_pred: Tensor, y_true: Tensor) -> tuple[Tensor, Tensor]:
    if isinstance(y_true, (list, tuple)):
        y_true = torch.cat([x.view(-1) for x in y_true])
    if isinstance(y_pred, (list, tuple)):
        _n = sum(x.numel() for x in y_pred)
        _npass = _n // y_true.numel()
        y_pred = torch.cat([x.view(_npass, -1) for x in y_pred], dim=1)
    y_true = y_true.view(-1)
    if y_pred.ndim > y_true.ndim:
        if y_pred.shape[1] != y_true.shape[0]:
            y_pred = y_pred.view(-1, y_true.shape[0])
        _npass = y_pred.shape[0]
        y_pred = y_pred.view(_npass, -1)
    else:
        y_pred = y_pred.view(-1)
    return y_pred, y_true


def _iqr(a):
    a = a.view(-1)
    k1 = 1 + round(0.25 * (a.numel() - 1))
    k2 = 1 + round(0.75 * (a.numel() - 1))
    v1 = a.kthvalue(k1).values.item()
    v2 = a.kthvalue(k2).values.item()
    return v2 - v1


def _freedman_diaconis_bins(a, max_bins=50):
    """Calculate number of hist bins using Freedman-Diaconis rule."""
    # From https://stats.stackexchange.com/questions/798/
    if a.numel() < 2:
        return 1
    h = 2 * _iqr(a) / (a.numel() ** (1 / 3))
    # fall back to sqrt(a) bins if iqr is 0
    n_bins = int(np.sqrt(a.numel())) if h == 0 else int(np.ceil((a.max().item() - a.min().item()) / h))
    return min(n_bins, max_bins)


def calculate_metrics(result, histogram=False, corrplot=False):
    keys = [k[:-5] for k in result if k.endswith("_pred")]
    for k in keys:
        y_pred = result.pop(k + "_pred")
        y_true = result.pop(k + "_true")
        y_pred, y_true = cat_flatten(y_pred, y_true)
        stats = regression_stats(y_pred, y_true)
        npass = stats["mae"].numel()
        if k.split(".")[-1] in ("energy", "forces"):
            f = 23.06  # eV to kcal/mol
        else:
            f = 1.0
        for i in range(npass):
            for m, v in stats.items():
                if m in ("mae", "rmse"):
                    v[i] = v[i] * f
                result.log(f"{k}_{m}_{i}", v[i])
        if histogram:
            err = y_pred - y_true
            for i in range(npass):
                result[f"{k}_{i}_hist"] = torch.histc(err, bins=_freedman_diaconis_bins(err))
    return result


class RegMultiMetric(Metric):
    def __init__(self, cfg: list[dict], loss_fn=None):
        super().__init__()
        self.cfg = cfg
        self.loss_fn = loss_fn

    def attach_loss(self, loss_fn):
        self.loss_fn = loss_fn

    @reinit__is_reduced
    def reset(self):
        super().reset()
        self.data = defaultdict(lambda: defaultdict(float))
        self.atoms = 0.0
        self.samples = 0.0
        self.loss = defaultdict(float)

    def _update_one(self, key: str, pred: Tensor, true: Tensor) -> None:
        e = true - pred
        e = e.view(pred.shape[0], -1) if pred.ndim > true.ndim else e.view(-1)
        d = self.data[key]
        d["sum_abs_err"] += e.abs().sum(-1).to(dtype=torch.double, device="cpu")  # type: ignore[attr-defined]
        d["sum_sq_err"] += e.pow(2).sum(-1).to(dtype=torch.double, device="cpu")  # type: ignore[attr-defined]
        d["sum_true"] += true.sum().to(dtype=torch.double, device="cpu")  # type: ignore[attr-defined]
        d["sum_sq_true"] += true.pow(2).sum().to(dtype=torch.double, device="cpu")  # type: ignore[attr-defined]

    @reinit__is_reduced
    def update(self, output) -> None:
        y_pred, y_true = output
        if y_pred is None:
            return
        for k in y_pred:
            if k not in y_true:
                continue
            with torch.no_grad():
                self._update_one(k, y_pred[k].detach(), y_true[k].detach())
            b = y_true[k].shape[0]
        self.samples += b

        _n = y_pred["_natom"]
        if _n.numel() > 1:
            self.atoms += _n.sum().item()
        else:
            self.atoms += y_pred["numbers"].shape[0] * y_pred["numbers"].shape[1]
        if self.loss_fn is not None:
            with torch.no_grad():
                loss_d = self.loss_fn(y_pred, y_true)
                for k, loss in loss_d.items():
                    if isinstance(loss, Tensor):
                        if loss.numel() > 1:
                            loss = loss.mean()
                        loss = loss.item()
                    self.loss[k] += loss * b

    def compute(self) -> dict[str, float]:
        if self.samples == 0:
            raise NotComputableError
        # Use custom reduction
        if idist.get_world_size() > 1:
            self.atoms = idist.all_reduce(self.atoms)
            self.samples = idist.all_reduce(self.samples)
            for k, loss in self.loss.items():
                self.loss[k] = idist.all_reduce(loss)  # type: ignore[attr-defined]
            for k1, v1 in self.data.items():
                for k2, v2 in v1.items():
                    self.data[k1][k2] = idist.all_reduce(v2)  # type: ignore[attr-defined]
        self._is_reduced = True

        # compute
        ret = {}
        for k in self.data:
            if k not in self.cfg:
                continue
            cfg = self.cfg[k]
            _n = self.atoms if cfg.get("peratom", False) else self.samples
            _n *= cfg.get("mult", 1.0)
            name = k
            abbr = cfg["abbr"]
            v = self.data[name]
            m = {}
            m["mae"] = v["sum_abs_err"] / _n
            m["rmse"] = (v["sum_sq_err"] / _n).sqrt()
            m["r2"] = 1.0 - v["sum_sq_err"] / (v["sum_sq_true"] - (v["sum_true"].pow(2)) / _n)  # type: ignore[attr-defined]
            for k, v in m.items():
                if k in ("mae", "rmse"):
                    v *= cfg.get("scale", 1.0)
                v = v.tolist()
                if isinstance(v, list):
                    for ii, vv in enumerate(v):
                        ret[f"{abbr}_{k}_{ii}"] = vv
                else:
                    ret[f"{abbr}_{k}"] = v
        if len(self.loss):
            for k, loss in self.loss.items():
                if not k.endswith("loss"):
                    k = k + "_loss"
                ret[k] = loss / self.samples

        logging.info(str(ret))

        return ret
