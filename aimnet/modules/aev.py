import math

import torch
from torch import Tensor, nn

from aimnet import nbops, ops

from aimnet.kernels import conv_sv_2d_sp


class AEVSV(nn.Module):
    """AEV module to expand distances and vectors toneighbors over shifted Gaussian basis functions.

    Parameters:
    -----------
    rmin : float, optional
        Minimum distance for the Gaussian basis functions. Default is 0.8.
    rc_s : float, optional
        Cutoff radius for scalar features. Default is 5.0.
    nshifts_s : int, optional
        Number of shifts for scalar features. Default is 16.
    eta_s : Optional[float], optional
        Width of the Gaussian basis functions for scalar features. Will estimate reasonable default.
    rc_v : Optional[float], optional
        Cutoff radius for vector features. Default is same as `rc_s`.
    nshifts_v : Optional[int], optional
        Number of shifts for vector features. Default is same as `nshifts_s`
    eta_v : Optional[float], optional
        Width of the Gaussian basis functions for vector features. Will estimate reasonable default.
    shifts_s : Optional[List[float]], optional
        List of shifts for scalar features. Default equidistant between `rmin` and `rc_s`
    shifts_v : Optional[List[float]], optional
        List of shifts for vector features. Default equidistant between `rmin` and `rc_v`
    """

    def __init__(
        self,
        rmin: float = 0.8,
        rc_s: float = 5.0,
        nshifts_s: int = 16,
        eta_s: float | None = None,
        rc_v: float | None = None,
        nshifts_v: int | None = None,
        eta_v: float | None = None,
        shifts_s: list[float] | None = None,
        shifts_v: list[float] | None = None,
    ):
        super().__init__()

        self._init_basis(rc_s, eta_s, nshifts_s, shifts_s, rmin, mod="_s")
        if rc_v is not None:
            if rc_v > rc_s:
                raise ValueError("rc_v must be less than or equal to rc_s")
            if nshifts_v is None:
                raise ValueError("nshifts_v must not be None")
            self._init_basis(rc_v, eta_v, nshifts_v, shifts_v, rmin, mod="_v")
            self._dual_basis = True
        else:
            # dummy init
            self._init_basis(rc_s, eta_s, nshifts_s, shifts_s, rmin, mod="_v")
            self._dual_basis = False

        self.dmat_fill = rc_s

    def _init_basis(self, rc, eta, nshifts, shifts, rmin, mod="_s"):
        self.register_parameter(
            "rc" + mod,
            nn.Parameter(torch.tensor(rc, dtype=torch.float), requires_grad=False),
        )
        if eta is None:
            eta = (1 / ((rc - rmin) / nshifts)) ** 2
        self.register_parameter(
            "eta" + mod,
            nn.Parameter(torch.tensor(eta, dtype=torch.float), requires_grad=False),
        )
        if shifts is None:
            shifts = torch.linspace(rmin, rc, nshifts + 1)[:nshifts]
        else:
            shifts = torch.as_tensor(shifts, dtype=torch.float)
        self.register_parameter("shifts" + mod, nn.Parameter(shifts, requires_grad=False))

    def forward(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        # shapes (..., m) and (..., m, 3)
        d_ij, r_ij = ops.calc_distances(data)
        data["d_ij"] = d_ij
        # shapes (..., m, g, 4) where 4 = 1 scalar + 3 vector
        g_sv = self._calc_aev(r_ij, d_ij, data)
        data["g_sv"] = g_sv
        return data

    def _calc_aev(self, r_ij: Tensor, d_ij: Tensor, data: dict[str, Tensor]) -> Tensor:
        fc_ij = ops.cosine_cutoff(d_ij, self.rc_s)  # (..., m)
        fc_ij = nbops.mask_ij_(fc_ij, data, 0.0)
        # (..., m, nshifts) * (..., m, 1) -> (..., m, shitfs)
        gs = ops.exp_expand(d_ij, self.shifts_s, self.eta_s) * fc_ij.unsqueeze(-1)
        u_ij = r_ij / d_ij.unsqueeze(-1)  # (..., m, 3) / (..., m, 1) -> (..., m, 3)
        # (..., m, 1, shifts), (..., m, 3, 1) -> (..., m, shifts, 3)
        gv = gs.unsqueeze(-1) * u_ij.unsqueeze(-2)
        g_sv = torch.cat([gs.unsqueeze(-1), gv], dim=-1)
        return g_sv


class ConvSV(nn.Module):
    """AIMNet2 type convolution: encoding of local environment which combines geometry of local environment and atomic features.

    Parameters:
    -----------
    nshifts_s : int
        Number of shifts (gaussian basis functions) for scalar convolution.
    nchannel : int
        Number of feature channels for atomic features.
    d2features : bool, optional
        Flag indicating whether to use 2D features. Default is False.
    do_vector : bool, optional
        Flag indicating whether to perform vector convolution. Default is True.
    nshifts_v : Optional[int], optional
        Number of shifts for vector convolution. If not provided, defaults to the value of nshifts_s.
    ncomb_v : Optional[int], optional
        Number of linear combinations for vector features. If not provided, defaults to the value of nshifts_v.
    """

    def __init__(
        self,
        nshifts_s: int,
        nchannel: int,
        d2features: bool = False,
        do_vector: bool = True,
        nshifts_v: int | None = None,
        ncomb_v: int | None = None,
    ):
        super().__init__()
        nshifts_v = nshifts_v or nshifts_s
        ncomb_v = ncomb_v or nshifts_v
        agh = _init_ahg(nchannel, nshifts_v, ncomb_v)
        self.register_parameter("agh", nn.Parameter(agh, requires_grad=True))
        self.do_vector = True
        self.nchannel = nchannel
        self.d2features = d2features
        self.nshifts_s = nshifts_s
        self.nshifts_v = nshifts_v
        self.ncomb_v = ncomb_v

    def output_size(self):
        n = self.nchannel * self.nshifts_s
        if self.do_vector:
            n += self.nchannel * self.ncomb_v
        return n

    def forward(self, data: dict[str, Tensor], a: Tensor) -> Tensor:
        g_sv = data["g_sv"]
        mode = nbops.get_nb_mode(data)
        if self.d2features:
            if mode > 0 and a.device.type == "cuda":
                avf_sv = conv_sv_2d_sp(a, data["nbmat"], g_sv)  # type: ignore[misc]
            else:
                avf_sv = torch.einsum("...mag,...mgd->...agd", a.unsqueeze(1), g_sv)
        else:
            if mode > 0:
                a_j = a.index_select(0, data["nbmat"].flatten()).unflatten(0, data["nbmat"].shape)
                avf_sv = torch.einsum("...ma,...mgd->...agd", a_j, g_sv)
            else:
                avf_sv = torch.einsum("...ma,...mgd->...agd", a.unsqueeze(1), g_sv)
        avf_s, avf_v = avf_sv.split([1, 3], dim=-1)
        avf_v = torch.einsum("agh,...agd->...ahd", self.agh, avf_v).pow(2).sum(-1)
        return torch.cat([avf_s.squeeze(-1).flatten(-2, -1), avf_v.flatten(-2, -1)], dim=-1)


def _init_ahg(b: int, m: int, n: int):
    ret = torch.zeros(b, m, n)
    for i in range(b):
        ret[i] = _init_ahg_one(m, n)  # pylinit: disable-arguments-out-of-order
    return ret


def _init_ahg_one(m: int, n: int):
    # make x8 times more vectors to select most diverse
    x = torch.arange(m).unsqueeze(0)
    a1, a2, a3, a4 = torch.randn(8 * n, 4).unsqueeze(-2).unbind(-1)
    y = a1 * torch.sin(a2 * 2 * x * math.pi / m) + a3 * torch.cos(a4 * 2 * x * math.pi / m)
    y -= y.mean(dim=-1, keepdim=True)
    y /= y.std(dim=-1, keepdim=True)

    dmat = torch.cdist(y, y)
    # most distant point
    ret = torch.zeros(n, m)
    mask = torch.ones(y.shape[0], dtype=torch.bool)
    i = dmat.sum(-1).argmax()
    ret[0] = y[i]
    mask[i] = False

    # simple maxmin implementation
    for j in range(1, n):
        mindist, _ = torch.cdist(ret[:j], y).min(dim=0)
        maxidx = torch.argsort(mindist)[mask][-1]
        ret[j] = y[maxidx]
        mask[maxidx] = False
    return ret.t()
