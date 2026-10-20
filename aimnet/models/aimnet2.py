from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn

from aimnet import nbops, ops
from aimnet.models.base import AIMNet2Base
from aimnet.modules import AEVSV, MLP, ConvSV, Embedding
from aimnet.modules.aev import _can_group_dense_convolutions


class AIMNet2(AIMNet2Base):
    def __init__(
        self,
        aev: dict,
        nfeature: int,
        d2features: bool,
        ncomb_v: int,
        hidden: tuple[list[int], ...],
        aim_size: int,
        outputs: list[nn.Module] | dict[str, nn.Module],
        num_charge_channels: int = 1,
    ):
        super().__init__()

        if num_charge_channels not in [1, 2]:
            raise ValueError("num_charge_channels must be 1 (closed shell) or 2 (NSE for open-shell).")
        self.num_charge_channels = num_charge_channels

        self.aev = AEVSV(**aev)
        nshifts_s = aev["nshifts_s"]
        nshifts_v = aev.get("nshifts_v") or nshifts_s
        if d2features:
            if nshifts_s != nshifts_v:
                raise ValueError("nshifts_s must be equal to nshifts_v for d2features")
            nfeature_tot = nshifts_s * nfeature
        else:
            nfeature_tot = nfeature
        self.nfeature = nfeature
        self.nshifts_s = nshifts_s
        self.d2features = d2features

        self.afv = Embedding(num_embeddings=64, embedding_dim=nfeature, padding_idx=0)

        with torch.no_grad():
            nn.init.orthogonal_(self.afv.weight[1:])
            if d2features:
                self.afv.weight = nn.Parameter(
                    self.afv.weight.clone().unsqueeze(-1).expand(64, nfeature, nshifts_s).flatten(-2, -1)
                )

        conv_param = {"nshifts_s": nshifts_s, "nshifts_v": nshifts_v, "ncomb_v": ncomb_v}
        self.conv_a = ConvSV(nchannel=nfeature, d2features=d2features, **conv_param)
        self.conv_q = ConvSV(nchannel=num_charge_channels, d2features=False, **conv_param)

        mlp_param: dict[str, Any] = {"activation_fn": nn.GELU(), "last_linear": True}
        mlps: list[nn.Module] = [
            MLP(
                n_in=self.conv_a.output_size() + nfeature_tot,
                n_out=nfeature_tot + 2 * num_charge_channels,
                hidden=hidden[0],
                **mlp_param,
            )
        ]
        mlp_param = {"activation_fn": nn.GELU(), "last_linear": False}
        for h in hidden[1:-1]:
            mlps.append(
                MLP(
                    n_in=self.conv_a.output_size() + self.conv_q.output_size() + nfeature_tot + num_charge_channels,
                    n_out=nfeature_tot + 2 * num_charge_channels,
                    hidden=h,
                    **mlp_param,
                )
            )
        mlp_param = {"activation_fn": nn.GELU(), "last_linear": False}
        mlps.append(
            MLP(
                n_in=self.conv_a.output_size() + self.conv_q.output_size() + nfeature_tot + num_charge_channels,
                n_out=aim_size,
                hidden=hidden[-1],
                **mlp_param,
            )
        )
        self.mlps = nn.ModuleList(mlps)

        self.outputs: nn.ModuleList | nn.ModuleDict
        if isinstance(outputs, Sequence):
            self.outputs = nn.ModuleList(outputs)
        elif isinstance(outputs, Mapping):
            self.outputs = nn.ModuleDict(outputs)
        else:
            raise TypeError("`outputs` is not either list or dict")

    def _preprocess_spin_polarized_charge(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        if "mult" not in data:
            raise ValueError("mult key is required for NSE if two channels for charge are not provided")
        _half_spin = 0.5 * (data["mult"] - 1.0)
        _half_q = 0.5 * data["charge"]
        data["charge"] = torch.stack([_half_q + _half_spin, _half_q - _half_spin], dim=-1)
        return data

    def _postprocess_spin_polarized_charge(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        data["spin_charges"] = data["charges"][..., 0] - data["charges"][..., 1]
        data["charges"] = data["charges"].sum(dim=-1)
        data["charge"] = data["charge"].sum(dim=-1)
        return data

    def _prepare_in_a(self, data: dict[str, Tensor]) -> Tensor:
        a_i = nbops.get_i(data["a"], data)
        avf_a = self.conv_a(data, data["a"])
        if self.d2features:
            a_i = a_i.flatten(-2, -1)
        _in = torch.cat([a_i.squeeze(-2), avf_a], dim=-1)
        return _in

    def _prepare_in_q(self, data: dict[str, Tensor]) -> Tensor:
        q_i = nbops.get_i(data["charges"], data)
        avf_q = self.conv_q(data, data["charges"])
        _in = torch.cat([q_i.squeeze(-2), avf_q], dim=-1)
        return _in

    def _prepare_in_aq(self, data: dict[str, Tensor]) -> Tensor:
        # A and Q within a pass are independent consumers of the same basis.
        # Group only their contraction, keeping the canonical charge iterations
        # and the individual ConvSV output projections. Hooks and overrides must
        # still observe the original module calls, so they use the fallback.
        if (
            not self.training
            and getattr(self._prepare_in_a, "__func__", None) is _CANONICAL_PREPARE_IN_A
            and getattr(self._prepare_in_q, "__func__", None) is _CANONICAL_PREPARE_IN_Q
            and _can_group_dense_convolutions(data, self.conv_a, self.conv_q)
        ):
            from aimnet.modules.aev_inference import dense_convolutions, dense_convolutions_carry, geometry_carry_basis

            features = (data["a"], data["charges"])
            flags = (self.conv_a.d2features, self.conv_q.d2features)
            if "_geometry_carry" in data:
                (avf_a, avf_q), data["_geometry_carry"] = dense_convolutions_carry(
                    features, geometry_carry_basis(data), flags
                )
            else:
                avf_a, avf_q = dense_convolutions(features, data["g_sv"].permute(0, 3, 2, 1, 4), flags)
            avf_a = self.conv_a._finish_convolution(data, avf_a, 0)
            avf_q = self.conv_q._finish_convolution(data, avf_q, 0)
            a_i = nbops.get_i(data["a"], data)
            if self.d2features:
                a_i = a_i.flatten(-2, -1)
            q_i = nbops.get_i(data["charges"], data)
            return torch.cat([a_i.squeeze(-2), avf_a, q_i.squeeze(-2), avf_q], dim=-1)
        return torch.cat([self._prepare_in_a(data), self._prepare_in_q(data)], dim=-1)

    def _update_q(self, data: dict[str, Tensor], x: Tensor, delta_q: bool = True) -> dict[str, Tensor]:
        _q, _f, delta_a = x.split(
            [
                self.num_charge_channels,
                self.num_charge_channels,
                x.shape[-1] - 2 * self.num_charge_channels,
            ],
            dim=-1,
        )
        # Charge conservation violation penalty for training loss
        data["_delta_Q"] = data["charge"] - nbops.mol_sum(_q, data)
        q = data["charges"] + _q if delta_q else _q
        data["charges_pre"] = q if self.num_charge_channels == 2 else q.squeeze(-1)
        f = _f.pow(2)
        q = ops.nse(data["charge"], q, f, data, epsilon=1.0e-6)
        data["charges"] = q
        data["a"] = data["a"] + delta_a.view_as(data["a"])
        return data

    def forward(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        return self.forward_prepared(self.prepare_input(data))

    def forward_prepared(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        """Canonical model computation on validated, masked inputs.

        Regular execution prepares inputs in ``forward``. Static-layout runners
        prepare their reusable input buffers separately and call this same body;
        they do not own a second implementation of the model equations.
        """

        # initial features
        a: Tensor = self.afv(data["numbers"])
        if self.d2features:
            a = a.unflatten(-1, (self.nfeature, self.nshifts_s))
        data["a"] = a

        # NSE case
        if self.num_charge_channels == 2:
            data = self._preprocess_spin_polarized_charge(data)
        else:
            # Ensure charge tensor has channel dimension for consistency with features
            data["charge"] = data["charge"].unsqueeze(-1)

        data.pop("_geometry_carry", None)
        data.pop("_geometry_carry_root", None)
        data = self.aev(data)

        if _can_carry_dense_geometry(self, data):
            data["_geometry_carry_root"] = data["g_sv"]
            data["_geometry_carry"] = data["g_sv"].permute(0, 3, 2, 1, 4)

        _npass = len(self.mlps)
        try:
            for ipass, mlp in enumerate(self.mlps):
                if ipass == 0:
                    _in = self._prepare_in_a(data)
                else:
                    _in = self._prepare_in_aq(data)

                _out = mlp(_in)
                if nbops.is_input_padded(data):
                    _out = nbops.mask_i_(_out, data, mask_value=0.0)

                if ipass == 0:
                    data = self._update_q(data, _out, delta_q=False)
                elif ipass < _npass - 1:
                    data = self._update_q(data, _out, delta_q=True)
                else:
                    data["aim"] = _out
        finally:
            data.pop("_geometry_carry", None)
            data.pop("_geometry_carry_root", None)

        # squeeze charges
        if self.num_charge_channels == 2:
            data = self._postprocess_spin_polarized_charge(data)
        else:
            data["charges"] = data["charges"].squeeze(-1)
            data["charge"] = data["charge"].squeeze(-1)

        for m in self.outputs.children():
            data = m(data)

        return data


_CANONICAL_PREPARE_IN_A = AIMNet2._prepare_in_a
_CANONICAL_PREPARE_IN_Q = AIMNet2._prepare_in_q
_CANONICAL_PREPARE_IN_AQ = AIMNet2._prepare_in_aq
_CANONICAL_UPDATE_Q = AIMNet2._update_q


def _can_carry_dense_geometry(model: AIMNet2, data: dict[str, Tensor]) -> bool:
    """Scope geometry continuation to an unobserved canonical dense loop.

    Public geometry remains the original tensor, and custom methods/hooks keep
    the existing independent-autograd path. Single structures do not benefit
    from this bandwidth optimization and retain the simpler grouped path.
    """
    if (
        type(model) is not AIMNet2
        or model.training
        or not _can_group_dense_convolutions(data, model.conv_a, model.conv_q)
    ):
        return False
    if data["g_sv"].shape[0] < 2:
        return False
    if torch.is_autocast_enabled(data["g_sv"].device.type):
        return False
    methods = (
        ("_prepare_in_a", _CANONICAL_PREPARE_IN_A),
        ("_prepare_in_q", _CANONICAL_PREPARE_IN_Q),
        ("_prepare_in_aq", _CANONICAL_PREPARE_IN_AQ),
        ("_update_q", _CANONICAL_UPDATE_Q),
    )
    if any(getattr(getattr(model, name), "__func__", None) is not method for name, method in methods):
        return False
    hook_names = ("_forward_pre_hooks", "_forward_hooks", "_backward_pre_hooks", "_backward_hooks")
    return not any(getattr(module, name, {}) for module in model.modules() for name in hook_names)
