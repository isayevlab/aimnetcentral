import os

import torch
from torch import Tensor, nn

from aimnet import constants, nbops, ops
from aimnet.modules.ops import dftd3_energy


def _calc_coulomb_sr(
    data: dict[str, Tensor],
    rc: Tensor,
    envelope: str,
    key_in: str,
    factor: float,
) -> Tensor:
    """Shared short-range Coulomb energy calculation.

    Computes pairwise Coulomb energy with envelope-weighted cutoff.

    Parameters
    ----------
    data : dict
        Data dictionary containing d_ij distances and charges.
    rc : Tensor
        Cutoff radius tensor.
    envelope : str
        Envelope function: "exp" or "cosine".
    key_in : str
        Key for charges in data dict.
    factor : float
        Unit conversion factor (half_Hartree * Bohr).

    Returns
    -------
    Tensor
        Short-range Coulomb energy per molecule.
    """
    d_ij = data["d_ij"]
    q = data[key_in]
    q_i, q_j = nbops.get_ij(q, data)
    q_ij = q_i * q_j
    if envelope == "exp":
        fc = ops.exp_cutoff(d_ij, rc)
    else:  # cosine
        fc = ops.cosine_cutoff(d_ij, rc.item())
    e_ij = fc * q_ij / d_ij
    e_ij = nbops.mask_ij_(e_ij, data, 0.0)
    # Accumulate in float64 for precision
    e_i = e_ij.sum(-1, dtype=torch.float64)
    return factor * nbops.mol_sum(e_i, data)


class LRCoulomb(nn.Module):
    """Long-range Coulomb energy module.

    Computes electrostatic energy using one of several methods:
    simple (all pairs), DSF (damped shifted force), or Ewald summation.

    Parameters
    ----------
    key_in : str
        Key for input charges in data dict. Default is "charges".
    key_out : str
        Key for output energy in data dict. Default is "e_h".
    rc : float
        Short-range cutoff radius. Default is 4.6 Angstrom.
    method : str
        Coulomb method: "simple", "dsf", or "ewald". Default is "simple".
    dsf_alpha : float
        Alpha parameter for DSF method. Default is 0.2.
    dsf_rc : float
        Cutoff for DSF method. Default is 15.0.
    ewald_accuracy : float
        Target accuracy for Ewald summation. Controls real-space and
        reciprocal-space cutoffs. Lower values give higher accuracy.
        Default is 1e-8.
    subtract_sr : bool
        Whether to subtract short-range contribution. Default is True.
    envelope : str
        Envelope function for SR cutoff: "exp" or "cosine". Default is "exp".

    Notes
    -----
    Energy accumulation uses float64 for numerical precision, particularly
    important for large systems where many small contributions can suffer
    from floating-point error accumulation.

    Neighbor list keys follow a suffix resolution pattern: methods first look
    for module-specific keys (e.g., nbmat_coulomb, shifts_coulomb), falling
    back to shared _lr suffix (nbmat_lr, shifts_lr) if not found.
    """

    def __init__(
        self,
        key_in: str = "charges",
        key_out: str = "e_h",
        rc: float = 4.6,
        method: str = "simple",
        dsf_alpha: float = 0.2,
        dsf_rc: float = 15.0,
        ewald_accuracy: float = 1e-8,
        subtract_sr: bool = True,
        envelope: str = "exp",
    ):
        super().__init__()
        self.key_in = key_in
        self.key_out = key_out
        self._factor = constants.half_Hartree * constants.Bohr
        self.register_buffer("rc", torch.tensor(rc))
        self.dsf_alpha = dsf_alpha
        self.dsf_rc = dsf_rc
        self.ewald_accuracy = ewald_accuracy
        self.subtract_sr = subtract_sr
        if envelope not in ("exp", "cosine"):
            raise ValueError(f"Unknown envelope {envelope}, must be 'exp' or 'cosine'")
        self.envelope = envelope
        if method in ("simple", "dsf", "ewald"):
            self.method = method
        else:
            raise ValueError(f"Unknown method {method}")

    def coul_simple(self, data: dict[str, Tensor]) -> Tensor:
        """Compute pairwise Coulomb energy.

        With subtract_sr=True (default): Returns LR only (FULL - SR)
        With subtract_sr=False: Returns FULL pairwise Coulomb
        """
        suffix = nbops.resolve_suffix(data, ["_coulomb", "_lr"])
        data = ops.lazy_calc_dij(data, suffix)
        d_ij = data[f"d_ij{suffix}"]
        q = data[self.key_in]
        q_i, q_j = nbops.get_ij(q, data, suffix=suffix)
        q_ij = q_i * q_j
        # Compute FULL pairwise Coulomb (no exp_cutoff weighting)
        e_ij = q_ij / d_ij
        e_ij = nbops.mask_ij_(e_ij, data, 0.0, suffix=suffix)
        e_i = e_ij.sum(-1, dtype=torch.float64)
        e = self._factor * nbops.mol_sum(e_i, data)
        # Same pattern as dsf/ewald - subtract SR to get LR
        if self.subtract_sr:
            e = e - self.coul_simple_sr(data)
        return e

    def coul_simple_sr(self, data: dict[str, Tensor]) -> Tensor:
        return _calc_coulomb_sr(data, self.rc, self.envelope, self.key_in, self._factor)

    def coul_dsf(self, data: dict[str, Tensor]) -> Tensor:
        suffix = nbops.resolve_suffix(data, ["_coulomb", "_lr"])
        data = ops.lazy_calc_dij(data, suffix)
        d_ij = data[f"d_ij{suffix}"]
        q = data[self.key_in]
        q_i, q_j = nbops.get_ij(q, data, suffix=suffix)
        J = ops.coulomb_matrix_dsf(d_ij, self.dsf_rc, self.dsf_alpha, data)
        e = (q_i * q_j * J).sum(-1, dtype=torch.float64)
        e = self._factor * nbops.mol_sum(e, data)
        if self.subtract_sr:
            e = e - self.coul_simple_sr(data)
        return e

    def coul_ewald(self, data: dict[str, Tensor]) -> Tensor:
        J = ops.coulomb_matrix_ewald(data["coord"], data["cell"], accuracy=self.ewald_accuracy)
        q_i, q_j = data["charges"].unsqueeze(-1), data["charges"].unsqueeze(-2)
        e = self._factor * (q_i * q_j * J).flatten(-2, -1).sum(-1, dtype=torch.float64)
        if self.subtract_sr:
            e = e - self.coul_simple_sr(data)
        return e

    def forward(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        if self.method == "simple":
            e = self.coul_simple(data)
        elif self.method == "dsf":
            e = self.coul_dsf(data)
        elif self.method == "ewald":
            e = self.coul_ewald(data)
        else:
            raise ValueError(f"Unknown method {self.method}")
        if self.key_out in data:
            data[self.key_out] = data[self.key_out].double() + e
        else:
            data[self.key_out] = e
        return data


class SRCoulomb(nn.Module):
    """Subtract short-range Coulomb contribution from energy.

    For models trained with "simple" Coulomb mode, the NN has implicitly learned
    the short-range Coulomb interaction. When using DSF or Ewald summation for
    the full Coulomb energy, we need to subtract this short-range contribution
    to avoid double-counting.

    Parameters
    ----------
    rc : float
        Cutoff radius for short-range Coulomb. Default is 4.6 Angstrom.
    key_in : str
        Key for input charges in data dict. Default is "charges".
    key_out : str
        Key for output energy in data dict. Default is "energy".
    envelope : str
        Envelope function for cutoff: "exp" (mollifier) or "cosine". Default is "exp".
    """

    def __init__(
        self,
        rc: float = 4.6,
        key_in: str = "charges",
        key_out: str = "energy",
        envelope: str = "exp",
    ):
        super().__init__()
        self.key_in = key_in
        self.key_out = key_out
        self._factor = constants.half_Hartree * constants.Bohr
        self.register_buffer("rc", torch.tensor(rc))
        if envelope not in ("exp", "cosine"):
            raise ValueError(f"Unknown envelope {envelope}, must be 'exp' or 'cosine'")
        self.envelope = envelope

    def forward(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        """Subtract short-range Coulomb from energy."""
        e_sr = _calc_coulomb_sr(data, self.rc, self.envelope, self.key_in, self._factor)

        # Subtract short-range Coulomb from energy (in float64)
        if self.key_out in data:
            data[self.key_out] = data[self.key_out].double() - e_sr
        else:
            data[self.key_out] = -e_sr
        return data


class DispParam(nn.Module):
    def __init__(
        self,
        ref_c6: dict[int, Tensor] | Tensor | None = None,
        ref_alpha: dict[int, Tensor] | Tensor | None = None,
        ptfile: str | None = None,
        key_in: str = "disp_param",
        key_out: str = "disp_param",
    ):
        super().__init__()
        if (ptfile is None and (ref_c6 is None or ref_alpha is None)) or (
            ptfile is not None and (ref_c6 is not None or ref_alpha is not None)
        ):
            raise ValueError("Either ptfile or ref_c6 and ref_alpha should be supplied.")
        # load data
        ref = torch.load(ptfile) if ptfile is not None else torch.zeros(87, 2)
        for i, p in enumerate([ref_c6, ref_alpha]):
            if p is not None:
                if isinstance(p, Tensor):
                    ref[: p.shape[0], i] = p
                else:
                    for k, v in p.items():
                        ref[k, i] = v
        # Element 0 represents dummy atoms with c6=0 and alpha=1
        ref[0, 0] = 0.0
        ref[0, 1] = 1.0
        self.register_buffer("disp_param0", ref)
        self.key_in = key_in
        self.key_out = key_out

    def forward(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        disp_param_mult = data[self.key_in].clamp(min=-4, max=4).exp()
        disp_param = self.disp_param0[data["numbers"]]
        vals = disp_param * disp_param_mult
        data[self.key_out] = vals
        return data


class D3TS(nn.Module):
    """DFT-D3-like pairwise dispersion with TS combination rule"""

    def __init__(self, a1: float, a2: float, s8: float, s6: float = 1.0, key_in="disp_param", key_out="energy"):
        super().__init__()
        self.register_buffer("r4r2", constants.get_r4r2())
        self.a1 = a1
        self.a2 = a2
        self.s6 = s6
        self.s8 = s8
        self.key_in = key_in
        self.key_out = key_out

    def forward(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        suffix = nbops.resolve_suffix(data, ["_dftd3", "_lr"])

        disp_param = data[self.key_in]
        disp_param_i, disp_param_j = nbops.get_ij(disp_param, data, suffix=suffix)
        c6_i, alpha_i = disp_param_i.unbind(dim=-1)
        c6_j, alpha_j = disp_param_j.unbind(dim=-1)

        # TS combination rule
        c6ij = 2 * c6_i * c6_j / (c6_i * alpha_j / alpha_i + c6_j * alpha_i / alpha_j).clamp(min=1e-4)
        c6ij = nbops.mask_ij_(c6ij, data, 0.0, suffix=suffix)

        rr = self.r4r2[data["numbers"]]
        rr_i, rr_j = nbops.get_ij(rr, data, suffix=suffix)
        rrij = 3 * rr_i * rr_j
        rrij = nbops.mask_ij_(rrij, data, 1.0, suffix=suffix)
        r0ij = self.a1 * rrij.sqrt() + self.a2

        ops.lazy_calc_dij(data, suffix)
        d_ij = data[f"d_ij{suffix}"] * constants.Bohr_inv
        e_ij = c6ij * (self.s6 / (d_ij.pow(6) + r0ij.pow(6)) + self.s8 * rrij / (d_ij.pow(8) + r0ij.pow(8)))

        e = -constants.half_Hartree * nbops.mol_sum(e_ij.sum(-1), data)

        if self.key_out in data:
            data[self.key_out] = data[self.key_out] + e
        else:
            data[self.key_out] = e

        return data


class DFTD3(nn.Module):
    """DFT-D3 implementation using nvalchemiops GPU-accelerated kernels.

    BJ damping, C6 and C8 terms, without 3-body term.

    This implementation uses nvalchemiops.interactions.dispersion.dftd3 for
    GPU-accelerated computation of dispersion energies and forces. It is
    differentiable through a custom autograd function.

    Parameters
    ----------
    s8 : float
        Scaling factor for C8 term.
    a1 : float
        BJ damping parameter 1.
    a2 : float
        BJ damping parameter 2.
    s6 : float, optional
        Scaling factor for C6 term. Default is 1.0.
    cutoff : float, optional
        Cutoff distance in Angstroms for smoothing. Default is 15.0.
    smoothing_fraction : float, optional
        Fraction of cutoff distance used for smoothing window width.
        Smoothing starts at cutoff * (1 - smoothing_fraction) and ends at cutoff.
        Example: With cutoff=15.0 and smoothing_fraction=0.2:
          - Smoothing starts at 12.0 Å (15.0 * 0.8)
          - Smoothing ends at 15.0 Å
        Default is 0.2 (20% of cutoff as smoothing window).
    key_out : str, optional
        Key for output energy in data dict. Default is "energy".
    compute_forces : bool, optional
        Whether to add forces to data dict. Default is False.
    compute_virial : bool, optional
        Whether to compute virial for cell gradients. Default is False.

    Attributes
    ----------
    smoothing_on : float
        Distance where smoothing starts (Angstroms).
    smoothing_off : float
        Distance where smoothing ends / cutoff (Angstroms).
    s6, s8, a1, a2 : float
        BJ damping parameters.

    Notes
    -----
    Neighbor list keys follow a suffix resolution pattern: methods first look
    for module-specific keys (e.g., nbmat_dftd3, shifts_dftd3), falling back
    to shared _lr suffix (nbmat_lr, shifts_lr) if not found.
    """

    def __init__(
        self,
        s8: float,
        a1: float,
        a2: float,
        s6: float = 1.0,
        cutoff: float = 15.0,
        smoothing_fraction: float = 0.2,
        key_out: str = "energy",
        compute_forces: bool = False,
        compute_virial: bool = False,
    ):
        super().__init__()
        self.key_out = key_out
        self.compute_forces = compute_forces
        self.compute_virial = compute_virial
        # BJ damping parameters
        self.s6 = s6
        self.s8 = s8
        self.a1 = a1
        self.a2 = a2

        # Smoothing parameters as module attributes
        self.smoothing_on: float = cutoff * (1 - smoothing_fraction)
        self.smoothing_off: float = cutoff

        # Load D3 reference parameters and convert to nvalchemiops format
        dirname = os.path.dirname(os.path.dirname(__file__))
        filename = os.path.join(dirname, "dftd3_data.pt")
        param = torch.load(filename, map_location="cpu", weights_only=True)

        c6ab_packed = param["c6ab"]
        c6ab = c6ab_packed[..., 0].contiguous()
        cn_ref = c6ab_packed[..., 1].contiguous()

        # Register buffers for D3 parameters
        self.register_buffer("rcov", param["rcov"].float())
        self.register_buffer("r4r2", param["r4r2"].float())
        self.register_buffer("c6ab", c6ab.float())
        self.register_buffer("cn_ref", cn_ref.float())

    def set_smoothing(self, cutoff: float, smoothing_fraction: float = 0.2) -> None:
        """Update smoothing parameters based on new cutoff and fraction.

        Parameters
        ----------
        cutoff : float
            Cutoff distance in Angstroms.
        smoothing_fraction : float
            Fraction of cutoff used as smoothing window width.
            Smoothing occurs from cutoff * (1 - smoothing_fraction) to cutoff.
            Example: smoothing_fraction=0.2 means smoothing over last 20%
            of cutoff distance (from 0.8*cutoff to cutoff). Default is 0.2.
        """
        self.smoothing_on = cutoff * (1 - smoothing_fraction)
        self.smoothing_off = cutoff

    def _load_from_state_dict(
        self,
        state_dict: dict,
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: list,
        unexpected_keys: list,
        error_msgs: list,
    ) -> None:
        """Handle loading from old state dict format with packed c6ab.

        Migrates from legacy format where c6ab had shape [95, 95, 5, 5, 3]
        with last dimension containing (c6ref, cnref_i, cnref_j) to new format
        where c6ab is [95, 95, 5, 5] and cn_ref is separate [95, 95, 5, 5].
        Also removes deprecated cnmax parameter if present.
        """
        c6ab_key = prefix + "c6ab"
        cn_ref_key = prefix + "cn_ref"
        cnmax_key = prefix + "cnmax"

        if c6ab_key in state_dict and state_dict[c6ab_key].ndim == 5:
            c6ab_packed = state_dict[c6ab_key]
            state_dict[c6ab_key] = c6ab_packed[..., 0].contiguous()
            state_dict[cn_ref_key] = c6ab_packed[..., 1].contiguous()

        state_dict.pop(cnmax_key, None)

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        nb_mode = nbops.get_nb_mode(data)
        coord = data["coord"]
        numbers = data["numbers"].to(torch.int32)
        cell = data.get("cell")

        # Prepare inputs based on nb_mode
        if nb_mode == 0:
            # Batched mode without neighbor matrix - construct full neighbor matrix
            # This only applies when no nbmat is provided in data
            B, N = coord.shape[:2]
            coord_flat = coord.flatten(0, 1)  # (B*N, 3)
            numbers_flat = numbers.flatten()  # (B*N,)
            batch_idx = torch.arange(B, device=coord.device, dtype=torch.int32).repeat_interleave(N)
            num_systems = B
            total_atoms = B * N
            max_neighbors = N - 1

            # Build dense all-to-all neighbor matrix for mode 0 (no pre-computed neighbor list)
            # Creates (N, N-1) template where each atom connects to all others except itself
            arange_n = torch.arange(N, device=coord.device, dtype=torch.int32)
            all_indices = arange_n.unsqueeze(0).expand(N, -1)
            mask = all_indices != arange_n.unsqueeze(1)
            template = all_indices[mask].view(N, N - 1)
            batch_offsets = torch.arange(B, device=coord.device, dtype=torch.int32).unsqueeze(1).unsqueeze(2) * N
            neighbor_matrix = (template.unsqueeze(0) + batch_offsets).view(total_atoms, max_neighbors)

            fill_value = total_atoms
            neighbor_matrix_shifts: Tensor | None = None
            cell_for_autograd: Tensor | None = None

        elif nb_mode == 1:
            # Flat mode with neighbor matrix
            suffix = nbops.resolve_suffix(data, ["_dftd3", "_lr"])
            N = coord.shape[0]
            coord_flat = coord
            numbers_flat = numbers
            neighbor_matrix = data[f"nbmat{suffix}"].to(torch.int32)

            mol_idx = data.get("mol_idx")
            if mol_idx is not None:
                batch_idx = mol_idx.to(torch.int32)
                num_systems = int(mol_idx.max().item()) + 1
            else:
                batch_idx = torch.zeros(N, dtype=torch.int32, device=coord.device)
                num_systems = 1

            shifts = data.get(f"shifts{suffix}")
            neighbor_matrix_shifts = shifts.to(torch.int32) if shifts is not None else None

            fill_value = N
            cell_for_autograd = cell

        elif nb_mode == 2:
            # Batched mode with neighbor matrix
            suffix = nbops.resolve_suffix(data, ["_dftd3", "_lr"])
            B, N = coord.shape[:2]
            coord_flat = coord.flatten(0, 1)
            numbers_flat = numbers.flatten()
            batch_idx = torch.arange(B, device=coord.device, dtype=torch.int32).repeat_interleave(N)
            num_systems = B

            nbmat = data[f"nbmat{suffix}"]
            offsets = torch.arange(B, device=coord.device).unsqueeze(1) * N
            neighbor_matrix = (nbmat + offsets.unsqueeze(-1)).flatten(0, 1).to(torch.int32)

            shifts = data.get(f"shifts{suffix}")
            if shifts is not None:
                neighbor_matrix_shifts = shifts.flatten(0, 1).to(torch.int32)
            else:
                neighbor_matrix_shifts = None

            fill_value = B * N
            cell_for_autograd = cell

        else:
            raise ValueError(f"Unsupported neighbor mode: {nb_mode}")

        # Compute energy using autograd function
        energy_ev = self._compute_energy_autograd(
            coord_flat,
            cell_for_autograd,
            numbers_flat,
            batch_idx,
            neighbor_matrix,
            neighbor_matrix_shifts,
            num_systems,
            fill_value,
        )

        # Add dispersion energy to output
        if self.key_out in data:
            data[self.key_out] = data[self.key_out] + energy_ev.to(data[self.key_out].dtype)
        else:
            data[self.key_out] = energy_ev

        # Optionally compute and add forces to data dict
        # Compute forces via autograd (will use saved forces from DFTD3Function)
        if self.compute_forces and not torch.jit.is_scripting() and coord_flat.requires_grad:
            # Forces are -grad of energy
            forces_flat = torch.autograd.grad(
                energy_ev.sum(),
                coord_flat,
                create_graph=self.training,
                retain_graph=True,
            )[0]
            forces = -forces_flat

            # Reshape if needed
            if nb_mode == 0 or nb_mode == 2:
                B, N = coord.shape[:2]
                forces = forces.view(B, N, 3)

            if "forces" in data:
                data["forces"] = data["forces"] + forces
            else:
                data["forces"] = forces

        return data

    def _compute_energy_autograd(
        self,
        coord: Tensor,
        cell: Tensor | None,
        numbers: Tensor,
        batch_idx: Tensor,
        neighbor_matrix: Tensor,
        neighbor_matrix_shifts: Tensor | None,
        num_systems: int,
        fill_value: int,
    ) -> Tensor:
        """Compute DFT-D3 energy using custom op for differentiability and TorchScript."""
        return dftd3_energy(
            coord=coord,
            cell=cell,
            numbers=numbers,
            batch_idx=batch_idx,
            neighbor_matrix=neighbor_matrix,
            shifts=neighbor_matrix_shifts,
            rcov=self.rcov,
            r4r2=self.r4r2,
            c6ab=self.c6ab,
            cn_ref=self.cn_ref,
            a1=self.a1,
            a2=self.a2,
            s6=self.s6,
            s8=self.s8,
            num_systems=num_systems,
            fill_value=fill_value,
            smoothing_on=self.smoothing_on,
            smoothing_off=self.smoothing_off,
            compute_virial=self.compute_virial,
        )
