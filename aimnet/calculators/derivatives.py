"""Derivative computation for :class:`~aimnet.calculators.AIMNet2Calculator`.

Free functions for autograd forces/stress, the dense Hessian, and the
combination of external (nvalchemiops-computed) derivative terms with
autograd-derived ones. The calculator facade delegates here; state such as
the saved-for-grad tensors is passed explicitly.
"""

import torch
from torch import Tensor

from aimnet.modules.lr import ExternalDerivativeTerms


def sum_optional_tensor(x: Tensor | None, y: Tensor | None) -> Tensor | None:
    """Elementwise sum of two ``Optional[Tensor]`` operands."""
    if x is None:
        return y
    if y is None:
        return x
    return x + y.to(dtype=x.dtype, device=x.device)


def combine_external_terms(
    a: ExternalDerivativeTerms | None,
    b: ExternalDerivativeTerms | None,
) -> ExternalDerivativeTerms | None:
    """Sum forces and virials of two external derivative terms.

    Both inputs follow the calculator-side contract used by
    :func:`get_derivatives`: ``forces`` add to the autograd-derived forces
    and ``virial`` enters as ``dedc -= virial.mT``. DSF Coulomb and DFTD3
    both publish detached terms in this convention, so combining them is a
    per-system elementwise sum.
    """
    if a is None:
        return b
    if b is None:
        return a
    return ExternalDerivativeTerms(
        forces=sum_optional_tensor(a.forces, b.forces),
        virial=sum_optional_tensor(a.virial, b.virial),
        hessian=sum_optional_tensor(a.hessian, b.hessian),
    )


def set_grad_tensors(
    data: dict[str, Tensor],
    *,
    forces: bool = False,
    stress: bool = False,
    hessian: bool = False,
) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Tensor] | None]:
    """Mark gradient inputs and (for stress) apply the strain scaling.

    Returns ``(data, saved_for_grad, external_strain_inputs)``:
    ``saved_for_grad`` holds the tensors :func:`get_derivatives`
    differentiates with respect to; ``external_strain_inputs`` carries the
    unstrained coordinates/cell for external LR modules, or ``None`` when
    no stress is requested.
    """
    saved_for_grad: dict[str, Tensor] = {}
    external_strain_inputs: dict[str, Tensor] | None = None
    if forces or hessian:
        data["coord"].requires_grad_(True)
        saved_for_grad["coord"] = data["coord"]
    if stress:
        assert "cell" in data and data["cell"] is not None, "Stress calculation requires cell"
        coord_unstrained = data["coord"]
        cell = data["cell"]
        cell_unstrained = cell
        if cell.ndim == 2:
            # Single system: (3, 3) scaling
            scaling = torch.eye(3, requires_grad=True, dtype=cell.dtype, device=cell.device)
            data["coord"] = data["coord"] @ scaling
            data["cell"] = cell @ scaling
        else:
            # Batched systems: (B, 3, 3) scaling - each system gets independent scaling
            B = cell.shape[0]
            scaling = torch.eye(3, dtype=cell.dtype, device=cell.device).unsqueeze(0).expand(B, -1, -1)
            scaling.requires_grad_(True)
            mol_idx = data["mol_idx"]
            # Apply per-atom scaling: coord[i] @ scaling[mol_idx[i]]
            atom_scaling = torch.index_select(scaling, 0, mol_idx)  # (N_total, 3, 3)
            data["coord"] = (data["coord"].unsqueeze(1) @ atom_scaling).squeeze(1)
            data["cell"] = cell @ scaling
        saved_for_grad["scaling"] = scaling
        external_strain_inputs = {
            "coord_unstrained": coord_unstrained,
            "cell_unstrained": cell_unstrained,
            "scaling": scaling,
        }
    return data, saved_for_grad, external_strain_inputs


def get_derivatives(
    data: dict[str, Tensor],
    *,
    forces: bool = False,
    stress: bool = False,
    hessian: bool = False,
    coulomb_terms: ExternalDerivativeTerms | None = None,
    saved_for_grad: dict[str, Tensor],
    create_graph: bool,
) -> dict[str, Tensor]:
    x = []
    if hessian:
        forces = True
    if forces and ("forces" not in data or (create_graph and not data["forces"].requires_grad)):
        forces = True
        x.append(saved_for_grad["coord"])
    if stress:
        x.append(saved_for_grad["scaling"])
    if x:
        tot_energy = data["energy"].sum()
        deriv = torch.autograd.grad(tot_energy, x, create_graph=create_graph)
        if forces:
            force = -deriv[0]
            if coulomb_terms is not None and coulomb_terms.forces is not None:
                force = force + coulomb_terms.forces.to(dtype=force.dtype, device=force.device)
            data["forces"] = force
        if stress:
            dedc = deriv[0] if not forces else deriv[1]
            if coulomb_terms is not None and coulomb_terms.virial is not None:
                virial = coulomb_terms.virial.to(dtype=dedc.dtype, device=dedc.device)
                if dedc.ndim == 2 and virial.ndim == 3:
                    virial = virial.sum(dim=0)
                # nvalchemiops virial convention is W = -dE/dstrain.
                # AIMNet applies row-vector strain as coord @ scaling,
                # so the stress numerator contribution is -W.T.
                dedc = dedc - virial.mT
            cell = data["cell"].detach()
            if cell.ndim == 2:
                volume = cell.det().abs()
            else:
                volume = torch.linalg.det(cell).abs().unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
            data["stress"] = dedc / volume
    if hessian:
        H = calculate_hessian(data["forces"], saved_for_grad["coord"])
        if coulomb_terms is not None and getattr(coulomb_terms, "hessian", None) is not None:
            # The LR coulomb hessian is computed in float64 via finite
            # differences. Accumulate in that (higher) precision rather than
            # downcasting it to H's dtype, which would discard the FD precision.
            H = H.to(dtype=coulomb_terms.hessian.dtype, device=H.device) + coulomb_terms.hessian.to(device=H.device)
        data["hessian"] = H
    return data


def calculate_hessian(forces: Tensor, coord: Tensor) -> Tensor:
    """Dense ``(N, 3, N, 3)`` Hessian of the energy w.r.t. real-atom coordinates.

    Autograd contract (IMPORTANT):
    The returned dense Hessian is a **detached value**: it carries no
    autograd graph back to the coordinates or model parameters. This is by
    design (it is materialized via ``torch.func.vmap`` over a vjp of the
    already-built force graph, and the periodic PME block is a fixed-charge
    finite-difference term that is non-differentiable; Ewald is a full
    relaxed-charge autograd term since the 0.4 energy-graph migration). Forces
    DO compose with an upstream coordinate-builder graph, but the Hessian
    does not, so you cannot backpropagate through ``eval(..., hessian=True)``.

    If you need the Hessian to *compose* (e.g. ``H @ v`` that scales with /
    differentiates through an outer computation) or to avoid forming the
    dense ``(N, 3, N, 3)`` tensor on large systems, use the matrix-free
    :meth:`AIMNet2Calculator.hessian_vector_product` instead. For a
    fully-differentiable Hessian, build one externally with
    ``torch.autograd.functional.hessian(energy_fn, coords)`` over a closure
    that calls the model on differentiable coordinates (note that the
    periodic Ewald/PME long-range block remains a fixed-charge FD term in
    either case).
    """
    # Coord includes padding atom (shape N+1), forces only for real atoms (shape N).
    # Hessian computed only for actual atoms: (N, 3, N, 3).
    #
    # vmap-over-vjp form (not is_grads_batched=True or autograd.functional.hessian):
    # torch.library.register_vmap on aimnet::conv_sv_2d_sp_{bwd,bwd_bwd} is consulted
    # ONLY by the functorch dispatch (torch.func.vmap). The legacy batching dispatch
    # would still raise "Batching rule not implemented."
    n = forces.numel()
    eye = torch.eye(n, device=forces.device, dtype=forces.dtype)

    def vjp(go: Tensor) -> Tensor:
        return torch.autograd.grad(
            forces.flatten(),
            coord,
            grad_outputs=go,
            retain_graph=True,
            allow_unused=True,
        )[0]

    hessian = -torch.func.vmap(vjp, 0)(eye)
    return hessian.view(-1, 3, coord.shape[0], 3)[:-1, :, :-1, :]
