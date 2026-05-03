"""TorchSim ``ModelInterface`` wrapper for AIMNet2."""

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor
from torch_sim.models.interface import ModelInterface
from torch_sim.state import SimState

from .calculator import AIMNet2Calculator


class AIMNet2TorchSim(ModelInterface):
    """Wrap an :class:`AIMNet2Calculator` as a TorchSim model.

    Parameters
    ----------
    base_calc
        Underlying AIMNet2 calculator. AIMNet2 inference uses float32
        internally, so the wrapper reports ``torch.float32`` regardless of the
        incoming TorchSim state dtype.
    compute_stress
        Request AIMNet2 stress on every forward call. This is required for NPT
        integrators and PBC cell relaxation. Leave it false for energy/force
        workflows to avoid retaining extra autograd state.
    """

    def __init__(
        self,
        base_calc: AIMNet2Calculator,
        *,
        compute_forces: bool = True,
        compute_stress: bool = False,
        validate_species: bool = True,
    ) -> None:
        super().__init__()
        self._base_calc = base_calc
        self._device = torch.device(base_calc.device)
        self._dtype = torch.float32
        self._compute_forces = compute_forces
        self._compute_stress = compute_stress
        self._validate_species = validate_species
        self._memory_scales_with = "n_atoms_x_density"
        self._update_implemented_properties()

    @property
    def base_calc(self) -> AIMNet2Calculator:
        """Underlying AIMNet2 calculator."""
        return self._base_calc

    @property
    def metadata(self) -> Mapping[str, Any] | None:
        """Underlying model metadata, when available."""
        return self._base_calc.metadata

    @ModelInterface.compute_forces.setter
    def compute_forces(self, value: bool) -> None:
        self._compute_forces = bool(value)
        self._update_implemented_properties()

    @ModelInterface.compute_stress.setter
    def compute_stress(self, value: bool) -> None:
        self._compute_stress = bool(value)
        self._update_implemented_properties()

    def forward(self, state: SimState, **kwargs: Any) -> dict[str, Tensor]:
        """Compute AIMNet2 outputs for a TorchSim state."""
        if state.device != self._device or state.dtype != self._dtype:
            state = state.to(self._device, self._dtype)

        data = self._state_to_aimnet2_data(state)
        results = self._base_calc(
            data,
            forces=self._compute_forces,
            stress=self._compute_stress,
            validate_species=self._validate_species,
        )
        if "charges" in results:
            results["partial_charges"] = results["charges"]
        return {key: value.detach() if torch.is_tensor(value) else value for key, value in results.items()}

    def _state_to_aimnet2_data(self, state: SimState) -> dict[str, Tensor]:
        data: dict[str, Tensor] = {
            "coord": state.positions.clone(),
            "numbers": state.atomic_numbers.to(torch.int64),
            "mol_idx": state.system_idx.to(torch.int64),
            "charge": self._system_tensor(state, "charge", default=0.0),
        }

        if self._base_calc.is_nse:
            data["mult"] = self._system_tensor(state, "mult", "spin", default=1.0)

        pbc = state.pbc
        cell = state.row_vector_cell
        has_periodic_cell = torch.as_tensor(pbc, device=self._device, dtype=torch.bool).any() and not torch.allclose(
            cell, torch.zeros_like(cell)
        )
        if has_periodic_cell:
            data["cell"] = cell.contiguous()
            data["pbc"] = pbc
        elif self._compute_stress:
            raise ValueError("AIMNet2 stress calculation requires a periodic TorchSim state with a non-zero cell.")

        return data

    def _system_tensor(self, state: SimState, *names: str, default: float) -> Tensor:
        value = None
        for name in names:
            value = getattr(state, name, None)
            if value is not None:
                break
        if value is None:
            return torch.full((state.n_systems,), default, dtype=torch.float32, device=self._device)

        tensor = torch.as_tensor(value, dtype=torch.float32, device=self._device).reshape(-1)
        if tensor.numel() == 1:
            tensor = tensor.expand(state.n_systems)
        elif tensor.numel() != state.n_systems:
            names_label = "/".join(names)
            raise ValueError(
                f"TorchSim system extra '{names_label}' must be scalar or have one value per system "
                f"({state.n_systems}); got {tensor.numel()} values."
            )
        return tensor

    def _update_implemented_properties(self) -> None:
        properties = ["energy"]
        if self._compute_forces:
            properties.append("forces")
        if self._compute_stress:
            properties.append("stress")
        properties.extend(["charges", "partial_charges"])
        if self._base_calc.is_nse:
            properties.append("spin_charges")
        self.implemented_properties = properties
