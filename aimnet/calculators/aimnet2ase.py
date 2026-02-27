import numpy as np
import torch

try:
    from ase.calculators.calculator import Calculator, PropertyNotImplementedError, all_changes  # type: ignore
except ImportError:
    raise ImportError("ASE is not installed. Please install ASE to use this module.") from None

from .calculator import AIMNet2Calculator


class AIMNet2ASE(Calculator):
    from typing import ClassVar

    implemented_properties: ClassVar[list[str]] = [
        "energy",
        "forces",
        "free_energy",
        "charges",
        "stress",
        "dipole_moment",
    ]

    def __init__(self, base_calc: AIMNet2Calculator | str = "aimnet2", charge=0, mult=1):
        super().__init__()
        if isinstance(base_calc, str):
            base_calc = AIMNet2Calculator(base_calc)
        self.base_calc = base_calc
        if self.base_calc.is_nse:
            self.implemented_properties = [*self.__class__.implemented_properties, "spin_charges"]
        self.reset()
        self.charge = charge
        self.mult = mult
        self.update_tensors()
        # list of implemented species
        if hasattr(base_calc, "implemented_species"):
            self.implemented_species = base_calc.implemented_species.cpu().numpy()  # type: ignore
        else:
            self.implemented_species = None

    def reset(self):
        super().reset()
        self._t_numbers = None
        self._t_charge = None
        self._t_mult = None

    def set_atoms(self, atoms):
        if self.implemented_species is not None and not np.in1d(atoms.numbers, self.implemented_species).all():
            raise ValueError("Some species are not implemented in the AIMNet2Calculator")
        self.reset()
        self.atoms = atoms

    def check_state(self, atoms, tol=1e-15):
        state = super().check_state(atoms, tol=tol)
        if (not state) and getattr(self, "atoms", None) is not None:
            # Check for specific keys in info that affect the calculation
            old_info = getattr(self.atoms, "info", {})
            new_info = getattr(atoms, "info", {})

            # Check charge
            if old_info.get("charge") != new_info.get("charge"):
                state.append("info")

            # Check spin/multiplicity (NSE models only)
            elif self.base_calc.is_nse:
                old_spin = old_info.get("spin", old_info.get("mult"))
                new_spin = new_info.get("spin", new_info.get("mult"))
                if old_spin != new_spin:
                    state.append("info")
        return state

    def set_charge(self, charge):
        self.charge = charge
        self._t_charge = None
        self.update_tensors()

    def set_mult(self, mult):
        self.mult = mult
        self._t_mult = None
        self.update_tensors()

    def _update_charge_spin_from_info(self):
        atoms = getattr(self, "atoms", None)
        if atoms is None:
            return
        info = getattr(atoms, "info", {})

        # Order of precedence for charge:
        # 1. atoms.info['charge']
        # 2. calculator.charge (passed to constructor or set_charge)
        charge = info.get("charge")
        if charge is not None and charge != self.charge:
            self.charge = charge
            self._t_charge = None

        if self.base_calc.is_nse:
            # Support both "mult" (AIMNet2 style) and "spin" (MACE style)
            # Both represent multiplicity (2S+1)
            mult = info.get("mult", info.get("spin"))
            if mult is not None and mult != self.mult:
                self.mult = mult
                self._t_mult = None

    def update_tensors(self):
        if self._t_numbers is None and getattr(self, "atoms", None):
            self._t_numbers = torch.tensor(self.atoms.numbers, dtype=torch.int64, device=self.base_calc.device)
        if self._t_charge is None:
            self._t_charge = torch.tensor(self.charge, dtype=torch.float32, device=self.base_calc.device)
        if self._t_mult is None:
            self._t_mult = torch.tensor(self.mult, dtype=torch.float32, device=self.base_calc.device)

    def get_dipole_moment(self, atoms):
        charges = self.get_charges()[:, np.newaxis]
        positions = atoms.get_positions()
        return np.sum(charges * positions, axis=0)

    def get_spin_charges(self, atoms=None):
        if "spin_charges" not in self.results:
            raise PropertyNotImplementedError("spin_charges is not available. Use an NSE model (e.g. 'aimnet2nse').")
        return self.results["spin_charges"]

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        if properties is None:
            properties = ["energy"]
        super().calculate(atoms, properties, system_changes)
        self._update_charge_spin_from_info()
        self.update_tensors()

        cell = self.atoms.cell.array if self.atoms.cell is not None and self.atoms.pbc.any() else None

        _in = {
            "coord": torch.tensor(self.atoms.positions, dtype=torch.float32, device=self.base_calc.device),
            "numbers": self._t_numbers,
            "charge": self._t_charge,
            "mult": self._t_mult,
        }

        _unsqueezed = False
        if cell is not None:
            _in["cell"] = cell
        else:
            for k, v in _in.items():
                _in[k] = v.unsqueeze(0)
            _unsqueezed = True

        results = self.base_calc(_in, forces="forces" in properties, stress="stress" in properties)

        for k, v in results.items():
            if _unsqueezed:
                v = v.squeeze(0)
            results[k] = v.detach().cpu().numpy()  # type: ignore

        self.results["energy"] = results["energy"].item()
        self.results["charges"] = results["charges"]
        self.results["dipole_moment"] = self.get_dipole_moment(self.atoms)

        if "forces" in properties:
            self.results["forces"] = results["forces"]
        if "stress" in properties:
            self.results["stress"] = results["stress"]
        if "spin_charges" in results:
            self.results["spin_charges"] = results["spin_charges"]
