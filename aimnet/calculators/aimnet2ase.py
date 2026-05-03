import warnings
from typing import ClassVar

import numpy as np
import torch

try:
    from ase.calculators.calculator import Calculator, PropertyNotImplementedError, all_changes  # type: ignore
except ImportError:
    raise ImportError("ASE is not installed. Please install ASE to use this module.") from None

from .calculator import AIMNet2Calculator


class AIMNet2ASE(Calculator):
    implemented_properties: ClassVar[list[str]] = [
        "energy",
        "forces",
        "free_energy",
        "charges",
        "stress",
        "dipole_moment",
    ]

    def __init__(
        self,
        base_calc: AIMNet2Calculator | str = "aimnet2",
        charge=0,
        mult=1,
        validate_species: bool = True,
    ):
        super().__init__()
        if isinstance(base_calc, str):
            base_calc = AIMNet2Calculator(base_calc)
        self.base_calc = base_calc
        self.validate_species = validate_species
        if self.base_calc.is_nse:
            self.__dict__["implemented_properties"] = [*self.__class__.implemented_properties, "spin_charges"]
        self.reset()
        self.charge = charge
        self.mult = mult
        self.update_tensors()
        # list of implemented species — read from model metadata
        _meta = getattr(base_calc.model, "_metadata", None)
        _species = _meta.get("implemented_species") if _meta is not None else None
        self.implemented_species = np.array(_species, dtype=np.int64) if _species else None

    def reset(self):
        super().reset()
        self._t_numbers = None
        self._t_charge = None
        self._t_mult = None

    def set_atoms(self, atoms):
        if self.implemented_species is not None and not np.isin(atoms.numbers, self.implemented_species).all():
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

    def _update_charge_spin_from_info(self, atoms=None):
        if atoms is None:
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

    def update_tensors(self, atoms=None):
        if atoms is None:
            atoms = getattr(self, "atoms", None)
        if atoms is not None:
            new_numbers = torch.as_tensor(atoms.numbers, dtype=torch.int64, device=self.base_calc.device)
            if (
                self._t_numbers is None
                or self._t_numbers.shape != new_numbers.shape
                or not torch.equal(self._t_numbers, new_numbers)
            ):
                self._t_numbers = new_numbers
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

    def get_hessian(self, atoms=None):
        """Return Cartesian Hessian as a (3N, 3N) ndarray in eV/Å^2.

        Designed for use as ``Sella(atoms, hessian_function=atoms.calc.get_hessian)``.
        Computed via double-backward through the AIMNet2 energy graph; cost scales
        as O(3N) backward passes per call. Not supported when ``compile_model=True``
        or for batched / multi-molecule input.

        This method intentionally bypasses the standard ASE
        ``Calculator.calculate(properties=['hessian'])`` flow and ``self.results``
        cache. The Sella callback contract is ``(atoms) -> ndarray``, so a direct
        method is the simplest match. ``"hessian"`` is therefore not advertised in
        ``implemented_properties``; if that ever changes, the two paths must be
        reconciled.

        When called with an explicit ``atoms`` argument that differs from
        ``self.atoms``, the passed ``atoms.info`` is consulted for charge/mult
        precedence (and the calculator's stored ``self.charge``/``self.mult`` may
        be updated as a side effect, mirroring the ``calculate()`` behavior).
        """
        if atoms is None:
            atoms = getattr(self, "atoms", None)
            if atoms is None:
                raise PropertyNotImplementedError(
                    "get_hessian() requires an attached Atoms object or an explicit argument."
                )
        if atoms.pbc.any():
            raise PropertyNotImplementedError(
                "Hessian for periodic systems is not supported by AIMNet2ASE.get_hessian(). "
                "For periodic transition states, use pysisyphus dimer or climbing-image NEB."
            )
        if len(atoms) > 100:
            warnings.warn(
                f"Computing AIMNet2 Hessian for {len(atoms)} atoms; "
                "the forces+hessian path retains a much larger autograd graph than forces alone "
                "(peak GPU memory is roughly 5-10x a forces-only call). Risk of OOM on smaller GPUs.",
                stacklevel=2,
            )

        self._update_charge_spin_from_info(atoms)
        self.update_tensors(atoms)

        # Pass coord as 2D (N, 3) — not batched — so mol_flatten takes the
        # ndim==2 path and calculate_hessian sees the expected (N+1, 3) coord
        # after padding. Batching (unsqueeze(0)) triggers the ndim==3 path
        # which may skip flattening when N < nb_threshold on GPU, causing
        # calculate_hessian to produce an incorrect (N, 3, 1, 3) shape.
        coord = torch.tensor(atoms.positions, dtype=self.base_calc.keys_in["coord"], device=self.base_calc.device)
        _in = {
            "coord": coord,
            "numbers": self._t_numbers,
            "charge": self._t_charge,
            "mult": self._t_mult,
        }

        results = self.base_calc(
            _in,
            forces=True,
            hessian=True,
            validate_species=self.validate_species,
        )
        H = results["hessian"].detach()  # (N, 3, N, 3)
        N = H.shape[0]
        return H.reshape(N * 3, N * 3).cpu().numpy()

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
            _in["pbc"] = self.atoms.pbc
        else:
            for k, v in _in.items():
                _in[k] = v.unsqueeze(0)
            _unsqueezed = True

        results = self.base_calc(
            _in,
            forces="forces" in properties,
            stress="stress" in properties,
            validate_species=self.validate_species,
        )

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
