from typing import ClassVar

import torch

from .calculator import AIMNet2Calculator

try:
    import pysisyphus.run  # type: ignore
    from pysisyphus.calculators.Calculator import Calculator  # type: ignore
    from pysisyphus.constants import ANG2BOHR, AU2EV, BOHR2ANG  # type: ignore
    from pysisyphus.elem_data import ATOMIC_NUMBERS  # type: ignore
except ImportError:
    raise ImportError("Pysisyphus is not installed. Please install Pysisyphus to use this module.") from None

EV2AU = 1 / AU2EV


class AIMNet2Pysis(Calculator):
    implemented_properties: ClassVar = ["energy", "forces", "free_energy", "charges", "stress"]

    def __init__(self, model: AIMNet2Calculator | str = "aimnet2", charge=0, mult=1, **kwargs):
        super().__init__(charge=charge, mult=mult, **kwargs)
        if isinstance(model, str):
            model = AIMNet2Calculator(model)
        self.model = model

    def _prepare_input(self, atoms, coord):
        device = self.model.device
        numbers = torch.as_tensor([ATOMIC_NUMBERS[a.lower()] for a in atoms], device=device)
        coord = torch.as_tensor(coord, dtype=torch.float, device=device).view(-1, 3) * BOHR2ANG
        charge = torch.as_tensor([self.charge], dtype=torch.float, device=device)
        mult = torch.as_tensor([self.mult], dtype=torch.float, device=device)
        return {"coord": coord, "numbers": numbers, "charge": charge, "mult": mult}

    @staticmethod
    def _results_get_energy(results):
        return results["energy"].item() * EV2AU

    @staticmethod
    def _results_get_forces(results):
        return (results["forces"].detach() * (EV2AU / ANG2BOHR)).flatten().to(torch.double).cpu().numpy()

    @staticmethod
    def _results_get_hessian(results):
        return (
            (results["hessian"].flatten(0, 1).flatten(-2, -1) * (EV2AU / ANG2BOHR / ANG2BOHR))
            .to(torch.double)
            .cpu()
            .numpy()
        )

    def get_energy(self, atoms, coords):
        _in = self._prepare_input(atoms, coords)
        res = self.model(_in)
        energy = self._results_get_energy(res)
        return {"energy": energy}

    def get_forces(self, atoms, coords):
        _in = self._prepare_input(atoms, coords)
        res = self.model(_in, forces=True)
        energy = self._results_get_energy(res)
        forces = self._results_get_forces(res)
        return {"energy": energy, "forces": forces}

    def get_hessian(self, atoms, coords):
        _in = self._prepare_input(atoms, coords)
        res = self.model(_in, forces=True, hessian=True)
        energy = self._results_get_energy(res)
        forces = self._results_get_forces(res)
        hessian = self._results_get_hessian(res)
        return {"energy": energy, "forces": forces, "hessian": hessian}


def run_pysis():
    pysisyphus.run.CALC_DICT["aimnet"] = AIMNet2Pysis
    pysisyphus.run.run()
