import numpy as np
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
    def __init__(
        self, model: AIMNet2Calculator | str = "aimnet2", charge=0, mult=1, validate_species: bool = True, **kwargs
    ):
        super().__init__(charge=charge, mult=mult, **kwargs)
        if isinstance(model, str):
            model = AIMNet2Calculator(model)
        self.model = model
        self.validate_species = validate_species
        # Cache the most recent results dict so that the get_energy → get_forces
        # double-call pattern (AFIR, some IRC paths) doesn't run the model twice.
        self._cache_key: tuple[tuple[str, ...], bytes] | None = None
        self._cache_results: dict | None = None

    def _cache_get(self, atoms, coord) -> dict | None:
        if self._cache_results is None:
            return None
        if self._cache_key == (tuple(atoms), np.asarray(coord).tobytes()):
            return self._cache_results
        return None

    def _cache_put(self, atoms, coord, results: dict) -> None:
        self._cache_key = (tuple(atoms), np.asarray(coord).tobytes())
        self._cache_results = results

    def _prepare_input(self, atoms, coord):
        device = self.model.device
        numbers = torch.as_tensor([ATOMIC_NUMBERS[a.lower()] for a in atoms], device=device)
        # CPU-side cast + Bohr→Å scalar before H2D; halves PCIe bandwidth (float64→float32)
        # and folds the unit conversion into the upload, eliminating a small GPU kernel launch.
        coord_np = (np.asarray(coord, dtype=np.float32) * BOHR2ANG).reshape(-1, 3)
        coord = torch.from_numpy(coord_np).to(device, non_blocking=True)
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
        cached = self._cache_get(atoms, coords)
        if cached is not None and "energy" in cached:
            return {"energy": self._results_get_energy(cached)}
        _in = self._prepare_input(atoms, coords)
        res = self.model(_in, validate_species=self.validate_species)
        self._cache_put(atoms, coords, res)
        return {"energy": self._results_get_energy(res)}

    def get_forces(self, atoms, coords):
        cached = self._cache_get(atoms, coords)
        if cached is not None and "forces" in cached:
            return {"energy": self._results_get_energy(cached), "forces": self._results_get_forces(cached)}
        _in = self._prepare_input(atoms, coords)
        res = self.model(_in, forces=True, validate_species=self.validate_species)
        self._cache_put(atoms, coords, res)
        return {"energy": self._results_get_energy(res), "forces": self._results_get_forces(res)}

    def get_hessian(self, atoms, coords):
        _in = self._prepare_input(atoms, coords)
        res = self.model(_in, forces=True, hessian=True, validate_species=self.validate_species)
        self._cache_put(atoms, coords, res)
        return {
            "energy": self._results_get_energy(res),
            "forces": self._results_get_forces(res),
            "hessian": self._results_get_hessian(res),
        }


def run_pysis():
    pysisyphus.run.CALC_DICT["aimnet"] = AIMNet2Pysis
    pysisyphus.run.run()
