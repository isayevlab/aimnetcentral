import numpy as np
import torch

from .calculator import AIMNet2Calculator

try:
    import pysisyphus.run  # type: ignore
    from pysisyphus.calculators.Calculator import Calculator  # type: ignore
    from pysisyphus.constants import ANG2BOHR, AU2EV, BOHR2ANG  # type: ignore
    from pysisyphus.elem_data import ATOMIC_NUMBERS  # type: ignore
except ImportError as exc:
    _PYSIS_IMPORT_ERROR: ImportError | None = exc

    class Calculator:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            pass

    ANG2BOHR = 1.0
    AU2EV = 1.0
    BOHR2ANG = 1.0
    ATOMIC_NUMBERS: dict[str, int] = {}
    pysisyphus = None  # type: ignore[assignment]
else:
    _PYSIS_IMPORT_ERROR = None

EV2AU = 1 / AU2EV


class AIMNet2Pysis(Calculator):
    def __init__(
        self, model: AIMNet2Calculator | str = "aimnet2", charge=0, mult=1, validate_species: bool = True, **kwargs
    ):
        if _PYSIS_IMPORT_ERROR is not None:
            raise ImportError(
                'AIMNet2Pysis requires PySisyphus. Install it with `pip install "aimnet[pysis]"`.'
            ) from _PYSIS_IMPORT_ERROR

        super().__init__(charge=charge, mult=mult, **kwargs)
        if isinstance(model, str):
            model = AIMNet2Calculator(model)
        self.model = model
        self.validate_species = validate_species
        # AFIR and some IRC paths call get_forces then get_energy at the same coord;
        # the cache serves the second call without a redundant model forward.
        self._cache_key: tuple[tuple[str, ...], bytes] | None = None
        self._cache_results: dict | None = None

    def _prepare_input(self, atoms, coord):
        device = self.model.device
        numbers = torch.as_tensor([ATOMIC_NUMBERS[a.lower()] for a in atoms], device=device)
        # CPU-side float64→float32 cast and Bohr→Å scalar before H2D halves PCIe bandwidth.
        coord_np = (np.asarray(coord, dtype=np.float32) * BOHR2ANG).reshape(-1, 3)
        coord = torch.from_numpy(coord_np).to(device)
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
            (results["hessian"].detach().flatten(0, 1).flatten(-2, -1) * (EV2AU / ANG2BOHR / ANG2BOHR))
            .to(torch.double)
            .cpu()
            .numpy()
        )

    def get_energy(self, atoms, coords):
        key = (tuple(atoms), np.asarray(coords).tobytes())
        if self._cache_key == key and self._cache_results is not None:
            return {"energy": self._results_get_energy(self._cache_results)}
        _in = self._prepare_input(atoms, coords)
        res = self.model(_in, validate_species=self.validate_species)
        self._cache_key, self._cache_results = key, res
        return {"energy": self._results_get_energy(res)}

    def get_forces(self, atoms, coords):
        key = (tuple(atoms), np.asarray(coords).tobytes())
        # Cache hit only when populated by a forces=True call; an energy-only cache miss here is correct.
        if self._cache_key == key and self._cache_results is not None and "forces" in self._cache_results:
            return {
                "energy": self._results_get_energy(self._cache_results),
                "forces": self._results_get_forces(self._cache_results),
            }
        _in = self._prepare_input(atoms, coords)
        res = self.model(_in, forces=True, validate_species=self.validate_species)
        self._cache_key, self._cache_results = key, res
        return {"energy": self._results_get_energy(res), "forces": self._results_get_forces(res)}

    def get_hessian(self, atoms, coords):
        key = (tuple(atoms), np.asarray(coords).tobytes())
        _in = self._prepare_input(atoms, coords)
        res = self.model(_in, forces=True, hessian=True, validate_species=self.validate_species)
        self._cache_key, self._cache_results = key, res
        return {
            "energy": self._results_get_energy(res),
            "forces": self._results_get_forces(res),
            "hessian": self._results_get_hessian(res),
        }


def run_pysis():
    if _PYSIS_IMPORT_ERROR is not None:
        raise ImportError(
            'AIMNet2Pysis requires PySisyphus. Install it with `pip install "aimnet[pysis]"`.'
        ) from _PYSIS_IMPORT_ERROR

    pysisyphus.run.CALC_DICT["aimnet"] = AIMNet2Pysis
    pysisyphus.run.run()
