"""Pysisyphus integration smoke tests.

Verifies the AIMNet2Pysis wrapper end-to-end: geometry optimization on water,
TS search on HCN <-> CNH via rsprfo with analytic Hessian, and the result-level
cache that prevents redundant model forwards on the get_forces -> get_energy
double-call pattern. All tests skip cleanly when pysisyphus is not installed.
"""

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.pysis

pytest.importorskip("pysisyphus", reason="pysisyphus not installed")

from pysisyphus.constants import ANG2BOHR  # noqa: E402
from pysisyphus.Geometry import Geometry  # noqa: E402
from pysisyphus.optimizers.RFOptimizer import RFOptimizer  # noqa: E402
from pysisyphus.tsoptimizers.RSPRFOptimizer import RSPRFOptimizer  # noqa: E402

from aimnet.calculators import AIMNet2Pysis  # noqa: E402


class _CountingModelStub:
    """Records call_count; returns shape-correct zero results so the cache logic
    can be exercised without loading a real AIMNet2 model.
    """

    def __init__(self):
        self.device = torch.device("cpu")
        self.call_count = 0

    def __call__(self, data, forces=False, hessian=False, validate_species=True):
        self.call_count += 1
        n = data["coord"].shape[0]
        out = {"energy": torch.zeros(1)}
        if forces:
            out["forces"] = torch.zeros(n, 3)
        if hessian:
            out["hessian"] = torch.zeros(n, 3, n, 3)
        return out


class _CountingPysis(AIMNet2Pysis):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hessian_calls = 0

    def get_hessian(self, atoms, coords):
        self.hessian_calls += 1
        return super().get_hessian(atoms, coords)


def _water_geom():
    atoms = ("o", "h", "h")
    coords = (
        np.array([
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [-0.24, 0.93, 0.0],
        ]).flatten()
        * ANG2BOHR
    )
    return Geometry(atoms, coords)


def _hcn_ts_guess():
    """Distorted HCN, near the HCN <-> CNH TS region."""
    atoms = ("h", "c", "n")
    coords = (
        np.array([
            [-1.18, 0.55, 0.0],
            [0.00, 0.00, 0.0],
            [1.17, 0.00, 0.0],
        ]).flatten()
        * ANG2BOHR
    )
    return Geometry(atoms, coords)


class TestPysisSmoke:
    def test_water_geom_opt_converges(self):
        geom = _water_geom()
        geom.set_calculator(AIMNet2Pysis("aimnet2"))
        opt = RFOptimizer(geom, max_cycles=30, thresh="gau", dump=False)
        opt.run()
        assert opt.is_converged

    def test_hcn_ts_search_runs_with_analytic_hessian(self):
        """Smoke test: rsprfo with hessian_init='calc' must complete and return a
        converged geometry. Whether the topology is a clean first-order saddle
        depends on the model — `aimnet2-rxn` is recommended for serious TS work.
        """
        geom = _hcn_ts_guess()
        calc = _CountingPysis("aimnet2")
        geom.set_calculator(calc)
        ts_opt = RSPRFOptimizer(
            geom,
            max_cycles=40,
            hessian_init="calc",
            hessian_recalc=3,
            trust_radius=0.1,
            trust_max=0.2,
            thresh="gau",
            dump=False,
        )
        ts_opt.run()
        assert ts_opt.is_converged
        assert calc.hessian_calls >= 1

    def test_cache_serves_get_energy_after_get_forces(self):
        """get_forces then get_energy at the same coord must run the model once."""
        stub = _CountingModelStub()
        calc = AIMNet2Pysis(model=stub)
        geom = _water_geom()
        calc.get_forces(geom.atoms, geom.coords)
        calc.get_energy(geom.atoms, geom.coords)
        assert stub.call_count == 1

    def test_cache_invalidates_on_coord_change(self):
        """Different coords must trigger a new model forward."""
        stub = _CountingModelStub()
        calc = AIMNet2Pysis(model=stub)
        geom = _water_geom()
        calc.get_forces(geom.atoms, geom.coords)
        perturbed = geom.coords.copy()
        perturbed[0] += 0.01
        calc.get_forces(geom.atoms, perturbed)
        assert stub.call_count == 2
