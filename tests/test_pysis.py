"""Pysisyphus integration smoke tests.

Verifies the AIMNet2Pysis wrapper end-to-end: geometry optimization on water,
TS search on HCN <-> CNH via rsprfo with analytic Hessian, and the result-level
cache that prevents redundant model forwards on the get_forces -> get_energy
double-call pattern. All tests skip cleanly when pysisyphus is not installed.
"""

from unittest.mock import patch

import numpy as np
import pytest

pytestmark = pytest.mark.pysis

pytest.importorskip("pysisyphus", reason="pysisyphus not installed")

from pysisyphus.constants import ANG2BOHR  # noqa: E402
from pysisyphus.Geometry import Geometry  # noqa: E402
from pysisyphus.optimizers.RFOptimizer import RFOptimizer  # noqa: E402
from pysisyphus.tsoptimizers.RSPRFOptimizer import RSPRFOptimizer  # noqa: E402

from aimnet.calculators import AIMNet2Pysis  # noqa: E402


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
        geom.set_calculator(AIMNet2Pysis("aimnet2"))
        ts_opt = RSPRFOptimizer(geom, max_cycles=40, hessian_init="calc", hessian_recalc=3, thresh="gau", dump=False)
        ts_opt.run()
        assert ts_opt.is_converged

    def test_cache_serves_get_energy_after_get_forces(self):
        """get_forces then get_energy at the same coord must run the model once."""
        calc = AIMNet2Pysis("aimnet2")
        geom = _water_geom()
        with patch.object(calc, "model", wraps=calc.model) as model_spy:
            calc.get_forces(geom.atoms, geom.coords)
            calc.get_energy(geom.atoms, geom.coords)
            assert model_spy.call_count == 1

    def test_cache_invalidates_on_coord_change(self):
        """Different coords must trigger a new model forward."""
        calc = AIMNet2Pysis("aimnet2")
        geom = _water_geom()
        with patch.object(calc, "model", wraps=calc.model) as model_spy:
            calc.get_forces(geom.atoms, geom.coords)
            perturbed = geom.coords.copy()
            perturbed[0] += 0.01
            calc.get_forces(geom.atoms, perturbed)
            assert model_spy.call_count == 2
