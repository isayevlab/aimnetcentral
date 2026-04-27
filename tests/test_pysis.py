"""Pysisyphus integration smoke tests.

Verifies the AIMNet2Pysis wrapper end-to-end: geometry optimization on water,
TS search on HCN <-> CNH via rsprfo with analytic Hessian, and the result-level
cache that prevents redundant model forwards on the get_energy -> get_forces
double-call pattern. All tests skip cleanly when pysisyphus is not installed.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.pysis

pytest.importorskip("pysisyphus", reason="pysisyphus not installed")


def _water_geom():
    from pysisyphus.constants import ANG2BOHR
    from pysisyphus.Geometry import Geometry

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
    """Linearly distorted HCN, near the HCN <-> CNH TS region."""
    from pysisyphus.constants import ANG2BOHR
    from pysisyphus.Geometry import Geometry

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
        from pysisyphus.optimizers.RFOptimizer import RFOptimizer

        from aimnet.calculators import AIMNet2Pysis

        geom = _water_geom()
        geom.set_calculator(AIMNet2Pysis("aimnet2"))
        opt = RFOptimizer(geom, max_cycles=30, thresh="gau", dump=False)
        opt.run()
        assert opt.is_converged

    def test_hcn_ts_with_analytic_hessian(self):
        from pysisyphus.tsoptimizers.RSPRFOptimizer import RSPRFOptimizer

        from aimnet.calculators import AIMNet2Pysis

        geom = _hcn_ts_guess()
        geom.set_calculator(AIMNet2Pysis("aimnet2"))
        ts_opt = RSPRFOptimizer(geom, max_cycles=40, hessian_init="calc", hessian_recalc=3, thresh="gau", dump=False)
        ts_opt.run()
        assert ts_opt.is_converged
        eigvals = np.linalg.eigvalsh(geom.cart_hessian.reshape(9, 9))
        assert (eigvals < 0).sum() == 1, f"Expected one negative eigenvalue, got {(eigvals < 0).sum()}"

    def test_double_call_cache_hit(self):
        """get_energy after get_forces at the same coord must serve from cache."""
        from aimnet.calculators import AIMNet2Pysis

        calc = AIMNet2Pysis("aimnet2")
        geom = _water_geom()
        atoms = geom.atoms
        coords = geom.coords

        calc.get_forces(atoms, coords)
        cached_results_id = id(calc._cache_results)
        assert calc._cache_results is not None
        assert "forces" in calc._cache_results

        calc.get_energy(atoms, coords)
        assert id(calc._cache_results) == cached_results_id, "cache must not be replaced on hit"

    def test_cache_invalidates_on_coord_change(self):
        from aimnet.calculators import AIMNet2Pysis

        calc = AIMNet2Pysis("aimnet2")
        geom = _water_geom()
        atoms = geom.atoms

        calc.get_forces(atoms, geom.coords)
        first_results = calc._cache_results

        perturbed = geom.coords.copy()
        perturbed[0] += 0.01
        calc.get_forces(atoms, perturbed)
        assert calc._cache_results is not first_results, "cache must be replaced when coords change"
