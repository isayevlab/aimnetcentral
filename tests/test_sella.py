"""Sella saddle-point optimizer smoke tests.

These tests verify that AIMNet2's analytic Hessian (exposed via
AIMNet2ASE.get_hessian) is callable from Sella's hessian_function= hook.
They are gated by both the ``sella`` and ``ase`` markers; CI without Sella
installed will deselect them automatically.
"""

import numpy as np
import pytest

pytestmark = [pytest.mark.sella, pytest.mark.ase]


def _h2o2_guess():
    """A trivially perturbed H2O2 starting from a non-stationary geometry."""
    from ase import Atoms

    return Atoms(
        "H2O2",
        positions=[
            [0.000, 0.700, 0.350],
            [0.000, -0.700, 0.350],
            [0.000, 0.700, -0.350],
            [0.000, -0.700, -0.350],
        ],
    )


class TestSellaIntegration:
    def test_callback_consumed_by_sella(self):
        """Sella(..., hessian_function=callback) must run at least one step."""
        pytest.importorskip("ase", reason="ASE not installed")
        sella = pytest.importorskip("sella", reason="Sella not installed")

        from aimnet.calculators import AIMNet2ASE

        atoms = _h2o2_guess()
        atoms.calc = AIMNet2ASE("aimnet2")

        dyn = sella.Sella(
            atoms,
            order=0,
            internal=True,
            hessian_function=atoms.calc.get_hessian,
        )
        # Two steps is enough to prove the Hessian callback is consumed without
        # error; we are not benchmarking convergence here.
        dyn.run(fmax=0.05, steps=2)

        # Energy must be finite and forces present afterwards.
        e = atoms.get_potential_energy()
        f = atoms.get_forces()
        assert np.isfinite(e)
        assert np.isfinite(f).all()

    def test_default_sella_no_hessian_callback(self):
        """Sella without a Hessian callback must also work via standard ASE."""
        pytest.importorskip("ase", reason="ASE not installed")
        sella = pytest.importorskip("sella", reason="Sella not installed")

        from aimnet.calculators import AIMNet2ASE

        atoms = _h2o2_guess()
        atoms.calc = AIMNet2ASE("aimnet2")

        dyn = sella.Sella(atoms, order=0, internal=True)
        dyn.run(fmax=0.05, steps=2)

        assert np.isfinite(atoms.get_potential_energy())
