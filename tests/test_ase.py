import os

from ase.io import read

from aimnet.calculators import AIMNet2ASE

MODELS = ("aimnet2", "aimnet2_b973c")

file = os.path.join(os.path.dirname(__file__), "data", "caffeine.xyz")


def test_calculator():
    for model in MODELS:
        atoms = read(file)
        atoms.calc = AIMNet2ASE(model)
        e = atoms.get_potential_energy()
        assert isinstance(e, float)

        assert hasattr(atoms, "get_charges")
        q = atoms.get_charges()
        assert q.shape == (len(atoms),)

        assert hasattr(atoms, "get_dipole_moment")
        dm = atoms.get_dipole_moment()
        assert dm.shape == (3,)
