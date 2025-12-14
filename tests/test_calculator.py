import os

import pytest

from aimnet.calculators import AIMNet2Calculator

file = os.path.join(os.path.dirname(__file__), "data", "caffeine.xyz")


@pytest.mark.ase
def test_from_zoo():
    pytest.importorskip("ase", reason="ASE not installed. Install with: pip install aimnet[ase]")
    import ase.io

    def load_mol(filepath):
        atoms = ase.io.read(filepath)
        data = {
            "coord": atoms.get_positions(),  # type: ignore
            "numbers": atoms.get_atomic_numbers(),  # type: ignore
            "charge": 0.0,
        }
        return data

    calc = AIMNet2Calculator("aimnet2", nb_threshold=0)
    data = load_mol(file)
    res = calc(data)
    assert "energy" in res
    res = calc(data, forces=True)
    assert "forces" in res
    res = calc(data, hessian=True)
    assert "hessian" in res


# def test_non_nopbc():
#     calc = AIMNet2Calculator('aimnet2', nb_threshold=0)
#     data = load_mol(file)
#     data['coord'] = data['coord'][:10]
#     data['numbers'] = data['numbers'][:10]
#     res = calc(data)
#     assert 'energy' in res
#     res = calc(data, forces=True)
#     assert 'forces' in res
#     res = calc(data, hessian=True)
#     assert 'hessian' in res
