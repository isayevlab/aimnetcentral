"""Guards for the torch.compile-only fast paths in nbops and ops.

The eager and compiled branches of `get_nb_mode`, `is_input_padded`,
`mol_sum` and `ops.nse` must agree. Under torch.compile these functions
deliberately avoid `Tensor.item()` (a graph break) and read tensor metadata
instead, so nothing else pins them together.
"""

import pytest
import torch

from aimnet import nbops, ops


class TestInferNbMode:
    """infer_nb_mode must reproduce what set_nb_mode writes."""

    def test_matches_set_nb_mode_no_nbmat(self, device):
        data = nbops.set_nb_mode({"numbers": torch.tensor([[6, 1, 1]], device=device)})
        assert nbops.infer_nb_mode(data) == int(data["_nb_mode"].item())

    def test_matches_set_nb_mode_2d_nbmat(self, device):
        nbmat = torch.randint(0, 5, (5, 3), device=device)
        data = nbops.set_nb_mode({"nbmat": nbmat})
        assert nbops.infer_nb_mode(data) == int(data["_nb_mode"].item()) == 1

    def test_matches_set_nb_mode_3d_nbmat(self, device):
        nbmat = torch.randint(0, 5, (2, 5, 3), device=device)
        data = nbops.set_nb_mode({"nbmat": nbmat})
        assert nbops.infer_nb_mode(data) == int(data["_nb_mode"].item()) == 2

    def test_packed_dict_without_nbmat(self, device):
        """mol_sum-style dicts carry a flat 1D numbers and no nbmat."""
        data = {
            "numbers": torch.tensor([6, 1, 1, 6, 1, 0], device=device),
            "mol_idx": torch.tensor([0, 0, 0, 1, 1, 2], device=device),
        }
        assert nbops.infer_nb_mode(data) == 1

    def test_invalid_nbmat_shape(self, device):
        data = {"nbmat": torch.randint(0, 5, (2, 3, 4, 5), device=device)}
        with pytest.raises(ValueError, match="Invalid neighbor matrix shape"):
            nbops.infer_nb_mode(data)


class TestIsInputPadded:
    """Eager must reproduce the _input_padded flag exactly."""

    @pytest.mark.parametrize("padded", [False, True])
    def test_matches_flag(self, device, padded):
        numbers = torch.tensor([[6, 1, 1, 0]] if padded else [[6, 1, 1]], device=device)
        data = nbops.calc_masks(nbops.set_nb_mode({"numbers": numbers}))
        assert nbops.is_input_padded(data) is bool(data["_input_padded"].item())
        assert nbops.is_input_padded(data) is padded


def _packed_data(n_mol, n_atom_per_mol, device, nfeat=2):
    """Packed (mode 1) data dict with the usual trailing padding atom."""
    mol_idx = torch.arange(n_mol, device=device).repeat_interleave(n_atom_per_mol)
    mol_idx = torch.cat([mol_idx, mol_idx[-1:]])
    numbers = torch.full((mol_idx.shape[0],), 6, dtype=torch.long, device=device)
    numbers[-1] = 0
    nbmat = torch.zeros((mol_idx.shape[0], 4), dtype=torch.int32, device=device)
    data = {
        "numbers": numbers,
        "mol_idx": mol_idx,
        "nbmat": nbmat,
        "charge": torch.zeros(n_mol, device=device),
    }
    return nbops.calc_masks(nbops.set_nb_mode(data))


@pytest.mark.parametrize("n_mol", [1, 2, 5])
def test_mol_sum_compiled_matches_eager(device, n_mol):
    """The compiled branch reads the count from `charge`; the eager one from
    the `_num_mol` cache. A single molecule additionally takes the sum path,
    which keeps a degenerate size-1 scatter out of the graph entirely."""
    if device.type != "cuda":
        pytest.skip("compiled parity is only meaningful on the GPU backend")
    data = _packed_data(n_mol, 4, device)
    x = torch.randn(data["mol_idx"].shape[0], 3, device=device)

    torch._dynamo.reset()
    compiled = torch.compile(nbops.mol_sum)
    got = compiled(x, dict(data))
    ref = nbops.mol_sum(x, dict(data))
    assert got.shape == ref.shape == (n_mol, 3)
    torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("n_mol", [1, 3])
def test_nse_compiled_matches_eager(device, n_mol):
    """ops.nse broadcasts per-molecule values back to atoms. For a single
    molecule the compiled path must broadcast rather than gather: a gather's
    backward is a scatter into a size-1 buffer, which inductor miscompiles."""
    if device.type != "cuda":
        pytest.skip("compiled parity is only meaningful on the GPU backend")
    data = _packed_data(n_mol, 4, device)
    n_atom = data["mol_idx"].shape[0]
    q_u = torch.randn(n_atom, 1, device=device, requires_grad=True)
    f_u = torch.rand(n_atom, 1, device=device) + 0.5
    Q = torch.zeros(n_mol, 1, device=device)

    ref = ops.nse(Q, q_u, f_u, dict(data))
    ref_g = torch.autograd.grad(ref.sum(), q_u)[0]

    torch._dynamo.reset()
    got = torch.compile(ops.nse)(Q, q_u, f_u, dict(data))
    got_g = torch.autograd.grad(got.sum(), q_u)[0]

    torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(got_g, ref_g, rtol=1e-5, atol=1e-5)
