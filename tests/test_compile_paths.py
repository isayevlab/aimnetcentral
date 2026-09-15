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


def test_symbolic_trace_context_is_scoped(device):
    numbers = torch.tensor([[6, 1, 1]], device=device)
    data = nbops.calc_masks(nbops.set_nb_mode({"numbers": numbers}))

    assert nbops.is_input_padded(data) is False
    with nbops._symbolic_trace_context():
        assert nbops.get_nb_mode(data) == 0
        assert nbops.is_input_padded(data) is True
        traced = nbops.calc_masks(nbops.set_nb_mode({"numbers": numbers}))
        assert traced["_natom"].shape == traced["mol_sizes"].shape == (1,)
        torch.testing.assert_close(traced["_natom"], torch.tensor([3], device=device))
    assert nbops.is_input_padded(data) is False


@pytest.mark.parametrize("padded", [False, True])
def test_mode0_metadata_shape_and_values(device, padded):
    """Eager mode 0 preserves its scalar unpadded metadata contract."""
    numbers = torch.tensor(
        [[6, 1, 1, 8], [8, 1, 1, 1]] if not padded else [[6, 1, 1, 0], [8, 1, 0, 0]],
        device=device,
    )
    data = nbops.calc_masks(nbops.set_nb_mode({"numbers": numbers}))

    assert data["_input_padded"].shape == ()
    assert data["_input_padded"].dtype == torch.bool
    assert bool(data["_input_padded"].item()) is padded
    if padded:
        expected = torch.tensor([3, 2], device=device)
        assert data["_natom"].shape == data["mol_sizes"].shape == (2,)
    else:
        expected = torch.tensor(4, device=device)
        assert data["_natom"].shape == data["mol_sizes"].shape == ()
    torch.testing.assert_close(data["_natom"], expected)
    torch.testing.assert_close(data["mol_sizes"], expected)


@pytest.mark.parametrize("padded", [False, True])
def test_mode0_metadata_compiled_uses_per_system_counts(device, padded):
    """The fullgraph preparation path keeps static per-system loss metadata."""
    if device.type != "cuda":
        pytest.skip("compiled parity is only meaningful on the GPU backend")
    numbers = torch.tensor(
        [[6, 1, 1, 8], [8, 1, 1, 1]] if not padded else [[6, 1, 1, 0], [8, 1, 0, 0]],
        device=device,
    )

    def metadata(values):
        data = nbops.calc_masks(nbops.set_nb_mode({"numbers": values}))
        return data["_input_padded"], data["_natom"], data["mol_sizes"], data["mask_ij"]

    eager = metadata(numbers)
    torch._dynamo.reset()
    compiled = torch.compile(metadata, dynamic=True, fullgraph=True)
    actual = compiled(numbers)

    expected_counts = torch.tensor([4, 4] if not padded else [3, 2], device=device)
    expected_mask = (
        torch.eye(numbers.shape[1], dtype=torch.bool, device=device).unsqueeze(0).expand(numbers.shape[0], -1, -1)
    )
    padding_mask = numbers.eq(0)
    expected_mask = expected_mask | (padding_mask.unsqueeze(-2) | padding_mask.unsqueeze(-1))
    assert bool(actual[0].item()) is bool(eager[0].item()) is padded
    torch.testing.assert_close(actual[3], expected_mask)
    assert actual[1].shape == actual[2].shape == (2,)
    torch.testing.assert_close(actual[1], expected_counts)
    torch.testing.assert_close(actual[2], expected_counts)


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


@pytest.mark.parametrize("n_mol", [1, 3])
def test_mode1_calc_masks_compiled_matches_eager(device, n_mol):
    """Mode 1 metadata uses the charge length as its fixed output size."""
    if device.type != "cuda":
        pytest.skip("compiled parity is only meaningful on the GPU backend")
    prepared = _packed_data(n_mol, 4, device)
    data = {key: prepared[key] for key in ("numbers", "mol_idx", "nbmat", "charge")}

    def metadata(values):
        values = nbops.calc_masks(nbops.set_nb_mode(values))
        return values["mol_sizes"]

    eager = metadata(dict(data))
    torch._dynamo.reset()
    compiled = torch.compile(metadata, dynamic=True, fullgraph=True)
    actual = compiled(dict(data))

    torch.testing.assert_close(actual, eager)


@pytest.mark.parametrize("mode", [1, 2])
def test_packed_and_global_loss_metadata_uses_mol_sizes(device, mode):
    from aimnet.train.loss import energy_loss_fn

    data = _packed_data(2, 3, device) if mode == 1 else nbops.calc_masks(nbops.set_nb_mode(_mode2_data(device)))
    prediction = torch.tensor([1.0, 2.0], device=device)
    target = torch.tensor([0.0, 0.0], device=device)
    actual = energy_loss_fn({"energy": prediction, "_natom": data["_natom"]}, {"energy": target})
    expected = ((prediction - target).square() / data["mol_sizes"].sqrt()).mean()

    torch.testing.assert_close(data["_natom"], data["mol_sizes"])
    torch.testing.assert_close(actual, expected)


def _mode2_data(device):
    B, N, M = 2, 4, 3
    sentinel = B * N
    nbmat = torch.full((B, N, M), sentinel, dtype=torch.int32, device=device)
    for batch in range(B):
        offset = batch * N
        nbmat[batch, 0, :2] = torch.tensor([offset + 1, offset + 2], device=device)
        nbmat[batch, 1, :2] = torch.tensor([offset, offset + 2], device=device)
        nbmat[batch, 2, :2] = torch.tensor([offset, offset + 1], device=device)
    return {
        "numbers": torch.tensor([[6, 1, 1, 0], [8, 1, 1, 0]], device=device),
        "nbmat": nbmat,
        "nbmat_lr": nbmat,
    }


def test_mode2_calc_masks_compiled_matches_eager(device):
    """Compiled mode 2 avoids alias-identity dedup without changing masks."""
    if device.type != "cuda":
        pytest.skip("compiled parity is only meaningful on the GPU backend")

    def derive(data):
        prepared = nbops.calc_masks(nbops.set_nb_mode(data))
        return prepared["mask_ij"], prepared["_nbmat_gather"], prepared["_nbmat_kernel"]

    eager = derive(_mode2_data(device))
    torch._dynamo.reset()
    compiled = torch.compile(derive, fullgraph=True)
    actual = compiled(_mode2_data(device))

    for got, expected in zip(actual, eager, strict=True):
        torch.testing.assert_close(got, expected)


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
