"""Supported feature-only inference: layouts, equivalence, and skipped physics."""

import copy
from pathlib import Path

import pytest
import torch
from torch import nn

from aimnet.calculators import AIMNet2Calculator
from aimnet.models.aimnet2 import AIMNet2
from aimnet.modules.core import AtomicSum, Output


@pytest.fixture(scope="module")
def tiny_model():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(163)
        model = (
            AIMNet2(
                aev={"rc_s": 5.0, "nshifts_s": 4},
                nfeature=4,
                d2features=True,
                ncomb_v=4,
                hidden=([8], [8], [8]),
                aim_size=8,
                outputs={
                    "energy_mlp": Output({"hidden": [8]}, 8, 1, "aim", "energy"),
                    "atomic_sum": AtomicSum("energy", "energy"),
                },
            )
            .eval()
            .requires_grad_(False)
        )
        # Untrained flexibility outputs near zero amplify NSE's epsilon bias.
        # Keep this synthetic fixture well-conditioned without changing the model.
        with torch.no_grad():
            for mlp in model.mlps[:-1]:
                final = [layer for layer in mlp if isinstance(layer, nn.Linear)][-1]
                final.weight[1].zero_()
                final.bias[1] = 1.0
        return model


def inputs(padded=True):
    return {
        "coord": torch.tensor([
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [0.0, 0.0, 0.0] if padded else [0.1, 1.2, 0.2]],
        ]),
        "numbers": torch.tensor([[8, 1, 1], [8, 1, 0 if padded else 1]]),
        "charge": torch.tensor([0.0, -1.0 if padded else 0.0]),
    }


def full_features(calc, data):
    """Use the unchanged full model as the numerical reference (including heads)."""
    with torch.no_grad():
        result = calc.model({key: value.to(calc.device).clone() for key, value in data.items()})
    n = data["numbers"].numel()
    shape = data["numbers"].shape
    return {
        "aim": result["aim"].reshape(-1, result["aim"].shape[-1])[:n].reshape(*shape, -1),
        "charges": result["charges"].reshape(-1)[:n].reshape(shape),
    }


def check_equivalence_and_layout(calc, padded=True):
    data = inputs(padded)
    original = {key: value.clone() for key, value in data.items()}
    actual = calc.extract_features(data)
    expected = full_features(calc, data)
    assert torch.equal(actual["atom_mask"].cpu(), data["numbers"] > 0)
    for key in ("aim", "charges"):
        torch.testing.assert_close(actual[key], expected[key], rtol=2e-5, atol=1e-5)
        assert not actual[key].requires_grad
        assert torch.count_nonzero(actual[key][~actual["atom_mask"]]) == 0
        for index in range(2):
            mask = data["numbers"][index] > 0
            single = calc.extract_features({
                "coord": data["coord"][index][mask],
                "numbers": data["numbers"][index][mask],
                "charge": data["charge"][index : index + 1],
            })
            # Torch 2.14 CUDA AQuaRef has a 2.83e-5 full-model batch/single
            # rounding difference. Feature/full equivalence above stays tighter;
            # this bound covers that independently reproduced baseline behavior.
            torch.testing.assert_close(actual[key][index][mask], single[key], rtol=2e-5, atol=4e-5)
    for key in data:
        torch.testing.assert_close(data[key], original[key], rtol=0, atol=0)
    torch.testing.assert_close(actual["charges"].sum(-1).cpu(), data["charge"], atol=2e-5, rtol=0)


def test_cpu_features_match_full_and_single(tiny_model):
    check_equivalence_and_layout(AIMNet2Calculator(copy.deepcopy(tiny_model), device="cpu"))


def test_features_skip_heads_and_long_range(tiny_model, monkeypatch):
    calc = AIMNet2Calculator(copy.deepcopy(tiny_model), device="cpu", needs_coulomb=True)
    expected = calc.extract_features(inputs())

    def fail(*args, **kwargs):
        raise AssertionError("Feature-only inference evaluated an energy head or long-range work")

    for head in calc.model.outputs.values():
        monkeypatch.setattr(head, "forward", fail)
    monkeypatch.setattr(calc.external_coulomb, "forward", fail)
    for name in ("_nblist_lr", "_nblist_coulomb", "_nblist_dftd3"):
        monkeypatch.setattr(calc, name, fail)
    actual = calc.extract_features(inputs())
    for key in actual:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


def test_flattened_features_preserve_molecule_and_atom_order(tiny_model):
    calc = AIMNet2Calculator(copy.deepcopy(tiny_model), device="cpu")
    batch = inputs()
    expected = calc.extract_features(batch)
    mask = batch["numbers"] > 0
    flat = {
        "coord": batch["coord"][mask],
        "numbers": batch["numbers"][mask],
        "charge": batch["charge"],
        "mol_idx": torch.tensor([0, 0, 0, 1, 1]),
    }
    actual = calc.extract_features(flat)
    for key in ("aim", "charges"):
        torch.testing.assert_close(actual[key], expected[key][mask], atol=1e-5, rtol=2e-5)


def test_features_translation_rotation_permutation(tiny_model):
    calc = AIMNet2Calculator(copy.deepcopy(tiny_model), device="cpu")
    data = {key: value[0] for key, value in inputs().items()}
    data["charge"] = data["charge"].reshape(1)
    expected = calc.extract_features(data)
    permutation = torch.tensor([2, 0, 1])
    rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    changed = {
        "coord": data["coord"][permutation] @ rotation + 7.0,
        "numbers": data["numbers"][permutation],
        "charge": data["charge"],
    }
    actual = calc.extract_features(changed)
    for key in ("aim", "charges"):
        torch.testing.assert_close(actual[key], expected[key][permutation], atol=1e-5, rtol=2e-5)


@pytest.mark.parametrize("bad_number", [-1.0, 1.5, float("nan")])
def test_features_reject_invalid_atomic_numbers(tiny_model, bad_number):
    calc = AIMNet2Calculator(copy.deepcopy(tiny_model), device="cpu")
    data = inputs()
    data["numbers"] = data["numbers"].float()
    data["numbers"][0, 0] = bad_number
    with pytest.raises(ValueError, match="Atomic numbers"):
        calc.extract_features(data)


def test_features_reject_unsupported_model():
    calc = AIMNet2Calculator(nn.Identity(), device="cpu")
    with pytest.raises(NotImplementedError, match="forward_features"):
        calc.extract_features(inputs())


@pytest.mark.gpu
@pytest.mark.parametrize("threshold", [0, 120])
@pytest.mark.parametrize("padded", [True, False])
def test_gpu_dense_sparse_features(tiny_model, threshold, padded):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    check_equivalence_and_layout(AIMNet2Calculator(copy.deepcopy(tiny_model), device="cuda", nb_threshold=threshold))


@pytest.mark.weights
@pytest.mark.slow
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("name", ["aimnet2_wb97m_d3_0.pt", "aimnet2-qr_b97md4_qzvp_2.pt"])
def test_real_weight_features(device, name):
    checkpoint = Path.home() / ".cache/aimnet" / name
    if not checkpoint.is_file():
        pytest.skip("Preseed the local checkpoint explicitly")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    imports = {"aimnet.modules.core.SRRep"} if "-qr_" in name else None
    check_equivalence_and_layout(AIMNet2Calculator(str(checkpoint), device=device, model_import_paths=imports))


def test_explicit_mode2_padding_preserves_features(tiny_model):
    calc = AIMNet2Calculator(copy.deepcopy(tiny_model), device="cpu")
    original = inputs()
    expected = calc.extract_features(original)
    data = {
        "coord": torch.cat((original["coord"], torch.zeros(2, 1, 3)), dim=1),
        "numbers": torch.cat((original["numbers"], torch.zeros(2, 1, dtype=torch.long)), dim=1),
        "charge": original["charge"],
        "nbmat": torch.full((2, 4, 3), 8, dtype=torch.int32),
    }
    for batch, count in enumerate((3, 2)):
        for atom in range(count):
            neighbors = [batch * 4 + other for other in range(count) if other != atom]
            data["nbmat"][batch, atom, : len(neighbors)] = torch.tensor(neighbors, dtype=torch.int32)
    actual = calc.extract_features(data)
    for key in ("aim", "charges"):
        torch.testing.assert_close(actual[key][:, :3], expected[key], rtol=2e-5, atol=1e-5)
        assert torch.count_nonzero(actual[key][:, 3]) == 0


@pytest.mark.parametrize("indices", [[0, 2, 2], [1, 1, 1], [0, 1, 0], [0, 0.5, 1]])
def test_features_reject_ambiguous_molecule_indices(tiny_model, indices):
    calc = AIMNet2Calculator(copy.deepcopy(tiny_model), device="cpu")
    data = {key: value[0] for key, value in inputs().items()}
    data["charge"] = torch.tensor([0.0, 0.0, 0.0])
    data["mol_idx"] = indices
    with pytest.raises(ValueError, match="mol_idx"):
        calc.extract_features(data)


def _mode1_case(case):
    water = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    if case == "caller_zero_before_model_dummy":
        # Four caller atoms, five nbmat rows: the model dummy is row 4, so the
        # caller's zero atom at row 3 would be treated as a real atom.
        return {
            "coord": torch.cat((water, torch.tensor([[5.0, 5.0, 5.0]]))),
            "numbers": torch.tensor([8, 1, 1, 0]),
            "charge": torch.tensor([-1.0]),
            "nbmat": torch.tensor([[1, 2], [0, 2], [0, 1], [4, 4], [4, 4]]),
        }
    if case == "real_atom_in_dummy_row":
        return {
            "coord": water,
            "numbers": torch.tensor([8, 1, 1]),
            "charge": torch.tensor([0.0]),
            "nbmat": torch.tensor([[1, 2], [0, 2], [0, 1]]),
        }
    if case == "rectangular_coord":
        return {
            "coord": torch.stack((water, torch.tensor([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [5.0, 5.0, 5.0]]))),
            "numbers": torch.tensor([[8, 1, 1], [8, 1, 0]]),
            "charge": torch.tensor([0.0, -1.0]),
            "nbmat": torch.tensor([[1, 2], [0, 2], [0, 1], [4, 6], [3, 6], [6, 6], [6, 6]]),
        }
    if case == "too_few_rows":
        return {
            "coord": water,
            "numbers": torch.tensor([8, 1, 1]),
            "charge": torch.tensor([0.0]),
            "nbmat": torch.tensor([[1], [0]]),
        }
    raise AssertionError(case)


@pytest.mark.parametrize(
    "case", ["caller_zero_before_model_dummy", "real_atom_in_dummy_row", "rectangular_coord", "too_few_rows"]
)
def test_features_reject_misaligned_mode1_neighbors(tiny_model, case):
    calc = AIMNet2Calculator(copy.deepcopy(tiny_model), device="cpu")
    with pytest.raises(ValueError, match="mode-1"):
        calc.extract_features(_mode1_case(case))


@pytest.mark.parametrize("caller_dummy", [False, True])
def test_explicit_mode1_neighbors_match_builtin(tiny_model, caller_dummy):
    calc = AIMNet2Calculator(copy.deepcopy(tiny_model), device="cpu")
    water = {
        "coord": torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        "numbers": torch.tensor([8, 1, 1]),
        "charge": torch.tensor([-1.0]),
    }
    expected = calc.extract_features(water)
    data = {**water, "nbmat": torch.tensor([[1, 2], [0, 2], [0, 1], [3, 3]])}
    if caller_dummy:
        data["coord"] = torch.cat((data["coord"], torch.zeros(1, 3)))
        data["numbers"] = torch.cat((data["numbers"], torch.zeros(1, dtype=torch.long)))
    actual = calc.extract_features(data)
    for key in ("aim", "charges"):
        torch.testing.assert_close(actual[key][:3], expected[key], rtol=2e-5, atol=1e-5)
    torch.testing.assert_close(actual["charges"].sum(), torch.tensor(-1.0), atol=2e-5, rtol=0)


@pytest.mark.filterwarnings("ignore:Switching to DSF Coulomb")
@pytest.mark.parametrize("shared_shape", [(3, 3), (1, 3, 3)])
@pytest.mark.parametrize("method", ["dsf", "ewald"])
def test_shared_cell_broadcasts_to_every_system(tiny_model, shared_shape, method):
    """A single cell applies to each system of a batch: features, energy, forces, and stress."""
    calc = AIMNet2Calculator(copy.deepcopy(tiny_model), device="cpu", needs_coulomb=True)
    calc.set_lrcoulomb_method(method)
    water = torch.tensor([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
    data = {
        "coord": torch.stack((water + 3.0, water + 4.5)),
        "numbers": torch.tensor([[8, 1, 1], [8, 1, 1]]),
        "charge": torch.tensor([0.0, -1.0]),
    }
    cell = torch.eye(3) * 12.0
    shared = cell.reshape(shared_shape)
    per_system = {**data, "cell": torch.stack((cell, cell))}
    broadcast = {**data, "cell": shared}
    flat = {
        "coord": data["coord"].reshape(-1, 3),
        "numbers": data["numbers"].reshape(-1),
        "charge": data["charge"],
        "mol_idx": torch.tensor([0, 0, 0, 1, 1, 1]),
        "cell": shared,
    }
    expected = calc.extract_features(per_system)
    for key in ("aim", "charges"):
        torch.testing.assert_close(calc.extract_features(broadcast)[key], expected[key], rtol=0, atol=1e-6)
        torch.testing.assert_close(calc.extract_features(flat)[key], expected[key].flatten(0, 1), rtol=0, atol=1e-6)
    reference = calc.eval({k: v.clone() for k, v in per_system.items()}, forces=True, stress=True)
    assert reference["stress"].shape == (2, 3, 3)
    for candidate, layout in ((broadcast, lambda x: x), (flat, lambda x: x.flatten(0, 1))):
        result = calc.eval({k: v.clone() for k, v in candidate.items()}, forces=True, stress=True)
        torch.testing.assert_close(result["energy"], reference["energy"], rtol=0, atol=1e-5)
        torch.testing.assert_close(result["charges"], layout(reference["charges"]), rtol=0, atol=1e-6)
        torch.testing.assert_close(result["forces"], layout(reference["forces"]), rtol=0, atol=1e-5)
        torch.testing.assert_close(result["stress"], reference["stress"], rtol=0, atol=1e-6)
