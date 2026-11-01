"""Stress training through the built-in SizeGroupedDataset loader (#138)."""

import copy

import numpy as np
import pytest

pytestmark = pytest.mark.train

torch = pytest.importorskip("torch")
pytest.importorskip("ignite")
OmegaConf = pytest.importorskip("omegaconf").OmegaConf


def _model_config(outputs_extra: dict | None = None) -> dict:
    outputs = {
        "energy_mlp": {
            "class": "aimnet.modules.Output",
            "kwargs": {
                "n_in": 16,
                "n_out": 1,
                "key_in": "aim",
                "key_out": "energy",
                "mlp": {"activation_fn": "torch.nn.GELU", "last_linear": True, "hidden": [16]},
            },
        },
        "atomic_shift": {"class": "aimnet.modules.AtomicShift", "kwargs": {"key_in": "energy", "key_out": "energy"}},
        "atomic_sum": {"class": "aimnet.modules.AtomicSum", "kwargs": {"key_in": "energy", "key_out": "energy"}},
    }
    outputs.update(outputs_extra or {})
    return {
        "class": "aimnet.models.AIMNet2",
        "kwargs": {
            "nfeature": 4,
            "d2features": True,
            "ncomb_v": 4,
            "hidden": [[16], [16], [16]],
            "aim_size": 16,
            # Deliberately not the calculator's 5.0 A fallback: the loader must
            # read the cutoff from the model.
            "aev": {"rc_s": 4.0, "nshifts_s": 8},
            "outputs": outputs,
        },
    }


def _build_model(outputs_extra: dict | None = None):
    from aimnet.train.utils import build_model

    torch.manual_seed(0)
    return build_model(OmegaConf.create(_model_config(outputs_extra)))


def _periodic_groups(seed: int = 0) -> dict[int, dict[str, np.ndarray]]:
    """Two size groups of small triclinic cells; the 3-atom group makes N equal to the cell rank."""
    rng = np.random.default_rng(seed)
    groups = {}
    for natoms, nsamples in ((3, 3), (4, 3)):
        cell = np.stack([np.diag(rng.uniform(3.6, 4.4, 3)) + rng.uniform(-0.3, 0.3, (3, 3)) for _ in range(nsamples)])
        frac = rng.uniform(0.0, 1.0, (nsamples, natoms, 3))
        # Some atoms sit outside the home cell, as unwrapped MD frames do.
        frac[:, 0] += 1.0
        groups[natoms] = {
            "coord": np.einsum("bni,bij->bnj", frac, cell).astype(np.float32),
            "numbers": rng.choice([1, 6, 8], (nsamples, natoms)).astype(np.int64),
            "charge": np.zeros(nsamples, dtype=np.float32),
            "cell": cell.astype(np.float32),
            "energy": rng.normal(size=nsamples).astype(np.float32),
            "forces": rng.normal(scale=0.1, size=(nsamples, natoms, 3)).astype(np.float32),
            "stress": rng.normal(scale=1e-3, size=(nsamples, 3, 3)).astype(np.float32),
        }
    return groups


def _write_dataset(path) -> None:
    from aimnet.data import SizeGroupedDataset

    SizeGroupedDataset(_periodic_groups()).save_h5(str(path))


def _train_config(data_file) -> OmegaConf:
    from aimnet.train import train as train_module

    cfg = OmegaConf.load(train_module._default_config)
    overrides = {
        "run_name": "stress-smoke",
        "data": {
            "train": str(data_file),
            "val": str(data_file),
            "sae": {"energy": None},
            "x": ["coord", "numbers", "charge", "cell"],
            "y": ["energy", "forces", "stress"],
            "samplers": {
                "train": {"kwargs": {"batch_size": 4, "batches_per_epoch": 2}},
                "val": {"kwargs": {"batch_size": 4}},
            },
            "loaders": {"train": {"pin_memory": False}, "val": {"pin_memory": False}},
        },
        "trainer": {"epochs": 1},
        "scheduler": None,
        "checkpoint": None,
        "wandb": None,
    }
    cfg = OmegaConf.merge(cfg, overrides)
    cfg.loss.kwargs.components = {
        "energy": {"fn": "aimnet.train.loss.energy_loss_fn", "weight": 1.0},
        "forces": {
            "fn": "aimnet.train.loss.peratom_loss_fn",
            "weight": 0.2,
            "kwargs": {"key_true": "forces", "key_pred": "forces"},
        },
        "stress": {
            "fn": "aimnet.train.loss.mse_loss_fn",
            "weight": 100.0,
            "kwargs": {"key_true": "stress", "key_pred": "stress"},
        },
    }
    return cfg


def test_stress_training_runs_through_builtin_loader(tmp_path, monkeypatch):
    from aimnet.train import train as train_module

    data_file = tmp_path / "periodic.h5"
    _write_dataset(data_file)
    save = tmp_path / "weights.pt"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    train_module.run(0, 1, _model_config(), OmegaConf.to_yaml(_train_config(data_file)), None, str(save))

    state = torch.load(save, weights_only=True)
    assert state and all(torch.isfinite(v).all() for v in state.values() if v.is_floating_point())


def _loader_batches(cutoffs, batch_size=6):
    from aimnet.data import SizeGroupedDataset, SizeGroupedSampler
    from aimnet.train.neighbor_lists import PeriodicNeighborCollate

    ds = SizeGroupedDataset(_periodic_groups())
    ds.merge_groups(min_size=100)  # pads the 3-atom group into the 4-atom one with numbers == 0
    sampler = SizeGroupedSampler(ds, batch_size=batch_size)
    loader = ds.get_loader(
        sampler,
        ["coord", "numbers", "charge", "cell"],
        ["energy", "forces", "stress"],
        collate_fn=PeriodicNeighborCollate(cutoffs),
    )
    return list(loader), ds


def test_collate_builds_packed_periodic_lists_with_trailing_dummy():
    from aimnet.train.neighbor_lists import neighbor_cutoffs
    from aimnet.train.utils import _CompiledTrainingRunner

    cutoffs = neighbor_cutoffs(_build_model())
    assert cutoffs == {"": pytest.approx(4.0)}
    (batch,), ds = _loader_batches(cutoffs)
    x, y = batch
    real_atoms = int((ds.groups[0]["numbers"] != 0).sum())
    L = real_atoms + 1

    _CompiledTrainingRunner(torch.nn.Identity(), ("stress",))._validate_stress_input(x)
    assert x["coord"].shape == (L, 3)
    assert x["numbers"].shape == (L,) and int(x["numbers"][-1]) == 0 and bool((x["numbers"][:-1] != 0).all())
    assert x["mol_idx"].shape == (L,) and int(x["mol_idx"][-1]) == 5
    assert bool((x["mol_idx"][1:] >= x["mol_idx"][:-1]).all())
    assert x["nbmat"].shape[0] == L and bool((x["nbmat"][-1] == L - 1).all())
    assert x["shifts"].shape == (*x["nbmat"].shape, 3) and not x["shifts"].is_floating_point()
    assert x["cell"].shape == (6, 3, 3) and x["charge"].shape == (6,)
    # Per-atom labels follow the packed layout; per-system labels keep theirs.
    assert y["forces"].shape == (L, 3) and bool((y["forces"][-1] == 0).all())
    assert y["energy"].shape == (6,) and y["stress"].shape == (6, 3, 3)


def test_collate_rejects_missing_cell_and_precomputed_lists():
    from aimnet.train.neighbor_lists import PeriodicNeighborCollate

    collate = PeriodicNeighborCollate({"": 4.0})
    sample = {
        "coord": np.zeros((1, 2, 3), dtype=np.float32),
        "numbers": np.ones((1, 2), dtype=np.int64),
        "charge": np.zeros(1, dtype=np.float32),
    }
    with pytest.raises(ValueError, match="cell"):
        collate([(sample, {"energy": np.zeros(1, dtype=np.float32)})])
    sample["cell"] = np.eye(3, dtype=np.float32)[None] * 5
    sample["nbmat"] = np.zeros((1, 2, 1), dtype=np.int64)
    with pytest.raises(ValueError, match="nbmat"):
        collate([(sample, {"energy": np.zeros(1, dtype=np.float32)})])


def _lrcoulomb(**kwargs) -> dict:
    return {
        "lrcoulomb": {
            "class": "aimnet.modules.LRCoulomb",
            "kwargs": {"rc": 4.6, "key_in": "charges", "key_out": "energy", **kwargs},
        }
    }


def test_neighbor_cutoffs_follow_long_range_modules():
    from aimnet.train.neighbor_lists import neighbor_cutoffs

    # The short-range Coulomb subtraction runs on nbmat, so its 4.6 A radius
    # widens the 4.0 A AEV list.
    ewald = neighbor_cutoffs(_build_model(_lrcoulomb(method="ewald")))
    assert ewald == {"": pytest.approx(4.6), "_coulomb": ("ewald", pytest.approx(1e-6))}
    d3ts = {"d3": {"class": "aimnet.modules.D3TS", "kwargs": {"a1": 0.4, "a2": 4.8, "s8": 0.3}}}
    assert neighbor_cutoffs(_build_model(d3ts)) == {"": pytest.approx(4.0), "_dftd3": pytest.approx(15.0)}
    with pytest.raises(ValueError, match="ewald"):
        neighbor_cutoffs(_build_model(_lrcoulomb(method="simple")))
    with pytest.raises(NotImplementedError, match="dsf"):
        neighbor_cutoffs(_build_model(_lrcoulomb(method="dsf")))
    # Pair repulsion also runs on nbmat; with cutoff_fn="none" it sums every listed pair.
    srrep = {"srrep": {"class": "aimnet.modules.core.SRRep", "kwargs": {"key_out": "energy", "rc": 5.2}}}
    assert neighbor_cutoffs(_build_model(srrep)) == {"": pytest.approx(5.2)}


def _assert_matches_calculator(model, calc) -> None:
    from aimnet.train.neighbor_lists import neighbor_cutoffs
    from aimnet.train.utils import build_compiled_training_runner, prepare_batch

    runner = build_compiled_training_runner(model, ("energy", "forces", "stress"), compile_training=False)
    (batch,), ds = _loader_batches(neighbor_cutoffs(model))
    pred = runner(prepare_batch(dict(batch[0]), device="cpu"))

    group = ds.groups[0]
    offset = 0
    for b in range(len(group)):
        real = group["numbers"][b] != 0
        ref = calc(
            {
                "coord": torch.as_tensor(group["coord"][b][real]),
                "numbers": torch.as_tensor(group["numbers"][b][real]),
                "charge": torch.zeros(1),
                "cell": torch.as_tensor(group["cell"][b]),
            },
            forces=True,
            stress=True,
        )
        n = int(real.sum())
        torch.testing.assert_close(pred["energy"][b].detach(), ref["energy"].reshape(()), rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(pred["forces"][offset : offset + n].detach(), ref["forces"], rtol=1e-5, atol=1e-6)
        # The batch keeps the unwrapped coordinates (some atoms one cell or more
        # outside), so float32 pair vectors c_j - c_i + s @ cell round slightly
        # differently from the calculator's wrapped ones: bound by the stress scale.
        stress_ref = ref["stress"].reshape(3, 3)
        stress_atol = 2e-5 * stress_ref.abs().max().item()
        torch.testing.assert_close(pred["stress"][b].detach(), stress_ref, rtol=1e-5, atol=stress_atol)
        offset += n


def test_loader_stress_matches_calculator():
    from aimnet.calculators import AIMNet2Calculator

    model = _build_model()
    _assert_matches_calculator(model, AIMNet2Calculator(copy.deepcopy(model), device="cpu"))


def test_loader_ewald_stress_matches_calculator():
    from aimnet.calculators import AIMNet2Calculator

    model = _build_model(_lrcoulomb(method="ewald"))
    # The same network with Coulomb owned by the calculator instead of the model.
    base = _build_model()
    base.load_state_dict({k: v for k, v in model.state_dict().items() if not k.startswith("outputs.lrcoulomb")})
    calc = AIMNet2Calculator(base, device="cpu", needs_coulomb=True)
    calc.set_lrcoulomb_method("ewald")
    _assert_matches_calculator(model, calc)


def _dense_batch(natoms: int, nsamples: int = 2) -> dict:
    """Unmerged dense samples of one size; the first atom of each cell sits outside it."""
    rng = np.random.default_rng(natoms)
    cell = np.stack([np.diag(rng.uniform(3.6, 4.4, 3)) + rng.uniform(-0.3, 0.3, (3, 3)) for _ in range(nsamples)])
    frac = rng.uniform(0.0, 1.0, (nsamples, natoms, 3))
    frac[:, 0] += (1.0, -1.0, 2.0)
    batch = {
        "coord": np.einsum("bni,bij->bnj", frac, cell),
        "numbers": rng.choice([1, 6, 8], (nsamples, natoms)),
        "charge": np.zeros(nsamples),
        "cell": cell,
        "forces": np.ones((nsamples, natoms, 3)),
        "stress": np.zeros((nsamples, 3, 3)),
    }
    return {k: torch.as_tensor(v, dtype=torch.long if k == "numbers" else torch.float32) for k, v in batch.items()}


def _inputs(batch: dict) -> dict:
    return {k: batch[k] for k in ("coord", "numbers", "charge", "cell")}


def test_packed_batch_keeps_original_coordinates_for_dipole():
    from aimnet.calculators import AIMNet2Calculator
    from aimnet.train.neighbor_lists import PeriodicNeighborCollate, neighbor_cutoffs
    from aimnet.train.utils import build_compiled_training_runner, prepare_batch

    model = _build_model({"dipole": {"class": "aimnet.modules.Dipole", "kwargs": {"key_in": "charges"}}})
    calc = AIMNet2Calculator(copy.deepcopy(model), device="cpu")
    batch = _dense_batch(4, nsamples=3)
    x, _ = PeriodicNeighborCollate(neighbor_cutoffs(model)).pack(_inputs(batch), {})
    torch.testing.assert_close(x["coord"][:-1], batch["coord"].flatten(0, 1), rtol=0, atol=0)

    runner = build_compiled_training_runner(model, ("energy", "forces", "stress"), compile_training=False)
    pred = runner(prepare_batch(dict(x), device="cpu"))
    for b in range(3):
        ref = calc(
            {
                "coord": batch["coord"][b],
                "numbers": batch["numbers"][b],
                "charge": torch.zeros(1),
                "cell": batch["cell"][b],
            },
            forces=True,
            stress=True,
        )
        atoms = slice(4 * b, 4 * b + 4)
        torch.testing.assert_close(pred["charges"][atoms].detach(), ref["charges"], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(pred["forces"][atoms].detach(), ref["forces"], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(pred["stress"][b].detach(), ref["stress"].reshape(3, 3), rtol=1e-5, atol=1e-8)
        # Charges are image-invariant, so the dipole is that of the coordinates as given.
        expected = (ref["charges"].unsqueeze(-1) * batch["coord"][b]).sum(0)
        torch.testing.assert_close(pred["dipole"][b].detach(), expected, rtol=1e-5, atol=1e-5)


def test_collate_rejects_partial_periodicity():
    from aimnet.train.neighbor_lists import PeriodicNeighborCollate

    x = _inputs(_dense_batch(4))
    x["pbc"] = torch.tensor([[True, True, False]] * 2)
    with pytest.raises(ValueError, match="3D"):
        PeriodicNeighborCollate({"": 4.0}).pack(x, {})


def test_collate_classifies_per_atom_keys_explicitly():
    from aimnet.train.neighbor_lists import PeriodicNeighborCollate

    collate = PeriodicNeighborCollate({"": 4.0})
    # One-atom cells: (B, 1, 3) forces are per atom; (B, 3, 3) cell and stress are per system.
    one = _dense_batch(1)
    x, y = collate.pack(_inputs(one), {"forces": one["forces"], "stress": one["stress"]})
    assert x["coord"].shape == (3, 3) and y["forces"].shape == (3, 3)
    assert y["stress"].shape == (2, 3, 3) and x["cell"].shape == (2, 3, 3)

    # Three-atom cells: an unknown (B, 3, 3) label could be either, so it is refused ...
    three = _inputs(_dense_batch(3))
    with pytest.raises(ValueError, match="hessian_like"):
        collate.pack(dict(three), {"hessian_like": torch.zeros(2, 3, 3)})
    # ... unless declared per atom; an unknown label that cannot be per atom is kept per system.
    declared = PeriodicNeighborCollate({"": 4.0}, atom_keys=("hessian_like",))
    _, y3 = declared.pack(dict(three), {"hessian_like": torch.zeros(2, 3, 3), "band_gap": torch.zeros(2, 5)})
    assert y3["hessian_like"].shape == (7, 3) and y3["band_gap"].shape == (2, 5)
