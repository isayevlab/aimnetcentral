import pytest

pytest.importorskip("ignite")


pytestmark = pytest.mark.train


def test_build_model_does_not_wrap_forces_when_false():
    torch = pytest.importorskip("torch")
    OmegaConf = pytest.importorskip("omegaconf").OmegaConf
    from aimnet.modules import Forces
    from aimnet.train.utils import build_model

    cfg = OmegaConf.create({"class": "torch.nn.Identity"})
    model = build_model(cfg, forces=False)
    assert isinstance(model, torch.nn.Identity)
    assert not isinstance(model, Forces)


def test_build_model_wraps_forces_when_true():
    OmegaConf = pytest.importorskip("omegaconf").OmegaConf
    from aimnet.modules import Forces
    from aimnet.train.utils import build_model

    cfg = OmegaConf.create({"class": "torch.nn.Identity"})
    model = build_model(cfg, forces=True)
    assert isinstance(model, Forces)


def test_state_dict_roundtrip_weights_only(tmp_path):
    torch = pytest.importorskip("torch")

    sd = {"w": torch.randn(3, 3), "b": torch.zeros(3)}
    p = tmp_path / "sd.pt"
    torch.save(sd, p)
    loaded = torch.load(p, map_location="cpu", weights_only=True)
    assert set(loaded) == {"w", "b"}
    torch.testing.assert_close(loaded["w"], sd["w"])


def test_regression_stats_and_metric_compute():
    torch = pytest.importorskip("torch")
    import numpy as np

    from aimnet.train.metrics import RegMultiMetric, regression_stats

    pred = torch.tensor([1.0, 2.0, 3.0, 4.0])
    true = torch.tensor([1.5, 2.0, 2.5, 5.0])
    stats = regression_stats(pred, true)
    err = (true - pred).numpy()
    assert np.isclose(stats["mae"].item(), np.abs(err).mean())
    assert np.isclose(stats["rmse"].item(), np.sqrt((err**2).mean()))

    cfg = {"energy": {"abbr": "E", "peratom": False}}
    metric = RegMultiMetric(cfg)
    metric.reset()
    # Four samples with one energy each: E_mae = sum|err| / n_samples.
    y_pred = {
        "energy": pred,
        "_natom": torch.tensor([2.0, 2.0, 2.0, 2.0]),
        "numbers": torch.tensor([[1, 8]] * 4),
    }
    y_true = {"energy": true}
    metric.update((y_pred, y_true))
    result = metric.compute()
    assert np.isclose(result["E_mae"], np.abs(err).mean())
    assert np.isclose(result["E_rmse"], np.sqrt((err**2).mean()))


def test_reg_multi_metric_raises_when_empty():
    pytest.importorskip("torch")
    from ignite.exceptions import NotComputableError

    from aimnet.train.metrics import RegMultiMetric

    metric = RegMultiMetric({})
    metric.reset()
    with pytest.raises(NotComputableError):
        metric.compute()


def test_export_model_helpers(tmp_path):
    torch = pytest.importorskip("torch")
    from torch import nn

    from aimnet.train.export_model import (
        bake_sae_into_model,
        get_implemented_species,
        load_sae,
        mask_not_implemented_species,
    )

    sae_file = tmp_path / "sae.yaml"
    sae_file.write_text("1: -0.5\n8: -75.0\n")
    sae = load_sae(str(sae_file))
    assert sae == {1: -0.5, 8: -75.0}
    assert get_implemented_species(sae) == [1, 8]

    bad_file = tmp_path / "bad.yaml"
    bad_file.write_text("- 1\n- 2\n")
    with pytest.raises(TypeError, match="dictionary"):
        load_sae(str(bad_file))

    class Shifts(nn.Module):
        def __init__(self):
            super().__init__()
            self.shifts = nn.Embedding(10, 1)
            nn.init.zeros_(self.shifts.weight)

    model = nn.Module()
    model.outputs = nn.Module()
    model.outputs.atomic_shift = Shifts()
    model.afv = nn.Embedding(10, 4)
    model = bake_sae_into_model(model, sae)
    assert model.outputs.atomic_shift.shifts.weight.dtype == torch.float64
    assert model.outputs.atomic_shift.shifts.weight[8, 0].item() == -75.0
    model = mask_not_implemented_species(model, [1, 8])
    assert torch.isnan(model.afv.weight[2]).all()
    assert not torch.isnan(model.afv.weight[1]).any()


def test_train_utils_param_helpers():
    torch = pytest.importorskip("torch")
    OmegaConf = pytest.importorskip("omegaconf").OmegaConf

    from aimnet.modules import Forces
    from aimnet.train.utils import _to_config_dict, set_trainable_parameters, unwrap_module

    with pytest.raises(TypeError, match="dictionary"):
        _to_config_dict(OmegaConf.create([1, 2]), "Broken")

    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
    model = set_trainable_parameters(model, force_train=["0\\."], force_no_train=["1\\."])
    assert all(p.requires_grad for p in model[0].parameters())
    assert not any(p.requires_grad for p in model[1].parameters())

    inner = torch.nn.Linear(2, 2)
    assert unwrap_module(Forces(inner)) is inner
