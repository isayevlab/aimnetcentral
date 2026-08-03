import os
import stat
from pathlib import Path
from unittest.mock import Mock

import pytest

pytestmark = pytest.mark.train


def _write_export_inputs(tmp_path, model_yaml: str) -> tuple[Path, Path, Path]:
    torch = pytest.importorskip("torch")
    weights = tmp_path / "weights.pt"
    model_config = tmp_path / "model.yaml"
    sae = tmp_path / "model.sae"
    torch.save({}, weights)
    model_config.write_text(model_yaml)
    sae.write_text("1: -0.5\n")
    return weights, model_config, sae


def _patch_minimal_export_model(monkeypatch) -> None:
    from aimnet.train import export_model as export_module

    monkeypatch.setattr(export_module, "bake_sae_into_model", lambda model, _sae: model)
    monkeypatch.setattr(export_module, "mask_not_implemented_species", lambda model, _species: model)
    monkeypatch.setattr(export_module, "extract_cutoff", Mock(return_value=5.0))


def test_build_model_does_not_wrap_forces_when_false():
    pytest.importorskip("ignite")
    torch = pytest.importorskip("torch")
    OmegaConf = pytest.importorskip("omegaconf").OmegaConf
    from aimnet.modules import Forces
    from aimnet.train.utils import build_model

    cfg = OmegaConf.create({"class": "torch.nn.Identity"})
    model = build_model(cfg, forces=False)
    assert isinstance(model, torch.nn.Identity)
    assert not isinstance(model, Forces)


def test_build_model_wraps_forces_when_true():
    pytest.importorskip("ignite")
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
    pytest.importorskip("ignite")
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
    pytest.importorskip("ignite")
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


def test_export_model_rejects_conflicting_coulomb_flag_before_building(monkeypatch, tmp_path):
    import click

    from aimnet.train import export_model as export_module

    weights, model_config, sae = _write_export_inputs(tmp_path, "{}")

    monkeypatch.setattr(
        export_module,
        "strip_lr_modules_from_yaml",
        Mock(return_value=({"class": "aimnet.models.AIMNet2"}, "sr_embedded", False, None, 4.6, "exp", None)),
    )
    build_module = Mock(side_effect=AssertionError("model must not be built"))
    monkeypatch.setattr(export_module, "build_module", build_module)

    with pytest.raises(click.ClickException, match=r"--no-coulomb.*sr_embedded"):
        export_module.export_model.callback(
            str(weights),
            str(tmp_path / "export.pt"),
            str(model_config),
            str(sae),
            False,
            None,
        )

    build_module.assert_not_called()


def test_export_model_rejects_enabled_dispersion_without_complete_d3(monkeypatch, tmp_path):
    import click

    from aimnet.train import export_model as export_module

    weights, model_config, sae = _write_export_inputs(tmp_path, "{}")
    output = tmp_path / "export.pt"
    output.write_bytes(b"original artifact")
    monkeypatch.setattr(
        export_module,
        "strip_lr_modules_from_yaml",
        Mock(return_value=({"class": "aimnet.models.AIMNet2"}, "none", True, {"s8": 1.0}, None, "exp", None)),
    )
    build_module = Mock(side_effect=AssertionError("model must not be built"))
    monkeypatch.setattr(export_module, "build_module", build_module)

    with pytest.raises(click.ClickException, match=r"complete D3 parameters.*a1.*a2"):
        export_module.export_model.callback(
            str(weights),
            str(output),
            str(model_config),
            str(sae),
            None,
            None,
        )

    build_module.assert_not_called()
    assert output.read_bytes() == b"original artifact"


def test_export_model_atomic_save_preserves_existing_destination(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    from aimnet.train.export_model import _save_artifact_atomically

    destination = tmp_path / "existing.pt"
    destination.write_bytes(b"original artifact")

    def corrupt_then_fail(_artifact, target):
        target.write(b"partial artifact")
        raise OSError("simulated write failure")

    monkeypatch.setattr(torch, "save", corrupt_then_fail)

    with pytest.raises(OSError, match="simulated write failure"):
        _save_artifact_atomically({}, destination)

    assert destination.read_bytes() == b"original artifact"


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits are required")
def test_export_model_atomic_save_preserves_existing_permissions(tmp_path):
    from aimnet.train.export_model import _save_artifact_atomically

    destination = tmp_path / "existing.pt"
    destination.write_bytes(b"original artifact")
    destination.chmod(0o640)

    _save_artifact_atomically({"value": 1}, destination)

    assert stat.S_IMODE(destination.stat().st_mode) == 0o640


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits are required")
@pytest.mark.parametrize("file_umask", [0, 0o077])
def test_export_model_atomic_save_uses_private_permissions_for_new_file(tmp_path, file_umask):
    from aimnet.train.export_model import _save_artifact_atomically

    destination = tmp_path / "new.pt"

    previous_umask = os.umask(file_umask)
    try:
        _save_artifact_atomically({"value": 1}, destination)
    finally:
        os.umask(previous_umask)

    assert stat.S_IMODE(destination.stat().st_mode) == 0o600


def test_export_model_validates_canonical_artifact_before_replacing_output(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    from aimnet.train import export_model as export_module

    weights, model_config, sae = _write_export_inputs(tmp_path, "{}")
    output = tmp_path / "export.pt"
    output.write_bytes(b"original artifact")

    monkeypatch.setattr(
        export_module,
        "strip_lr_modules_from_yaml",
        Mock(return_value=({"class": "aimnet.models.AIMNet2"}, "none", False, None, None, "exp", None)),
    )
    monkeypatch.setattr(export_module, "build_module", Mock(return_value=torch.nn.Identity()))
    monkeypatch.setattr(export_module, "bake_sae_into_model", lambda model, _sae: model)
    monkeypatch.setattr(export_module, "mask_not_implemented_species", lambda model, _species: model)
    monkeypatch.setattr(export_module, "extract_cutoff", Mock(return_value=5.0))
    monkeypatch.setattr(export_module, "validate_state_dict_keys", Mock(return_value=([], [])))
    validate = Mock(side_effect=ValueError("invalid canonical artifact"))
    monkeypatch.setattr(export_module, "validate_v2_artifact_with_policy", validate)

    with pytest.raises(ValueError, match="invalid canonical artifact"):
        export_module.export_model.callback(
            str(weights),
            str(output),
            str(model_config),
            str(sae),
            None,
            None,
        )

    validate.assert_called_once()
    assert output.read_bytes() == b"original artifact"


def test_export_model_rejects_forbidden_yaml_before_construction(monkeypatch, tmp_path):
    from aimnet.train import export_model as export_module

    config = """
class: aimnet.modules.AtomicSum
fn: os.system
kwargs:
  key_in: energy
  key_out: energy
"""
    weights, model_config, sae = _write_export_inputs(tmp_path, config)
    output = tmp_path / "export.pt"
    output.write_bytes(b"original artifact")
    build_module = Mock(side_effect=AssertionError("model must not be built"))
    monkeypatch.setattr(export_module, "build_module", build_module)

    with pytest.raises(ValueError, match="forbidden"):
        export_module.export_model.callback(
            str(weights),
            str(output),
            str(model_config),
            str(sae),
            None,
            None,
        )

    build_module.assert_not_called()
    assert output.read_bytes() == b"original artifact"


def test_export_model_includes_embedded_d3ts_flag(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    from aimnet.train import export_model as export_module

    config = """
class: aimnet.modules.AtomicSum
kwargs:
  key_in: energy
  key_out: energy
  outputs:
    d3ts:
      class: custom.D3TS
"""
    core_config = {
        "class": "aimnet.modules.AtomicSum",
        "kwargs": {"key_in": "energy", "key_out": "energy"},
    }
    weights, model_config, sae = _write_export_inputs(tmp_path, config)
    output = tmp_path / "export.pt"
    monkeypatch.setattr(
        export_module,
        "strip_lr_modules_from_yaml",
        Mock(return_value=(core_config, "none", False, None, None, "exp", None)),
    )
    _patch_minimal_export_model(monkeypatch)

    export_module.export_model.callback(
        str(weights),
        str(output),
        str(model_config),
        str(sae),
        None,
        None,
    )

    artifact = torch.load(output, map_location="cpu", weights_only=True)
    assert artifact["has_embedded_d3ts"] is True
    assert artifact["has_embedded_lr"] is True


def test_export_builtin_constructor_round_trips_through_default_calculator(monkeypatch, tmp_path):
    from aimnet.calculators import AIMNet2Calculator
    from aimnet.modules import AtomicSum
    from aimnet.train import export_model as export_module

    config = """
class: aimnet.modules.AtomicSum
kwargs:
  key_in: energy
  key_out: energy
"""
    weights, model_config, sae = _write_export_inputs(tmp_path, config)
    output = tmp_path / "export.pt"
    _patch_minimal_export_model(monkeypatch)
    monkeypatch.setattr(
        export_module,
        "strip_lr_modules_from_yaml",
        Mock(
            return_value=(
                {"class": "aimnet.modules.AtomicSum", "kwargs": {"key_in": "energy", "key_out": "energy"}},
                "none",
                False,
                None,
                None,
                "exp",
                None,
            )
        ),
    )

    export_module.export_model.callback(
        str(weights),
        str(output),
        str(model_config),
        str(sae),
        None,
        None,
    )
    calc = AIMNet2Calculator(str(output), device="cpu")

    assert isinstance(calc.model, AtomicSum)
    assert calc.external_coulomb is None
    assert calc.external_dftd3 is None


def test_export_custom_constructor_requires_explicit_import_round_trip(monkeypatch, tmp_path):
    from aimnet.calculators import AIMNet2Calculator
    from aimnet.train import export_model as export_module

    config = "class: torch.nn.Identity\n"
    weights, model_config, sae = _write_export_inputs(tmp_path, config)
    output = tmp_path / "export.pt"
    _patch_minimal_export_model(monkeypatch)
    monkeypatch.setattr(
        export_module,
        "strip_lr_modules_from_yaml",
        Mock(return_value=({"class": "torch.nn.Identity"}, "none", False, None, None, "exp", None)),
    )

    with pytest.raises(ValueError, match="Untrusted"):
        export_module.export_model.callback(
            str(weights),
            str(output),
            str(model_config),
            str(sae),
            None,
            None,
        )

    export_module.export_model.callback(
        str(weights),
        str(output),
        str(model_config),
        str(sae),
        None,
        None,
        ("torch.nn.Identity",),
    )
    with pytest.raises(ValueError, match="Untrusted"):
        AIMNet2Calculator(str(output), device="cpu")

    calc = AIMNet2Calculator(
        str(output),
        device="cpu",
        model_import_paths={"torch.nn.Identity"},
    )
    assert isinstance(calc.model, pytest.importorskip("torch").nn.Identity)


def test_train_utils_param_helpers():
    pytest.importorskip("ignite")
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


def test_mse_loss_fn_matches_torch_mse():
    torch = pytest.importorskip("torch")
    from aimnet.train.loss import mse_loss_fn

    pred = {"energy": torch.tensor([1.0, 2.0, 3.0])}
    true = {"energy": torch.tensor([1.5, 2.0, 2.0])}
    loss = mse_loss_fn(pred, true, key_pred="energy", key_true="energy")
    expected = torch.nn.functional.mse_loss(true["energy"], pred["energy"])
    assert torch.allclose(loss, expected)
