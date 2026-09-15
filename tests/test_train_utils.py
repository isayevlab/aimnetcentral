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
    # The mocked core_config below stands in for what strip_lr_modules_from_yaml
    # actually returns: a "d3ts"-keyed output entry passes through unchanged
    # (aimnet/models/utils.py's rebuild-outputs loop only special-cases
    # "lrcoulomb"/"dftd3"/"d3bj"). It must keep a D3TS entry -- using the
    # allowlisted class spelling, since the real exported model_yaml goes
    # through the artifact-validation import policy -- so this stays
    # consistent with the has_embedded_d3ts flag the export computes from the
    # original (unmocked) model_config above.
    core_config = {
        "class": "aimnet.modules.AtomicSum",
        "kwargs": {
            "key_in": "energy",
            "key_out": "energy",
            "outputs": {
                "d3ts": {
                    "class": "aimnet.modules.D3TS",
                    "kwargs": {"a1": 0.5, "a2": 3.0, "s8": 1.0},
                },
            },
        },
    }
    weights, model_config, sae = _write_export_inputs(tmp_path, config)
    output = tmp_path / "export.pt"
    monkeypatch.setattr(
        export_module,
        "strip_lr_modules_from_yaml",
        Mock(return_value=(core_config, "none", False, None, None, "exp", None)),
    )
    monkeypatch.setattr(export_module, "build_module", Mock(return_value=torch.nn.Identity()))
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


def test_default_training_config_disables_compile():
    OmegaConf = pytest.importorskip("omegaconf").OmegaConf

    config_path = Path(__file__).parents[1] / "aimnet" / "train" / "default_train.yaml"
    cfg = OmegaConf.load(config_path)
    assert cfg.trainer.compile is False


def test_compiled_training_rejects_ddp_and_cpu(monkeypatch):
    pytest.importorskip("ignite")
    torch = pytest.importorskip("torch")
    OmegaConf = pytest.importorskip("omegaconf").OmegaConf
    from aimnet.train import train as train_module

    cfg = OmegaConf.create({"trainer": {"compile": True}})
    with pytest.raises(RuntimeError, match="DDP"):
        train_module.run(0, 2, {}, cfg, None, None)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        train_module.run(0, 1, {}, cfg, None, None)


def test_compiled_stress_training_rejects_custom_evaluator():
    torch = pytest.importorskip("torch")
    OmegaConf = pytest.importorskip("omegaconf").OmegaConf
    from aimnet.train.utils import build_engine

    model = torch.nn.Linear(1, 1)
    cfg = OmegaConf.create({
        "trainer": {
            "trainer": "aimnet.train.utils.default_trainer",
            "evaluator": "custom.evaluator",
            "compile": True,
        },
        "data": {"y": ["stress"]},
    })
    with pytest.raises(
        RuntimeError,
        match=r"trainer\.compile=True with stress targets requires aimnet\.train\.utils\.default_evaluator\.",
    ):
        build_engine(model, torch.optim.SGD(model.parameters(), lr=0.1), None, None, None, cfg, [])


@pytest.mark.gpu
@pytest.mark.parametrize("mode", [0, 1, 2])
def test_compiled_aimnet2_force_training_updates_parameters_for_two_steps(mode):
    pytest.importorskip("ignite")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for compiled training")

    from aimnet.models.aimnet2 import AIMNet2
    from aimnet.modules import AtomicSum, Forces, Output
    from aimnet.train.utils import default_trainer

    def make_model():
        return Forces(
            AIMNet2(
                aev={"rc_s": 3.0, "nshifts_s": 2},
                nfeature=2,
                d2features=False,
                ncomb_v=1,
                hidden=([4], [4], [4]),
                aim_size=3,
                outputs=[
                    Output({"hidden": [2]}, n_in=3, n_out=1, key_in="aim", key_out="energy"),
                    AtomicSum("energy", "energy"),
                ],
            )
        ).cuda()

    def make_loss(observed):
        def loss_fn(pred, true):
            energy_loss = (pred["energy"] - true["energy"]).square().mean()
            forces_loss = (pred["forces"] - true["forces"]).square().mean()
            observed.append({
                **{key: pred[key].detach().clone() for key in ("energy", "forces")},
                "energy_loss": energy_loss.detach().clone(),
                "forces_loss": forces_loss.detach().clone(),
            })
            return {"loss": energy_loss + forces_loss}

        return loss_fn

    if mode == 0:
        x = {
            "coord": torch.randn(2, 4, 3),
            "numbers": torch.tensor([[1, 6, 1, 8], [1, 1, 6, 0]]),
            "charge": torch.zeros(2),
        }
    elif mode == 1:
        x = {
            "coord": torch.randn(7, 3),
            "numbers": torch.tensor([8, 1, 1, 8, 1, 1, 0]),
            "charge": torch.zeros(2),
            "nbmat": torch.tensor([[1, 2], [0, 2], [0, 1], [4, 5], [3, 5], [3, 4], [6, 6]]),
            "mol_idx": torch.tensor([0, 0, 0, 1, 1, 1, 1]),
        }
    else:
        sentinel = 8
        x = {
            "coord": torch.randn(2, 4, 3),
            "numbers": torch.tensor([[8, 1, 1, 0], [8, 1, 1, 0]]),
            "charge": torch.zeros(2),
            "nbmat": torch.tensor([
                [[1, 2], [0, 2], [0, 1], [sentinel, sentinel]],
                [[5, 6], [4, 6], [4, 5], [sentinel, sentinel]],
            ]),
        }
    batch = (x, {"energy": torch.zeros(2), "forces": torch.zeros_like(x["coord"])})
    reordered_batch = (
        {key: batch[0][key] for key in reversed(tuple(batch[0]))},
        batch[1],
    )
    eager_model = make_model()
    compiled_model = make_model()
    initial = {name: parameter.detach().clone() for name, parameter in eager_model.named_parameters()}
    compiled_model.load_state_dict(eager_model.state_dict())
    eager_optimizer = torch.optim.SGD(eager_model.parameters(), lr=0.01)
    compiled_optimizer = torch.optim.SGD(compiled_model.parameters(), lr=0.01)

    eager_outputs = []
    compiled_outputs = []
    eager_trainer = default_trainer(eager_model, eager_optimizer, make_loss(eager_outputs), device="cuda")
    compiled_trainer = default_trainer(
        compiled_model, compiled_optimizer, make_loss(compiled_outputs), device="cuda", compile_training=True
    )
    eager_trainer.run([batch, reordered_batch], max_epochs=1)
    compiled_trainer.run([batch, reordered_batch], max_epochs=1)

    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in compiled_model.parameters()
    )
    assert any(
        not torch.equal(parameter.detach(), initial[name]) for name, parameter in compiled_model.named_parameters()
    )
    for eager_output, compiled_output in zip(eager_outputs, compiled_outputs, strict=True):
        for key in ("energy", "forces", "energy_loss", "forces_loss"):
            torch.testing.assert_close(eager_output[key], compiled_output[key], rtol=1e-5, atol=1e-6)
    for (name, eager_parameter), (compiled_name, compiled_parameter) in zip(
        eager_model.named_parameters(), compiled_model.named_parameters(), strict=True
    ):
        assert name == compiled_name
        assert (eager_parameter.grad is None) == (compiled_parameter.grad is None)
        if eager_parameter.grad is not None:
            torch.testing.assert_close(eager_parameter.grad, compiled_parameter.grad, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(eager_parameter, compiled_parameter, rtol=1e-5, atol=1e-6)

    changed_keys_batch = (
        {key: value for key, value in batch[0].items() if key != "charge"} | {"extra": torch.zeros(2)},
        batch[1],
    )
    with pytest.raises(
        ValueError,
        match=r"Compiled training input keys changed after the first batch; missing \('charge',\), unexpected \('extra',\)\.",
    ):
        compiled_trainer.run([changed_keys_batch], max_epochs=1)


@pytest.mark.gpu
@pytest.mark.parametrize("mode", [0, 1, 2])
def test_compiled_stress_training_default_validator_returns_stress(monkeypatch, mode):
    pytest.importorskip("ignite")
    torch = pytest.importorskip("torch")
    OmegaConf = pytest.importorskip("omegaconf").OmegaConf
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for compiled training")

    from aimnet.models.aimnet2 import AIMNet2
    from aimnet.modules import AtomicSum, Forces, Output
    from aimnet.train import utils as train_utils
    from aimnet.train.loss import MTLoss
    from aimnet.train.utils import default_evaluator, default_trainer

    class Metrics:
        def attach(self, *_args):
            pass

    class EagerStress(Forces):
        def forward(self, data):
            coord = data[self.x].detach().requires_grad_(True)
            strain = (
                torch.eye(3, dtype=coord.dtype, device=coord.device)
                .unsqueeze(0)
                .repeat(data["cell"].shape[0] if coord.ndim == 2 else coord.shape[0], 1, 1)
                .requires_grad_(True)
            )
            if coord.ndim == 2:
                data[self.x] = torch.einsum("ni,nij->nj", coord, strain[data["mol_idx"]])
            else:
                data[self.x] = torch.einsum("bni,bij->bnj", coord, strain)
            data["cell"] = data["cell"] @ strain
            data = self.module(data)
            forces, stress = torch.autograd.grad(data[self.y].sum(), (coord, strain), create_graph=self.training)
            data[self.key_out] = -forces
            volume = torch.linalg.det(data["cell"].detach()).abs().unsqueeze(-1).unsqueeze(-1)
            data["stress"] = stress / volume
            return data

    monkeypatch.setattr(train_utils.idist, "get_local_rank", lambda: 1)

    def make_model(*, eager_stress=False):
        return (EagerStress if eager_stress else Forces)(
            AIMNet2(
                aev={"rc_s": 3.0, "nshifts_s": 2},
                nfeature=2,
                d2features=False,
                ncomb_v=1,
                hidden=([4], [4], [4]),
                aim_size=3,
                outputs=[
                    Output({"hidden": [2]}, n_in=3, n_out=1, key_in="aim", key_out="energy"),
                    AtomicSum("energy", "energy"),
                ],
            )
        ).cuda()

    def make_loss(observed):
        base = MTLoss({
            "energy": {"fn": "aimnet.train.loss.energy_loss_fn", "weight": 1.0},
            "forces": {
                "fn": "aimnet.train.loss.peratom_loss_fn",
                "weight": 1.0,
                "kwargs": {"key_pred": "forces", "key_true": "forces"},
            },
            "stress": {
                "fn": "aimnet.train.loss.mse_loss_fn",
                "weight": 1.0,
                "kwargs": {"key_pred": "stress", "key_true": "stress"},
            },
        })

        def loss_fn(pred, true):
            values = base(pred, true)
            observed.append({key: value.detach().clone() for key, value in values.items()})
            return values

        loss_fn.components = base.components
        return loss_fn

    cfg = OmegaConf.create({
        "trainer": {
            "trainer": "aimnet.train.utils.default_trainer",
            "evaluator": "aimnet.train.utils.default_evaluator",
            "compile": True,
        },
        "data": {"y": ["energy", "forces", "stress"]},
        "checkpoint": False,
    })
    if mode == 0:
        x = {
            "coord": torch.randn(2, 4, 3),
            "numbers": torch.tensor([[1, 6, 1, 8], [1, 1, 6, 0]]),
            "charge": torch.zeros(2),
            "cell": torch.eye(3).expand(2, -1, -1).clone() * 8,
        }
    elif mode == 1:
        x = {
            "coord": torch.randn(7, 3),
            "numbers": torch.tensor([8, 1, 1, 8, 1, 1, 0]),
            "charge": torch.zeros(2),
            "nbmat": torch.tensor([[1, 2], [0, 2], [0, 1], [4, 5], [3, 5], [3, 4], [6, 6]]),
            "mol_idx": torch.tensor([0, 0, 0, 1, 1, 1, 1]),
            "cell": torch.eye(3).expand(2, -1, -1).clone() * 8,
        }
    else:
        sentinel = 8
        nbmat = torch.tensor([
            [[1, 2], [0, 2], [0, 1], [sentinel, sentinel]],
            [[5, 6], [4, 6], [4, 5], [sentinel, sentinel]],
        ])
        x = {
            "coord": torch.randn(2, 4, 3),
            "numbers": torch.tensor([[8, 1, 1, 0], [8, 1, 1, 0]]),
            "charge": torch.zeros(2),
            "nbmat": nbmat,
            "cell": torch.eye(3).expand(2, -1, -1).clone() * 8,
            "pbc": torch.ones(2, 3, dtype=torch.bool),
            "shifts": torch.zeros(*nbmat.shape, 3),
        }
    batch = (
        x,
        {
            "energy": torch.zeros(2),
            "forces": torch.zeros_like(x["coord"]),
            "stress": torch.zeros(2, 3, 3),
        },
    )
    eager_model = make_model(eager_stress=True)
    compiled_model = make_model()
    initial = {name: parameter.detach().clone() for name, parameter in eager_model.named_parameters()}
    compiled_model.load_state_dict(eager_model.state_dict())
    eager_optimizer = torch.optim.SGD(eager_model.parameters(), lr=0.01)
    compiled_optimizer = torch.optim.SGD(compiled_model.parameters(), lr=0.01)
    eager_losses = []
    compiled_losses = []

    eager_trainer = default_trainer(eager_model, eager_optimizer, make_loss(eager_losses), device="cuda")
    eager_validator = default_evaluator(eager_model, device="cuda", stress=True)
    compiled_trainer, compiled_validator = train_utils.build_engine(
        compiled_model, compiled_optimizer, None, make_loss(compiled_losses), Metrics(), cfg, [batch]
    )
    eager_trainer.run([batch], max_epochs=1)
    eager_validator.run([batch])
    compiled_trainer.run([batch], max_epochs=1)

    eager_pred, _ = eager_validator.state.output
    compiled_pred, _ = compiled_validator.state.output
    assert eager_pred["stress"].shape == compiled_pred["stress"].shape == (2, 3, 3)
    assert torch.isfinite(eager_pred["stress"]).all()
    assert torch.isfinite(compiled_pred["stress"]).all()
    for eager_loss, compiled_loss in zip(eager_losses, compiled_losses, strict=True):
        for key in ("energy", "forces", "stress", "loss"):
            assert torch.isfinite(compiled_loss[key])
            torch.testing.assert_close(eager_loss[key], compiled_loss[key], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(eager_pred["stress"], compiled_pred["stress"], rtol=1e-5, atol=1e-6)
    for (name, eager_parameter), (compiled_name, compiled_parameter) in zip(
        eager_model.named_parameters(), compiled_model.named_parameters(), strict=True
    ):
        assert name == compiled_name
        assert (eager_parameter.grad is None) == (compiled_parameter.grad is None)
        if eager_parameter.grad is not None:
            assert torch.isfinite(compiled_parameter.grad).all()
            torch.testing.assert_close(eager_parameter.grad, compiled_parameter.grad, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(eager_parameter, compiled_parameter, rtol=1e-5, atol=1e-6)
    assert any(
        not torch.equal(parameter.detach(), initial[name]) for name, parameter in compiled_model.named_parameters()
    )
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in compiled_model.parameters()
        if parameter.requires_grad
    )


def test_mse_loss_fn_matches_torch_mse():
    torch = pytest.importorskip("torch")
    from aimnet.train.loss import mse_loss_fn

    pred = {"energy": torch.tensor([1.0, 2.0, 3.0])}
    true = {"energy": torch.tensor([1.5, 2.0, 2.0])}
    loss = mse_loss_fn(pred, true, key_pred="energy", key_true="energy")
    expected = torch.nn.functional.mse_loss(true["energy"], pred["energy"])
    assert torch.allclose(loss, expected)


def test_peratom_loss_preserves_unpadded_mode0_result():
    torch = pytest.importorskip("torch")
    from aimnet import nbops
    from aimnet.train.loss import peratom_loss_fn

    predicted = torch.arange(24, dtype=torch.float32).view(2, 4, 3)
    expected = predicted + 0.5
    metadata = nbops.calc_masks(nbops.set_nb_mode({"numbers": torch.tensor([[1, 6, 1, 8], [1, 1, 6, 8]])}))
    y_pred = {
        "forces": predicted,
        "_natom": metadata["_natom"],
        "_input_padded": metadata["_input_padded"],
    }
    y_true = {"forces": expected}

    assert y_pred["_natom"].shape == ()
    actual = peratom_loss_fn(y_pred, y_true, key_pred="forces", key_true="forces")
    torch.testing.assert_close(actual, torch.nn.functional.mse_loss(predicted, expected))

    padded_predicted = predicted.clone()
    padded_expected = expected.clone()
    padded_predicted[0, -1] = 0
    padded_expected[0, -1] = 0
    padded_metadata = nbops.calc_masks(
        nbops.set_nb_mode({"numbers": torch.tensor([[1, 6, 1, 0], [1, 1, 6, 8]])})
    )
    padded_actual = peratom_loss_fn(
        {
            "forces": padded_predicted,
            "_natom": padded_metadata["_natom"],
            "_input_padded": padded_metadata["_input_padded"],
        },
        {"forces": padded_expected},
        key_pred="forces",
        key_true="forces",
    )
    diff2 = (padded_predicted - padded_expected).pow(2).view(2, -1)
    padded_expected_loss = (diff2 * (padded_metadata["_natom"].unsqueeze(-1) / diff2.shape[-1])).mean()
    torch.testing.assert_close(padded_actual, padded_expected_loss)


@pytest.mark.parametrize("tail_shape", [(), (3,)])
def test_peratom_loss_packed_matches_dense_and_global_representations(tail_shape):
    torch = pytest.importorskip("torch")
    from aimnet.train.loss import peratom_loss_fn

    real = torch.tensor([1.0, 2.0, 3.0, 4.0]).view(2, 2, *([1] * len(tail_shape)))
    if tail_shape:
        real = real.expand(2, 2, *tail_shape)
    target = torch.zeros_like(real)
    expected = torch.nn.functional.mse_loss(real, target)
    dense = peratom_loss_fn(
        {
            "value": real,
            "numbers": torch.tensor([[1, 8], [6, 1]]),
            "_natom": torch.tensor(2),
            "_input_padded": torch.tensor(False),
        },
        {"value": target},
        key_pred="value",
        key_true="value",
    )
    packed = peratom_loss_fn(
        {
            "value": torch.cat([real.reshape(4, *tail_shape), torch.full((1, *tail_shape), 100.0)]),
            "mol_idx": torch.tensor([0, 0, 1, 1, 1]),
            "_natom": torch.tensor([2, 2]),
            "_input_padded": torch.tensor(True),
        },
        {"value": torch.cat([target.reshape(4, *tail_shape), target[:1, :1].reshape(1, *tail_shape)])},
        key_pred="value",
        key_true="value",
    )
    global_neighbors = peratom_loss_fn(
        {
            "value": torch.cat([real, torch.full((2, 1, *tail_shape), 100.0)], dim=1),
            "numbers": torch.tensor([[1, 8, 0], [6, 1, 0]]),
            "nbmat": torch.zeros(2, 3, 1, dtype=torch.long),
            "_natom": torch.tensor([2, 2]),
            "_input_padded": torch.tensor(True),
        },
        {
            "value": torch.cat([target, torch.zeros((2, 1, *tail_shape))], dim=1),
        },
        key_pred="value",
        key_true="value",
    )

    torch.testing.assert_close(dense, expected)
    torch.testing.assert_close(packed, expected)
    torch.testing.assert_close(global_neighbors, expected)


def test_peratom_loss_packed_variable_sizes_excludes_only_final_dummy():
    torch = pytest.importorskip("torch")
    from aimnet.train.loss import peratom_loss_fn

    predicted = torch.arange(18, dtype=torch.float32).view(6, 3)
    predicted[-1] = 1000
    target = torch.zeros_like(predicted)
    actual = peratom_loss_fn(
        {
            "forces": predicted,
            "mol_idx": torch.tensor([0, 0, 1, 1, 1, 1]),
            "_natom": torch.tensor([2, 3]),
            "_input_padded": torch.tensor(True),
        },
        {"forces": target},
        key_pred="forces",
        key_true="forces",
    )

    torch.testing.assert_close(actual, torch.nn.functional.mse_loss(predicted[:-1], target[:-1]))
