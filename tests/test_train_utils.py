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


def test_mse_loss_fn_matches_torch_mse():
    torch = pytest.importorskip("torch")
    from aimnet.train.loss import mse_loss_fn

    pred = {"energy": torch.tensor([1.0, 2.0, 3.0])}
    true = {"energy": torch.tensor([1.5, 2.0, 2.0])}
    loss = mse_loss_fn(pred, true, key_pred="energy", key_true="energy")
    expected = torch.nn.functional.mse_loss(true["energy"], pred["energy"])
    assert torch.allclose(loss, expected)
