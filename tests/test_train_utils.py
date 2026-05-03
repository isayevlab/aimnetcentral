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
