import importlib.util

from .calculator import AIMNet2Calculator

__all__ = ["AIMNet2Calculator"]

if importlib.util.find_spec("ase") is not None:
    __all__.append("AIMNet2ASE")

if importlib.util.find_spec("pysisyphus") is not None:
    __all__.append("AIMNet2Pysis")

if importlib.util.find_spec("torch_sim") is not None:
    __all__.append("AIMNet2TorchSim")


def __getattr__(name: str):
    if name == "AIMNet2ASE":
        from .aimnet2ase import AIMNet2ASE

        return AIMNet2ASE
    if name == "AIMNet2Pysis":
        from .aimnet2pysis import AIMNet2Pysis

        return AIMNet2Pysis
    if name == "AIMNet2TorchSim":
        from .aimnet2torchsim import AIMNet2TorchSim

        return AIMNet2TorchSim
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
