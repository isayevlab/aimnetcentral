import importlib.util

from .calculator import AIMNet2Calculator

__all__ = ["AIMNet2Calculator"]

if importlib.util.find_spec("ase") is not None:
    from .aimnet2ase import AIMNet2ASE  # noqa: F401

    __all__.append("AIMNet2ASE")

if importlib.util.find_spec("pysisyphus") is not None:
    from .aimnet2pysis import AIMNet2Pysis  # noqa: F401

    __all__.append("AIMNet2Pysis")
