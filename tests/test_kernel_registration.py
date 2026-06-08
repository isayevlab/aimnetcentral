"""CPU smoke test: custom ops register correctly across torch versions.

Registration of op schema + fake kernel happens at import time regardless of
device (the kernels are gated to CUDA only for *execution*, and wp.init() does
not require a GPU), so this runs on a CPU CI runner and catches torch-version
drift in torch.library schema inference.
"""
import torch

EXPECTED_OPS = {
    "aimnet::conv_sv_2d_sp_fwd",
    "aimnet::conv_sv_2d_sp_bwd",
    "aimnet::conv_sv_2d_sp_bwd_bwd",
}


def test_torch_version_is_supported():
    major, minor = (int(x) for x in torch.__version__.split(".")[:2])
    assert (major, minor) >= (2, 8), f"torch {torch.__version__} below supported floor 2.8"


def test_custom_ops_register_on_import():
    from aimnet.kernels import load_ops

    # set-equality, not ordered list: load_ops() ordering is an implementation
    # detail, but full membership must hold (do not weaken to a length check).
    assert set(load_ops()) == EXPECTED_OPS, f"unexpected registered ops: {load_ops()}"


def test_ops_namespace_present():
    import aimnet.kernels  # noqa: F401  triggers registration

    assert hasattr(torch.ops, "aimnet")
    for name in ("conv_sv_2d_sp_fwd", "conv_sv_2d_sp_bwd", "conv_sv_2d_sp_bwd_bwd"):
        assert hasattr(torch.ops.aimnet, name), f"missing op {name}"
