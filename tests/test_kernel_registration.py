"""CPU smoke test: custom ops register correctly across torch versions.

Registration of op schema + fake kernel happens at import time regardless of
device (the kernels are gated to CUDA only for *execution*, and wp.init() does
not require a GPU), so this runs on a CPU CI runner and catches torch-version
drift in torch.library schema inference.
"""

import pytest
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


def test_conv_sv_generalized_schemas():
    import aimnet.kernels.conv_sv_2d_sp_wp  # noqa: F401

    fwd = str(torch.ops.aimnet.conv_sv_2d_sp_fwd.default._schema)
    bwd = str(torch.ops.aimnet.conv_sv_2d_sp_bwd.default._schema)
    bwd_bwd = str(torch.ops.aimnet.conv_sv_2d_sp_bwd_bwd.default._schema)
    assert "Tensor a, Tensor idx, Tensor g, SymInt padding_value, SymInt num_centers" in fwd
    assert "Tensor grad_output, Tensor a, Tensor idx, Tensor g, SymInt padding_value, SymInt num_centers" in bwd
    assert (
        "Tensor grad_output, Tensor grad2_a, Tensor grad2_g, Tensor a, Tensor idx, Tensor g, "
        "SymInt padding_value, SymInt num_centers"
    ) in bwd_bwd


def test_conv_sv_generalized_fake_outputs():
    from torch._subclasses.fake_tensor import FakeTensorMode

    import aimnet.kernels.conv_sv_2d_sp_wp  # noqa: F401

    with FakeTensorMode():
        a = torch.empty(4, 3, 2, device="cuda")
        idx = torch.empty(4, 5, dtype=torch.int32, device="cuda")
        g = torch.empty(4, 5, 2, 4, device="cuda")
        out = torch.ops.aimnet.conv_sv_2d_sp_fwd(a, idx, g, 4, 4)
        grad_a, grad_g = torch.ops.aimnet.conv_sv_2d_sp_bwd(out, a, idx, g, 4, 4)
        second = torch.ops.aimnet.conv_sv_2d_sp_bwd_bwd(out, grad_a, grad_g, a, idx, g, 4, 4)
    assert out.shape == (4, 3, 2, 4)
    assert grad_a.shape == a.shape
    assert grad_g.shape == g.shape
    assert [value.shape for value in second] == [out.shape, a.shape, g.shape]


def test_conv_sv_generalized_fake_rejects_num_centers():
    from torch._subclasses.fake_tensor import FakeTensorMode

    import aimnet.kernels.conv_sv_2d_sp_wp  # noqa: F401

    with FakeTensorMode():
        a = torch.empty(4, 3, 2, device="cuda")
        idx = torch.empty(4, 5, dtype=torch.int32, device="cuda")
        g = torch.empty(4, 5, 2, 4, device="cuda")
        with pytest.raises(ValueError, match="num_centers"):
            torch.ops.aimnet.conv_sv_2d_sp_fwd(a, idx, g, 4, 5)


def test_conv_sv_generalized_forward_vmap_rejected():
    from torch._subclasses.fake_tensor import FakeTensorMode

    import aimnet.kernels.conv_sv_2d_sp_wp  # noqa: F401

    with FakeTensorMode():
        a = torch.empty(2, 4, 3, 2, device="cuda")
        idx = torch.empty(4, 5, dtype=torch.int32, device="cuda")
        g = torch.empty(4, 5, 2, 4, device="cuda")
        with pytest.raises((RuntimeError, torch.AcceleratorError), match=r"(Batching rule|vmap|CUDA error)"):
            torch.func.vmap(lambda value: torch.ops.aimnet.conv_sv_2d_sp_fwd(value, idx, g, 4, 4))(a)
