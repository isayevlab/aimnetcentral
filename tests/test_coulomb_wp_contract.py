"""CUDA-only Coulomb kernel rejects invalid inputs before any Warp launch.

CUDA-shaped cases use PyTorch FakeTensors: this module runs on CPU-only hosts
without allocating GPU memory or compiling/launching Warp kernels.
"""

import pytest
import torch
from torch._subclasses import fake_tensor
from torch._subclasses.fake_tensor import FakeTensorMode

from aimnet.kernels import coulomb_wp


@pytest.fixture(autouse=True)
def never_launch_warp(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid input reached a CUDA stream or Warp launch")

    monkeypatch.setattr(coulomb_wp, "_get_stream", forbidden)
    monkeypatch.setattr(coulomb_wp.wp, "launch", forbidden)
    # PyTorch 2.11 initializes a real device context while constructing CUDA
    # FakeTensors if CUDA is available. These tests only inspect metadata and
    # reject before execution, so suppress that unnecessary allocation. This
    # also lets cuda:1 metadata be tested on a physical single-GPU machine.
    monkeypatch.setattr(fake_tensor, "init_gpu_context", lambda device: None)


def test_cpu_tensors_are_rejected():
    before = torch.cuda.is_initialized()
    with pytest.raises(ValueError, match="same CUDA device"):
        coulomb_wp.full_coulomb_rows(torch.zeros(3, 3), torch.zeros(3), 3, 3)
    assert torch.cuda.is_initialized() == before


@pytest.mark.parametrize(
    "coord_shape,charge_shape",
    [
        ((3, 2), (3,)),
        ((3, 3), (3, 1)),
        ((3, 3), (2,)),
        ((9,), (3,)),
        ((2, 3, 3), (6,)),
    ],
)
def test_inconsistent_shapes_are_rejected_without_cuda(coord_shape, charge_shape):
    before = torch.cuda.is_initialized()
    with FakeTensorMode(), pytest.raises(ValueError, match=r"coord\[T, 3\] and charges\[T\]"):
        coord = torch.zeros(coord_shape, device="cuda:0")
        charge = torch.zeros(charge_shape, device="cuda:0")
        coulomb_wp.full_coulomb_rows(coord, charge, 3, 3)
    assert torch.cuda.is_initialized() == before


@pytest.mark.parametrize("n_mol,n_real", [(0, 3), (-1, 3), (3, 0), (3, -1), (3, 6), (2, 3), (3.0, 3), (3, 3.0)])
def test_invalid_atom_counts_are_rejected_without_cuda(n_mol, n_real):
    before = torch.cuda.is_initialized()
    with FakeTensorMode(), pytest.raises(ValueError, match="divide n_real <= T"):
        coord, charge = torch.zeros((3, 3), device="cuda:0"), torch.zeros(3, device="cuda:0")
        coulomb_wp.full_coulomb_rows(coord, charge, n_mol, n_real)
    assert torch.cuda.is_initialized() == before


@pytest.mark.parametrize(
    "coord_dtype,charge_dtype",
    [
        (torch.float64, torch.float32),
        (torch.float32, torch.float64),
        (torch.float16, torch.float16),
    ],
)
def test_unsupported_precision_is_rejected_without_cuda(coord_dtype, charge_dtype):
    before = torch.cuda.is_initialized()
    with FakeTensorMode(), pytest.raises(TypeError, match="requires float32"):
        coord = torch.zeros((3, 3), device="cuda:0", dtype=coord_dtype)
        charge = torch.zeros(3, device="cuda:0", dtype=charge_dtype)
        coulomb_wp.full_coulomb_rows(coord, charge, 3, 3)
    assert torch.cuda.is_initialized() == before


@pytest.mark.parametrize("charge_device", ["cpu", "cuda:1"])
def test_mixed_devices_are_rejected_without_cuda(charge_device):
    before = torch.cuda.is_initialized()
    with FakeTensorMode(), pytest.raises(ValueError, match="same CUDA device"):
        coord = torch.zeros((3, 3), device="cuda:0")
        charge = torch.zeros(3, device=charge_device)
        coulomb_wp.full_coulomb_rows(coord, charge, 3, 3)
    assert torch.cuda.is_initialized() == before
