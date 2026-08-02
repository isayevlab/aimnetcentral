"""Isolated CUDA tests for mode-2 validation failures."""

import os
import subprocess
import sys

import pytest
import torch

from aimnet.calculators import AIMNet2Calculator
from aimnet.models.base import AIMNet2Base

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable"),
]


def _cuda_mode2_data() -> dict[str, torch.Tensor]:
    device = torch.device("cuda")
    return {
        "coord": torch.zeros((2, 4, 3), device=device),
        "numbers": torch.tensor([[6, 1, 1, 0], [8, 1, 1, 0]], device=device),
        "charge": torch.zeros(2, device=device),
        "nbmat": torch.tensor(
            [
                [[1, 2, 8], [0, 2, 8], [0, 1, 8], [8, 8, 8]],
                [[5, 6, 8], [4, 6, 8], [4, 5, 8], [8, 8, 8]],
            ],
            device=device,
            dtype=torch.int64,
        ),
    }


def _run_invalid_child(case: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["AIMNET_MODE2_INVALID_CASE"] = case
    return subprocess.run(  # noqa: S603
        [sys.executable, "-m", "pytest", __file__, "-k", "test_global_mode2_invalid_child", "-q"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_global_mode2_invalid_child():
    case = os.environ.get("AIMNET_MODE2_INVALID_CASE")
    if case is None:
        pytest.skip("child process only")
    data = _cuda_mode2_data()
    if case == "padded_center":
        data["nbmat"][0, -1, 0] = 0
    elif case in {"cross_batch_direct", "cross_batch_hessian_preremap"}:
        if case == "cross_batch_hessian_preremap":
            data["nbmat"][0, 0, 0] = 4
        else:
            data["nbmat"][1, 0, 0] = 0
    elif case == "shift_int32_overflow":
        data["cell"] = torch.eye(3, device="cuda").repeat(2, 1, 1)
        data["shifts"] = torch.zeros((2, 4, 3, 3), device="cuda")
        data["shifts"][0, 0, 0, 0] = float(torch.iinfo(torch.int32).max) + 1
    else:
        raise AssertionError(f"unknown invalid case: {case}")
    if case == "cross_batch_hessian_preremap":
        AIMNet2Calculator("aimnet2", nb_threshold=0, device="cuda").eval(data, hessian=True, validate_species=False)
    else:
        AIMNet2Base().prepare_input(data)
    torch.cuda.synchronize()


def test_global_mode2_cuda_invalid_input_isolated():
    result = _run_invalid_child("cross_batch_direct")
    assert result.returncode != 0
    assert "batch interval" in (result.stdout + result.stderr).lower()


def test_global_mode2_cuda_rejects_non_sentinel_padded_center():
    result = _run_invalid_child("padded_center")
    assert result.returncode != 0
    assert "padded center" in (result.stdout + result.stderr).lower()


def test_global_mode2_cuda_rejects_hessian_preremap():
    result = _run_invalid_child("cross_batch_hessian_preremap")
    assert result.returncode != 0
    assert "batch interval" in (result.stdout + result.stderr).lower()


def test_global_mode2_cuda_rejects_shift_overflow():
    result = _run_invalid_child("shift_int32_overflow")
    assert result.returncode != 0
    assert "int32" in (result.stdout + result.stderr).lower()


def test_global_mode2_valid_cuda_path_has_no_host_sync():
    data = _cuda_mode2_data()
    previous_mode = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        AIMNet2Base().prepare_input(data)
    finally:
        torch.cuda.set_sync_debug_mode(previous_mode)
