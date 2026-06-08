"""Cross-torch-version numerical determinism + TF32 behavior.

Guards the CPU-runnable numerical path changed in Phase 2 (calc_distances uses
torch.linalg.vector_norm). The full-model energy/force golden is GPU-only
(real model is network/ASE-gated) and lives in the GPU validation task.
"""

import pytest
import torch

from aimnet import nbops, ops

# Reference values captured on the locked torch (see Task 2 Step 1).
REF_DIJ_SUM = 12.056054711341858
REF_DIJ_01 = 0.9577755928039551


def _water_data():
    coord = torch.tensor([[[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]]])
    numbers = torch.tensor([[8, 1, 1]])
    data = {"coord": coord, "numbers": numbers, "charge": torch.tensor([0.0])}
    data = nbops.set_nb_mode(data)
    return nbops.calc_masks(data)


def test_calc_distances_matches_reference():
    d_ij, _ = ops.calc_distances(_water_data())
    assert d_ij.double().sum().item() == REF_DIJ_SUM
    assert d_ij[0, 0, 1].double().item() == REF_DIJ_01


def test_calc_distances_gradient_is_finite():
    data = _water_data()
    data["coord"].requires_grad_(True)
    d_ij, _ = ops.calc_distances(data)
    d_ij.sum().backward()
    assert data["coord"].grad is not None
    assert torch.isfinite(data["coord"].grad).all()


def test_enable_tf32_sets_matmul_precision():
    # aimnet.train.utils imports ignite/omegaconf (the `train` extra); skip where
    # those are absent so the determinism tests above stay collectable in the
    # core/torch-matrix selection, which installs only the dev group.
    pytest.importorskip("ignite")
    from aimnet.train.utils import enable_tf32

    enable_tf32(True)
    assert torch.backends.cuda.matmul.allow_tf32 is True
    enable_tf32(False)
    assert torch.backends.cuda.matmul.allow_tf32 is False
