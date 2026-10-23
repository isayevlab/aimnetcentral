"""Public shared-kernel replay on moving, permuted mixed-size molecular batches."""

import pytest
import torch
from conftest import CAFFEINE_FILE, TAXOL_FILE, load_mol

from aimnet.calculators import AIMNet2Calculator

pytestmark = [pytest.mark.gpu, pytest.mark.weights]


@pytest.mark.parametrize("model", ["aimnet2", "aimnet2-2025"])
def test_grouped_public_mixed_batch_parity_symmetry_and_clear(model):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    molecules = [load_mol(CAFFEINE_FILE), load_mol(TAXOL_FILE)]
    size = max(len(m["numbers"]) for m in molecules) + 2
    data = {
        "coord": torch.zeros(2, size, 3, device="cuda"),
        "numbers": torch.zeros(2, size, dtype=torch.long, device="cuda"),
        "charge": torch.tensor([0.0, 1.0], device="cuda"),
    }
    for row, molecule in enumerate(molecules):
        n = len(molecule["numbers"])
        data["coord"][row, :n] = torch.as_tensor(molecule["coord"], device="cuda")
        data["numbers"][row, :n] = torch.as_tensor(molecule["numbers"], device="cuda")
    regular = AIMNet2Calculator(model, cuda_graph=False, inference_kernels=False)
    eager = AIMNet2Calculator(model, cuda_graph=False, inference_kernels=True)
    graph = AIMNet2Calculator(model, cuda_graph=True, inference_kernels=True)

    def compare(actual, expected):
        for key, atol in (("energy", 1e-3), ("forces", 1e-4), ("charges", 1e-5)):
            torch.testing.assert_close(actual[key], expected[key], rtol=1e-5 if key == "forces" else 0.0, atol=atol)

    try:
        for step in range(4):
            data["coord"].add_(0.003 * torch.sin(torch.arange(size, device="cuda"))[None, :, None])
            if step == 3:
                data["charge"].copy_(torch.tensor([-1.0, 0.0], device="cuda"))
            ref = regular(data, forces=True)
            compare(eager(data, forces=True), ref)
            out = graph(data, forces=True)
            compare(out, ref)
            torch.testing.assert_close(out["charges"].sum(-1), data["charge"], atol=1e-5, rtol=0.0)
            assert torch.count_nonzero(out["forces"][data["numbers"] == 0]) == 0
            if step >= 1:
                assert graph._cuda_graph.memory_stats()["engines"] == 1
        # Rotate, translate and permute real and padded slots without recapture.
        permutation = torch.arange(size - 1, -1, -1, device="cuda")
        rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], device="cuda")
        transformed = {
            "coord": data["coord"][:, permutation] @ rotation + 1.75,
            "numbers": data["numbers"][:, permutation],
            "charge": data["charge"],
        }
        result = graph(transformed, forces=True)
        compare(result, regular(transformed, forces=True))
        torch.testing.assert_close(result["energy"], out["energy"], atol=1e-3, rtol=0.0)
        torch.testing.assert_close(result["charges"], out["charges"][:, permutation], atol=1e-5, rtol=0.0)
        torch.testing.assert_close(result["forces"], out["forces"][:, permutation] @ rotation, atol=1e-4, rtol=1e-5)
        assert graph._cuda_graph.memory_stats()["engines"] == 1
    finally:
        graph._cuda_graph.clear()
    assert graph._cuda_graph.memory_stats()["pool_reserved"] == 0
