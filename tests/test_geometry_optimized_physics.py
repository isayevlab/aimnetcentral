"""Public optimized dense E/F consistency across a molecular separation scan."""

import pytest
import torch

from aimnet.calculators import AIMNet2Calculator

pytestmark = [pytest.mark.gpu, pytest.mark.weights]


def _water_dimers(distance: float):
    # Coordinates are angstroms. Both rows contain two actual water monomers;
    # their different orientations and O--O separations avoid duplicate inputs.
    water = torch.tensor([[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.2399872, 0.927297, 0.0]], device="cuda")
    rotation = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.8, -0.6], [0.0, 0.6, 0.8]], device="cuda")
    # The first O--O pair is exactly `distance`, including the 5 A cutoff scan.
    first = torch.cat((water, water + torch.tensor([distance, 0.0, 0.0], device="cuda")))
    second = torch.cat((
        water @ rotation,
        water @ rotation.T + torch.tensor([distance + 0.31, -0.2, 0.15], device="cuda"),
    ))
    return {
        "coord": torch.stack((first, second)),
        "numbers": torch.tensor([[8, 1, 1, 8, 1, 1], [8, 1, 1, 8, 1, 1]], device="cuda"),
        "charge": torch.zeros(2, device="cuda"),
    }


def _snapshot(output):
    # Replay-owned tensors may be overwritten by the following calculator call.
    return {key: output[key].detach().clone() for key in ("energy", "forces", "charges")}


def _compare(actual, expected):
    for key, atol in (("energy", 1e-3), ("forces", 1e-4), ("charges", 1e-5)):
        torch.testing.assert_close(actual[key], expected[key], rtol=1e-5 if key == "forces" else 0.0, atol=atol)


def test_optimized_public_cutoff_scan_finite_difference_and_inference_first_call():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    regular = AIMNet2Calculator("aimnet2-2025", cuda_graph=False, inference_kernels=False)
    graph = AIMNet2Calculator("aimnet2-2025", cuda_graph=True, inference_kernels=True)
    try:
        initial = _water_dimers(3.25)
        # Host constants and cached buffers created by an inference-only first
        # call must remain usable by the subsequent differentiable force call.
        with torch.inference_mode():
            for _ in range(3):
                initial_energy = graph(initial, forces=False)["energy"].clone()
        for _ in range(3):
            initial_forces = _snapshot(graph(initial, forces=True))
        torch.testing.assert_close(initial_forces["energy"], initial_energy, rtol=0.0, atol=1e-3)
        _compare(initial_forces, regular(initial, forces=True))
        # Energy and force graphs have separate cache keys. This must cover a
        # genuine force replay, not merely an existing energy-only cache entry.
        assert graph._cuda_graph.memory_stats()["engines"] >= 2
        assert any(getattr(entry.runner, "f", None) is not None for entry in graph._cuda_graph._engines.values())

        force_at_fd = {}
        for distance in (2.5, 3.5, 4.999, 5.0, 5.001, 6.0, 8.0):
            data = _water_dimers(distance)
            expected = _snapshot(regular(data, forces=True))
            actual = _snapshot(graph(data, forces=True))
            _compare(actual, expected)
            assert torch.isfinite(actual["energy"]).all()
            assert torch.isfinite(actual["forces"]).all()
            assert torch.isfinite(actual["charges"]).all()
            torch.testing.assert_close(actual["charges"].sum(-1), data["charge"], rtol=0.0, atol=1e-5)
            torch.testing.assert_close(actual["forces"].sum(-2), torch.zeros(2, 3, device="cuda"), rtol=0.0, atol=1e-4)
            if distance == 3.5:
                force_at_fd = {"regular": expected["forces"], "graph": actual["forces"]}

        # A rigid displacement of atoms 3:6 has dE/dd = -sum(F_x[3:6]).
        # h=1e-3 A balances truncation error against FP32 neural-energy noise;
        # 1e-2 eV/A is deliberately looser than analytic-path parity above.
        # Check both paths so finite-difference failure is not attributed to an
        # optimized kernel when it is already present in the reference model.
        step = 1e-3
        for name, calculator in (("regular", regular), ("graph", graph)):
            energies = []
            for sign in (-1, 1):
                data = _water_dimers(3.5)
                data["coord"][:, 3:, 0].add_(sign * step)
                energies.append(calculator(data, forces=False)["energy"].detach().double().clone())
            finite_difference = -(energies[1] - energies[0]) / (2.0 * step)
            analytical = force_at_fd[name][:, 3:, 0].double().sum(-1)
            torch.testing.assert_close(finite_difference, analytical, rtol=0.0, atol=1e-2)
    finally:
        graph._cuda_graph.clear()
