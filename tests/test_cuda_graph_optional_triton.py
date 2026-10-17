"""An unavailable optional Triton module must retain genuine CUDA-graph replay."""

import builtins

import pytest
import torch
from conftest import CAFFEINE_FILE, load_mol

from aimnet.calculators import AIMNet2Calculator

pytestmark = [pytest.mark.gpu, pytest.mark.weights]

if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)


@pytest.mark.parametrize("model", ["aimnet2", "aimnet2-2025"])
@pytest.mark.parametrize("forces", [False, True], ids=["energy", "energy_forces"])
def test_missing_triton_keeps_dense_graph_replay(monkeypatch, model, forces):
    """Exercise the real ImportError branch, not merely ``fused_aev=False``."""
    graph = AIMNet2Calculator(model, device="cuda", compile_model=False, cuda_graph=True)
    regular = AIMNet2Calculator(model, device="cuda", compile_model=False, cuda_graph=False)
    molecule = load_mol(CAFFEINE_FILE)
    coord = torch.as_tensor(molecule["coord"], dtype=torch.float32, device="cuda")
    data = {
        "coord": coord,
        "numbers": torch.as_tensor(molecule["numbers"], dtype=torch.int64, device="cuda"),
        "charge": torch.zeros((), device="cuda"),
    }
    assert coord.shape[0] <= graph.nb_threshold
    assert graph._cuda_graph is not None
    original_import = builtins.__import__
    blocked_imports = []

    def without_optional_triton(name, *args, **kwargs):
        if name == "aimnet.kernels.aev_sv_dense_triton":
            blocked_imports.append(name)
            raise ImportError("optional Triton kernel deliberately unavailable in this test")
        return original_import(name, *args, **kwargs)

    try:
        with monkeypatch.context() as patch:
            # Intercept this exact import even if another test has already
            # cached the module. All unrelated imports retain their behavior.
            patch.setattr(builtins, "__import__", without_optional_triton)
            for _ in range(3):
                graph(data, forces=forces)
            assert blocked_imports, "the optional import failure must actually be exercised"
            (entry,) = graph._cuda_graph._engines.values()
            assert entry.runner.engine._aev_kernel is None
            original_launch = entry.runner.launch
            replays = []

            def record_launch(x):
                replays.append(True)
                return original_launch(x)

            patch.setattr(entry.runner, "launch", record_launch)
            perturbation = 0.01 * torch.sin(torch.arange(coord.numel(), device="cuda")).reshape_as(coord)
            for step in range(1, 4):
                moved = {**data, "coord": coord + step * perturbation}
                actual = graph(moved, forces=forces)
                expected = regular(moved, forces=forces)
                assert actual.keys() == expected.keys()
                for key in expected:
                    assert actual[key].shape == expected[key].shape, key
                    assert actual[key].dtype == expected[key].dtype, key
                    assert actual[key].device == expected[key].device, key
                    atol, rtol = {"energy": (1e-3, 0.0), "charges": (1e-5, 0.0)}.get(key, (1e-4, 1e-5))
                    torch.testing.assert_close(actual[key], expected[key], atol=atol, rtol=rtol)
                assert len(graph._cuda_graph._engines) == 1
                assert next(iter(graph._cuda_graph._engines.values())) is entry
                assert entry.runner.engine._aev_kernel is None
            assert len(replays) == 3, "parity must come from replay, not silently using regular evaluation"
    finally:
        graph._cuda_graph.clear()
