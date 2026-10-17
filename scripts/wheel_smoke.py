"""Installed-wheel CPU smoke test; run with ``python -I`` outside the checkout.

This intentionally exercises the public calculator with one released .pt model,
including shipped model configuration and long-range parameter assets. Model
weights use the normal AIMNet cache/download mechanism.
"""

import importlib
import inspect
import json
import sys
from pathlib import Path

import torch

import aimnet
from aimnet.calculators import AIMNet2Calculator

package = Path(aimnet.__file__).resolve()
assert package.is_relative_to(Path(sys.prefix).resolve()), package
assert "site-packages" in package.parts, package
assert inspect.signature(AIMNet2Calculator).parameters["compile_model"].default is False

for module in (
    "aimnet.calculators.cuda_graph.cache",
    "aimnet.calculators.cuda_graph.contracts",
    "aimnet.calculators.cuda_graph.dense",
    "aimnet.calculators.cuda_graph.sparse",
    "aimnet.kernels.coulomb_wp",
):
    importlib.import_module(module)

data = {
    "coord": torch.tensor([[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.2399872, 0.92662721, 0.0]]),
    "numbers": torch.tensor([8, 1, 1]),
    "charge": torch.tensor(0.0),
}
regular = AIMNet2Calculator("aimnet2-2025", device="cpu", compile_model=False, cuda_graph=False)
automatic = AIMNet2Calculator("aimnet2-2025", device="cpu", compile_model=False)
expected = regular(data, forces=True)
actual = automatic(data, forces=True)
for key in ("energy", "forces", "charges"):
    assert torch.isfinite(actual[key]).all(), key
    torch.testing.assert_close(actual[key], expected[key], atol=0.0, rtol=0.0)
assert abs(float(actual["charges"].detach().sum())) < 1e-5

print(
    json.dumps(
        {
            "package": str(package),
            "torch": torch.__version__,
            "passed": True,
            "units": {"energy": "eV", "forces": "eV/angstrom", "charges": "e"},
            "cpu_auto_equals_regular": True,
        },
        indent=2,
    )
)
