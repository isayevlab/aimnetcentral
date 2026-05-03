import os
from time import perf_counter

import ase.io
import torch_sim as ts

from aimnet.calculators import AIMNet2Calculator, AIMNet2TorchSim


def torch_show_device_info():
    import torch

    print(f"Torch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA available, version {torch.version.cuda}, device: {torch.cuda.get_device_name()}")  # type: ignore
    else:
        print("CUDA not available")


torch_show_device_info()

xyzfile = os.path.join(os.path.dirname(__file__), "taxol.xyz")
atoms = ase.io.read(xyzfile, index=0)

base_calc = AIMNet2Calculator("aimnet2")
calc = AIMNet2TorchSim(base_calc)

t0 = perf_counter()
n_systems = 50
systems = [atoms.copy() for _ in range(n_systems)]

final_state = ts.optimize(
    system=systems,
    model=calc,
    optimizer=ts.Optimizer.fire,
    autobatcher=True,
    pbar=True,
)
t1 = perf_counter()
print(f"Optimized {final_state.n_systems} systems in {t1 - t0:.1f} s")
