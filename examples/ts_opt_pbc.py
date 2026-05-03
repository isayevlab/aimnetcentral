import os

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

ciffile = os.path.join(os.path.dirname(__file__), "2019828.cif")
atoms = ase.io.read(ciffile)

base_calc = AIMNet2Calculator("aimnet2")
calc = AIMNet2TorchSim(base_calc, compute_stress=True)

final_state = ts.optimize(
    system=[atoms, atoms.copy()],
    model=calc,
    optimizer=ts.Optimizer.fire,
)

n_steps = 1000
final_state = ts.integrate(
    system=final_state,
    model=calc,
    integrator=ts.Integrator.nvt_vrescale,
    n_steps=n_steps,
    temperature=300,
    timestep=0.0005,  # ps, equivalent to 0.5 fs
    pbar=True,
)
