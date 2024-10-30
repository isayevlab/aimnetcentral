import os
from time import perf_counter

import ase.io
from ase.optimize import LBFGS

from aimnet.calculators import AIMNet2ASE


def torch_show_device_into():
    import torch

    print(f"Torch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA available, version {torch.version.cuda}, device: {torch.cuda.get_device_name()}")  # type: ignore
    else:
        print("CUDA not available")


torch_show_device_into()
ciffile = os.path.join(os.path.dirname(__file__), "2019828.cif")

# read the first one
atoms = ase.io.read(ciffile)

# create the calculator with default model
calc = AIMNet2ASE()
# switch to DSF Coulomb
calc.base_calc.set_lrcoulomb_method("dsf")

# attach the calculator to the atoms object
atoms.calc = calc  # type: ignore

# setup optimizer
opt = LBFGS(atoms)

# run optimization
t0 = perf_counter()
print(f"Running optimization for {len(atoms)} atoms crystal.")
opt.run(fmax=0.01, steps=5000)
t1 = perf_counter()

print(f"Total optimition steps: {opt.nsteps}")
print(f"Completed optimization in {t1 - t0:.1f} s ({(t1 - t0) / opt.nsteps:.3f} s/step)")
