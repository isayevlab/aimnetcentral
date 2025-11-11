import os
import sys

from time import perf_counter

import ase.io
from ase import units
from ase.md.langevin import Langevin
from ase.md import MDLogger


from aimnet.calculators import AIMNet2ASE


def torch_show_device_into():
    import torch

    print(f"Torch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA available, version {torch.version.cuda}, device: {torch.cuda.get_device_name()}")  # type: ignore
    else:
        print("CUDA not available")


torch_show_device_into()
# 59 conformations of taxol
xyzfile = os.path.join(os.path.dirname(__file__), "taxol.xyz")

# read the first one
atoms = ase.io.read(xyzfile, index=0)

# create the calculator with default model and enable/disable torch compile with cudagraphs
COMPILE_CUDAGRAPHS=False
if COMPILE_CUDAGRAPHS:
    print('running with torch_compile+cudagraphs')
else:
    print('running without torch_compile+cudagraphs')

calc = AIMNet2ASE(compile_cuda_graphs=COMPILE_CUDAGRAPHS)

# attach the calculator to the atoms object
atoms.calc = calc  # type: ignore

# do a single point calculation to trigger compile and do a warmup step
forces = atoms.get_forces()
energy = atoms.get_potential_energy()
print('energy:', energy)


# setup MD
temperature_K: float = 300
timestep: float = 1.0 * units.fs
friction: float = 0.01 / units.fs
traj_interval: int = 1000
log_interval: int   = 1000
nsteps: int = 10000

dyn = Langevin(atoms, timestep, temperature_K=temperature_K, friction=friction)
dyn.attach(
    lambda: ase.io.write("traj.xyz", atoms, append=True), interval=traj_interval
)
dyn.attach(MDLogger(dyn, atoms,  sys.stdout), interval=log_interval)


# Run the dynamics
t1 = perf_counter()
dyn.run(steps=nsteps)
t2 = perf_counter()

print(f"Completed MD in {t2 - t1:.1f} s ({(t2 - t1)*1000 / nsteps:.3f} ms/step)")



