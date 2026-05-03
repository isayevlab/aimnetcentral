# TorchSim

**Status: supported (in-tree calculator).**

[TorchSim](https://torchsim.github.io/torch-sim/) is a Torch-native atomistic simulation engine with high-level runners for static evaluation, geometry optimization, molecular dynamics, and autobatching.

!!! note "Installation"

    TorchSim requires Python 3.12+. Install AIMNet with `pip install "aimnet[torchsim]"`.

## Quick Start

```python
import ase.io
import torch_sim as ts

from aimnet.calculators import AIMNet2Calculator, AIMNet2TorchSim

atoms = ase.io.read("molecule.xyz")
base_calc = AIMNet2Calculator("aimnet2")
model = AIMNet2TorchSim(base_calc)

state = ts.static(system=atoms, model=model)
print(state[0]["potential_energy"])
```

## Charge and Multiplicity

For charged systems or NSE multiplicities, convert ASE objects to a TorchSim state with `system_extras_map` so the wrapper receives per-system values:

```python
atoms.info["charge"] = 0
atoms.info["mult"] = 2

state = ts.io.atoms_to_state(
    [atoms],
    device=model.device,
    dtype=model.dtype,
    system_extras_map={"charge": "charge", "mult": "mult"},
)
results = model(state)
```

Use `mult` for AIMNet2-NSE spin multiplicity. The wrapper also accepts a `spin` extra as a compatibility fallback.

## Periodic Systems

Periodic states are detected from TorchSim's cell and PBC tensors. Construct the wrapper with `compute_stress=True` when using NPT integrators or cell relaxation:

```python
model = AIMNet2TorchSim(base_calc, compute_stress=True)
```
