# TorchSim

**Status: supported (in-tree calculator).**

[TorchSim](https://torchsim.github.io/torch-sim/) is a Torch-native atomistic simulation engine with high-level runners for static evaluation, geometry optimization, molecular dynamics, and autobatching.

!!! note "Installation"

    TorchSim requires Python 3.12+. For the ASE-based examples below, install both extras:
    `pip install "aimnet[torchsim,ase]"`.
    AIMNet itself supports Python 3.11, but the TorchSim extra is only available on Python 3.12+.

## Quick Start

```python
import ase.io
import torch_sim as ts

from aimnet.calculators import AIMNet2Calculator, AIMNet2TorchSim

atoms = ase.io.read("molecule.xyz")
base_calc = AIMNet2Calculator("aimnet2")
model = AIMNet2TorchSim(base_calc)

results = ts.static(system=atoms, model=model)
print(results[0]["potential_energy"])
print(results[0]["forces"])
```

`ts.static` returns a list of property dictionaries, one per input system. AIMNet energies are in eV and forces are in eV/Angstrom.

For energy-only static batches, construct the wrapper with `compute_forces=False`:

```python
model = AIMNet2TorchSim(base_calc, compute_forces=False)
results = ts.static(system=[atoms, atoms.copy()], model=model)
```

Geometry optimization and molecular dynamics require the default `compute_forces=True`.

## Optimization

TorchSim runners accept a single ASE `Atoms`, a list of `Atoms`, or a `SimState`. Use `autobatcher=True` for heterogeneous batches when memory is the limiting factor:

```python
systems = [atoms.copy() for _ in range(50)]

final_state = ts.optimize(
    system=systems,
    model=model,
    optimizer=ts.Optimizer.fire,
    autobatcher=True,
)

print(final_state.n_systems)
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

Use `mult` for AIMNet2-NSE spin multiplicity. The wrapper also accepts a `spin` extra as a compatibility fallback. AIMNet partial atomic charges are returned as both `charges` and TorchSim-compatible `partial_charges` output fields.

## Periodic Systems

Periodic states are detected from TorchSim's cell and PBC tensors. Construct the wrapper with `compute_stress=True` when using NPT integrators or cell relaxation:

```python
model = AIMNet2TorchSim(base_calc, compute_stress=True)

results = ts.static(system=periodic_atoms, model=model)
print(results[0]["stress"])
```

Stress requires a periodic TorchSim state with non-zero cell vectors. `ts.static` keeps the per-system batch dimension for stress, so a single periodic system returns stress with shape `(1, 3, 3)`.

## Examples

Runnable examples are available in the repository. CI smoke tests cover static evaluation and periodic stress; the optimization and dynamics scripts are intended as manual examples.

- [`examples/ts_opt.py`](https://github.com/isayevlab/aimnetcentral/blob/main/examples/ts_opt.py) optimizes a batch of molecular systems with TorchSim autobatching.
- [`examples/ts_opt_pbc.py`](https://github.com/isayevlab/aimnetcentral/blob/main/examples/ts_opt_pbc.py) runs periodic optimization and NVT dynamics with stress enabled.
