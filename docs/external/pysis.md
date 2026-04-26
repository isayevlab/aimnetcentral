# pysisyphus

AIMNet2 ships a pysisyphus `Calculator` implementation (`AIMNet2Pysis`)
for use inside pysisyphus workflows: optimisations, intrinsic reaction
coordinate following, transition-state searches, NEB, growing-string, etc.

## Install

```bash
pip install "aimnet[pysis]"
```

## Use from Python

```python
from pysisyphus.helpers import geom_loader
from aimnet.calculators import AIMNet2Pysis

geom = geom_loader("reactant.xyz")
geom.set_calculator(AIMNet2Pysis("aimnet2", charge=0, mult=1))
energy = geom.energy        # Hartree (pysisyphus convention)
forces = geom.forces        # Hartree/Bohr
```

## Use from a pysisyphus YAML run

`aimnet2pysis` is a console script that wraps `pysis` and registers the
calculator under the YAML key `aimnet`. Run your pysisyphus input file
with `aimnet2pysis input.yaml` instead of `pysis input.yaml`, and select
the calculator with:

```yaml
calc:
  type: aimnet
  model: aimnet2
  charge: 0
  mult: 1
```

## Units

Pysisyphus uses Hartree / Bohr internally; the wrapper converts AIMNet2's
native eV / Angstrom output transparently.

## See also

- [Reaction paths and transition states](../advanced/reaction_paths.md)
