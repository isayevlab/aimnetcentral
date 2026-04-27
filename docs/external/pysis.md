# pysisyphus

**Status: supported (in-tree calculator).**

AIMNet2 ships a pysisyphus `Calculator` implementation (`AIMNet2Pysis`) for use inside pysisyphus workflows: optimisations, intrinsic reaction coordinate following, transition-state searches, NEB, growing-string, etc.

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

`aimnet2pysis` is a console script that wraps `pysis` and registers the calculator under the YAML key `aimnet`. Run your pysisyphus input file with `aimnet2pysis input.yaml` instead of `pysis input.yaml`, and select the calculator with:

```yaml
calc:
  type: aimnet
  model: aimnet2
  charge: 0
  mult: 1
```

## Recommended configurations

AIMNet2 returns analytic Hessians via `torch.func.vmap`, so configure pysisyphus to use them. The default `hessian_init: fischer` (a model Hessian) is conservative and well-suited to expensive ab-initio methods, but for cheap analytic-Hessian MLIPs it leaves performance on the table.

### Geometry optimisation

```yaml
geom:
  type: cart
  fn: input.xyz

calc:
  type: aimnet
  model: aimnet2
  charge: 0
  mult: 1

opt:
  type: rfo
  hessian_init: calc
  hessian_recalc: 5
  thresh: gau_tight
```

`hessian_recalc: 5` recomputes the analytic Hessian every 5 steps; for AIMNet2 this is cheap and improves robustness on shallow potential surfaces.

### Transition state search

```yaml
geom:
  type: cart
  fn: ts_guess.xyz

calc:
  type: aimnet
  model: aimnet2
  charge: 0
  mult: 1

tsopt:
  type: rsprfo
  hessian_init: calc
  hessian_recalc: 3
  thresh: gau_tight
```

For TS work, prefer AIMNet2-rxn (`model: aimnet2-rxn`) — it is trained on transition-state data and behaves better near saddle points than the default model.

### Validating against a finite-difference Hessian

Pysisyphus can override the calculator's analytic Hessian and use central differences instead, which is useful when you suspect the analytic path is wrong (e.g., when reporting numerical Hessians for a paper):

```yaml
calc:
  type: aimnet
  model: aimnet2
  force_num_hess: true
```

This switches `get_hessian` to a finite-difference loop driven by `get_forces`. Slower (`O(6N)` extra force calls), but verifies the analytic path numerically.

## Units

Pysisyphus uses Hartree / Bohr internally; the wrapper converts AIMNet2's native eV / Angstrom output transparently.

## Model coverage

All AIMNet2 model families are accessible by passing the model name to the constructor: wb97m-d3, b97-3c, NSE, rxn, Pd. AIMNet2-rxn is particularly relevant for reaction-path / TS work in pysisyphus -- see the linked guide.

## See also

- [Reaction paths and transition states](../advanced/reaction_paths.md)
- [Calculator API reference](../calculator.md)
- [Sella TS optimizer](sella.md) — alternative analytic-Hessian-aware TS optimizer
