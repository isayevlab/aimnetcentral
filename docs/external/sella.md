# Sella

**Status: supported (uses AIMNet2's existing ASE calculator).**

[Sella](https://github.com/zadorlab/sella) is a saddle-point optimizer for ASE that targets transition states (`order=1`) and minima (`order=0`) with partitioned rational function optimization on internal coordinates. Sella consumes any ASE `Calculator`, so AIMNet2 works through the existing `AIMNet2ASE` class. The optional extra installs Sella ≥ 2.4.0 alongside ASE.

## Install

```bash
pip install "aimnet[sella]"
```

`sella>=2.4.0` is required. The 2.4.0 release (March 2026) introduced MLIP-targeted vectorization that made Sella usable for large systems (~22x wall-clock improvement on a 50-atom benchmark, [PR #64](https://github.com/zadorlab/sella/pull/64)). Older Sella versions do not accept the `hessian_function=` callback shown below and will raise `TypeError`.

## Recommended configuration

```python
import ase.io
from sella import Sella
from aimnet.calculators import AIMNet2ASE

atoms = ase.io.read("ts_guess.xyz")
atoms.calc = AIMNet2ASE("aimnet2")

dyn = Sella(
    atoms,
    order=1,                                    # 1 = saddle, 0 = minimum
    internal=True,                              # internal coordinates (recommended)
    hessian_function=atoms.calc.get_hessian,    # analytic Hessian callback
)
dyn.run(fmax=0.01)
```

## Why pass `hessian_function`

By default Sella refines its Hessian via an iterative Davidson eigensolver that costs ~10–30 extra gradient calls per refinement (every `nsteps_per_diag=3` steps in saddle mode). Wiring `AIMNet2ASE.get_hessian` into `hessian_function=` replaces each refresh with one analytic Hessian call (O(3N) backward passes per refresh through the AIMNet2 energy graph). This pattern was validated by [Yuan et al. (Nature Comms 2024)](https://www.nature.com/articles/s41467-024-52481-5) for NewtonNet, where it cut TS optimization step counts by 2–3x.

The Hessian is computed in eV/Å² and shaped `(3N, 3N)` to match Sella's convention.

## Limitations

- Gas-phase only. The `internal=True` path assumes molecular topology; periodic TS searches are not supported by `AIMNet2ASE.get_hessian`.
- `compile_model=True` is incompatible with the Hessian path — Dynamo + double-backward through GELU hangs (`AIMNet2Calculator` raises `RuntimeError` if you combine them).
- Multi-molecule batching is not available for the Hessian; each `Sella` instance must hold one structure.

## Minima with Sella

For minima (not TS), the [Rowan optimizer benchmark](https://rowansci.com/blog/which-optimizer-should-you-use-with-nnps) (September 2025) recommends `Sella(order=0, internal=True)` as a strong plug-and-play optimizer for AIMNet2 — often converging in fewer steps than LBFGS, and without requiring an analytic Hessian.

## See also

- [ASE calculator interface](ase.md)
- [pysisyphus integration](pysis.md) — alternative for IRC, NEB, growing-string
- [Reaction paths and transition states](../advanced/reaction_paths.md)
- [Sella v2.4.0 release notes](https://github.com/zadorlab/sella/releases/tag/v2.4.0)
- [Sella JCTC 2022 paper](https://pubs.acs.org/doi/10.1021/acs.jctc.2c00395)
