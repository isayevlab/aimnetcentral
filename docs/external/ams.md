# SCM AMS (Amsterdam Modeling Suite)

**Status: supported upstream.**

The [Amsterdam Modeling Suite](https://www.scm.com/amsterdam-modeling-suite/)
ships first-class support for AIMNet2 through the
[`MLPotential`](https://www.scm.com/doc/MLPotential/) engine, introduced
in **AMS2024.1**. Two AIMNet2 variants are exposed as named models:

- `AIMNet2-wB97MD3`
- `AIMNet2-B973c`

Per the SCM documentation, "these are currently the only ML potential
models that support charged systems (ions), and that predict atomic
charges and dipole moments and that give IR intensities when calculating
normal modes" -- a meaningful capability gap over the other backends in
the same engine (TorchANI, M3GNet, MACE, NequIP, FAIRChem).

Element coverage: H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br, I. The AMS
integration exposes AIMNet2 for **aperiodic systems only** -- this is a
property of AMS's `MLPotential` wrapper, not of AIMNet2 itself, which
supports PBC via the in-tree `AIMNet2Calculator`.

**Model coverage in this engine**: AIMNet2-wB97MD3 and AIMNet2-B973c
only. The NSE (open-shell) and rxn (reactive) AIMNet2 model families are
not currently exposed by AMS.

## Minimal AMS input

```text
Task SinglePoint

System
  Atoms
    O 0.0 0.0 0.0
    H 0.0 0.7 0.6
    H 0.0 -0.7 0.6
  End
End

Engine MLPotential
  Model AIMNet2-wB97MD3
EndEngine
```

The `Model` keyword is sufficient -- the AIMNet2 backend is activated
implicitly when an AIMNet2-prefixed model is selected. Other backends
in the same engine (FAIRChem, M3GNet, MACE, NequIP, TorchANI) are
selected via `Backend` directly.

## Workflows

`Engine MLPotential` works with the standard AMS driver tasks:
single-point, geometry optimization, transition-state search, vibrational
analysis, molecular dynamics. AIMNet2's atomic-charge and dipole
predictions feed AMS's normal-mode IR intensity reporting.

## See also

- [SCM MLPotential general docs](https://www.scm.com/doc/MLPotential/general.html)
- [Models & backends](https://www.scm.com/doc/MLPotential/ModelsAndBackends.html)
- [SCM webinar on AIMNet2 (2024)](https://www.scm.com/news/ams-webinar-2024-aimnet2/)
