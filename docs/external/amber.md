# AMBER (`torchani-amber`)

**Status: supported upstream with caveats.**

AIMNet2 (and Nutmeg) are supported in AMBER through the
[`torchani-amber`](https://github.com/roitberg-group/torchani-amber)
integration from the Roitberg group. Despite the `torchani-` prefix the
project explicitly handles AimNet2 and Nutmeg models in addition to
ANI-family potentials. Reference paper:
[Bridging neural network potentials and classical biomolecular simulations](https://doi.org/10.1021/acs.jpcb.5c05725).

## Mechanism

`torchani-amber` is a **compiled-in C++/Fortran integration**, not a
plugin. It hooks into AMBER's existing external-potential paths:

| Mode | AMBER mechanism | Use case |
|---|---|---|
| Full ML | `iextpot = 1` + `&extpot` (`extprog = "TORCHANI"`) | All-atom ML simulation |
| ML/MM | `ifqnt = 1` + `&qmmm` (`qm_theory = 'EXTERN'`) | QM/MM with the QM region treated by the NNP |

Both modes are pure mdin-file configuration -- no Python code at runtime.

## Install

Not pip-installable. The integration must be **built into AMBER from
source**:

1. Clone `torchani-amber` (`git clone --recurse-submodules ...`).
2. Build its bundled `torchani_sandbox` plus the C++ extensions
   (`run-cmake`).
3. Build AmberTools 25/26 from source with `CMAKE_PREFIX_PATH` pointing
   at the `torchani-amber` install. AMBER auto-links it.
4. For ML/MM with charged systems on AimNet2 and Nutmeg, AmberTools 25
   needs an additional patch (replace
   `amber_src/AmberTools/sander/qm2_extern_torchani_module.F90`).
   AmberTools 26 includes the patch.

A conda environment recipe (PyTorch, CUDA, GFortran, OpenMPI) is
provided in the upstream repo.

## User-facing input

Minimal AIMNet2 example (full ML):

```
&cntrl
    iextpot = 1,
/
&extpot
    extprog = "TORCHANI",
/
&ani
    model_type = "aimnet2",
    use_double_precision = .true.,
    use_cuda_device = .true.,
/
```

ML/MM example with charge-coupled embedding:

```
&cntrl
    ifqnt = 1,
/
&qmmm
    qm_theory = "EXTERN",
/
&ani
    model_type = "aimnet2",
    mlmm_coupling = 1,
/
```

Custom `.pt` files can be loaded by passing a full path as `model_type`,
which is where the [TorchScript-export work](../superpowers/plans/2026-04-26-torchscript-export.md)
in this repo would let users ship a self-contained AIMNet2 jit asset for
`torchani-amber` rather than relying on the upstream-bundled model.

## Caveats

- Hard rebuild of AmberTools required. Once linked, the AMBER binaries
  depend on the torchani libraries even for non-ML runs.
- Supported AMBER versions: AmberTools 25 (with patch) and 26.
- The integration is tied to the `torchani_sandbox` build; the AIMNet2
  model selection is constrained to what `torchani-amber` exposes
  upstream until a custom jit `.pt` path is provided.

## Alternatives

If a full AmberTools rebuild is not acceptable, classical "AMBER force
field plus ML region" workflows can sometimes be done via
[OpenMM + openmm-ml](openmm.md) on AMBER-format topologies (loaded via
ParmEd / OpenMM's `AmberPrmtopFile`). That route is pip-installable and
needs no AMBER source build.
