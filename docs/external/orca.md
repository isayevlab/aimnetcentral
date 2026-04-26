# ORCA (`!ExtOpt`)

**Status: supported upstream.**

ORCA 6.1 added a generic external-methods interface (`!ExtOpt`) that lets
the ORCA optimizer, NEB-TS, and GOAT conformer sampler drive arbitrary
external energy/gradient providers, including neural-network potentials.
AIMNet2 is one of the explicitly named supported backends, alongside UMA
and `g-xTB`. Wrapper scripts live in the upstream
[ORCA-External-Tools](https://www.faccts.de/docs/orca/6.1/tutorials/workflows/extopt.html)
project.

## Install the AIMNet2 wrapper

ORCA-External-Tools ships an installer that creates an isolated venv
with the AIMNet2 dependencies:

```bash
python install.py -e aimnet2
```

To control the install location:

```bash
python install.py --venv-dir <path/to/venv> --script-dir <path/to/bin> -e aimnet2
```

The installer creates a `bin/` directory containing the
`oet_client` / `oet_server` scripts. The venv path **must not** be moved
after installation; the scripts can be moved freely.

## Standalone vs server/client

Two execution modes:

- **Standalone**: each ORCA energy/gradient call spawns a fresh Python
  process. Simple, but Python + Torch import overhead is paid every
  call -- noticeable in optimizations and especially in `!GOAT`.
- **Server/client (recommended)**: `oet_server` keeps the AIMNet2 model
  resident in memory; ORCA hits it via the lightweight `oet_client`
  per call. Once started with an AIMNet2 model, the same server can
  service multiple ORCA jobs.

Start a server in a separate shell:

```bash
oet_server aimnet2 --nthreads 4
```

Then in the ORCA input:

```text
! ExtOpt Opt PAL8

%method
  ProgExt "/full/path/to/oet_client"
end

*XYZfile 0 1 mol.xyz
```

For standalone mode, point `ProgExt` at the AIMNet2 standalone wrapper
script (also installed by `install.py`) instead of `oet_client`.

!!! warning "Closed-shell only"
    Charge / multiplicity in the `*XYZfile` line are passed through to
    the wrapper. Charged closed-shell systems work (e.g.
    `*XYZfile -1 1 mol.xyz`); open-shell multiplicities (`mult > 1`)
    are not exposed by the upstream `aimnet2` wrapper -- the wb97m-d3
    model is closed-shell only. Verify with the wrapper's own docs if
    you need radicals.

## Compatible workflows

ORCA's external-methods interface composes with:

- `!Opt` -- geometry optimization
- `!NEB-TS` -- nudged-elastic-band transition-state search
- `!GOAT` -- conformer sampling
- `!FREQ` / `!NUMFREQ` -- numerical frequencies

## Model coverage

The upstream ORCA-External-Tools `aimnet2` wrapper exposes the wb97m-d3
model only. NSE (open-shell) and rxn (reactive) AIMNet2 families are
not currently wrapped.

## See also

- [ORCA 6.1 ExtOpt tutorial](https://www.faccts.de/docs/orca/6.1/tutorials/workflows/extopt.html)
- [OPI (ORCA Python Interface) external-methods notebook](https://www.faccts.de/docs/opi/nightly/docs/contents/notebooks/extopt.html)
