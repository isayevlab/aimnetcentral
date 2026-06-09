# GPU Validation (torch / warp-lang / nvalchemiops coupling)

The CPU CI matrix verifies torch-core API and CPU numerics across PyTorch 2.8–2.12, but it cannot validate three GPU-only concerns: that `warp-lang` and `nvalchemiops` install coherently with each torch version, that the Warp custom op kernels execute, and that full-model energies/forces stay reproducible across the range. This manual tool covers them on a CUDA machine.

## Running it

```bash
make gpu-validate
```

This loops over PyTorch 2.8, 2.9, 2.10, 2.11, 2.12. For each it creates a fresh virtualenv, installs a **resolver-coherent** environment (the matching CUDA torch wheel and its `triton`, plus `aimnet` + `warp-lang` + `nvalchemiops`), runs `pytest -m gpu`, and computes deterministic energies/forces for a fixed set of systems (water, methane, caffeine, a periodic spiro crystal). A same-run torch 2.9 baseline is then used to diff every other version.

Inspect the commands without running anything:

```bash
DRY_RUN=1 bash scripts/gpu_validate.sh
```

## Configuration

| Variable | Default | Notes |
| --- | --- | --- |
| `CUDA_INDEX` | `https://download.pytorch.org/whl/cu126` | Pick the channel matching your driver. torch 2.12 defaults to CUDA 13; if a `cu126` wheel is unavailable for a version, that leg reports `INSTALL-FAIL` — re-run that version with the appropriate `--index-url`. |
| `PYTHON` | `python3.12` | aimnet itself supports Python ≥ 3.11, but validation defaults to 3.12 because `nvalchemiops`'s torch extra activates only at Python ≥ 3.12 — i.e. 3.12+ is where the coupling under test actually engages. |
| `TORCH_VERSIONS` | `2.8 2.9 2.10 2.11 2.12` | Space-separated minor versions. |
| `BASELINE` | `2.9` | torch version used as the energy/force reference. |
| `ENERGY_ATOL` | `1e-5` | Hartree. Matches the repo's `ENERGY_ATOL`. |
| `FORCE_ATOL` | `1e-4` | Hartree/Å. Looser than `FORCE_ATOL=1e-5` to absorb legitimate cross-version GPU reduction-order variation; tighten if your hardware is stable. |

## Reading the matrix

Each row is one torch version with `install`, `gpu-suite`, `maxDE`, `maxDF`, and a verdict:

- `PASS` — installs, suite green, energies/forces within tolerance of baseline.
- `BASELINE` — the reference version.
- `INSTALL-FAIL` — the coherent env did not install (often the CUDA channel; sometimes a real warp/nvalchemiops-vs-torch break — the signal this tool exists to find).
- `SUITE-FAIL` — `pytest -m gpu` failed.
- `DRIFT` — energies or forces diverged beyond tolerance.
- `NO-RESULTS` — installed but the observables dump did not produce output.

A nonzero exit code means at least one leg failed. Paste the matrix into the release or PR note as the GPU-side evidence for the supported range. The real `aimnet2` model downloads from GCS on first use, so the box needs network (or a pre-populated `~/.cache/aimnet`).
