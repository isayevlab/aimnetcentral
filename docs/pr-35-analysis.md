# Deep Analysis: PR #35 — Integrate nvalchemi-toolkit-ops and new model format

**Author:** Roman Zubatyuk (@zubatyuk)  
**Scope:** +11,272 / -1,498 lines across 51 files  
**Branch:** `feature/alchemi-nblist-external-lr` → `main`

---

## Executive Summary

This PR is a major architectural overhaul of AIMNet2 that:

1. **Replaces the custom Numba neighbor list** with `nvalchemi-toolkit-ops` (GPU-accelerated neighbor lists and DFT-D3)
2. **Introduces NVIDIA Warp kernels** for `conv_sv_2d_sp` sparse convolutions (~5x speedup potential)
3. **Adds a new model format** (`.pt` with embedded YAML) replacing legacy JIT `.jpt`
4. **Externalizes long-range modules** (LRCoulomb, DFTD3) for runtime configuration
5. **Enables batched PBC** with stress tensor support for cell optimization and NPT

---

## 1. Architecture Changes

### 1.1 Neighbor List: Numba → nvalchemiops

| Aspect | Before | After |
|--------|--------|-------|
| Implementation | `nbmat.py` + `nb_kernel_cpu.py` + `nb_kernel_cuda.py` (Numba) | `nvalchemiops.neighborlist.neighbor_list` |
| Overflow handling | `TooManyNeighborsError` + manual `max_density` scaling | `NeighborOverflowError` + `AdaptiveNeighborList` with hysteresis |
| PBC support | Single-cell only | Batched cells via `cell (B, 3, 3)`, `mol_idx` for atom→cell mapping |
| Buffer sizing | Fixed `calc_max_nb(cutoff, density)` | Dynamic ~75% utilization, 16-aligned |

**Removed files:** `nbmat.py`, `nb_kernel_cpu.py`, `nb_kernel_cuda.py` (659 lines total)

**New:** `AdaptiveNeighborList` in `calculator.py` wraps `nvalchemiops` with:
- `_round_to_16()` for memory alignment
- Retry loop on `NeighborOverflowError` (1.5x buffer increase)
- Shrink when utilization < 2/3 of target (hysteresis to avoid thrashing)

### 1.2 Warp Kernels for conv_sv_2d_sp

**New:** `aimnet/kernels/conv_sv_2d_sp_wp.py` (478 lines)

- Custom CUDA kernels via NVIDIA Warp for sparse 2D convolution
- PyTorch custom ops: `aimnet::conv_sv_2d_sp_fwd`, `_bwd`, `_bwd_bwd`
- **Critical fix:** Kernel dimensions use `B-1` to exclude padding row (gradient correctness)
- CPU fallback for `d2features` when Warp unavailable
- `weights_only=True` in `torch.load` for security (model loading)

### 1.3 Model Format: .jpt → .pt

| Format | Structure | Loading |
|--------|-----------|---------|
| Legacy `.jpt` | JIT-compiled TorchScript | `torch.jit.load()` |
| New `.pt` | `{state_dict, config_yaml, metadata}` | `load_model()` auto-detection |

**New modules:**
- `aimnet/models/utils.py` — `load_model()`, `ModelMetadata` TypedDict, inspection helpers
- `aimnet/models/convert.py` — JIT → v2 conversion
- `aimnet/train/export_model.py` — Export trained models

**Registry:** All model URLs updated to `aimnet2v2/` path. New CPCM models added (`wb97m_cpcms_v2_0`–`7`).

---

## 2. External Long-Range Modules

### 2.1 Decoupling

- **LRCoulomb** and **DFTD3** moved from embedded model components to attachable calculator modules
- Metadata drives attachment: `needs_coulomb`, `needs_dispersion`, `d3_params`, `coulomb_mode`
- `set_lrcoulomb_method()` now updates external module; legacy embedded Coulomb gets a warning

### 2.2 Separate Neighbor Lists

When Coulomb and DFTD3 cutoffs differ by >20%, separate `AdaptiveNeighborList` instances:
- `_nblist_dftd3`, `_nblist_coulomb` (or shared `_nblist_lr`)
- Data keys: `nbmat_lr`, `nbmat_coulomb`, `nbmat_dftd3`, `shifts_*`

### 2.3 DFTD3 via nvalchemiops

- `aimnet/modules/lr.py`: New `DFTD3` class wrapping `nvalchemiops.interactions.dispersion.dftd3`
- `_DFTD3Function` autograd wrapper with conditional smoothing
- `resolve_suffix()` for neighbor list lookup (supports `_dftd3`, `_coulomb`, etc.)

---

## 3. Batched PBC and Stress

### 3.1 move_coord_to_cell

Extended for batched cells:
- `cell (3, 3)` + `coord (N, 3)` or `(B, N, 3)` — existing behavior
- `cell (B, 3, 3)` + flat `coord (N_total, 3)` — requires `mol_idx` for atom→cell mapping

### 3.2 Stress Tensor

- `get_derivatives()`: Volume computation supports batched cells `(B, 3, 3)`
- `volume = torch.linalg.det(cell).abs().unsqueeze(-1).unsqueeze(-1)` for batched

### 3.3 PBC + Multiple Molecules

- Previous: `NotImplementedError` for PBC with multiple molecules
- Current: Removed; PBC with `mol_idx` and batched cells is supported

---

## 4. Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `nvalchemi-toolkit-ops` | ≥0.2 | Neighbor lists, DFT-D3 |
| `warp-lang` | ≥1.11 | CUDA kernels for conv_sv_2d_sp |
| `numba` | — | **Removed** from core deps |
| `numpy` | — | Constraint `<3.0` removed (lock shows unconstrained) |
| `ase` | 3.22.1 → 3.27.0 | Bumped in lock |

**Note:** `nvalchemi-toolkit-ops` depends on `warp-lang`; both add ~120MB+ to install. Warp has platform-specific wheels (Linux x86_64, ARM64, macOS ARM64, Windows).

---

## 5. API Changes

### 5.1 AIMNet2Calculator

- **New params:** `needs_coulomb`, `needs_dispersion`, `device`, `compile_model`, `compile_kwargs`, `train`
- **Changed:** `nb_threshold` default 320 → 120
- **New methods:** `set_lr_cutoff()`, `set_dftd3_cutoff()`
- **New properties:** `has_external_coulomb`, `has_external_dftd3`, `coulomb_method`, `coulomb_cutoff`, `dftd3_cutoff`
- **set_lrcoulomb_method:** Added `ewald_accuracy` (default 1e-8)

### 5.2 prepare_input

- Skips `make_nbmat` if `nbmat` already in data (user-provided neighbor list)
- PBC: Always flattens when `cell` present (for correct neighbor list with shifts)

### 5.3 Base Keys

- `_optional_keys` extended: `shifts_lr` added
- `_optional_keys_dtype` updated with comments

---

## 6. Test Coverage

### New / Extended Tests

- `test_conv_sv_2d_sp.py` — Warp kernel forward, backward, double-backward, shapes
- `test_dftd3.py` — DFTD3 energy, gradients, smoothing
- `test_pbc.py` — DSF/Ewald with CIF crystals, stress, torch.compile
- `test_model.py` — `load_model()`, metadata, conversion
- `test_calculator.py` — External modules, `set_lrcoulomb_method`, `set_dftd3_cutoff`
- `test_ops.py` — `resolve_suffix()`, `get_i()`, batched PBC distances

### Removed

- `test_nbmat.py` (136 lines) — Replaced by nvalchemiops-based flow

### Fixtures

- `conftest.py`: `pbc_crystal_small`, `pbc_crystal_large` from CIF
- New CIF files: `1100172.cif`, `2000054.cif`

---

## 7. Risks and Considerations

### 7.1 Breaking Changes

1. **Model registry URLs** — All point to `aimnet2v2/`; legacy `.jpt` URLs no longer used. Users with cached `.jpt` must re-download or convert.
2. **nb_threshold** — Default 320 → 120 changes when flattening kicks in; may affect performance for medium-sized systems.
3. **numba removed** — Any code depending on `aimnet.calculators.nbmat` or `nb_kernel_*` will break.

### 7.2 Compatibility

- **CPU-only:** Warp and nvalchemiops have CPU fallbacks where implemented; verify CPU path for neighbor lists and DFTD3.
- **Windows:** `warp-lang` has `win_amd64` wheel; nvalchemiops availability on Windows should be confirmed.
- **Legacy models:** `load_model()` supports both formats; embedded Coulomb/D3 models still work but `set_lrcoulomb_method` only affects external modules.

### 7.3 Security

- `weights_only=True` in `torch.load` for v2 models (good practice)
- DispParam: `ptfile` stripped during export; validation hook checks buffer values after load

### 7.4 Performance

- Warp kernels: ~5x speedup for `conv_sv_2d_sp` on GPU (per PR description)
- Adaptive neighbor list: Hysteresis may cause occasional buffer resizing in MD; 75% target balances memory vs. recomputation
- Batched PBC: More complex `move_coord_to_cell` and stress; benchmark on representative workloads

---

## 8. Documentation

- New: `docs/calculator.md`, `docs/cli.md`, `docs/getting_started.md`, `docs/long_range.md`, `docs/model_format.md`
- Updated: `docs/index.md`, `docs/api/calculators.md`, `docs/train.md`
- Removed: `docs/reference.md` (empty)
- README: Technical details on batching, AdaptiveNeighborList, buffer management

---

## 9. Recommendations for Review

1. **Verify nvalchemiops API** — Confirm `neighbor_list` and `dftd3` signatures match usage; check error handling for `NeighborOverflowError`.
2. **Warp kernel correctness** — `B-1` padding exclusion is critical; validate on edge cases (single atom, max neighbors).
3. **Model conversion** — Test `aimnet convert` and `aimnet export` on real trained models; ensure metadata round-trip.
4. **PBC + batched cells** — Test `mol_idx` mapping with multiple cells; validate stress tensor for NPT.
5. **CI/CD** — Ensure `nvalchemi-toolkit-ops` and `warp-lang` install cleanly on all supported platforms (Linux, Windows, macOS).
6. **Backward compatibility** — Run existing examples and ASE integration with both legacy and new model formats.

---

## 10. Code Review Agent Dispatch

To trigger the Claude Code Review workflow manually:

1. **Via GitHub UI:** Actions → "Claude Code Review" → "Run workflow" (after adding `workflow_dispatch`)
2. **Via gh CLI:** `gh workflow run "Claude Code Review" -f pr_number=35`
3. **Via PR comment:** Post `@claude please review this PR` to trigger the Claude workflow (if configured)
