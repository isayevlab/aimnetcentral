# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-05-03

### Added

- Added `AIMNet2TorchSim`, an optional TorchSim `ModelInterface` wrapper for static evaluation, optimization, molecular dynamics, and autobatched workloads via the Python 3.12+ `torchsim` extra (`torch-sim-atomistic>=0.6,<0.7`).
- Added TorchSim external/API documentation and runnable `examples/ts_opt.py` and `examples/ts_opt_pbc.py` scripts.
- Added dedicated CI coverage for the Sella optional extra.
- Added TorchSim CI coverage on Python 3.13.

### Changed

- Split Sella tests out of the ASE-only CI lane.
- Updated the CodeQL security workflow to `github/codeql-action` v4.
- Clarified README installation guidance now that AIMNet's core dependencies already include the GPU-accelerated nvalchemiops package.

### Fixed

- Made ASE and PySisyphus calculator modules importable in docs builds even when optional dependencies are not installed.
- Marked the local Hugging Face metadata propagation test with the `hf` marker so the HF CI lane runs it.
- Clarified that `aimnet2-rxn` supports only net-neutral systems.
- Fixed the reaction-path Hessian example to avoid `compile_model=True`, which is incompatible with Hessian requests.
- Corrected PySisyphus unit conversion documentation from Bohr to Angstrom input conversion.
- Repaired malformed Markdown fences in the batch-processing tutorial.

### Documentation

- Expanded API docs coverage for `DataGroup`, config helpers, AEV modules, and long-range modules.
- Added a public import inventory to the API overview.
- Added `aimnet2-rxn` to the README Hugging Face repo list, docs index, and pre-trained model changelog inventory.
- Aligned CUDA wheel examples on the CUDA 12.6 PyTorch index.
- Renamed the molecular dynamics NPT section to match ASE `NPT` rather than Berendsen.

## [0.1.1] - 2026-04-05

### Breaking Changes

- Minimum PyTorch version raised from 2.4 to **2.8**
- Minimum `nvalchemi-toolkit-ops` version raised from 0.2 to **0.3**
- Creating new TorchScript modules via `torch.jit.script()` is **no longer supported**; loading legacy `.jpt` files remains fully functional

### Changed

- Modernized nvalchemiops import paths for v0.3 API (`nvalchemiops.torch.neighbors`, `nvalchemiops.torch.interactions.dispersion`)
- Replaced deprecated `torch.inverse()` with `torch.linalg.inv()`
- Replaced `.transpose(-1, -2)` with `.mT` for matrix transpose operations
- Made `torch.jit.optimized_execution()` conditional on `ScriptModule` (preserves legacy `.jpt` inference, no-op for eager/compiled models)
- Removed `torch.jit.is_scripting()` guards from neighbor mask computation and DFTD3 force calculation
- Relaxed ASE dependency from `==3.27.0` to `>=3.27.0,<4`
- Bumped `codecov/codecov-action` from v5 to v6 in CI

### Fixed

- Corrected AIMNet2-Pd DFT reference from wB97M-D3/CPCM to **B97-3c/CPCM** (THF) in documentation
- Model loading now uses `weights_only=True` by default, falling back to full deserialization only for legacy `.jpt` TorchScript archives
- Model download validates HTTP response status before writing to disk

### Documentation

- Modernized README with prominent install instructions (pip, uv, conda/mamba) and `nvalchemi-toolkit-ops[torch]` install guidance
- Updated TorchScript compatibility notes across documentation and docstrings

## [0.1.0] - 2026-02-04

Initial public wheel of AIMNet2.

### Core Features

- `AIMNet2Calculator` with automatic dense/sparse mode selection based on system size
- ASE integration via `AIMNet2ASE` calculator for molecular dynamics and optimization
- PySisyphus integration via `aimnet2pysis` CLI for reaction path calculations
- Periodic boundary conditions with full stress tensor support

### Long-Range Interactions

- DFT-D3 dispersion corrections with BJ damping
- Long-range Coulomb methods: Simple cutoff, DSF (Damped-Shifted Force), Ewald summation
- Configurable cutoffs and accuracy parameters

### Performance

- GPU acceleration with NVIDIA Warp kernels for `conv_sv_2d_sp` operations
- Adaptive neighbor lists from `nvalchemi-toolkit-ops` for efficient large-system calculations
- Automatic dense (O(N^2)) / sparse (O(N)) mode switching

### Training & Model Export

- CLI tools: `aimnet train`, `aimnet export`
- New `.pt` model format with embedded YAML config and metadata
- Model conversion utilities for legacy `.jpt` format

### Pre-trained Models

- **aimnet2**: wB97M-D3 default model (H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br, I)
- **aimnet2_b973c**: B97-3c functional (H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br, I)
- **aimnet2_2025**: B97-3c with improved intermolecular interactions (H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br, I)
- **aimnet2nse**: Open-shell chemistry (H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br, I)
- **aimnet2pd**: Palladium-containing complexes (H, B, C, N, O, F, Si, P, S, Cl, Se, Br, Pd, I)
- **aimnet2-rxn**: Reactive chemistry, transition states, and IRC paths (H, C, N, O)
