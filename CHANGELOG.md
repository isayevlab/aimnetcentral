# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
- **aimnet2_cpcm**: CPCM implicit solvation (H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br, I)
