# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Security

- Upgraded transitive lockfile dependencies to patched versions for all open Dependabot alerts with available fixes (GitPython, Mako, Pillow, cryptography, idna, msgpack, pymdown-extensions, setuptools, tornado, urllib3). Remaining open alerts: paramiko (no patch released) and two low-severity torch advisories fixed only in torch 2.13, outside the supported 2.8-2.12 matrix.
- Verified official model-registry downloads and cached artifacts against their SHA-256 digests before loading. Registry names and aliases now take precedence over same-named implicit local paths, preventing unverified local files from shadowing registry models. Official distributions currently bundle no model artifacts and continue to download them on demand.
- Restricted v2 `.pt` artifact deserialization to weights and basic data. Legacy `.jpt` files are loaded only through the TorchScript loader and must come from a trusted source.
- Validated Python class and function references in model YAML against a default trusted set. Direct local and Hugging Face artifacts can extend or replace that set, or explicitly select unsafe loading for trusted custom code; registry models always use the fixed default set.
- Enriched and structurally validated complete Hugging Face metadata before weight access, including unambiguous `SRCoulomb` parameter recovery. Registry-backed HF fallback now treats digest-verified registry YAML and metadata as authoritative and rejects conflicting family metadata.
- Bumped the locked torch from 2.9.1 to 2.12.1 (with triton 3.7.1), staying inside the supported 2.8-2.12 matrix. This clears the low-severity `torch.lstm_cell` memory-corruption advisory (patched in 2.10.0; the earlier note that both torch advisories required 2.13 no longer matches the updated advisory data) and moves the default CI lane onto an inductor with the upstream fusion-legality fix (pytorch/pytorch#172301). The `torch.jit.script` advisory remains open: patched only in 2.13, still outside the supported matrix.

### Added

- Added `aimnet download <model...|--all>` to prefetch registry model weights into the local cache for offline/HPC use.
- Added `deterministic=True` calculator option: routes external DFT-D3 and DSF Coulomb through their differentiable pure-torch paths, making repeated identical evaluations bitwise reproducible on the same machine/build (issue #93). Ewald/PME kernels are not covered and warn once.
- Added weekly and manually dispatched strict fleet CI covering every official registry digest, strict-policy artifact load, and exact role-specific YAML defaults.
- Added `aimnet info` reporting package/torch/warp versions, CUDA availability, registered kernel ops, and the model cache location.
- Added a `weights` pytest marker on modules that need model weights, enabling a verified fully-offline test run (`-m "not weights ..."`) for packaging environments.

### Changed

- **BREAKING:** Model loading now routes TorchScript only for `.jpt` files, gives bare registry names precedence over same-named implicit local paths, rejects inconsistent v2 metadata and incomplete weights, requires lowercase registry SHA-256 digests, and uses exact role-specific default YAML imports instead of the broad `torch.nn.*` namespace. See the model-format migration notes for remedies.
- Split artifact checks into envelope, structural, canonical-distribution, and effective-calculator validation. Structurally valid direct and complete custom HF artifacts may explicitly disable external Coulomb or dispersion, while registry, registry-backed HF, and exported artifacts retain canonical action-flag requirements.
- Made `aimnet export` validate canonical artifacts before atomically replacing the destination, preserve existing output on failure, record embedded D3TS metadata, and accept explicit trusted custom constructor paths through `--model-import-path`.
- Marked legacy `.jpt` runtime metadata as embedded-LR while retaining format version 1 and embedded-module defaults. Consolidated v2 construction, state loading, and source-routing predicates without expanding the stable top-level models API.
- Hardened `make test`: the parallel run now hides CUDA and caps per-worker threads (`CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1`); previously xdist workers either all initialized the first GPU (OOM on CUDA boxes) or oversubscribed the CPU with per-worker torch thread pools. A new `make test-gpu` target runs the GPU-marked tests serially on CUDA.
- Bumped `nvalchemi-toolkit-ops` to `>=0.4.0` and `warp-lang` to `>=1.13,<2` (installs 1.15). Energies, charges, and Hessians are bit-identical to 0.3.1; explicit force/virial outputs shift within float32 accumulation noise (max 6.7e-05 eV/A on a periodic system, 40x inside the project's cross-version acceptance of 1e-4 Hartree/A). The 0.4.0 direct-output flags used by the Ewald/PME path (`compute_forces`/`compute_virial`) are deprecated upstream and now emit `DeprecationWarning`; migrating to the autograd-based API is tracked as follow-up work.

- Marked the expensive tail of the test suite (~60 test nodes ≥2 s each: torch.compile, model-format roundtrips, dense-Hessian/HVP comparisons, multi-model ASE runs) with the `slow` marker; the default CPU test run now completes in under two minutes. Run `pytest -m slow` for the marked tail. The cheapest representative of each critical path (dense Hessian, HVP correctness, legacy `.jpt` loading, torch.compile smoke) stays in the default set.
- Silenced the spurious "Warp CUDA error 100" stderr line emitted at import on CPU-only hosts.
- Made optional-dependency install hints installer-neutral (conda equivalents where they exist; pysisyphus named directly since it is pip-only).

### Fixed

- Excluded the local weight cache (`aimnet/calculators/assets/`) from wheels and sdists explicitly, instead of relying on hatchling's `.gitignore` handling.
- Fixed `DataGroup.cv_split` corrupting cross-validation folds: building each fold's train split mutated the shared parts in place via `cat()`, so later folds contained duplicated samples and validation splits larger than the dataset. Folds are now built without mutating the shared parts.
- Fixed a crash when torch has CUDA but warp-lang does not (possible with conda-forge variant packages): the AEV kernel gate now checks warp CUDA availability and falls back to the pure-torch path with a one-time warning.

### Removed

- Removed unused `DataGroup.to_dict`, `DataGroup.merge`, `DataGroup.rename_key`, `SizeGroupedDataset.merge`, and `SizeGroupedDataset.rename_datakey` (no callers in-repo or in downstream projects).
- Removed unused `aimnet.train.utils.make_seed` and `aimnet.ops.lazy_calc_dij_lr`.
- Removed unused `LRCoulomb.coul_ewald` and `LRCoulomb.coul_pme` convenience wrappers; use `LRCoulomb.forward` with `method="ewald"`/`"pme"`.
- Removed unused `aimnet.modules.core.DSequential` and `aimnet.constants.get_dftd3_param`.
- Removed unused `aimnet.models.utils.has_dftd3_in_config` (and its `aimnet.models` re-export).

### Added

- Added unit tests for the public `SizeGroupedDataset.save_h5`, `AIMNet2Calculator.set_lr_cutoff`, and `aimnet.train.loss.mse_loss_fn` APIs, which are kept.
- Added targeted unit tests for previously untested code: `RegMultiMetric`/`regression_stats`, the `aimnet export` helper functions (`load_sae`, `bake_sae_into_model`, `mask_not_implemented_species`), `SizeGroupedDataset.cv_split`/`concatenate`, `train.utils` config/parameter helpers, and `LRCoulomb` constructor validation.

### Documentation

- Documented the rxn dipole origin convention: `center_coord=False` is origin-safe because the family is net-neutral-only; any future charged-system family must ship `center_coord=True`.

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
