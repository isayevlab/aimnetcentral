# Model Architecture

This page describes the internal architecture of AIMNet2 models for ML researchers and developers. For model selection and usage, see the [Model Selection Guide](guide.md).

## Overview

AIMNet2 is an atom-centered neural network potential. The forward pass proceeds through these stages:

1. **AEV Computation** -- Expand interatomic distances and directions into a symmetry-adapted descriptor (Atomic Environment Vectors).
2. **Multi-Pass Charge Equilibration** -- Iteratively refine per-atom charges using ConvSV convolutions and MLPs, enforcing total charge conservation.
3. **Output Pipeline** -- Map final atomic features to per-atom energies, apply self-atomic energy shifts, sum to molecular energy, and add long-range corrections (Coulomb, DFT-D3).

```
coordinates ──> AEV (AEVSV) ──> [MLP_0] ──> [MLP_1] ──> ... ──> [MLP_final]
   |                 |              |            |                     |
   |            g_sv (radial +   initial     refine               AIM vector
   |             directional)   charges      charges
   |                                                                  |
   |                                                           Output + AtomicShift
   |                                                                  |
   |                                                             AtomicSum
   |                                                                  |
   +--------- LRCoulomb (external) -----> molecular energy <--- DFTD3 (external)
```

## AEV Descriptors (AEVSV)

The Atomic Environment Vector module (`AEVSV` in `aimnet/modules/aev.py`) encodes the local geometry around each atom as a fixed-length descriptor.

### Radial Basis

Interatomic distances are expanded over a set of shifted Gaussian basis functions:

$$
g_s(r_{ij}) = \exp\bigl(-\eta\,(r_{ij} - r_s)^2\bigr) \cdot f_c(r_{ij})
$$

Default parameters:

| Parameter   | Value | Description                                        |
| ----------- | ----- | -------------------------------------------------- |
| `rc_s`      | 5.0 A | Cutoff radius                                      |
| `nshifts_s` | 16    | Number of Gaussian shifts                          |
| `rmin`      | 0.8 A | Minimum distance for shifts                        |
| `eta`       | Auto  | Width, computed as `1 / ((rc - rmin) / nshifts)^2` |

The shifts are placed at equal intervals between `rmin` and `rc_s`. A cosine cutoff envelope `f_c(r)` smoothly brings the basis functions to zero at the cutoff radius.

### Scalar + Vector Decomposition

AIMNet2 decomposes the AEV into scalar (radial) and vector (directional) components. For each neighbor `j` of atom `i`:

- **Scalar component:** The Gaussian-expanded distance weighted by the cutoff envelope. Shape: `(nshifts,)` per neighbor.
- **Vector component:** The scalar component multiplied by the unit direction vector `r_ij / |r_ij|`. Shape: `(nshifts, 3)` per neighbor.

These are concatenated into a single tensor `g_sv` of shape `(num_neighbors, nshifts, 4)` -- one scalar channel plus three vector channels.

!!! note "Why Scalar + Vector?"

    Separating scalar and vector parts allows the network to learn both distance-dependent features (radial symmetry functions) and orientation- dependent features (angular information) within a unified framework, without requiring explicit angular symmetry functions.

## ConvSV Convolution

The `ConvSV` module (`aimnet/modules/aev.py`) is the core convolution operation that combines local geometry (from the AEV) with atomic features. It appears twice in AIMNet2: once for atomic features (`conv_a`) and once for charge features (`conv_q`).

### Mechanism

Given atomic features `a` (shape `(N, nchannel)` or `(N, nchannel, nshifts)` for 2D features) and the AEV descriptor `g_sv` (shape `(N, num_neighbors, nshifts, 4)`):

1. **Gather neighbor features:** Collect features `a_j` for all neighbors of each atom using the neighbor list (`nbmat`).

2. **Contract features with geometry via einsum:**

   For 1D features:

   ```
   avf_sv[i, a, g, d] = sum_j  a_j[i, m, a] * g_sv[i, m, g, d]
   ```

   For 2D features (d2features=True):

   ```
   avf_sv[i, a, g, d] = sum_j  a_j[i, m, a, g] * g_sv[i, m, g, d]
   ```

3. **Split scalar and vector parts:**

   - `avf_s` -- Scalar part (d=0), flattened to `(nchannel * nshifts_s,)`.
   - `avf_v` -- Vector part (d=1,2,3), processed through learned linear combinations via the `agh` parameter tensor, then squared and summed over the spatial dimension to produce rotationally invariant features. Output: `(nchannel * ncomb_v,)`.

4. **Concatenate:** The scalar and vector outputs are concatenated into a single feature vector of size `nchannel * nshifts_s + nchannel * ncomb_v`.

!!! info "GPU Acceleration"

    When 2D features are used on CUDA, ConvSV dispatches to a custom Warp kernel (`conv_sv_2d_sp`) for sparse gather-and-contract, avoiding the memory overhead of materializing the full neighbor feature tensor.

## Multi-Pass Charge Equilibration

The core of AIMNet2 is an iterative charge equilibration loop. The model runs `N` MLP passes (typically 3), each refining the atomic charges while updating atomic features.

### Pass Structure

**Pass 0 (initialization):**

- Input: `ConvSV(a)` -- convolution of initial atomic embeddings with geometry.
- Output: Initial charges `q`, charge flexibility `f`, and feature update `delta_a`.
- Charges are set directly (not added to previous values).

**Passes 1 to N-2 (refinement):**

- Input: `ConvSV(a) + ConvSV(q)` -- convolution of both atomic and charge features.
- Output: Charge correction `delta_q`, updated flexibility `f`, feature update `delta_a`.
- Charges are updated as `q = q + delta_q`.

**Pass N-1 (final):**

- Input: Same as refinement passes.
- Output: AIM vector (Atomic Interaction Model) -- the final per-atom representation passed to the output pipeline.

### Charge Conservation (NSE)

After each pass, charges are redistributed to enforce total charge conservation using the NSE (Neutral Spin Equilibrated) scheme:

$$
q_i^{\text{final}} = q_i^{\text{raw}} + f_i \cdot \frac{Q_{\text{total}} - \sum_j q_j^{\text{raw}}}{\sum_j f_j + \epsilon}
$$

where:

- `q_i^raw` is the unconstrained per-atom charge from the MLP.
- `f_i` is the per-atom charge flexibility (always positive, via squaring).
- `Q_total` is the target total charge.
- The ratio redistributes the charge deficit proportionally to each atom's flexibility.

This ensures exact charge conservation at every pass without constraining the network outputs directly.

### Open-Shell Extension (num_charge_channels=2)

For the NSE model (`aimnet2nse`), charges have two channels: alpha-spin and beta-spin. The model sets `num_charge_channels=2`.

**Preprocessing:** The total charge `Q` and multiplicity `M` are converted to two-channel targets:

```
Q_alpha = 0.5 * (Q + (M - 1))
Q_beta  = 0.5 * (Q - (M - 1))
```

**During equilibration:** Both channels are equilibrated independently using the same NSE formula, maintaining conservation for each spin channel.

**Postprocessing:** The two channels are combined:

- `charges = q_alpha + q_beta` (total charge per atom)
- `spin_charges = q_alpha - q_beta` (spin density per atom)

## Output Pipeline

After the multi-pass loop produces the AIM vector, the output pipeline converts it to physical observables:

### 1. Output MLP

An `Output` module applies a final MLP to the AIM vector to produce per-atom raw energies:

```
aim_vector (per atom) --[MLP]--> per-atom energy (scalar)
```

### 2. AtomicShift (Self-Atomic Energies)

`AtomicShift` adds element-specific energy offsets (SAE values) to the per-atom energies. These are stored as a learnable `nn.Embedding` indexed by atomic number:

```
e_atom = e_raw + SAE[atomic_number]
```

The SAE values are stored in float64 precision to avoid numerical issues when computing energy differences between large molecules.

### 3. AtomicSum

`AtomicSum` sums per-atom energies within each molecule to produce molecular energies:

```
E_mol = sum_i  e_atom_i    (for atoms i in molecule)
```

### 4. Long-Range Corrections

Two external modules add long-range physics that the short-range NN cannot capture:

**LRCoulomb** -- Electrostatic energy from predicted charges. Three methods:

| Method   | Use Case                    | Neighbor List                 |
| -------- | --------------------------- | ----------------------------- |
| `simple` | Non-periodic, small systems | All pairs                     |
| `dsf`    | Periodic systems            | Finite cutoff (default 15 A)  |
| `ewald`  | Periodic, high accuracy     | Computed from accuracy target |

The Coulomb module uses the charges predicted by the equilibration loop and subtracts the short-range Coulomb component (already learned by the NN) to avoid double-counting.

**DFTD3** -- Grimme's DFT-D3 dispersion correction with BJ damping. Uses reference C6 coefficients and coordination-number-dependent interpolation. Applied with a smoothed cutoff at 15 A (default).

## Data Flow Summary

The complete data flow through the model:

```
Input: coord, numbers, charge [, mult]
  |
  v
AEV (AEVSV): coord + nbmat --> g_sv (N, nnb, nshifts, 4)
  |
  v
Embedding: numbers --> a (N, nfeature [, nshifts])
  |
  v
Pass 0: ConvSV(a, g_sv) --> MLP --> q_initial, f, delta_a
  |                                    |
  |                               NSE(Q, q, f)
  v
Pass 1: ConvSV(a, g_sv) + ConvSV(q, g_sv) --> MLP --> delta_q, f, delta_a
  |                                                      |
  |                                                 NSE(Q, q+dq, f)
  v
Pass 2 (final): ConvSV(a, g_sv) + ConvSV(q, g_sv) --> MLP --> aim_vector
  |
  v
Output MLP: aim --> e_atom_raw
  |
  v
AtomicShift: e_atom_raw + SAE[Z] --> e_atom
  |
  v
AtomicSum: sum(e_atom) --> E_mol
  |
  v
LRCoulomb(charges) + DFTD3(coord) --> E_total
```

## Glossary

| Term | Full Name | Description |
| --- | --- | --- |
| **AEV** | Atomic Environment Vector | Fixed-length descriptor encoding the local geometry around each atom via Gaussian radial basis functions and directional components. |
| **NSE** | Neutral Spin Equilibrated | Charge redistribution scheme that enforces total charge (and optionally spin) conservation by distributing the deficit proportionally to per-atom flexibility values. |
| **SAE** | Self-Atomic Energy | Element-specific energy offset (a.k.a. atomic reference energy). Stored as a learnable embedding and added to per-atom NN outputs before summation. |
| **DSF** | Damped Shifted Force | Electrostatic method for periodic systems. Applies a damping function and shift to the Coulomb potential at a finite cutoff, avoiding the need for Ewald summation. |
| **SRCoulomb** | Short-Range Coulomb | The portion of Coulomb interaction within the NN cutoff radius (5.0 A). When using external Coulomb methods (DSF, Ewald), this component is subtracted to avoid double-counting with the NN. |
| **nb_threshold** | Neighbor Threshold | Atom count threshold (default 120) that controls whether molecules are processed in dense batched mode (small systems) or sparse flattened mode (large systems). |
| **ConvSV** | Scalar-Vector Convolution | Custom convolution combining neighbor atomic features with AEV geometry descriptors via einsum, producing rotationally invariant output through vector squaring. |
| **AIM** | Atomic Interaction Model | The final per-atom feature vector produced by the last MLP pass, which encodes all information needed to predict atomic properties. |
| **D3TS** | D3 with Tkatchenko-Scheffler combination rule | Embedded dispersion variant using TS mixing rules for C6 coefficients instead of standard D3 interpolation. |
| **BJ damping** | Becke-Johnson damping | Damping function for DFT-D3 that uses element-pair-specific cutoff radii to avoid divergence at short distances. |

## Key Source Files

| File | Contents |
| --- | --- |
| `aimnet/models/aimnet2.py` | `AIMNet2` model class with multi-pass forward loop |
| `aimnet/models/base.py` | `AIMNet2Base` class, input preprocessing, `load_model()` |
| `aimnet/modules/aev.py` | `AEVSV` (AEV computation) and `ConvSV` (convolution) |
| `aimnet/modules/core.py` | `Output`, `AtomicShift`, `AtomicSum`, `MLP`, `Embedding` |
| `aimnet/modules/lr.py` | `LRCoulomb`, `SRCoulomb`, `DFTD3`, `D3TS` |
| `aimnet/ops.py` | `nse()` charge equilibration function |
| `aimnet/calculators/calculator.py` | `AIMNet2Calculator` inference wrapper |
