# Model Selection Guide

AIMNet2 ships with five model variants, each trained on a different DFT functional or element set. This guide helps you pick the right one.

!!! warning "What AIMNet2 Cannot Do"

    Before choosing a model, verify that your system falls within AIMNet2's domain of applicability:

    - **Unsupported elements** -- No alkali metals (Li, Na, K, ...), no alkaline
      earth metals (Be, Mg, Ca, ...), and no transition metals other than Pd. See the [element table](#supported-elements) below.
    - **Molecular training data only** -- All models were trained on isolated
      molecules and molecular clusters. They are **not** parameterized for bulk materials, extended surfaces, or metallic systems.
    - **No implicit solvation** -- Most AIMNet2 models operate in vacuum.
      Solvent effects must be modeled explicitly (e.g., by including solvent molecules) or accounted for separately. The exception is [AIMNet2-Pd](aimnet2pd.md), which includes CPCM implicit solvation for THF baked into the model.
    - **No multi-reference chemistry** -- Systems with strong static correlation
      (e.g., stretched transition metal complexes, biradicaloids with near-degenerate states) are outside the training domain. Use multi-reference methods (e.g., CASSCF, CASPT2, NEVPT2) for these cases.

## Supported Elements

All standard models cover the same 14 elements:

**H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br, I**

The Pd model replaces As with Pd:

**H, B, C, N, O, F, Si, P, S, Cl, Se, Br, Pd, I**

!!! tip "Check Elements First"

    The most common source of errors is running a system with unsupported elements. Always verify your element set before selecting a model:

    ```python
    from aimnet.calculators import AIMNet2Calculator

    calc = AIMNet2Calculator("aimnet2")
    # The calculator will raise an error at inference time
    # if unsupported elements are encountered.
    ```

## Quick-Pick Table

| Model | Alias | Functional | Best For | Key Strength |
| --- | --- | --- | --- | --- |
| AIMNet2 | `aimnet2` | wB97M-D3 | General organic chemistry | Reliable default, broad coverage |
| **AIMNet2-2025** | **`aimnet2_2025`** | **B97-3c (improved)** | **General-purpose B97-3c** | **Recommended B97-3c model; supersedes B97-3c** |
| AIMNet2-B97-3c | `aimnet2_b973c` | B97-3c | Legacy B97-3c screening | Superseded by AIMNet2-2025 |
| AIMNet2-NSE | `aimnet2nse` | wB97M-D3 | Open-shell systems | Radicals, triplet states, bond dissociation |
| AIMNet2-Pd | `aimnet2pd` | wB97M-D3/CPCM (THF) | Pd catalysis | Pd organometallics with implicit THF solvation |

## Decision Flowchart

Follow these steps to choose a model:

### Step 1: Are all your elements supported?

Check the [element table](#supported-elements) above. If your system contains elements not listed, AIMNet2 cannot be used.

### Step 2: Does your system contain palladium?

If yes, use **`aimnet2pd`**. This is the only model that supports Pd. Note that it does **not** support As (which the other models do).

### Step 3: Is your system open-shell?

Radicals, triplet states, or any system with unpaired electrons should use **`aimnet2nse`**. The NSE (Neutral Spin Equilibrated) scheme handles spin polarization through two charge channels instead of one.

!!! note "When to use NSE"

    Use `aimnet2nse` whenever you need to set `mult > 1`, or when bonds are breaking or forming (e.g., transition states, bond dissociation curves). Even for closed-shell transition states, NSE often gives more reliable energies because the NN can represent partial bond breaking.

### Step 4: Do you want a B97-3c-level model?

If you prefer a B97-3c reference level (faster DFT, suitable for screening and large-scale studies), use **`aimnet2_2025`**. This is the recommended B97-3c model — it supersedes the original `aimnet2_b973c` with improved intermolecular interaction accuracy while retaining the same intramolecular performance.

!!! note "AIMNet2-2025 supersedes AIMNet2-B97-3c" For all new work requiring a B97-3c-level model, use `aimnet2_2025`. The original `aimnet2_b973c` is retained for reproducibility of published results but is no longer the recommended choice. AIMNet2-2025 provides strictly better accuracy for non-covalent interactions with no regression for covalent chemistry.

### Step 5: General-purpose?

- **General-purpose** calculations: use **`aimnet2`** (the default). Trained on wB97M-D3, a well-validated range-separated hybrid functional.
- **Legacy B97-3c**: `aimnet2_b973c` is available for reproducing previous results but new projects should use `aimnet2_2025` instead.

## Loading Models

All models are loaded through `AIMNet2Calculator`:

```python
from aimnet.calculators import AIMNet2Calculator

# Default model (wB97M-D3, member 0)
calc = AIMNet2Calculator("aimnet2")

# B97-3c model
calc = AIMNet2Calculator("aimnet2_b973c")

# 2025 improved model
calc = AIMNet2Calculator("aimnet2_2025")

# NSE open-shell model
calc = AIMNet2Calculator("aimnet2nse")

# Palladium model
calc = AIMNet2Calculator("aimnet2pd")
```

Each alias loads ensemble member `_0` by default. To load a specific member, use the full model name:

```python
# Load member 2 of the wB97M-D3 ensemble
calc = AIMNet2Calculator("aimnet2_wb97m_d3_2")
```

## Ensemble Models and Uncertainty Estimation

Each model variant consists of an **ensemble of 4 independently trained members** (indexed `_0` through `_3`). When you load a model by its alias (e.g., `aimnet2`), you get **only member `_0`**.

### Why Use the Full Ensemble?

Running all 4 members and computing the variance of predictions gives a built-in uncertainty estimate. High variance signals that the model is less confident about the prediction, which is useful for:

- Identifying out-of-distribution structures
- Active learning workflows
- Deciding when to fall back to DFT

### Computing Ensemble Uncertainty

```python
import torch
from aimnet.calculators import AIMNet2Calculator

# Load all 4 ensemble members
calcs = [AIMNet2Calculator(f"aimnet2_wb97m_d3_{i}") for i in range(4)]

# Run inference with each member
data = {
    "coord": coords,    # (N, 3) tensor
    "numbers": numbers,  # (N,) tensor
    "charge": charge,    # (1,) tensor
}

energies = []
for calc in calcs:
    result = calc(data)
    energies.append(result["energy"])

energies = torch.stack(energies)

# Ensemble mean and variance
mean_energy = energies.mean(dim=0)
energy_variance = energies.var(dim=0)

print(f"Energy: {mean_energy.item():.6f} +/- {energy_variance.sqrt().item():.6f} eV")
```

!!! tip "Single Member for Production"

    For production calculations where uncertainty is not needed, using a single member is 4x faster. The individual members have very similar accuracy on average.

## Model Aliases Reference

The following table shows all available model names and their aliases:

| Alias | Resolves To | Ensemble Members |
| --- | --- | --- |
| `aimnet2` | `aimnet2_wb97m_d3_0` | `aimnet2_wb97m_d3_{0,1,2,3}` |
| `aimnet2_wb97m` | `aimnet2_wb97m_d3_0` | (same as above) |
| `aimnet2_b973c` | `aimnet2_b973c_d3_0` | `aimnet2_b973c_d3_{0,1,2,3}` |
| `aimnet2_2025` | `aimnet2_b973c_2025_d3_0` | `aimnet2_b973c_2025_d3_{0,1,2,3}` |
| `aimnet2nse` | `aimnet2nse_0` | `aimnet2nse_{0,1,2,3}` |
| `aimnet2pd` | `aimnet2-pd_0` | `aimnet2-pd_{0,1,2,3}` |

## Direct Model Downloads

All model files are downloaded automatically the first time you call `AIMNet2Calculator("alias")`. To download manually or inspect files, use the links below. All files are in PyTorch v2 `.pt` format.

### AIMNet2 (wB97M-D3) — alias `aimnet2` / `aimnet2_wb97m`

| Registry name | File | Download |
| --- | --- | --- |
| `aimnet2_wb97m_d3_0` | aimnet2_wb97m_d3_0.pt | [download](https://storage.googleapis.com/aimnetcentral/aimnet2v2/AIMNet2/aimnet2_wb97m_d3_0.pt) |
| `aimnet2_wb97m_d3_1` | aimnet2_wb97m_d3_1.pt | [download](https://storage.googleapis.com/aimnetcentral/aimnet2v2/AIMNet2/aimnet2_wb97m_d3_1.pt) |
| `aimnet2_wb97m_d3_2` | aimnet2_wb97m_d3_2.pt | [download](https://storage.googleapis.com/aimnetcentral/aimnet2v2/AIMNet2/aimnet2_wb97m_d3_2.pt) |
| `aimnet2_wb97m_d3_3` | aimnet2_wb97m_d3_3.pt | [download](https://storage.googleapis.com/aimnetcentral/aimnet2v2/AIMNet2/aimnet2_wb97m_d3_3.pt) |

### AIMNet2-B97-3c — alias `aimnet2_b973c`

| Registry name | File | Download |
| --- | --- | --- |
| `aimnet2_b973c_d3_0` | aimnet2_b973c_d3_0.pt | [download](https://storage.googleapis.com/aimnetcentral/aimnet2v2/AIMNet2/aimnet2_b973c_d3_0.pt) |
| `aimnet2_b973c_d3_1` | aimnet2_b973c_d3_1.pt | [download](https://storage.googleapis.com/aimnetcentral/aimnet2v2/AIMNet2/aimnet2_b973c_d3_1.pt) |
| `aimnet2_b973c_d3_2` | aimnet2_b973c_d3_2.pt | [download](https://storage.googleapis.com/aimnetcentral/aimnet2v2/AIMNet2/aimnet2_b973c_d3_2.pt) |
| `aimnet2_b973c_d3_3` | aimnet2_b973c_d3_3.pt | [download](https://storage.googleapis.com/aimnetcentral/aimnet2v2/AIMNet2/aimnet2_b973c_d3_3.pt) |

### AIMNet2-2025 — alias `aimnet2_2025`

| Registry name | File | Download |
| --- | --- | --- |
| `aimnet2_b973c_2025_d3_0` | aimnet2_2025_b973c_d3_0.pt | [download](https://storage.googleapis.com/aimnetcentral/aimnet2v2/AIMNet2/aimnet2_2025_b973c_d3_0.pt) |
| `aimnet2_b973c_2025_d3_1` | aimnet2_2025_b973c_d3_1.pt | [download](https://storage.googleapis.com/aimnetcentral/aimnet2v2/AIMNet2/aimnet2_2025_b973c_d3_1.pt) |
| `aimnet2_b973c_2025_d3_2` | aimnet2_2025_b973c_d3_2.pt | [download](https://storage.googleapis.com/aimnetcentral/aimnet2v2/AIMNet2/aimnet2_2025_b973c_d3_2.pt) |
| `aimnet2_b973c_2025_d3_3` | aimnet2_2025_b973c_d3_3.pt | [download](https://storage.googleapis.com/aimnetcentral/aimnet2v2/AIMNet2/aimnet2_2025_b973c_d3_3.pt) |

### AIMNet2-NSE — alias `aimnet2nse`

| Registry name | File | Download |
| --- | --- | --- |
| `aimnet2nse_0` | aimnet2nse_wb97m_0.pt | [download](https://storage.googleapis.com/aimnetcentral/aimnet2v2/AIMNet2NSE/aimnet2nse_wb97m_0.pt) |
| `aimnet2nse_1` | aimnet2nse_wb97m_1.pt | [download](https://storage.googleapis.com/aimnetcentral/aimnet2v2/AIMNet2NSE/aimnet2nse_wb97m_1.pt) |
| `aimnet2nse_2` | aimnet2nse_wb97m_2.pt | [download](https://storage.googleapis.com/aimnetcentral/aimnet2v2/AIMNet2NSE/aimnet2nse_wb97m_2.pt) |
| `aimnet2nse_3` | aimnet2nse_wb97m_3.pt | [download](https://storage.googleapis.com/aimnetcentral/aimnet2v2/AIMNet2NSE/aimnet2nse_wb97m_3.pt) |

### AIMNet2-Pd — alias `aimnet2pd`

| Registry name | File | Download |
| --- | --- | --- |
| `aimnet2-pd_0` | aimnet2-pd_0.pt | [download](https://storage.googleapis.com/aimnetcentral/aimnet2v2/AIMNet2Pd/aimnet2-pd_0.pt) |
| `aimnet2-pd_1` | aimnet2-pd_1.pt | [download](https://storage.googleapis.com/aimnetcentral/aimnet2v2/AIMNet2Pd/aimnet2-pd_1.pt) |
| `aimnet2-pd_2` | aimnet2-pd_2.pt | [download](https://storage.googleapis.com/aimnetcentral/aimnet2v2/AIMNet2Pd/aimnet2-pd_2.pt) |
| `aimnet2-pd_3` | aimnet2-pd_3.pt | [download](https://storage.googleapis.com/aimnetcentral/aimnet2v2/AIMNet2Pd/aimnet2-pd_3.pt) |

## What's Next

- **[Your First Calculation](../tutorials/single_point.md)** -- Run a single-point energy calculation with your chosen model
- **[Architecture Overview](architecture.md)** -- Understand AIMNet2's neural network internals
- **[Calculator API Reference](../calculator.md)** -- Full details on constructor parameters and methods

## References

- **AIMNet2 (wB97M-D3, B97-3c):** Anstine, D.M., Zubatyuk, R., Isayev, O. _AIMNet2: A Neural Network Potential to Meet your Neutral, Charged, Organic, and Elemental-Organic Needs._ Chemical Science 2025, 16, 10228-10244. [DOI: 10.1039/D4SC08572H](https://doi.org/10.1039/D4SC08572H)

- **AIMNet2-NSE (open-shell):** Kalita, B.; Zubatyuk, R.; Anstine, D. M.; Bergeler, M.; Settels, V.; Stork, C.; Spicher, S.; Isayev, O. AIMNet2-NSE: A Transferable Reactive Neural Network Potential for Open-Shell Chemistry. _Angew. Chem. Int. Ed._ **2026**. [DOI: 10.1002/anie.202516763](https://doi.org/10.1002/anie.202516763)

- **AIMNet2-Pd:** Anstine, D. M.; Zubatyuk, R.; Gallegos, L.; Paton, R.; Wiest, O.; Nebgen, B.; Jones, T.; Gomes, G.; Tretiak, S.; Isayev, O. Transferable Machine Learning Interatomic Potential for Pd-Catalyzed Cross-Coupling Reactions. _ChemRxiv_ **2025**. [DOI: 10.26434/chemrxiv-2025-n36r6](https://doi.org/10.26434/chemrxiv-2025-n36r6)
