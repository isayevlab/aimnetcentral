# Non-Covalent Interactions

## What You'll Learn

- How to compute interaction energies for molecular complexes
- Scanning potential energy surfaces for non-covalent dimers
- Why BSSE is different for ML potentials versus quantum chemistry
- Comparing `aimnet2` and `aimnet2_2025` for intermolecular accuracy
- Configuring long-range electrostatics for non-covalent systems

## Prerequisites

- Familiarity with [AIMNet2Calculator](../calculator.md) and basic single-point calculations
- Understanding of [long-range methods](../long_range.md) (DSF, Ewald)
- ASE installation for geometry optimization examples

## Spotlight Model: AIMNet2-2025

The `aimnet2_2025` model was specifically trained with improved sampling of intermolecular configurations, making it the best choice for non-covalent interaction studies. It shares the same element coverage as the standard models (H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br, I) but provides significantly better accuracy for hydrogen bonding, pi-stacking, and dispersion-dominated complexes.

See the [Model Selection Guide](../models/guide.md) for choosing between models.

## Interaction Energy: The Supramolecular Approach

The interaction energy of a complex AB is computed as the difference between the energy of the complex and the sum of the isolated monomers:

```
E_int = E(AB) - E(A) - E(B)
```

Each monomer is computed at its geometry within the complex (no re-optimization). This is sometimes called the "supramolecular" or "counterpoise-free" approach.

```python
import torch
from aimnet.calculators import AIMNet2Calculator

calc = AIMNet2Calculator("aimnet2_2025")

def interaction_energy(coords_ab, numbers_ab, charge_ab,
                       coords_a, numbers_a, charge_a,
                       coords_b, numbers_b, charge_b):
    """Compute interaction energy E(AB) - E(A) - E(B) in eV."""
    e_ab = calc({"coord": coords_ab, "numbers": numbers_ab,
                 "charge": charge_ab})["energy"]
    e_a = calc({"coord": coords_a, "numbers": numbers_a,
                "charge": charge_a})["energy"]
    e_b = calc({"coord": coords_b, "numbers": numbers_b,
                "charge": charge_b})["energy"]
    return (e_ab - e_a - e_b).item()
```

## Worked Example: Water Dimer Binding Curve

The water dimer is the classic benchmark for hydrogen bonding. We scan the O...O distance from 2.0 to 6.0 Angstrom while keeping monomer geometries fixed.

```python
import torch
import numpy as np
from aimnet.calculators import AIMNet2Calculator

calc = AIMNet2Calculator("aimnet2_2025")

# Water monomer A (acceptor): O at origin, H atoms in xz-plane
water_a_coords = torch.tensor([
    [0.0000,  0.0000,  0.0000],   # O
    [0.7572,  0.0000,  0.5865],   # H
    [-0.7572, 0.0000,  0.5865],   # H
])
water_a_numbers = torch.tensor([8, 1, 1])

# Compute monomer A energy once
e_a = calc({
    "coord": water_a_coords,
    "numbers": water_a_numbers,
    "charge": torch.tensor(0.0),
})["energy"].item()

# Water monomer B (donor): one H pointing toward monomer A's O
# O along +x axis at distance R, with one H directed back toward A
water_b_coords_template = torch.tensor([
    [0.0000,  0.0000,  0.0000],   # O (will be shifted to d_oo along x)
    [-0.5865, 0.0000, -0.7572],   # H (pointing toward monomer A)
    [-0.5865, 0.0000,  0.7572],   # H (pointing away)
])
water_b_numbers = torch.tensor([8, 1, 1])

e_b = calc({
    "coord": water_b_coords_template,
    "numbers": water_b_numbers,
    "charge": torch.tensor(0.0),
})["energy"].item()

# Scan O...O distance
distances = np.arange(2.0, 6.05, 0.1)  # Angstrom
energies = []

for d_oo in distances:
    # Shift monomer B along x-axis so its O is at (d_oo, 0, 0)
    shifted_b = water_b_coords_template.clone()
    shifted_b[:, 0] += d_oo

    # Build dimer: A (acceptor at origin) + B (donor at d_oo along x)
    dimer_coords = torch.cat([water_a_coords, shifted_b], dim=0)
    dimer_numbers = torch.cat([water_a_numbers, water_b_numbers])

    e_dimer = calc({
        "coord": dimer_coords,
        "numbers": dimer_numbers,
        "charge": torch.tensor(0.0),
    })["energy"].item()

    e_int = e_dimer - e_a - e_b
    energies.append(e_int)

# Convert to kcal/mol for comparison with literature
ev_to_kcal = 23.0609
energies_kcal = [e * ev_to_kcal for e in energies]

# Find minimum
min_idx = np.argmin(energies_kcal)
print(f"Minimum at O...O = {distances[min_idx]:.1f} A")
print(f"Interaction energy = {energies_kcal[min_idx]:.2f} kcal/mol")
# Reference: approximately -5.0 kcal/mol at ~2.9 A (CCSD(T)/CBS)
```

## Comparing Models on the Water Dimer

The `aimnet2_2025` model shows improved accuracy over the original `aimnet2` for intermolecular interactions. You can compare them directly:

```python
import torch
import numpy as np
from aimnet.calculators import AIMNet2Calculator

calc_orig = AIMNet2Calculator("aimnet2")
calc_2025 = AIMNet2Calculator("aimnet2_2025")

# Use the same water dimer geometry at near-equilibrium distance
# Acceptor at origin, donor at 2.9 A along x with one H pointing toward acceptor O
water_a = torch.tensor([
    [0.0000,  0.0000,  0.0000],
    [0.7572,  0.0000,  0.5865],
    [-0.7572, 0.0000,  0.5865],
])
water_b = torch.tensor([
    [2.9000,  0.0000,  0.0000],
    [2.3135,  0.0000, -0.7572],
    [2.3135,  0.0000,  0.7572],
])
dimer = torch.cat([water_a, water_b], dim=0)
numbers_mono = torch.tensor([8, 1, 1])
numbers_dimer = torch.tensor([8, 1, 1, 8, 1, 1])
charge = torch.tensor(0.0)

ev_to_kcal = 23.0609

for name, calc in [("aimnet2", calc_orig), ("aimnet2_2025", calc_2025)]:
    e_dimer = calc({"coord": dimer, "numbers": numbers_dimer,
                     "charge": charge})["energy"].item()
    e_a = calc({"coord": water_a, "numbers": numbers_mono,
                 "charge": charge})["energy"].item()
    e_b = calc({"coord": water_b, "numbers": numbers_mono,
                 "charge": charge})["energy"].item()
    e_int = (e_dimer - e_a - e_b) * ev_to_kcal
    print(f"{name}: E_int = {e_int:.2f} kcal/mol")
```

## Benzene Dimer: Parallel-Displaced Configuration

The benzene dimer is the classic pi-stacking benchmark. The global minimum is the parallel-displaced (PD) geometry, not the sandwich or T-shaped configurations.

```python
import torch
import numpy as np
from aimnet.calculators import AIMNet2Calculator

calc = AIMNet2Calculator("aimnet2_2025")

# Benzene monomer (planar, in xy-plane centered at origin)
benzene_c = torch.tensor([
    [ 1.3915,  0.0000, 0.0],
    [ 0.6957,  1.2048, 0.0],
    [-0.6957,  1.2048, 0.0],
    [-1.3915,  0.0000, 0.0],
    [-0.6957, -1.2048, 0.0],
    [ 0.6957, -1.2048, 0.0],
])
benzene_h = torch.tensor([
    [ 2.4715,  0.0000, 0.0],
    [ 1.2358,  2.1404, 0.0],
    [-1.2358,  2.1404, 0.0],
    [-2.4715,  0.0000, 0.0],
    [-1.2358, -2.1404, 0.0],
    [ 1.2358, -2.1404, 0.0],
])
mono_a = torch.cat([benzene_c, benzene_h], dim=0)
numbers_mono = torch.tensor([6]*6 + [1]*6)

# Monomer energy
e_mono = calc({
    "coord": mono_a, "numbers": numbers_mono, "charge": torch.tensor(0.0),
})["energy"].item()

# Parallel-displaced dimer: second benzene shifted by 1.6 A in x
# and separated by 3.4 A in z (approximate PD minimum)
mono_b = mono_a.clone()
mono_b[:, 0] += 1.6  # lateral displacement
mono_b[:, 2] += 3.4  # vertical separation

dimer = torch.cat([mono_a, mono_b], dim=0)
numbers_dimer = torch.cat([numbers_mono, numbers_mono])

e_dimer = calc({
    "coord": dimer, "numbers": numbers_dimer, "charge": torch.tensor(0.0),
})["energy"].item()

e_int = (e_dimer - 2.0 * e_mono) * 23.0609  # kcal/mol
print(f"Benzene dimer (PD) interaction energy: {e_int:.2f} kcal/mol")
# Reference: approximately -2.7 kcal/mol (CCSD(T)/CBS)
```

!!! tip "Optimizing Dimer Geometries"

    For a more rigorous treatment, optimize the dimer geometry while keeping monomers rigid (rigid-body optimization) or allow full relaxation. The fixed-monomer approach shown here is standard for benchmarking but may not capture full relaxation effects.

## BSSE in ML Potentials

!!! note "No Basis Set Superposition Error in the Traditional Sense"

    In quantum chemistry, BSSE arises because the basis functions of monomer A artificially improve the description of monomer B (and vice versa) when computing the dimer energy. The counterpoise correction removes this artifact.

    ML potentials like AIMNet2 have **no basis set** and therefore **no BSSE** in the traditional sense. The interaction energy computed as `E(AB) - E(A) - E(B)` does not suffer from basis set incompleteness.

    However, a subtlety remains: the DFT reference data used for training **was** computed with a finite basis set (def2-TZVPP for `aimnet2`/`aimnet2nse`; def2-mTZVP for `aimnet2_2025`/`aimnet2_b973c`). If the training data was not counterpoise-corrected, the model may have learned an implicit BSSE from the reference energies. For AIMNet2 models, these basis sets minimize this effect, but it is not entirely absent.

    **In practice:** Do not apply counterpoise corrections to AIMNet2 results. The supramolecular approach (`E(AB) - E(A) - E(B)`) is the correct way to compute interaction energies with ML potentials.

## Long-Range Method Guidance

Non-covalent interactions are sensitive to the treatment of long-range electrostatics. The default "simple" (all-pairs) Coulomb method is appropriate for gas-phase dimers and clusters.

For larger aggregates or periodic systems with non-covalent contacts:

```python
from aimnet.calculators import AIMNet2Calculator

calc = AIMNet2Calculator("aimnet2_2025")

# For non-periodic clusters: simple method is fine (default)
# For periodic systems: use DSF or Ewald
calc.set_lrcoulomb_method("dsf", cutoff=15.0)

# For highest accuracy on electrostatic-dominated interactions:
calc.set_lrcoulomb_method("ewald", ewald_accuracy=1e-8)
```

!!! warning "Dispersion Cutoff for Large Systems"

    For large molecular clusters or periodic systems with significant dispersion contributions (e.g., molecular crystals), ensure the DFT-D3 cutoff is large enough. The default 15.0 Angstrom is adequate for most cases, but van der Waals dominated systems may benefit from a larger cutoff:

    ```python
    calc.set_dftd3_cutoff(cutoff=20.0, smoothing_fraction=0.2)
    ```

    Test convergence by comparing energies at different cutoffs.

## Types of Non-Covalent Interactions

AIMNet2 models handle several classes of non-covalent interactions:

| Interaction Type | Example | Typical Strength | Model Recommendation |
| --- | --- | --- | --- |
| Hydrogen bonding | Water dimer, DNA base pairs | 3-20 kcal/mol | `aimnet2_2025` |
| pi-pi stacking | Benzene dimer, nucleobases | 1-5 kcal/mol | `aimnet2_2025` |
| CH-pi | Methane-benzene | 0.5-2 kcal/mol | `aimnet2_2025` |
| Halogen bonding | R-X...Y (X = Cl, Br, I) | 2-10 kcal/mol | `aimnet2_2025` |
| Dispersion (vdW) | Alkane dimers | 0.5-3 kcal/mol | `aimnet2_2025` |

!!! note "Accuracy Expectations"

    For well-represented interaction types (hydrogen bonds, pi-stacking), AIMNet2-2025 typically achieves errors of 0.3-0.5 kcal/mol compared to CCSD(T)/CBS references. Larger errors may occur for interaction types or geometries that are underrepresented in the training data.

## What's Next

- [Charged Systems](charged_systems.md) -- handling ions and charged complexes in non-covalent assemblies
- [Model Selection Guide](../models/guide.md) -- comparing all available models
- [Long-Range Methods](../long_range.md) -- details on DSF and Ewald methods
