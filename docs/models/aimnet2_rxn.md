# AIMNet2-rxn

A neural network interatomic potential specialized for **closed-shell organic reactions** (H, C, N, O), trained on ~4.7M reaction-relevant geometries at ωB97M-V/def2-TZVPP. Use for transition-state searches, NEB / batched-NEB, IRC profiles, and reaction-coordinate energy work.

## Loading

```python
from aimnet.calculators import AIMNet2Calculator

# From the GCS-backed registry (alias):
calc = AIMNet2Calculator("aimnet2rxn", ensemble_member=0)

# From Hugging Face Hub:
calc = AIMNet2Calculator("isayevlab/aimnet2-rxn", ensemble_member=0)
```

Both paths produce equivalent calculators. The HF path requires `pip install "aimnet[hf]"`.

## Calculator-enforced safeguards (this family)

The calculator applies the following checks automatically when `validate_species=True` (the default). Each can be bypassed with `validate_species=False`:

- **Element scope**: input atomic numbers must be a subset of `[1, 6, 7, 8]`. Other elements raise `ValueError` with pointers to alternative families.
- **Net charge**: only net-neutral systems (zwitterions OK). Non-zero `charge` raises `ValueError` pointing at `aimnet2-wb97m-d3` for ions.
- **AFV row sanitization**: at conversion time, atomic-feature-vector rows for elements outside `[1, 6, 7, 8]` are NaN-padded so `validate_species=False` produces NaN-propagation rather than plausible-looking nonsense.

Two further safeguards fire regardless of `validate_species`:

- **Hessian + `torch.compile`**: setting both raises `RuntimeError` (Dynamo + double-backward through GELU is known to hang). Reconstruct with `compile_model=False` for TS / IRC / vibrational work.
- **Coulomb cutoff lock**: calling `set_lrcoulomb_method(method, cutoff=…)` with a cutoff different from the model's `coulomb_sr_rc` (4.6 Å) emits a `UserWarning` because the SR/LR cancellation point was physically frozen during training.

A separate one-time `UserWarning` fires if the same Python process constructs calculators from two different AIMNet2 families (rxn vs. wb97m-d3 etc.), because the energy scales are not comparable.

## Canonical model card

Full content (energy convention, training data details, full limitations list, citation) lives at the Hugging Face model card:

[https://huggingface.co/isayevlab/aimnet2-rxn](https://huggingface.co/isayevlab/aimnet2-rxn)

The HF README is the canonical source — this page summarizes only the integration mechanics.
