# AIMNet Precision And Performance Summary

- Rows: 6 complete, 0 skipped
- Device: NVIDIA L40S (ada, cc 8.9)
- Torch/CUDA: 2.9.1+cu128 / 12.8
- AIMNet: 0.2.0.post1.dev34+g8092749f9.d20260617
- Workloads: caffeine_forces, charged_pair_forces, generated_organic_128, methane_forces, pbc_water_stress, water_forces
- Policies: strict

## Decision

No non-strict policy met the combined accuracy, stability, and meaningful-speedup gates.

Strict baseline pass rate: 6/6 workloads.

## Result Matrix

| Workload | Policy | Median ms | Speedup | Accuracy | Stability | Meaningful |
| --- | --- | ---: | ---: | --- | --- | --- |
| caffeine_forces | strict | 12.946 | n/a | pass | pass | n/a |
| charged_pair_forces | strict | 12.917 | n/a | pass | pass | n/a |
| generated_organic_128 | strict | 16.304 | n/a | pass | pass | n/a |
| methane_forces | strict | 12.854 | n/a | pass | pass | n/a |
| pbc_water_stress | strict | 20.155 | n/a | pass | pass | n/a |
| water_forces | strict | 12.658 | n/a | pass | pass | n/a |
