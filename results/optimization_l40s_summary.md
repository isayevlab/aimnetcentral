# AIMNet Optimization Benchmark Summary

- Rows: 18 complete, 0 skipped
- Device: NVIDIA L40S (ada, cc 8.9)
- Torch/CUDA: 2.9.1+cu128 / 12.8
- AIMNet: 0.2.0.post1.dev34+g8092749f9.d20260617
- Workloads: caffeine_forces, charged_pair_forces, generated_organic_128, methane_forces, pbc_water_stress, water_forces
- Modes: forced-sparse, gpu-resident, strict

## Decision

Accepted non-strict execution-mode/workload combinations:
- caffeine_forces / gpu-resident: speedup 1.087x

Strict baseline pass rate: 6/6 workloads.

## Result Matrix

| Workload | Mode | Median ms | Speedup | Accuracy | Stability | Meaningful |
| --- | --- | ---: | ---: | --- | --- | --- |
| caffeine_forces | forced-sparse | 14.028 | 0.853x | pass | n/a | fail |
| caffeine_forces | gpu-resident | 11.010 | 1.087x | pass | n/a | pass |
| caffeine_forces | strict | 11.971 | n/a | pass | n/a | n/a |
| charged_pair_forces | forced-sparse | 13.779 | 0.850x | pass | n/a | fail |
| charged_pair_forces | gpu-resident | 11.668 | 1.004x | pass | n/a | fail |
| charged_pair_forces | strict | 11.715 | n/a | pass | n/a | n/a |
| generated_organic_128 | forced-sparse | 14.073 | 1.006x | pass | n/a | n/a |
| generated_organic_128 | gpu-resident | 14.007 | 1.011x | pass | n/a | fail |
| generated_organic_128 | strict | 14.156 | n/a | pass | n/a | n/a |
| methane_forces | forced-sparse | 13.671 | 0.791x | pass | n/a | fail |
| methane_forces | gpu-resident | 11.692 | 0.925x | pass | n/a | fail |
| methane_forces | strict | 10.818 | n/a | pass | n/a | n/a |
| pbc_water_stress | forced-sparse | 18.463 | 1.004x | pass | n/a | n/a |
| pbc_water_stress | gpu-resident | 19.007 | 0.975x | pass | n/a | fail |
| pbc_water_stress | strict | 18.541 | n/a | pass | n/a | n/a |
| water_forces | forced-sparse | 13.385 | 0.854x | pass | n/a | fail |
| water_forces | gpu-resident | 11.497 | 0.994x | pass | n/a | fail |
| water_forces | strict | 11.425 | n/a | pass | n/a | n/a |

## Non-Strict Mode Aggregates

| Mode | Speedup min | Speedup median | Speedup max | Accuracy pass | Stability pass | Meaningful pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| forced-sparse | 0.791 | 0.852 | 0.854 | 6/6 | 6/6 | 0/6 |
| gpu-resident | 0.925 | 0.999 | 1.087 | 6/6 | 6/6 | 1/6 |

## Rejections

- water_forces / forced-sparse: speedup 0.854x below threshold
- water_forces / gpu-resident: speedup 0.994x below threshold
- methane_forces / forced-sparse: speedup 0.791x below threshold
- methane_forces / gpu-resident: speedup 0.925x below threshold
- charged_pair_forces / forced-sparse: speedup 0.850x below threshold
- charged_pair_forces / gpu-resident: speedup 1.004x below threshold
- caffeine_forces / forced-sparse: speedup 0.853x below threshold
- pbc_water_stress / forced-sparse: same effective execution path as strict
- pbc_water_stress / gpu-resident: speedup 0.975x below threshold
- generated_organic_128 / forced-sparse: same effective execution path as strict
- generated_organic_128 / gpu-resident: speedup 1.011x below threshold
