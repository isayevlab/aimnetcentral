# AIMNet Optimization Benchmark Summary

- Rows: 12 complete, 0 skipped
- Device: NVIDIA L40S (ada, cc 8.9)
- Torch/CUDA: 2.9.1+cu128 / 12.8
- AIMNet: 0.2.0.post1.dev34+g8092749f9.d20260617
- Workloads: water_scaling_5001, water_scaling_501, water_scaling_99, water_scaling_999
- Modes: forced-sparse, gpu-resident, strict

## Decision

Accepted non-strict execution-mode/workload combinations:
- water_scaling_99 / gpu-resident: speedup 1.219x
- water_scaling_501 / gpu-resident: speedup 1.600x

Strict baseline pass rate: 4/4 workloads.

## Result Matrix

| Workload | Mode | Median ms | Speedup | Accuracy | Stability | Meaningful |
| --- | --- | ---: | ---: | --- | --- | --- |
| water_scaling_5001 | forced-sparse | 80.328 | 0.679x | pass | n/a | n/a |
| water_scaling_5001 | gpu-resident | 54.865 | 0.994x | pass | n/a | fail |
| water_scaling_5001 | strict | 54.550 | n/a | pass | n/a | n/a |
| water_scaling_501 | forced-sparse | 21.112 | 1.242x | pass | n/a | n/a |
| water_scaling_501 | gpu-resident | 16.387 | 1.600x | pass | n/a | pass |
| water_scaling_501 | strict | 26.224 | n/a | pass | n/a | n/a |
| water_scaling_99 | forced-sparse | 31.011 | 0.436x | pass | n/a | fail |
| water_scaling_99 | gpu-resident | 11.078 | 1.219x | pass | n/a | pass |
| water_scaling_99 | strict | 13.506 | n/a | pass | n/a | n/a |
| water_scaling_999 | forced-sparse | 21.009 | 0.682x | pass | n/a | n/a |
| water_scaling_999 | gpu-resident | 15.379 | 0.932x | pass | n/a | fail |
| water_scaling_999 | strict | 14.330 | n/a | pass | n/a | n/a |

## Non-Strict Mode Aggregates

| Mode | Speedup min | Speedup median | Speedup max | Accuracy pass | Stability pass | Meaningful pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| forced-sparse | 0.436 | 0.436 | 0.436 | 4/4 | 4/4 | 0/4 |
| gpu-resident | 0.932 | 1.107 | 1.600 | 4/4 | 4/4 | 2/4 |

## Rejections

- water_scaling_99 / forced-sparse: speedup 0.436x below threshold
- water_scaling_501 / forced-sparse: same effective execution path as strict
- water_scaling_999 / forced-sparse: same effective execution path as strict
- water_scaling_999 / gpu-resident: speedup 0.932x below threshold
- water_scaling_5001 / forced-sparse: same effective execution path as strict
- water_scaling_5001 / gpu-resident: speedup 0.994x below threshold
