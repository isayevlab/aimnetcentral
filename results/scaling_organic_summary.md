# AIMNet Optimization Benchmark Summary

- Rows: 12 complete, 0 skipped
- Device: NVIDIA L40S (ada, cc 8.9)
- Torch/CUDA: 2.9.1+cu128 / 12.8
- AIMNet: 0.2.0.post1.dev34+g8092749f9.d20260617
- Workloads: organic_scaling_100, organic_scaling_1000, organic_scaling_500, organic_scaling_5000
- Modes: forced-sparse, gpu-resident, strict

## Decision

Accepted non-strict execution-mode/workload combinations:
- organic_scaling_100 / gpu-resident: speedup 1.549x
- organic_scaling_1000 / gpu-resident: speedup 1.116x

Strict baseline pass rate: 3/4 workloads.

## Result Matrix

| Workload | Mode | Median ms | Speedup | Accuracy | Stability | Meaningful |
| --- | --- | ---: | ---: | --- | --- | --- |
| organic_scaling_100 | forced-sparse | 23.594 | 0.797x | pass | n/a | fail |
| organic_scaling_100 | gpu-resident | 12.143 | 1.549x | pass | n/a | pass |
| organic_scaling_100 | strict | 18.805 | n/a | pass | n/a | n/a |
| organic_scaling_1000 | forced-sparse | 16.353 | 1.213x | pass | n/a | n/a |
| organic_scaling_1000 | gpu-resident | 17.766 | 1.116x | pass | n/a | pass |
| organic_scaling_1000 | strict | 19.835 | n/a | pass | n/a | n/a |
| organic_scaling_500 | forced-sparse | 19.982 | 0.802x | pass | n/a | n/a |
| organic_scaling_500 | gpu-resident | 31.107 | 0.515x | pass | n/a | fail |
| organic_scaling_500 | strict | 16.028 | n/a | pass | n/a | n/a |
| organic_scaling_5000 | forced-sparse | 59.437 | 1.130x | fail | n/a | n/a |
| organic_scaling_5000 | gpu-resident | 58.508 | 1.148x | fail | n/a | n/a |
| organic_scaling_5000 | strict | 67.143 | n/a | fail | n/a | n/a |

## Non-Strict Mode Aggregates

| Mode | Speedup min | Speedup median | Speedup max | Accuracy pass | Stability pass | Meaningful pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| forced-sparse | 0.797 | 0.797 | 0.797 | 3/4 | 4/4 | 0/4 |
| gpu-resident | 0.515 | 1.116 | 1.549 | 3/4 | 4/4 | 2/4 |

## Rejections

- organic_scaling_100 / forced-sparse: speedup 0.797x below threshold
- organic_scaling_500 / forced-sparse: same effective execution path as strict
- organic_scaling_500 / gpu-resident: speedup 0.515x below threshold
- organic_scaling_1000 / forced-sparse: same effective execution path as strict
- organic_scaling_5000 / forced-sparse: accuracy gate failed; same effective execution path as strict
- organic_scaling_5000 / gpu-resident: accuracy gate failed; same effective execution path as strict
