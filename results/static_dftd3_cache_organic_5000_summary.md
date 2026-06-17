# AIMNet Optimization Benchmark Summary

- Rows: 3 complete, 0 skipped
- Device: NVIDIA L40S (ada, cc 8.9)
- Torch/CUDA: 2.9.1+cu128 / 12.8
- AIMNet: 0.2.0.post1.dev34+g8092749f9.d20260617
- Workloads: organic_scaling_5000
- Modes: gpu-resident, static-cache, strict

## Decision

Accepted non-strict execution-mode/workload combinations:
- organic_scaling_5000 / static-cache: speedup 2.119x

Strict baseline pass rate: 1/1 workloads.

## Result Matrix

| Workload | Mode | Median ms | Speedup | Accuracy | Stability | Meaningful |
| --- | --- | ---: | ---: | --- | --- | --- |
| organic_scaling_5000 | gpu-resident | 57.179 | 1.000x | pass | n/a | fail |
| organic_scaling_5000 | static-cache | 26.980 | 2.119x | pass | n/a | pass |
| organic_scaling_5000 | strict | 57.158 | n/a | pass | n/a | n/a |

## Non-Strict Mode Aggregates

| Mode | Speedup min | Speedup median | Speedup max | Accuracy pass | Stability pass | Meaningful pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gpu-resident | 1.000 | 1.000 | 1.000 | 1/1 | 1/1 | 0/1 |
| static-cache | 2.119 | 2.119 | 2.119 | 1/1 | 1/1 | 1/1 |

## Rejections

- organic_scaling_5000 / gpu-resident: speedup 1.000x below threshold
