# AIMNet Optimization Benchmark Summary

- Rows: 3 complete, 0 skipped
- Device: NVIDIA L40S (ada, cc 8.9)
- Torch/CUDA: 2.9.1+cu128 / 12.8
- AIMNet: 0.2.0.post1.dev34+g8092749f9.d20260617
- Workloads: water_scaling_5001
- Modes: gpu-resident, static-cache, strict

## Decision

Accepted non-strict execution-mode/workload combinations:
- water_scaling_5001 / static-cache: speedup 1.744x

Strict baseline pass rate: 1/1 workloads.

## Result Matrix

| Workload | Mode | Median ms | Speedup | Accuracy | Stability | Meaningful |
| --- | --- | ---: | ---: | --- | --- | --- |
| water_scaling_5001 | gpu-resident | 51.215 | 0.999x | pass | n/a | fail |
| water_scaling_5001 | static-cache | 29.334 | 1.744x | pass | n/a | pass |
| water_scaling_5001 | strict | 51.155 | n/a | pass | n/a | n/a |

## Non-Strict Mode Aggregates

| Mode | Speedup min | Speedup median | Speedup max | Accuracy pass | Stability pass | Meaningful pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gpu-resident | 0.999 | 0.999 | 0.999 | 1/1 | 1/1 | 0/1 |
| static-cache | 1.744 | 1.744 | 1.744 | 1/1 | 1/1 | 1/1 |

## Rejections

- water_scaling_5001 / gpu-resident: speedup 0.999x below threshold
