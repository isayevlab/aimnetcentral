# AIMNet Optimization Benchmark Summary

- Rows: 6 complete, 0 skipped
- Device: NVIDIA L40S (ada, cc 8.9)
- Torch/CUDA: 2.9.1+cu128 / 12.8
- AIMNet: 0.2.0.post1.dev34+g8092749f9.d20260617
- Workloads: organic_scaling_5000, water_scaling_5001
- Modes: gpu-resident, neighbor-skin, strict

## Decision

No non-strict execution mode met the combined accuracy, stability, and meaningful-speedup gates.

Strict baseline pass rate: 2/2 workloads.

## Result Matrix

| Workload | Mode | Median ms | Speedup | Accuracy | Stability | Meaningful |
| --- | --- | ---: | ---: | --- | --- | --- |
| organic_scaling_5000 | gpu-resident | 56.550 | 1.004x | pass | n/a | fail |
| organic_scaling_5000 | neighbor-skin | 52.840 | 1.074x | pass | n/a | fail |
| organic_scaling_5000 | strict | 56.751 | n/a | pass | n/a | n/a |
| water_scaling_5001 | gpu-resident | 51.588 | 0.991x | pass | n/a | fail |
| water_scaling_5001 | neighbor-skin | 48.503 | 1.054x | pass | n/a | fail |
| water_scaling_5001 | strict | 51.099 | n/a | pass | n/a | n/a |

## Non-Strict Mode Aggregates

| Mode | Speedup min | Speedup median | Speedup max | Accuracy pass | Stability pass | Meaningful pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gpu-resident | 0.991 | 0.997 | 1.004 | 2/2 | 2/2 | 0/2 |
| neighbor-skin | 1.054 | 1.064 | 1.074 | 2/2 | 2/2 | 0/2 |

## Rejections

- water_scaling_5001 / gpu-resident: speedup 0.991x below threshold
- water_scaling_5001 / neighbor-skin: speedup 1.054x below threshold
- organic_scaling_5000 / gpu-resident: speedup 1.004x below threshold
- organic_scaling_5000 / neighbor-skin: speedup 1.074x below threshold
