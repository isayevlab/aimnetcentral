# AIMNet Optimization Benchmark Summary

- Rows: 24 complete, 0 skipped
- Device: NVIDIA L40S (ada, cc 8.9)
- Torch/CUDA: 2.9.1+cu128 / 12.8
- AIMNet: 0.2.0.post1.dev34+g8092749f9.d20260617
- Workloads: organic_scaling_100, organic_scaling_1000, organic_scaling_500, organic_scaling_5000, water_scaling_5001, water_scaling_501, water_scaling_99, water_scaling_999
- Modes: forced-sparse, gpu-resident, strict

## Decision

No non-strict execution mode met the combined accuracy, stability, and meaningful-speedup gates.

Strict baseline pass rate: 8/8 workloads.

## Result Matrix

| Workload | Mode | Median ms | Speedup | Accuracy | Stability | Meaningful |
| --- | --- | ---: | ---: | --- | --- | --- |
| organic_scaling_100 | forced-sparse | 14.314 | 0.849x | pass | n/a | fail |
| organic_scaling_100 | gpu-resident | 11.935 | 1.018x | pass | n/a | fail |
| organic_scaling_100 | strict | 12.152 | n/a | pass | n/a | n/a |
| organic_scaling_1000 | forced-sparse | 14.475 | 1.019x | pass | n/a | n/a |
| organic_scaling_1000 | gpu-resident | 14.274 | 1.034x | pass | n/a | fail |
| organic_scaling_1000 | strict | 14.756 | n/a | pass | n/a | n/a |
| organic_scaling_500 | forced-sparse | 14.376 | 1.007x | pass | n/a | n/a |
| organic_scaling_500 | gpu-resident | 14.574 | 0.994x | pass | n/a | fail |
| organic_scaling_500 | strict | 14.483 | n/a | pass | n/a | n/a |
| organic_scaling_5000 | forced-sparse | 57.402 | 1.000x | pass | n/a | n/a |
| organic_scaling_5000 | gpu-resident | 57.186 | 1.004x | pass | n/a | fail |
| organic_scaling_5000 | strict | 57.388 | n/a | pass | n/a | n/a |
| water_scaling_5001 | forced-sparse | 51.145 | 1.001x | pass | n/a | n/a |
| water_scaling_5001 | gpu-resident | 51.226 | 0.999x | pass | n/a | fail |
| water_scaling_5001 | strict | 51.191 | n/a | pass | n/a | n/a |
| water_scaling_501 | forced-sparse | 14.879 | 0.983x | pass | n/a | n/a |
| water_scaling_501 | gpu-resident | 14.633 | 1.000x | pass | n/a | fail |
| water_scaling_501 | strict | 14.627 | n/a | pass | n/a | n/a |
| water_scaling_99 | forced-sparse | 14.633 | 0.829x | pass | n/a | fail |
| water_scaling_99 | gpu-resident | 11.957 | 1.014x | pass | n/a | fail |
| water_scaling_99 | strict | 12.129 | n/a | pass | n/a | n/a |
| water_scaling_999 | forced-sparse | 14.870 | 1.014x | pass | n/a | n/a |
| water_scaling_999 | gpu-resident | 14.828 | 1.017x | pass | n/a | fail |
| water_scaling_999 | strict | 15.078 | n/a | pass | n/a | n/a |

## Non-Strict Mode Aggregates

| Mode | Speedup min | Speedup median | Speedup max | Accuracy pass | Stability pass | Meaningful pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| forced-sparse | 0.829 | 0.839 | 0.849 | 8/8 | 8/8 | 0/8 |
| gpu-resident | 0.994 | 1.009 | 1.034 | 8/8 | 8/8 | 0/8 |

## Rejections

- organic_scaling_100 / forced-sparse: speedup 0.849x below threshold
- organic_scaling_100 / gpu-resident: speedup 1.018x below threshold
- organic_scaling_500 / forced-sparse: same effective execution path as strict
- organic_scaling_500 / gpu-resident: speedup 0.994x below threshold
- organic_scaling_1000 / forced-sparse: same effective execution path as strict
- organic_scaling_1000 / gpu-resident: speedup 1.034x below threshold
- organic_scaling_5000 / forced-sparse: same effective execution path as strict
- organic_scaling_5000 / gpu-resident: speedup 1.004x below threshold
- water_scaling_99 / forced-sparse: speedup 0.829x below threshold
- water_scaling_99 / gpu-resident: speedup 1.014x below threshold
- water_scaling_501 / forced-sparse: same effective execution path as strict
- water_scaling_501 / gpu-resident: speedup 1.000x below threshold
- water_scaling_999 / forced-sparse: same effective execution path as strict
- water_scaling_999 / gpu-resident: speedup 1.017x below threshold
- water_scaling_5001 / forced-sparse: same effective execution path as strict
- water_scaling_5001 / gpu-resident: speedup 0.999x below threshold
