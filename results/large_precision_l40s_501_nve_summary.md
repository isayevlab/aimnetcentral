# AIMNet Precision And Performance Summary

- Rows: 3 complete, 0 skipped
- Device: NVIDIA L40S (ada, cc 8.9)
- Torch/CUDA: 2.9.1+cu128 / 12.8
- AIMNet: 0.2.0.post1.dev34+g8092749f9.d20260617
- Workloads: water_cluster_501
- Policies: bf16_learned, fp8_learned_experimental, strict

## Decision

No non-strict policy met the combined accuracy, stability, and meaningful-speedup gates.

Strict baseline pass rate: 1/1 workloads.

## Result Matrix

| Workload | Policy | Median ms | Speedup | Accuracy | Stability | Meaningful |
| --- | --- | ---: | ---: | --- | --- | --- |
| water_cluster_501 | bf16_learned | 17.250 | 0.865x | fail | fail | fail |
| water_cluster_501 | fp8_learned_experimental | 13.993 | 1.066x | fail | pass | fail |
| water_cluster_501 | strict | 14.915 | n/a | pass | pass | n/a |

## Non-Strict Policy Aggregates

| Policy | Speedup min | Speedup median | Speedup max | Accuracy pass | Stability pass | Meaningful pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bf16_learned | 0.865 | 0.865 | 0.865 | 0/1 | 0/1 | 0/1 |
| fp8_learned_experimental | 1.066 | 1.066 | 1.066 | 0/1 | 1/1 | 0/1 |

## Rejections

- water_cluster_501 / bf16_learned: accuracy gate failed; stability gate failed; speedup 0.865x below threshold
- water_cluster_501 / fp8_learned_experimental: accuracy gate failed; speedup 1.066x below threshold
