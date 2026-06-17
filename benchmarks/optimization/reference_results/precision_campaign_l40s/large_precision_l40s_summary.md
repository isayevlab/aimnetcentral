# AIMNet Precision And Performance Summary

- Rows: 9 complete, 0 skipped
- Device: NVIDIA L40S (ada, cc 8.9)
- Torch/CUDA: 2.9.1+cu128 / 12.8
- AIMNet: 0.2.0.post1.dev34+g8092749f9.d20260617
- Workloads: water_cluster_1500, water_cluster_501, water_cluster_999
- Policies: bf16_learned, fp8_learned_experimental, strict

## Decision

No non-strict policy met the combined accuracy, stability, and meaningful-speedup gates.

Strict baseline pass rate: 2/3 workloads.

## Result Matrix

| Workload | Policy | Median ms | Speedup | Accuracy | Stability | Meaningful |
| --- | --- | ---: | ---: | --- | --- | --- |
| water_cluster_1500 | bf16_learned | 16.240 | 0.963x | fail | pass | fail |
| water_cluster_1500 | fp8_learned_experimental | 16.530 | 0.946x | fail | pass | fail |
| water_cluster_1500 | strict | 15.637 | n/a | fail | pass | n/a |
| water_cluster_501 | bf16_learned | 15.175 | 0.967x | fail | pass | fail |
| water_cluster_501 | fp8_learned_experimental | 14.363 | 1.022x | fail | pass | fail |
| water_cluster_501 | strict | 14.674 | n/a | pass | pass | n/a |
| water_cluster_999 | bf16_learned | 17.077 | 0.897x | fail | pass | fail |
| water_cluster_999 | fp8_learned_experimental | 14.861 | 1.031x | fail | pass | fail |
| water_cluster_999 | strict | 15.325 | n/a | pass | pass | n/a |

## Non-Strict Policy Aggregates

| Policy | Speedup min | Speedup median | Speedup max | Accuracy pass | Stability pass | Meaningful pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bf16_learned | 0.897 | 0.963 | 0.967 | 0/3 | 3/3 | 0/3 |
| fp8_learned_experimental | 0.946 | 1.022 | 1.031 | 0/3 | 3/3 | 0/3 |

## Rejections

- water_cluster_501 / bf16_learned: accuracy gate failed; speedup 0.967x below threshold
- water_cluster_501 / fp8_learned_experimental: accuracy gate failed; speedup 1.022x below threshold
- water_cluster_999 / bf16_learned: accuracy gate failed; speedup 0.897x below threshold
- water_cluster_999 / fp8_learned_experimental: accuracy gate failed; speedup 1.031x below threshold
- water_cluster_1500 / bf16_learned: accuracy gate failed; speedup 0.963x below threshold
- water_cluster_1500 / fp8_learned_experimental: accuracy gate failed; speedup 0.946x below threshold
