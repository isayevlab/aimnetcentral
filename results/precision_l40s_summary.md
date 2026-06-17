# AIMNet Precision And Performance Summary

- Rows: 18 complete, 0 skipped
- Device: NVIDIA L40S (ada, cc 8.9)
- Torch/CUDA: 2.9.1+cu128 / 12.8
- AIMNet: 0.2.0.post1.dev34+g8092749f9.d20260617
- Workloads: caffeine_forces, charged_pair_forces, generated_organic_128, methane_forces, pbc_water_stress, water_forces
- Policies: bf16_learned, strict, tf32_learned

## Decision

No non-strict policy met the combined accuracy, stability, and meaningful-speedup gates.

Strict baseline pass rate: 6/6 workloads.

## Result Matrix

| Workload | Policy | Median ms | Speedup | Accuracy | Stability | Meaningful |
| --- | --- | ---: | ---: | --- | --- | --- |
| caffeine_forces | bf16_learned | 15.430 | 0.915x | fail | fail | fail |
| caffeine_forces | strict | 14.115 | n/a | pass | pass | n/a |
| caffeine_forces | tf32_learned | 14.685 | 0.961x | fail | fail | fail |
| charged_pair_forces | bf16_learned | 14.411 | 1.008x | fail | fail | fail |
| charged_pair_forces | strict | 14.520 | n/a | pass | pass | n/a |
| charged_pair_forces | tf32_learned | 14.962 | 0.970x | fail | pass | fail |
| generated_organic_128 | bf16_learned | 15.619 | 0.897x | fail | pass | fail |
| generated_organic_128 | strict | 14.013 | n/a | pass | pass | n/a |
| generated_organic_128 | tf32_learned | 14.910 | 0.940x | fail | pass | fail |
| methane_forces | bf16_learned | 14.538 | 0.957x | fail | fail | fail |
| methane_forces | strict | 13.914 | n/a | pass | pass | n/a |
| methane_forces | tf32_learned | 13.977 | 0.995x | fail | fail | fail |
| pbc_water_stress | bf16_learned | 19.975 | 0.969x | fail | fail | fail |
| pbc_water_stress | strict | 19.349 | n/a | pass | pass | n/a |
| pbc_water_stress | tf32_learned | 20.050 | 0.965x | fail | fail | fail |
| water_forces | bf16_learned | 15.061 | 0.949x | fail | fail | fail |
| water_forces | strict | 14.299 | n/a | pass | pass | n/a |
| water_forces | tf32_learned | 14.655 | 0.976x | pass | fail | fail |

## Non-Strict Policy Aggregates

| Policy | Speedup min | Speedup median | Speedup max | Accuracy pass | Stability pass | Meaningful pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bf16_learned | 0.897 | 0.953 | 1.008 | 0/6 | 1/6 | 0/6 |
| tf32_learned | 0.940 | 0.968 | 0.995 | 1/6 | 2/6 | 0/6 |

## Rejections

- water_forces / tf32_learned: stability gate failed; speedup 0.976x below threshold
- water_forces / bf16_learned: accuracy gate failed; stability gate failed; speedup 0.949x below threshold
- methane_forces / tf32_learned: accuracy gate failed; stability gate failed; speedup 0.995x below threshold
- methane_forces / bf16_learned: accuracy gate failed; stability gate failed; speedup 0.957x below threshold
- charged_pair_forces / tf32_learned: accuracy gate failed; speedup 0.970x below threshold
- charged_pair_forces / bf16_learned: accuracy gate failed; stability gate failed; speedup 1.008x below threshold
- caffeine_forces / tf32_learned: accuracy gate failed; stability gate failed; speedup 0.961x below threshold
- caffeine_forces / bf16_learned: accuracy gate failed; stability gate failed; speedup 0.915x below threshold
- pbc_water_stress / tf32_learned: accuracy gate failed; stability gate failed; speedup 0.965x below threshold
- pbc_water_stress / bf16_learned: accuracy gate failed; stability gate failed; speedup 0.969x below threshold
- generated_organic_128 / tf32_learned: accuracy gate failed; speedup 0.940x below threshold
- generated_organic_128 / bf16_learned: accuracy gate failed; speedup 0.897x below threshold
