# AIMNet Optimization Profile Summary

- Device: NVIDIA L40S (`cuda:0`)
- Profile steps: 5
- Modes: `strict`, `forced-sparse`
- Source branch: `opt/gpu-throughput`

| Workload | Mode | Median ms | CPU cudaLaunchKernel | CUDA events | Memcpy events |
| --- | --- | ---: | ---: | ---: | ---: |
| water_forces | strict | 11.523 | 1950 | 2115 | 110 |
| water_forces | forced-sparse | 13.642 | 2285 | 2700 | 285 |
| caffeine_forces | strict | 11.796 | 2070 | 2235 | 110 |
| caffeine_forces | forced-sparse | 13.890 | 2405 | 2820 | 285 |

The per-profile JSON files include full command, git, hardware, software,
timing, profiler event counts, and top CPU/CUDA event tables.
