# Offline and HPC Installation

AIMNet2 model weights are not bundled with the package. They are downloaded on first use from `https://storage.googleapis.com/aimnetcentral/` into a local cache and verified against SHA-256 digests pinned in the model registry.

## Pre-seeding the cache (air-gapped compute nodes)

On a node with internet access (e.g. a login node):

```bash
aimnet download aimnet2          # one model (registry name or alias)
aimnet download --all            # every registered model (~212 MB)
```

Then point compute jobs at the cache:

```bash
export AIMNET_CACHE_DIR=/shared/software/aimnet-cache   # default: ~/.cache/aimnet
```

The cache is read-only-safe: files are verified by checksum on load and are never rewritten unless the checksum fails. A shared, read-only `AIMNET_CACHE_DIR` serves any number of jobs and users.

## Warp kernel JIT cache

The CUDA AEV kernel is JIT-compiled by NVIDIA Warp at first use and cached (default: `~/.cache/warp`). On quota-limited or read-only home directories, redirect it:

```bash
export WARP_CACHE_PATH=/tmp/$USER/warp-cache
```

First kernel launch on a fresh node pays a one-time compile cost; subsequent runs start warm.

## Diagnostics

`aimnet info` prints the resolved cache location, torch/warp versions, CUDA availability, and which AEV kernel path (Warp CUDA kernel vs. pure-torch fallback) will run.
