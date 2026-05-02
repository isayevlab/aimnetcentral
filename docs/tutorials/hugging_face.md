# Loading Models from Hugging Face

AIMNet2 models are published on [Hugging Face Hub](https://huggingface.co/isayevlab) as safetensors checkpoints alongside a `config.json` that encodes the full model architecture and metadata. This lets you load any model with a single line — weights are downloaded and cached automatically.

## Installation

The HF integration is an optional dependency group. It adds `huggingface_hub` and `safetensors` but does not affect the default `pip install aimnet` path.

```bash
pip install "aimnet[hf]"
```

## Basic Usage

Pass a Hugging Face repo ID (`org/name`) directly to `AIMNet2Calculator`:

```python
from aimnet.calculators import AIMNet2Calculator

calc = AIMNet2Calculator("isayevlab/aimnet2-wb97m-d3")
```

The first call downloads the model weights to the HF cache directory (`~/.cache/huggingface/hub/` by default) and reuses them on subsequent runs.

## Available Models

| HF Repo | Equivalent Alias | DFT Functional | Best For |
| --- | --- | --- | --- |
| `isayevlab/aimnet2-wb97m-d3` | `aimnet2` | wB97M-D3 | General organic chemistry |
| `isayevlab/aimnet2-2025` | `aimnet2-2025` | B97-3c | Recommended for intermolecular interactions |
| `isayevlab/aimnet2-nse` | `aimnet2-nse` | wB97M-D3 | Open-shell systems / radicals |
| `isayevlab/aimnet2-pd` | `aimnet2-pd` | B97-3c/CPCM | Palladium chemistry |
| `isayevlab/aimnet2-rxn` | `aimnet2-rxn` | wB97M-D3 | Reactive chemistry / TS / IRC |

Each repo contains four ensemble members (`ensemble_0.safetensors` – `ensemble_3.safetensors`). Member 0 is loaded by default.

## Options

### Ensemble member

Each model family has four ensemble members trained from different random seeds. You can load a specific one for uncertainty estimation:

```python
calc_0 = AIMNet2Calculator("isayevlab/aimnet2-wb97m-d3", ensemble_member=0)
calc_1 = AIMNet2Calculator("isayevlab/aimnet2-wb97m-d3", ensemble_member=1)
calc_2 = AIMNet2Calculator("isayevlab/aimnet2-wb97m-d3", ensemble_member=2)
calc_3 = AIMNet2Calculator("isayevlab/aimnet2-wb97m-d3", ensemble_member=3)
```

### Pinned revision

Pin to a specific tag or branch for reproducible results:

```python
calc = AIMNet2Calculator("isayevlab/aimnet2-wb97m-d3", revision="v1.0")
```

### Private repos

Pass a HF access token for private or gated repos:

```python
calc = AIMNet2Calculator("myorg/private-model", token="hf_...")
```

Alternatively set the `HF_TOKEN` environment variable or log in with `huggingface-cli login`.

### Local directory

If you have a local directory with the same layout as an HF repo (`config.json` + `ensemble_N.safetensors`), pass the path directly:

```python
calc = AIMNet2Calculator("/path/to/local/repo")
```

## Mixing HF and Registry Models

HF loading and the built-in GCS registry are independent. You can use both in the same script without any conflicts — HF extras are only imported when an HF repo ID is detected:

```python
# Loads from Hugging Face (requires aimnet[hf]):
calc_hf = AIMNet2Calculator("isayevlab/aimnet2-wb97m-d3")

# Loads from GCS registry (no extra deps needed):
calc_registry = AIMNet2Calculator("aimnet2")
```

## Expected Repo Layout

When hosting your own models on HF Hub, the calculator expects:

```
config.json                 # architecture + metadata (see below)
ensemble_0.safetensors      # weights for member 0
ensemble_1.safetensors      # weights for member 1
ensemble_2.safetensors      # weights for member 2
ensemble_3.safetensors      # weights for member 3
```

### config.json fields

| Field | Required | Description |
| --- | --- | --- |
| `cutoff` | yes | Neighbor list cutoff in Å |
| `model_yaml` | yes\* | YAML string of the full model architecture |
| `needs_coulomb` | no | Whether external Coulomb correction is needed |
| `needs_dispersion` | no | Whether external D3 dispersion is needed |
| `coulomb_mode` | no | `"none"`, `"sr_embedded"`, or `"full_embedded"` |
| `implemented_species` | no | List of supported atomic numbers |

\*If `model_yaml` is absent, the loader falls back to the GCS registry using `member_names` (a list of registry keys) — useful for family-level uploads. A warning is issued in this case.

!!! note "Security"

    All `class:` entries in `model_yaml` are validated against an allowlist of `aimnet.*`
    classes before `build_module()` is called. Configs referencing arbitrary Python classes
    are rejected to prevent code execution via crafted `config.json` files.

## Interactive Demo

Try AIMNet2 directly in your browser without installing anything:

[Launch Demo on Hugging Face Spaces](https://huggingface.co/spaces/isayevlab/aimnet2-demo)
