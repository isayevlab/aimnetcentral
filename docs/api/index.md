# API Reference

This section contains the selected public API reference for AIMNet2. Lower-level kernels, tensor operators, and training internals are documented through source code and focused guides where they are intended for extension.

## Package Structure

- **[Calculators](calculators.md)** - Calculator interfaces for molecular simulations
- **[Modules](modules.md)** - Neural network modules and model components
- **[Data](data.md)** - Dataset handling and data loading utilities
- **[Config](config.md)** - Configuration and model building utilities

## Supported Public Imports

| Import path | Public surface |
| --- | --- |
| `aimnet.calculators` | `AIMNet2Calculator`, `AIMNet2ASE`, `AIMNet2Pysis`, `AIMNet2TorchSim` |
| `aimnet.data` | `DataGroup`, `SizeGroupedDataset`, `SizeGroupedSampler` |
| `aimnet.models` | `AIMNet2`, model loading helpers |
| `aimnet.modules` | Model building blocks intended for configuration and extension |
| `aimnet.config` | YAML/module construction helpers |

The `aimnet.ops`, `aimnet.nbops`, `aimnet.kernels`, `aimnet.train`, and `aimnet.cli` modules are importable but are primarily advanced or internal extension points. Their contracts may be narrower than the calculator and model APIs.

## Quick Links

### Calculators

The main entry points for using AIMNet2:

- `AIMNet2Calculator` - Core calculator for inference
- `AIMNet2ASE` - ASE calculator interface (requires `aimnet[ase]`)
- `AIMNet2Pysis` - PySisyphus calculator interface (requires `aimnet[pysis]`)
- `AIMNet2TorchSim` - TorchSim model interface (requires Python 3.12+ and `aimnet[torchsim]`)

### Command Line Interface

```bash
aimnet --help
```
