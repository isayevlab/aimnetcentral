# API Reference

This section contains the complete API reference for AIMNet2.

## Package Structure

- **[Calculators](calculators.md)** - Calculator interfaces for molecular simulations
- **[Modules](modules.md)** - Neural network modules and model components
- **[Data](data.md)** - Dataset handling and data loading utilities
- **[Config](config.md)** - Configuration and model building utilities

## Quick Links

### Calculators

The main entry points for using AIMNet2:

- `AIMNet2Calculator` - Core calculator for inference
- `AIMNet2ASE` - ASE calculator interface (requires `aimnet[ase]`)
- `AIMNet2Pysis` - PySisyphus calculator interface (requires `aimnet[pysis]`)

### Command Line Interface

```bash
aimnet --help
```
