"""Convert legacy JIT-compiled models to new state dict format.

This module provides CLI interface for converting legacy .jpt TorchScript
models to the new .pt format with metadata.

For programmatic use, import load_v1_model from aimnet.models.utils:
    from aimnet.models.utils import load_v1_model
    model, metadata = load_v1_model("model.jpt", "config.yaml", "model_new.pt")
"""

import click

from aimnet.models.utils import load_v1_model


@click.command()
@click.argument("jpt", type=click.Path(exists=True))
@click.argument("yaml_config", type=click.Path(exists=True))
@click.argument("output", type=str)
def convert_legacy_jpt(jpt: str, yaml_config: str, output: str):
    """Convert legacy JIT model to new state dict format.

    JPT: Path to the input JIT-compiled model file.
    YAML_CONFIG: Path to the model YAML configuration file.
    OUTPUT: Path to the output .pt file.

    Example:
        aimnet convert model.jpt config.yaml model_new.pt
    """
    load_v1_model(jpt, yaml_config, output)
