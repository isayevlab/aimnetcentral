import sys

import click

from .calculators.model_registry import clear_assets


@click.group()
def cli():
    """AIMNet2 command line tool"""


# Always available commands
cli.add_command(clear_assets, name="clear_model_cache")


# Register convert command (doesn't need heavy training dependencies)
try:
    from .models.convert import convert_legacy_jpt

    cli.add_command(convert_legacy_jpt, name="convert")
except ImportError:

    @cli.command(name="convert")
    def convert_stub():
        """Convert legacy JIT model to new format (requires aimnet[train])"""
        click.echo(
            "Training dependencies not installed.\nInstall with: pip install aimnet[train]",
            err=True,
        )
        sys.exit(1)


# Try to lazily register training commands (requires ignite, etc.)
try:
    from .train.calc_sae import calc_sae
    from .train.export_model import export_model
    from .train.train import train

    cli.add_command(train, name="train")
    cli.add_command(export_model, name="export")
    cli.add_command(calc_sae, name="calc_sae")
except ImportError:
    # If training dependencies are not available, register stub commands with helpful error messages

    @cli.command(name="train")
    def train_stub():
        """Train AIMNet2 models (requires aimnet[train])"""
        click.echo(
            "Training dependencies not installed.\nInstall with: pip install aimnet[train]",
            err=True,
        )
        sys.exit(1)

    @cli.command(name="export")
    def export_stub():
        """Export trained model to distributable format (requires aimnet[train])"""
        click.echo(
            "Training dependencies not installed.\nInstall with: pip install aimnet[train]",
            err=True,
        )
        sys.exit(1)

    @cli.command(name="calc_sae")
    def calc_sae_stub():
        """Calculate SAE (requires aimnet[train])"""
        click.echo(
            "Training dependencies not installed.\nInstall with: pip install aimnet[train]",
            err=True,
        )
        sys.exit(1)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    cli()
