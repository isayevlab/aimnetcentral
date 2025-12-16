import sys

import click

from .calculators.model_registry import clear_assets


@click.group()
def cli():
    """AIMNet2 command line tool"""


# Always available commands
cli.add_command(clear_assets, name="clear_model_cache")


# Try to lazily register training commands
try:
    from .train.calc_sae import calc_sae
    from .train.pt2jpt import jitcompile
    from .train.train import train

    cli.add_command(train, name="train")
    cli.add_command(jitcompile, name="jitcompile")
    cli.add_command(calc_sae, name="calc_sae")
except ImportError:
    # If training dependencies are not available, register stub commands with helpful error messages

    @cli.command(name="train")
    def train_stub():
        """Train AIMNet2 models (requires aimnet[train])"""
        click.echo(
            "❌ Training dependencies not installed.\nInstall with: pip install aimnet[train]",
            err=True,
        )
        sys.exit(1)

    @cli.command(name="jitcompile")
    def jitcompile_stub():
        """JIT compile models (requires aimnet[train])"""
        click.echo(
            "❌ Training dependencies not installed.\nInstall with: pip install aimnet[train]",
            err=True,
        )
        sys.exit(1)

    @cli.command(name="calc_sae")
    def calc_sae_stub():
        """Calculate SAE (requires aimnet[train])"""
        click.echo(
            "❌ Training dependencies not installed.\nInstall with: pip install aimnet[train]",
            err=True,
        )
        sys.exit(1)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    cli()
