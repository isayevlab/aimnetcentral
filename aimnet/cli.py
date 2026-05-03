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


def _missing_train_deps(exc: ImportError):
    click.echo(
        f"Training dependencies not installed or incomplete: {exc}\nInstall with: pip install aimnet[train]",
        err=True,
    )
    sys.exit(1)


@cli.command(name="train")
@click.option(
    "--config",
    type=click.Path(exists=True),
    default=None,
    multiple=True,
    help="Path to the extra configuration file (overrides values, can be specified multiple times).",
)
@click.option(
    "--model",
    type=click.Path(exists=True),
    default=None,
    help="Path to the model definition file. Defaults to the bundled AIMNet2 model YAML.",
)
@click.option("--load", type=click.Path(exists=True), default=None, help="Path to the model weights to load.")
@click.option("--save", type=click.Path(), default=None, help="Path to save the model weights.")
@click.option("--no-default-config", is_flag=True, default=False)
@click.argument("args", type=str, nargs=-1)
def train_cmd(config, model, load=None, save=None, args=None, no_default_config=False):
    """Train AIMNet2 model."""
    try:
        from .train.train import _default_model, train
    except ImportError as exc:
        _missing_train_deps(exc)
    train.callback(config, model or _default_model, load, save, args, no_default_config)  # type: ignore[union-attr]


@cli.command(name="export")
@click.argument("weights", type=click.Path(exists=True))
@click.argument("output", type=str)
@click.option("--model", "-m", type=click.Path(exists=True), required=True, help="Path to model definition YAML file")
@click.option("--sae", "-s", type=click.Path(exists=True), required=True, help="Path to the SAE YAML file")
@click.option(
    "--needs-coulomb/--no-coulomb", default=None, help="Override Coulomb detection. Default: auto-detect from YAML"
)
@click.option(
    "--needs-dispersion/--no-dispersion",
    default=None,
    help="Override dispersion detection. Default: auto-detect from YAML",
)
def export_cmd(weights, output, model, sae, needs_coulomb, needs_dispersion):
    """Export trained model to distributable state dict format."""
    try:
        from .train.export_model import export_model
    except ImportError as exc:
        _missing_train_deps(exc)
    export_model.callback(weights, output, model, sae, needs_coulomb, needs_dispersion)  # type: ignore[union-attr]


@cli.command(name="calc_sae")
@click.option("--samples", type=int, default=100000, help="Max number of samples to consider.")
@click.argument("ds", type=str)
@click.argument("output", type=str)
def calc_sae_cmd(ds, output, samples=100000):
    """Calculate energy SAE for a dataset."""
    try:
        from .train.calc_sae import calc_sae
    except ImportError as exc:
        _missing_train_deps(exc)
    calc_sae.callback(ds, output, samples)  # type: ignore[union-attr]


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    cli()
