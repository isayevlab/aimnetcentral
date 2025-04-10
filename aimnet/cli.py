import click

from .calculators.model_registry import clear_assets
from .train.calc_sae import calc_sae
from .train.pt2jpt import jitcompile
from .train.train import train


@click.group()
def cli():
    """AIMNet2 command line tool"""


cli.add_command(train, name="train")
cli.add_command(jitcompile, name="jitcompile")
cli.add_command(calc_sae, name="calc_sae")
cli.add_command(clear_assets, name="clear_model_cache")


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    cli()
