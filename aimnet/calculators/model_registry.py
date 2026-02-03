import logging
import os

import click
import requests
import yaml

logging.basicConfig(level=logging.INFO)


def load_model_registry(registry_file: str | None = None) -> dict[str, str]:
    registry_file = registry_file or os.path.join(os.path.dirname(__file__), "model_registry.yaml")
    with open(os.path.join(os.path.dirname(__file__), "model_registry.yaml")) as f:
        return yaml.load(f, Loader=yaml.SafeLoader)


def create_assets_dir():
    os.makedirs(os.path.join(os.path.dirname(__file__), "assets"), exist_ok=True)


def get_registry_model_path(model_name: str) -> str:
    model_registry = load_model_registry()
    create_assets_dir()
    if model_name in model_registry["aliases"]:
        model_name = model_registry["aliases"][model_name]  # type: ignore
    if model_name not in model_registry["models"]:
        raise ValueError(f"Model {model_name} not found in the registry.")
    cfg = model_registry["models"][model_name]  # type: ignore
    model_path = _maybe_download_asset(**cfg)  # type: ignore
    return model_path


def _maybe_download_asset(file: str, url: str) -> str:
    filename = os.path.join(os.path.dirname(__file__), "assets", file)
    if not os.path.exists(filename):
        print(f"Downloading {url} -> {filename}")
        with open(filename, "wb") as f:
            response = requests.get(url, timeout=60)
            f.write(response.content)
    return filename


def get_model_path(s: str) -> str:
    # direct file path
    if not os.path.isfile(s):
        s = get_registry_model_path(s)
    return s


@click.command(short_help="Clear assets directory.")
def clear_assets():
    from glob import glob

    for fil in glob(os.path.join(os.path.dirname(__file__), "assets", "*")):
        if os.path.isfile(fil):
            logging.warn(f"Removing {fil}")
            os.remove(fil)
