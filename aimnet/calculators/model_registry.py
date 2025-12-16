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
    if os.path.isfile(s):
        print("Found model file:", s)
    else:
        s = get_registry_model_path(s)
    return s


def get_model_definition_path(model_name: str) -> str:
    """Get the YAML definition file path for a model name.

    This maps model names to their architecture definition YAML files,
    which are needed for torch.compile (requires un-jitted model).

    Args:
        model_name: Model name or alias from the registry

    Returns:
        Path to the YAML model definition file
    """
    model_registry = load_model_registry()

    # Resolve alias first
    if model_name in model_registry.get("aliases", {}):
        model_name = model_registry["aliases"][model_name]

    # Determine which YAML definition to use based on model name
    # Models with D3 dispersion use aimnet2_dftd3_wb97m.yaml
    # Models without D3 (like NSE) use aimnet2.yaml
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")

    if "nse" in model_name.lower():
        # NSE models don't have D3
        yaml_file = "aimnet2.yaml"
    elif "d3" in model_name.lower() or "pd" in model_name.lower():
        # D3 and Pd models include DFTD3
        yaml_file = "aimnet2_dftd3_wb97m.yaml"
    else:
        # Default to D3 version for standard aimnet2 models
        yaml_file = "aimnet2_dftd3_wb97m.yaml"

    return os.path.join(models_dir, yaml_file)


@click.command(short_help="Clear assets directory.")
def clear_assets():
    from glob import glob

    for fil in glob(os.path.join(os.path.dirname(__file__), "assets", "*")):
        if os.path.isfile(fil):
            logging.warn(f"Removing {fil}")
            os.remove(fil)
