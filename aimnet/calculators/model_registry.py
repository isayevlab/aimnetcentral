import logging
import os
import shutil
import tempfile
from hashlib import sha256
from pathlib import Path
from typing import Any

import click
import requests
import yaml

logging.basicConfig(level=logging.INFO)


def load_model_registry(registry_file: str | None = None) -> dict[str, Any]:
    registry_file = registry_file or os.path.join(os.path.dirname(__file__), "model_registry.yaml")
    with open(registry_file) as f:
        return yaml.load(f, Loader=yaml.SafeLoader)


def get_cache_dir() -> str:
    """Return the model cache directory.

    ``AIMNET_CACHE_DIR`` has priority. Otherwise AIMNet uses
    ``~/.cache/aimnet``. The directory is created on demand.
    """
    cache_dir = os.environ.get("AIMNET_CACHE_DIR")
    if cache_dir is None:
        cache_dir = os.path.join(Path.home(), ".cache", "aimnet")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def create_assets_dir():
    return get_cache_dir()


def resolve_registry_model_name(model_name: str) -> str:
    model_registry = load_model_registry()
    if model_name in model_registry["aliases"]:
        model_name = model_registry["aliases"][model_name]  # type: ignore
    if model_name not in model_registry["models"]:
        raise ValueError(f"Model {model_name} not found in the registry.")
    return model_name


def get_registry_model_family(model_name: str) -> str:
    """Return the canonical family tag for a registered model name or alias."""
    model_name = resolve_registry_model_name(model_name)
    family_key, member = model_name.rsplit("_", 1)
    if not member.isdigit() or not family_key.startswith("aimnet2-"):
        raise ValueError(f"Model {model_name} does not follow the canonical registry naming convention.")
    return family_key.removeprefix("aimnet2-")


def get_registry_model_path(model_name: str) -> str:
    model_registry = load_model_registry()
    create_assets_dir()
    model_name = resolve_registry_model_name(model_name)
    cfg = model_registry["models"][model_name]  # type: ignore
    model_path = _maybe_download_asset(**cfg)  # type: ignore
    return model_path


def _hash_file(filename: str) -> str:
    h = sha256()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _validate_sha256(filename: str, expected: str | None) -> None:
    if expected is None:
        return
    digest = _hash_file(filename)
    if digest.lower() != expected.lower():
        raise ValueError(f"Checksum mismatch for {filename}: expected {expected}, got {digest}")


def _maybe_copy_bundled_asset(filename: str, file: str, sha256: str | None) -> bool:
    bundled = os.path.join(os.path.dirname(__file__), "assets", file)
    if not os.path.exists(bundled):
        return False
    _validate_sha256(bundled, sha256)
    shutil.copyfile(bundled, filename)
    return True


def _download_asset_atomic(filename: str, url: str, expected_sha256: str | None) -> None:
    dirname = os.path.dirname(filename)
    fd, tmp = tempfile.mkstemp(prefix=".download-", suffix=".tmp", dir=dirname)
    h = sha256()
    try:
        with os.fdopen(fd, "wb") as f:
            response = requests.get(url, timeout=60, stream=True)
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                h.update(chunk)
                f.write(chunk)
        if expected_sha256 is not None and h.hexdigest().lower() != expected_sha256.lower():
            raise ValueError(f"Checksum mismatch for {url}: expected {expected_sha256}, got {h.hexdigest()}")
        os.replace(tmp, filename)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _maybe_download_asset(file: str, url: str, sha256: str | None = None) -> str:
    filename = os.path.join(get_cache_dir(), file)
    if not os.path.exists(filename):
        print(f"Downloading {url} -> {filename}")
        if not _maybe_copy_bundled_asset(filename, file, sha256):
            _download_asset_atomic(filename, url, sha256)
    else:
        _validate_sha256(filename, sha256)
    return filename


def get_model_path(s: str) -> str:
    # direct file path
    if not os.path.isfile(s):
        s = get_registry_model_path(s)
    return s


@click.command(short_help="Clear assets directory.")
def clear_assets():
    from glob import glob

    for fil in glob(os.path.join(get_cache_dir(), "*")):
        if os.path.isfile(fil):
            logging.warning(f"Removing {fil}")
            os.remove(fil)
