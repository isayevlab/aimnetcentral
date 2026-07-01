"""Tests for the serialization ABI: import paths embedded in model configs and artifacts.

Released AIMNet2 model files (.pt) embed a YAML document (``model_yaml``) with
fully qualified class paths such as ``aimnet.models.aimnet2.AIMNet2``,
``aimnet.modules.Output``, and ``torch.nn.GELU``. At load time,
``aimnet.config.build_module`` resolves these strings with
``importlib.import_module`` (aimnet/config.py), and the Hugging Face loader
allowlists the ``aimnet.`` prefix (aimnet/calculators/hf_hub.py). Every dotted
path reachable this way is therefore a serialization ABI shared with all
released checkpoints: renaming or moving one of these classes breaks every
published .pt file that references it.

These tests pin that contract:

1. Every import path in YAML files shipped inside the ``aimnet`` package
   resolves through the same mechanism the loader uses (``config.get_module``).
2. An explicit frozen list of names embedded in released artifacts resolves.
3. The model YAML embedded in each bundled .pt asset resolves offline.
"""

import inspect
from pathlib import Path

import pytest
import torch
import yaml

import aimnet
from aimnet import config
from aimnet.calculators.hf_hub import _validate_model_yaml

_PACKAGE_ROOT = Path(aimnet.__file__).parent
_ASSETS_DIR = _PACKAGE_ROOT / "calculators" / "assets"

# YAML keys whose string values are dotted import paths resolved at runtime:
# - "class": build_module/get_init_module (aimnet/config.py)
# - "activation_fn", "weight_init_fn": MLP kwargs (aimnet/modules/core.py)
# - "fn": loss component functions (aimnet/train/loss.py)
# - "trainer", "evaluator": training loop entry points (aimnet/train/utils.py)
_IMPORT_PATH_KEYS = frozenset({"class", "activation_fn", "weight_init_fn", "fn", "trainer", "evaluator"})

# Top-level packages provided by the aimnet[train] extra. Import paths that
# require them are resolved when the extra is installed and skipped otherwise.
_OPTIONAL_EXTRA_ROOTS = frozenset({"ignite", "omegaconf", "wandb"})

# ---------------------------------------------------------------------------
# FROZEN SERIALIZATION ABI
#
# Every name below is embedded (or expected by loader machinery) in released
# model artifacts and therefore FROZEN. Renaming or moving any of these classes
# bricks every published .pt checkpoint that references it, because loading
# resolves these strings by import at runtime. Removing a name from this list
# is a conscious ABI-break decision and requires a deprecation alias at the old
# import path plus a model-format migration for all released artifacts.
# ---------------------------------------------------------------------------
_FROZEN_CLASS_PATHS = (
    # Model class. Released checkpoints reference BOTH spellings: the barrel
    # path (aimnet2_rxn_0.pt) and the fully qualified submodule path (all
    # other bundled artifacts), so both must keep resolving.
    "aimnet.models.AIMNet2",
    "aimnet.models.aimnet2.AIMNet2",
    # Output modules named in the model_yaml embedded in released checkpoints.
    "aimnet.modules.Output",
    "aimnet.modules.AtomicShift",
    "aimnet.modules.AtomicSum",
    "aimnet.modules.SRCoulomb",
    "aimnet.modules.Dipole",
    "aimnet.modules.Quadrupole",
    # Referenced by the shipped model definition YAMLs (aimnet/models/*.yaml)
    # and string-matched by the loader machinery: aimnet/models/utils.py
    # rewrites LRCoulomb/DFTD3 to SRCoulomb and detects D3TS by class name;
    # hf_hub._find_srcoulomb_params matches on the SRCoulomb suffix.
    "aimnet.modules.LRCoulomb",
    "aimnet.modules.DFTD3",
    "aimnet.modules.D3TS",
    # Public configuration building blocks (documented in docs/api as intended
    # for configuration and extension); external configs may reference them.
    "aimnet.modules.AEVSV",
    "aimnet.modules.Embedding",
    # Activation embedded via 'activation_fn' in every released model_yaml.
    "torch.nn.GELU",
)

# Frozen callables that are factories rather than classes.
# aimnet.modules.MLP is a function that builds an nn.Sequential, but it is a
# public config building block referenced by the same dotted-path mechanism.
_FROZEN_FACTORY_PATHS = ("aimnet.modules.MLP",)


def _iter_import_paths(obj):
    """Recursively yield (key, dotted_path) pairs from a parsed YAML tree."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in _IMPORT_PATH_KEYS and isinstance(value, str) and "." in value:
                yield key, value
            yield from _iter_import_paths(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from _iter_import_paths(item)


def _collect_yaml_import_paths():
    """Collect (relative_file, key, dotted_path) from every YAML in the package."""
    entries = set()
    for path in sorted(_PACKAGE_ROOT.rglob("*.yaml")) + sorted(_PACKAGE_ROOT.rglob("*.yml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        rel = path.relative_to(_PACKAGE_ROOT).as_posix()
        for key, dotted in _iter_import_paths(data):
            entries.add((rel, key, dotted))
    return sorted(entries)


_YAML_IMPORT_PATHS = _collect_yaml_import_paths()


def _resolve_or_skip_optional(dotted):
    """Resolve a dotted path via config.get_module, skipping missing optional extras."""
    try:
        return config.get_module(dotted)
    except ModuleNotFoundError as exc:
        missing_root = (exc.name or "").split(".")[0]
        if missing_root in _OPTIONAL_EXTRA_ROOTS:
            pytest.skip(f"'{dotted}' requires optional extra package '{missing_root}' (install aimnet[train])")
        raise


class TestShippedYamlImportPaths:
    """Every import path in YAML files shipped inside the package must resolve."""

    def test_collector_found_shipped_configs(self):
        """Guard against the glob silently matching nothing."""
        files = {rel for rel, _, _ in _YAML_IMPORT_PATHS}
        assert "models/aimnet2.yaml" in files
        assert "models/aimnet2_dftd3_wb97m.yaml" in files
        assert "models/aimnet2_rxn.yaml" in files
        assert "train/default_train.yaml" in files

    def test_model_registry_has_no_import_paths(self):
        """model_registry.yaml is download metadata only (files, URLs, aliases)."""
        registry_path = _PACKAGE_ROOT / "calculators" / "model_registry.yaml"
        registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
        assert list(_iter_import_paths(registry)) == []

    @pytest.mark.parametrize(
        ("yaml_file", "key", "dotted"),
        _YAML_IMPORT_PATHS,
        ids=[f"{rel}:{key}={dotted}" for rel, key, dotted in _YAML_IMPORT_PATHS],
    )
    def test_import_path_resolves(self, yaml_file, key, dotted):
        """Each dotted path resolves via the loader's own mechanism (get_module)."""
        obj = _resolve_or_skip_optional(dotted)
        assert callable(obj), f"{yaml_file}: '{key}: {dotted}' resolved to non-callable {type(obj)!r}"


class TestFrozenSerializationAbi:
    """The frozen list of names embedded in released model artifacts must resolve."""

    @pytest.mark.parametrize("dotted", _FROZEN_CLASS_PATHS)
    def test_frozen_class_path_resolves(self, dotted):
        """Each frozen name imports and is a class."""
        obj = config.get_module(dotted)
        assert inspect.isclass(obj), f"Frozen ABI path '{dotted}' resolved to non-class {type(obj)!r}"

    @pytest.mark.parametrize("dotted", _FROZEN_FACTORY_PATHS)
    def test_frozen_factory_path_resolves(self, dotted):
        """Each frozen factory imports and is callable."""
        obj = config.get_module(dotted)
        assert callable(obj), f"Frozen ABI path '{dotted}' resolved to non-callable {type(obj)!r}"

    def test_frozen_list_covers_shipped_model_yamls(self):
        """Every import path in aimnet/models/*.yaml must appear in the frozen list."""
        frozen = set(_FROZEN_CLASS_PATHS) | set(_FROZEN_FACTORY_PATHS)
        shipped = {dotted for rel, _, dotted in _YAML_IMPORT_PATHS if rel.startswith("models/")}
        missing = shipped - frozen
        assert not missing, (
            f"Shipped model YAMLs reference paths not in the frozen ABI list: {sorted(missing)}. "
            "Add them to _FROZEN_CLASS_PATHS (they become part of the serialization ABI)."
        )


_ASSET_FILES = sorted(_ASSETS_DIR.glob("*.pt")) if _ASSETS_DIR.is_dir() else []


class TestBundledAssetEmbeddedYaml:
    """The model YAML embedded in each bundled .pt asset must resolve offline."""

    @pytest.mark.parametrize("asset", _ASSET_FILES, ids=[p.name for p in _ASSET_FILES])
    def test_embedded_model_yaml_resolves(self, asset):
        """Load bundled .pt with weights_only=True (as load_model does) and resolve its YAML."""
        # weights_only=True is the secure code path load_model() tries first
        # for the v2 .pt format (aimnet/models/base.py). No network involved.
        data = torch.load(asset, map_location="cpu", weights_only=True)
        assert isinstance(data, dict), f"{asset.name}: expected v2 state-dict format, got {type(data)!r}"
        assert "model_yaml" in data, f"{asset.name}: missing embedded 'model_yaml'"

        model_yaml = data["model_yaml"]
        # Released YAML must stay within the HF loader allowlist (hf_hub.py).
        _validate_model_yaml(model_yaml)

        entries = list(_iter_import_paths(yaml.safe_load(model_yaml)))
        assert any(key == "class" for key, _ in entries), f"{asset.name}: no 'class' entries in embedded YAML"
        for key, dotted in entries:
            obj = config.get_module(dotted)
            assert callable(obj), f"{asset.name}: '{key}: {dotted}' resolved to non-callable {type(obj)!r}"

    @pytest.mark.parametrize("asset", _ASSET_FILES, ids=[p.name for p in _ASSET_FILES])
    def test_embedded_class_paths_are_frozen(self, asset):
        """Every class path embedded in a bundled artifact must be on the frozen list."""
        data = torch.load(asset, map_location="cpu", weights_only=True)
        embedded = {dotted for key, dotted in _iter_import_paths(yaml.safe_load(data["model_yaml"]))}
        frozen = set(_FROZEN_CLASS_PATHS) | set(_FROZEN_FACTORY_PATHS)
        missing = embedded - frozen
        assert not missing, (
            f"{asset.name} embeds paths not in the frozen ABI list: {sorted(missing)}. "
            "Add them to _FROZEN_CLASS_PATHS (they become part of the serialization ABI)."
        )
