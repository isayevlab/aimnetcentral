"""Tests for aimnet.config - configuration and module loading utilities."""

import pytest
import torch
from conftest import temp_file

from aimnet import config


class TestGetModule:
    """Tests for get_module function."""

    def test_get_module_valid_path(self):
        """Test loading a valid module path."""
        # Load torch.nn.ReLU
        relu_cls = config.get_module("torch.nn.ReLU")
        assert relu_cls is torch.nn.ReLU

    def test_get_module_function(self):
        """Test loading a function from a module."""
        # Load torch.zeros
        zeros_fn = config.get_module("torch.zeros")
        assert zeros_fn is torch.zeros

    def test_get_module_nested_path(self):
        """Test loading from nested module path."""
        # Load torch.nn.functional.relu
        relu_fn = config.get_module("torch.nn.functional.relu")
        assert relu_fn is torch.nn.functional.relu

    def test_get_module_invalid_module(self):
        """Test that invalid module raises ImportError."""
        with pytest.raises(ImportError):
            config.get_module("nonexistent_module.SomeClass")

    def test_get_module_invalid_attribute(self):
        """Test that invalid attribute raises AttributeError."""
        with pytest.raises(AttributeError):
            config.get_module("torch.nn.NonexistentClass")

    def test_get_module_aimnet_modules(self):
        """Test loading aimnet modules."""
        # Load an aimnet module
        nbops_mod = config.get_module("aimnet.nbops.set_nb_mode")
        from aimnet.nbops import set_nb_mode

        assert nbops_mod is set_nb_mode


class TestGetInitModule:
    """Tests for get_init_module function."""

    def test_get_init_module_no_args(self):
        """Test initializing module without arguments."""
        relu = config.get_init_module("torch.nn.ReLU")
        assert isinstance(relu, torch.nn.ReLU)

    def test_get_init_module_with_kwargs(self):
        """Test initializing module with keyword arguments."""
        linear = config.get_init_module("torch.nn.Linear", kwargs={"in_features": 10, "out_features": 5})
        assert isinstance(linear, torch.nn.Linear)
        assert linear.in_features == 10
        assert linear.out_features == 5

    def test_get_init_module_with_args(self):
        """Test initializing module with positional arguments."""
        linear = config.get_init_module("torch.nn.Linear", args=[10, 5])
        assert isinstance(linear, torch.nn.Linear)
        assert linear.in_features == 10
        assert linear.out_features == 5

    def test_get_init_module_with_args_and_kwargs(self):
        """Test initializing module with both args and kwargs."""
        linear = config.get_init_module("torch.nn.Linear", args=[10], kwargs={"out_features": 5, "bias": False})
        assert isinstance(linear, torch.nn.Linear)
        assert linear.in_features == 10
        assert linear.out_features == 5
        assert linear.bias is None


class TestLoadYaml:
    """Tests for load_yaml function."""

    def test_load_yaml_from_dict(self):
        """Test load_yaml with dict input (passthrough)."""
        input_dict = {"key1": "value1", "nested": {"key2": "value2"}}
        result = config.load_yaml(input_dict)
        assert result == input_dict

    def test_load_yaml_from_list(self):
        """Test load_yaml with list input (passthrough)."""
        input_list = [1, 2, {"key": "value"}]
        result = config.load_yaml(input_list)
        assert result == input_list

    def test_load_yaml_from_file(self):
        """Test load_yaml from file."""
        yaml_content = """
        model:
          type: test
          params:
            size: 100
        """
        with temp_file(suffix=".yaml", content=yaml_content) as path:
            result = config.load_yaml(str(path))
            assert "model" in result
            assert result["model"]["type"] == "test"
            assert result["model"]["params"]["size"] == 100

    def test_load_yaml_with_jinja2_template(self):
        """Test load_yaml with Jinja2 templating."""
        yaml_content = """
        model:
          hidden_size: {{ hidden_size }}
          layers: {{ num_layers }}
        """
        hyperpar = {"hidden_size": 256, "num_layers": 4}

        with temp_file(suffix=".yaml", content=yaml_content) as path:
            result = config.load_yaml(str(path), hyperpar)
            # Jinja2 renders, then YAML parses - numbers stay as numbers
            assert result["model"]["hidden_size"] == 256
            assert result["model"]["layers"] == 4

    def test_load_yaml_with_dict_hyperpar(self):
        """Test load_yaml with dict containing Jinja2 templates."""
        input_dict = {"value": "{{ x }}"}
        hyperpar = {"x": 42}
        result = config.load_yaml(input_dict, hyperpar)
        assert result["value"] == "42"

    def test_load_yaml_nested_yaml_include(self):
        """Test load_yaml with nested YAML file includes."""
        nested_content = """
        nested_key: nested_value
        """
        with temp_file(suffix=".yaml", content=nested_content) as nested_path:
            main_content = f"""
        main_key: main_value
        included: {nested_path}
        """
            with temp_file(suffix=".yaml", content=main_content) as main_path:
                result = config.load_yaml(str(main_path))
                assert result["main_key"] == "main_value"
                assert result["included"]["nested_key"] == "nested_value"

    def test_load_yaml_hyperpar_from_file(self):
        """Test load_yaml with hyperparameters loaded from file."""
        hyperpar_content = """
        learning_rate: 0.001
        batch_size: 32
        """
        config_content = """
        training:
          lr: {{ learning_rate }}
          bs: {{ batch_size }}
        """
        with (
            temp_file(suffix=".yaml", content=hyperpar_content) as hp_path,
            temp_file(suffix=".yaml", content=config_content) as cfg_path,
        ):
            result = config.load_yaml(str(cfg_path), str(hp_path))
            # YAML parses numbers as numbers
            assert result["training"]["lr"] == 0.001
            assert result["training"]["bs"] == 32

    def test_load_yaml_invalid_hyperpar_type(self):
        """Test that non-dict hyperpar from file raises TypeError."""
        hyperpar_content = "- item1\n- item2"  # YAML list, not dict

        with (
            temp_file(suffix=".yaml", content=hyperpar_content) as hp_path,
            pytest.raises(TypeError, match="Loaded hyperpar must be a dict"),
        ):
            config.load_yaml({"key": "value"}, str(hp_path))


class TestBuildModule:
    """Tests for build_module function."""

    def test_build_module_simple(self):
        """Test building a simple module from config."""
        cfg = {
            "class": "torch.nn.Linear",
            "kwargs": {"in_features": 10, "out_features": 5},
        }
        module = config.build_module(cfg)
        assert isinstance(module, torch.nn.Linear)
        assert module.in_features == 10

    def test_build_module_nested(self):
        """Test building module with nested module configs."""
        cfg = {
            "layer1": {
                "class": "torch.nn.Linear",
                "kwargs": {"in_features": 10, "out_features": 5},
            },
            "layer2": {
                "class": "torch.nn.ReLU",
            },
        }
        result = config.build_module(cfg)
        assert isinstance(result["layer1"], torch.nn.Linear)
        assert isinstance(result["layer2"], torch.nn.ReLU)

    def test_build_module_from_file(self):
        """Test building module from YAML file."""
        yaml_content = """
        class: torch.nn.Linear
        kwargs:
          in_features: 20
          out_features: 10
        """
        with temp_file(suffix=".yaml", content=yaml_content) as path:
            module = config.build_module(str(path))
            assert isinstance(module, torch.nn.Linear)
            assert module.in_features == 20

    def test_build_module_with_hyperpar(self):
        """Test building module with hyperparameters."""
        # Use integer values directly since YAML will preserve types
        cfg = {
            "class": "torch.nn.ReLU",  # ReLU has no required args
        }
        hyperpar = {"unused": 42}
        module = config.build_module(cfg, hyperpar)
        assert isinstance(module, torch.nn.ReLU)

    def test_build_module_with_hyperpar_from_file(self):
        """Test building module with hyperparameters from YAML file."""
        yaml_content = """
        class: torch.nn.Linear
        kwargs:
          in_features: {{ in_size }}
          out_features: {{ out_size }}
        """
        hyperpar = {"in_size": 32, "out_size": 16}

        with temp_file(suffix=".yaml", content=yaml_content) as path:
            module = config.build_module(str(path), hyperpar)
            assert isinstance(module, torch.nn.Linear)
            assert module.in_features == 32
            assert module.out_features == 16

    def test_build_module_invalid_hyperpar_type(self):
        """Test that non-dict hyperpar raises TypeError."""
        cfg = {"key": "value"}
        with pytest.raises(TypeError, match="Hyperpar must be a dictionary"):
            config.build_module(cfg, [1, 2, 3])


class TestDictConversions:
    """Tests for dict_to_dotted and dotted_to_dict functions."""

    def test_dict_to_dotted_simple(self):
        """Test converting nested dict to dotted notation."""
        d = {"a": {"b": {"c": 1}}}
        result = config.dict_to_dotted(d.copy())
        assert "a.b.c" in result
        assert result["a.b.c"] == 1

    def test_dict_to_dotted_multiple_keys(self):
        """Test dict_to_dotted with multiple keys."""
        d = {"x": {"y": 1}, "z": 2}
        result = config.dict_to_dotted(d.copy())
        assert "x.y" in result
        assert result["x.y"] == 1
        assert "z" in result
        assert result["z"] == 2

    def test_dict_to_dotted_empty_nested(self):
        """Test dict_to_dotted with empty nested dict."""
        d = {"a": {}}
        result = config.dict_to_dotted(d.copy())
        # Empty dict should be kept as-is (not flattened)
        assert "a" in result
        assert result["a"] == {}

    def test_dotted_to_dict_simple(self):
        """Test converting dotted notation to nested dict."""
        d = {"a.b.c": 1}
        result = config.dotted_to_dict(d.copy())
        assert "a" in result
        assert "b" in result["a"]
        assert "c" in result["a"]["b"]
        assert result["a"]["b"]["c"] == 1

    def test_dotted_to_dict_multiple_keys(self):
        """Test dotted_to_dict with multiple dotted keys."""
        d = {"x.y": 1, "x.z": 2, "w": 3}
        result = config.dotted_to_dict(d.copy())
        assert result["x"]["y"] == 1
        assert result["x"]["z"] == 2
        assert result["w"] == 3

    def test_dotted_to_dict_no_dots(self):
        """Test dotted_to_dict with no dotted keys."""
        d = {"a": 1, "b": 2}
        result = config.dotted_to_dict(d.copy())
        assert result == {"a": 1, "b": 2}

    def test_roundtrip_conversion(self):
        """Test that dict_to_dotted and dotted_to_dict are inverses."""
        original = {"a": {"b": 1, "c": 2}, "d": 3}

        # Convert to dotted and back
        dotted = config.dict_to_dotted(original.copy())
        restored = config.dotted_to_dict(dotted.copy())

        # Should get back original structure
        assert "a" in restored
        assert restored["a"]["b"] == 1
        assert restored["a"]["c"] == 2
        assert restored["d"] == 3


class TestIterRecBottomup:
    """Tests for _iter_rec_bottomup helper function."""

    def test_iter_dict(self):
        """Test iteration over dict."""
        d = {"a": 1, "b": 2}
        items = list(config._iter_rec_bottomup(d))

        # Should yield (container, key, value) tuples
        assert len(items) == 2
        keys_found = {item[1] for item in items}
        assert keys_found == {"a", "b"}

    def test_iter_list(self):
        """Test iteration over list."""
        lst = [1, 2, 3]
        items = list(config._iter_rec_bottomup(lst))

        assert len(items) == 3
        indices_found = {item[1] for item in items}
        assert indices_found == {0, 1, 2}

    def test_iter_nested(self):
        """Test iteration over nested structure."""
        d = {"a": {"b": 1}}
        items = list(config._iter_rec_bottomup(d))

        # Should yield bottom-up: first {"b": 1}, then {"a": {...}}
        assert len(items) == 2
        # First item should be the inner dict's item
        assert items[0][1] == "b"
        assert items[0][2] == 1

    def test_iter_invalid_type(self):
        """Test that invalid type raises TypeError."""
        with pytest.raises(TypeError, match="Unknown type"):
            list(config._iter_rec_bottomup("string"))
