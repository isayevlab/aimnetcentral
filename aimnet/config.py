import os
from collections.abc import Callable, Iterator
from importlib import import_module
from typing import Any

import yaml
from jinja2 import Template


def get_module(name: str) -> Callable:
    """
    Retrieves a module and function based on the given name.

    Args:
        name (str): The name of the module and function in the format 'module.function'.

    Returns:
        function: The function object.

    Raises:
        ImportError: If the module cannot be imported.
        AttributeError: If the function does not exist in the module.
    """
    parts = name.split(".")
    module_name, func_name = ".".join(parts[:-1]), parts[-1]
    module = import_module(module_name)
    func = getattr(module, func_name)
    return func  # type: ignore[no-any-return]


def get_init_module(name: str, args: list | None = None, kwargs: dict | None = None) -> Callable:
    """
    Get the initialized module based on the given name, arguments, and keyword arguments.

    Args:
        name (str): The name of the module.
        args (List, optional): The arguments to pass to the module constructor. Defaults to an empty list.
        kwargs (Dict, optional): The keyword arguments to pass to the module constructor. Defaults to an empty dictionary.

    Returns:
        The initialized module.

    """
    args = args if args is not None else []
    kwargs = kwargs if kwargs is not None else {}
    return get_module(name)(*args, **kwargs)  # type: ignore[no-any-return]


def load_yaml(
    config: dict[str, Any] | list | str, hyperpar: dict[str, Any] | str | None = None
) -> dict[str, Any] | list:
    """
    Load a YAML configuration file and apply optional hyperparameters.

    Args:
        config (Union[str, List, Dict]): The YAML configuration file path or a YAML object.
        hyperpar (Optional[Union[Dict, str, None]]): Optional hyperparameters to apply to the configuration.

    Returns:
        Union[List, Dict]: The loaded and processed configuration.

    Raises:
        FileNotFoundError: If a file specified in the configuration does not exist.

    """
    basedir = ""
    if isinstance(hyperpar, str):
        hyperpar = load_yaml(hyperpar)  # type: ignore[assignment]
        if not isinstance(hyperpar, dict):
            raise TypeError("Loaded hyperpar must be a dict")
    if isinstance(config, (list, dict)):
        if hyperpar:
            for d, k, v in _iter_rec_bottomup(config):
                if isinstance(v, str) and "{{" in v:
                    d[k] = Template(v).render(**hyperpar)  # type: ignore[assignment, index]
    else:
        with open(config, encoding="utf-8") as f:
            config = f.read()
        if hyperpar:
            config = Template(config).render(**hyperpar)
        config = yaml.load(config, Loader=yaml.FullLoader)  # noqa: S506
    # plugin yaml configs
    for d, k, v in _iter_rec_bottomup(config):  # type: ignore[arg-type]
        if isinstance(v, str) and any(v.endswith(x) for x in (".yml", ".yaml")):
            if not os.path.isfile(v):
                v = os.path.join(basedir, v)
            d[k] = load_yaml(v, hyperpar)  # type: ignore[assignment, index]
    return config  # type: ignore[return-value]


def _iter_rec_bottomup(
    d: dict[str, Any] | list,
) -> Iterator[tuple[dict[str, Any] | list, str | int, Any]]:
    if isinstance(d, list):
        it = enumerate(d)
    elif isinstance(d, dict):
        it = d.items()  # type: ignore[assignment]
    else:
        raise TypeError(f"Unknown type: {type(d)}")
    for k, v in it:
        if isinstance(v, (list, dict)):
            yield from _iter_rec_bottomup(v)
        yield d, k, v


def build_module(config: str | dict | list, hyperpar: str | dict | None = None) -> list | dict | Callable:
    """
    Build a module based on the provided configuration.
    Every (possibly nested) dictionary with a 'class' key will be replaced by an instance initialized with
    arguments and keywords provided as 'args' and 'kwargs' keys.

    Args:
        config (Union[str, Dict, List]): The configuration for building the module.
        hyperpar (Union[str, Dict, None], optional): The hyperparameters for the module. Defaults to None.

    Returns:
        Union[List, Dict, Callable]: The built module.

    Raises:
        AssertionError: If `hyperpar` is provided and is not a dictionary.

    """
    if isinstance(hyperpar, str):
        hyperpar = load_yaml(hyperpar)  # type: ignore[assignment]
    if hyperpar and not isinstance(hyperpar, dict):
        raise TypeError("Hyperpar must be a dictionary")
    config = load_yaml(config, hyperpar)
    for d, k, v in _iter_rec_bottomup(config):
        if isinstance(v, dict) and "class" in v:
            d[k] = get_init_module(  # type: ignore[index]
                v["class"],
                args=v.get("args", []),  # type: ignore[assignment]
                kwargs=v.get("kwargs", {}),
            )
    if "class" in config:
        config = get_init_module(  # type: ignore[assignment]
            config["class"],  # type: ignore[call-overload]
            args=config.get("args", []),  # type: ignore[union-attr]
            kwargs=config.get("kwargs", {}),  # type: ignore[union-attr]
        )
    return config  # type: ignore[assignment]


def dict_to_dotted(d, parent=""):
    if parent:
        parent += "."
    for k, v in list(d.items()):
        if isinstance(v, dict) and v:
            v = dict_to_dotted(v, parent + k)
            d.update(v)
            d.pop(k)
        else:
            d[parent + k] = d.pop(k)
    return d


def dotted_to_dict(d):
    for k, v in list(d.items()):
        if "." not in k:
            continue
        ks = k.split(".")
        ds = d
        for ksp in ks[:-1]:
            if ksp not in ds:
                ds[ksp] = {}
            ds = ds[ksp]
        ds[ks[-1]] = v
        d.pop(k)
    return d
