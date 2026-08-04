# Config

Configuration and model building utilities.

`load_yaml()` and `build_module()` expand nested `.yml`/`.yaml` references by
default for trusted training and plugin configurations. Production artifact
loaders pass `allow_file_references=False` so an artifact cannot redirect
module construction to an untrusted sidecar YAML file.

## Build Module

::: aimnet.config.build_module

## Module Lookup

::: aimnet.config.get_module

::: aimnet.config.get_init_module

## YAML Loading

::: aimnet.config.load_yaml

## Dotted Dictionaries

::: aimnet.config.dict_to_dotted

::: aimnet.config.dotted_to_dict
