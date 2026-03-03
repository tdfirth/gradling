from __future__ import annotations

import importlib
import pkgutil

from gradling.configs.config import (
    Config,
    config_fields,
    get_config,
    get_pipeline,
    list_configs,
    list_pipelines,
    normalize_scalar_type,
    pipeline,
)

__all__ = [
    "Config",
    "config_fields",
    "get_pipeline",
    "get_config",
    "list_configs",
    "list_pipelines",
    "load_config_modules",
    "pipeline",
    "normalize_scalar_type",
]

_MODULES_LOADED = False


def load_config_modules() -> None:
    global _MODULES_LOADED
    if _MODULES_LOADED:
        return

    package = importlib.import_module("gradling.configs")
    for module in pkgutil.iter_modules(package.__path__):
        if (
            module.name.startswith("_")
            or module.name == "config"
            or module.name.startswith("test_")
            or module.name.endswith("_test")
        ):
            continue
        importlib.import_module(f"{package.__name__}.{module.name}")

    _MODULES_LOADED = True
