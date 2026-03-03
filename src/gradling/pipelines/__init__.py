from __future__ import annotations

import importlib
import pkgutil

_MODULES_LOADED = False


def load_pipeline_modules() -> None:
    global _MODULES_LOADED
    if _MODULES_LOADED:
        return

    package = importlib.import_module("gradling.pipelines")
    for module in pkgutil.iter_modules(package.__path__):
        if (
            module.name.startswith("_")
            or module.name.startswith("test_")
            or module.name.endswith("_test")
        ):
            continue
        importlib.import_module(f"{package.__name__}.{module.name}")

    _MODULES_LOADED = True
