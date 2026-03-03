from __future__ import annotations

import inspect
import re
from collections.abc import Callable
from dataclasses import dataclass, fields, is_dataclass
from types import UnionType
from typing import Any, Union, dataclass_transform, get_args, get_origin

_REGISTRY: dict[str, type[Config]] = {}
_PIPELINE_REGISTRY: dict[type[Config], dict[str, Callable]] = {}


def _snake_case(name: str) -> str:
    first_pass = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", first_pass).lower()


@dataclass_transform()
class Config:
    __config_name__: str

    def __init_subclass__(cls, *, name: str | None = None, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Re-apply dataclass at every subclass level so inherited dataclass
        # types pick up newly declared fields in their generated __init__.
        dataclass(cls)

        key = (name or _snake_case(cls.__name__)).strip()
        if not key:
            msg = f"{cls.__name__} resolved to an empty config name"
            raise ValueError(msg)

        existing = _REGISTRY.get(key)
        if existing is not None and existing is not cls:
            msg = (
                f"Config name `{key}` is already registered by "
                f"{existing.__module__}.{existing.__name__}"
            )
            raise ValueError(msg)

        cls.__config_name__ = key
        _REGISTRY[key] = cls

    @classmethod
    def config_name(cls) -> str:
        return cls.__config_name__


def list_configs() -> list[tuple[str, type[Config]]]:
    return sorted(_REGISTRY.items(), key=lambda item: item[0])


def get_config(name: str) -> type[Config]:
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "(none)"
        msg = f"Unknown config `{name}`. Available configs: {available}"
        raise KeyError(msg)
    return _REGISTRY[name]


def pipeline[Fn: Callable[..., Any]](
    config_cls: type[Config], name: str | None = None
) -> Callable[[Fn], Fn]:
    if not issubclass(config_cls, Config):
        msg = f"`{config_cls}` must be a subclass of Config"
        raise TypeError(msg)

    def _decorator(fn: Fn) -> Fn:
        if not inspect.isfunction(fn):
            msg = "Pipelines must be plain functions declared with `def`."
            raise TypeError(msg)

        key = (name or fn.__name__).strip()

        if not key:
            msg = "Pipeline name cannot be empty"
            raise ValueError(msg)
        _PIPELINE_REGISTRY.setdefault(config_cls, {})[key] = fn
        return fn

    return _decorator


def list_pipelines(config_cls: type[Config]) -> dict[str, Callable]:
    discovered: dict[str, Callable] = {}
    for base in reversed(config_cls.mro()):
        if issubclass(base, Config):
            discovered.update(_PIPELINE_REGISTRY.get(base, {}))
    return dict(sorted(discovered.items()))


def get_pipeline(config_cls: type[Config], name: str) -> Callable:
    pipelines = list_pipelines(config_cls)
    if name not in pipelines:
        known = ", ".join(sorted(pipelines)) or "(none)"
        msg = (
            f"Unknown pipeline `{name}` for config `{config_cls.config_name()}`. "
            f"Known pipelines: {known}"
        )
        raise KeyError(msg)
    return pipelines[name]


def config_fields(config_cls: type[Config]):
    if not is_dataclass(config_cls):
        msg = f"{config_cls.__name__} is not a dataclass config"
        raise TypeError(msg)
    return [field for field in fields(config_cls) if field.init]


def normalize_scalar_type(type_hint: Any) -> Any:
    origin = get_origin(type_hint)
    if origin in (Union, UnionType):
        args = [arg for arg in get_args(type_hint) if arg is not type(None)]
        if len(args) == 1:
            return args[0]
    return type_hint
