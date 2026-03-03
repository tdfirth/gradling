from __future__ import annotations

import re
from typing import Any, ClassVar

import jax
from pydantic import BaseModel, ConfigDict


def _snake_case(name: str) -> str:
    first_pass = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", first_pass).lower()


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    __config_name__: ClassVar[str]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls.__config_name__ = _snake_case(cls.__name__)
        jax.tree_util.register_pytree_node_class(cls)

    @classmethod
    def config_name(cls) -> str:
        return cls.__config_name__

    def tree_flatten(self):
        names = tuple(type(self).model_fields.keys())
        values = tuple(getattr(self, name) for name in names)
        return values, names

    @classmethod
    def tree_unflatten(cls, names, values):
        data = dict(zip(names, values))
        return cls.model_validate(data)
