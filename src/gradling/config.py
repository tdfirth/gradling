from __future__ import annotations

import re
from typing import Any, ClassVar

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

    @classmethod
    def config_name(cls) -> str:
        return cls.__config_name__

    def __hash__(self) -> int:
        return hash(tuple(self.model_dump().items()))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.model_dump() == other.model_dump()
