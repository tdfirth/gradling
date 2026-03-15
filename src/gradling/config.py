from dataclasses import Field, asdict, dataclass, field, fields, replace
from typing import cast


def runtime_field[T](default: T, doc: str = "") -> T:
    return cast(T, field(default=default, metadata={"cli": False, "doc": doc}))


@dataclass
class Config:
    @classmethod
    def from_dict(cls, d: dict):
        valid = {k: v for k, v in d.items() if k in cls.field_names()}
        return cls(**valid)

    def to_dict(self) -> dict:
        return asdict(self)

    def replace(self, **kwargs):
        return replace(self, **kwargs)

    @property
    def fields(self):
        return fields(self)

    @classmethod
    def field_names(cls) -> set[str]:
        return {f.name for f in fields(cls)}

    @classmethod
    def cli_fields(cls) -> tuple[Field, ...]:
        return tuple(f for f in fields(cls) if f.metadata.get("cli") is not False)
