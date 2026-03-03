from collections.abc import Callable

from pydantic import BaseModel

from gradling.config import Config
from gradling.models.gpt import GPTConfig, sample, train


class Command[Cfg: Config](BaseModel):
    cfg: type[Cfg]
    fn: Callable[[Cfg], None]


class Model[Cfg: Config](BaseModel):
    cfg: type[Cfg]
    commands: dict[str, Command]
    description: str = ""


MODELS: dict[str, Model] = {
    "gpt": Model(
        cfg=GPTConfig,
        description="Character-level GPT model and commands.",
        commands={
            "train": Command(cfg=GPTConfig, fn=train),
            "sample": Command(cfg=GPTConfig, fn=sample),
        },
    )
}


__all__ = [
    "Command",
    "Model",
    "MODELS",
]
