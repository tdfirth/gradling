from collections.abc import Callable

from pydantic import BaseModel

from gradling.config import Config
from gradling.models import gpt, mlp


class Command[Cfg: Config](BaseModel):
    cfg: type[Cfg]
    fn: Callable[[Cfg], None]


class Model[Cfg: Config](BaseModel):
    cfg: type[Cfg]
    commands: dict[str, Command]
    description: str = ""


MODELS: dict[str, Model] = {
    "gpt": Model(
        cfg=gpt.GPTConfig,
        description="Character-level GPT model and commands.",
        commands={
            "train": Command(cfg=gpt.GPTConfig, fn=gpt.train),
            "sample": Command(cfg=gpt.GPTConfig, fn=gpt.sample),
        },
    ),
    "mlp": Model(
        cfg=mlp.MLPConfig,
        description="A simple MLP in vanilla jax.",
        commands={"train": Command(cfg=mlp.MLPConfig, fn=mlp.train)},
    ),
}


__all__ = [
    "Command",
    "Model",
    "MODELS",
]
