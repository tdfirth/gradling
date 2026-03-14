from collections.abc import Callable

from pydantic import BaseModel

from gradling.config import Config
from gradling.models import aiayn, mlp


class Command[Cfg: Config](BaseModel):
    cfg: type[Cfg]
    fn: Callable[[Cfg], None]


class Model[Cfg: Config](BaseModel):
    cfg: type[Cfg]
    commands: dict[str, Command]
    description: str = ""


MODELS: dict[str, Model] = {
    "mlp": Model(
        cfg=mlp.MLPConfig,
        description="A simple MLP in vanilla jax",
        commands={"train": Command(cfg=mlp.MLPConfig, fn=mlp.train)},
    ),
    "aiayn": Model(
        cfg=aiayn.AIAYNConfig,
        description="Decoder only transformer based on Attention is All You Need",
        commands={
            "train": Command(cfg=aiayn.AIAYNConfig, fn=aiayn.train),
            "data": Command(cfg=aiayn.DataConfig, fn=aiayn.data),
            "chat": Command(cfg=aiayn.ChatConfig, fn=aiayn.chat),
        },
    ),
}


__all__ = [
    "Command",
    "Model",
    "MODELS",
]
