from collections.abc import Callable

from pydantic import BaseModel

from gradling.config import Config
from gradling.models import aiayn, gpt2, mlp


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
            "sample": Command(cfg=aiayn.AIAYNConfig, fn=aiayn.sample),
        },
    ),
    "gpt2": Model(
        cfg=gpt2.GPT2Config,
        description="GPT2 architecture",
        commands={
            "train": Command(cfg=gpt2.GPT2Config, fn=gpt2.train),
            "data": Command(cfg=gpt2.DataConfig, fn=gpt2.data),
            "chat": Command(cfg=gpt2.ChatConfig, fn=gpt2.chat),
        },
    ),
}


__all__ = [
    "Command",
    "Model",
    "MODELS",
]
