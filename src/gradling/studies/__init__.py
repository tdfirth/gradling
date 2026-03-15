from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel

from gradling.config import Config
from gradling.metrics import RunIdentity, SinkFactory, log_only, with_wandb
from gradling.studies import aiayn, mlp


class Command[Cfg: Config](BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    cfg: type[Cfg]
    fn: Callable[..., None]
    sinks: SinkFactory = log_only


class Study[Cfg: Config](BaseModel):
    cfg: type[Cfg]
    commands: dict[str, Command]
    description: str = ""


STUDIES: dict[str, Study] = {
    "mlp": Study(
        cfg=mlp.MLPConfig,
        description="A simple MLP in vanilla jax",
        commands={
            "train": Command(cfg=mlp.MLPConfig, fn=mlp.train, sinks=with_wandb),
        },
    ),
    "aiayn": Study(
        cfg=aiayn.AIAYNConfig,
        description="Decoder only transformer based on Attention is All You Need",
        commands={
            "train": Command(cfg=aiayn.AIAYNConfig, fn=aiayn.train, sinks=with_wandb),
            "data": Command(cfg=aiayn.DataConfig, fn=aiayn.data),
            "chat": Command(cfg=aiayn.ChatConfig, fn=aiayn.chat),
        },
    ),
}


__all__ = [
    "Command",
    "RunIdentity",
    "Study",
    "STUDIES",
]
