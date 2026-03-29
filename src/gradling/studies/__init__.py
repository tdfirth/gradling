from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass

from gradling.config import Config
from gradling.metrics import RunIdentity, SinkFactory, log_only, with_wandb
from gradling.studies.aiayn.config import AIAYNConfig
from gradling.studies.aiayn.config import ChatConfig as AIAYNChatConfig
from gradling.studies.gpt2.config import (
    ChatConfig as GPTChatConfig,
)
from gradling.studies.gpt2.config import (
    EvalConfig,
    GPTConfig,
)
from gradling.studies.mlp.config import MLPConfig


def _lazy(module: str, attr: str) -> Callable[..., None]:
    def wrapper(*args, **kwargs):
        mod = importlib.import_module(module)
        return getattr(mod, attr)(*args, **kwargs)

    return wrapper


@dataclass
class Command[Cfg: Config]:
    model_config = {"arbitrary_types_allowed": True}
    cfg: type[Cfg]
    fn: Callable[..., None]
    sinks: SinkFactory = log_only


@dataclass
class Study[Cfg: Config]:
    cfg: type[Cfg]
    commands: dict[str, Command]
    description: str = ""


STUDIES: dict[str, Study] = {
    "mlp": Study(
        cfg=MLPConfig,
        description="A simple MLP in vanilla jax",
        commands={
            "train": Command(
                cfg=MLPConfig,
                fn=_lazy("gradling.studies.mlp.train", "train"),
                sinks=with_wandb,
            ),
        },
    ),
    "aiayn": Study(
        cfg=AIAYNConfig,
        description="Decoder only transformer based on Attention is All You Need",
        commands={
            "train": Command(
                cfg=AIAYNConfig,
                fn=_lazy("gradling.studies.aiayn.train", "train"),
                sinks=with_wandb,
            ),
            "chat": Command(
                cfg=AIAYNChatConfig,
                fn=_lazy("gradling.studies.aiayn.chat", "chat"),
            ),
        },
    ),
    "gpt2": Study(
        cfg=GPTConfig,
        description="Reimplementation of GPT-2 124M",
        commands={
            "train": Command(
                cfg=GPTConfig,
                fn=_lazy("gradling.studies.gpt2.train", "train"),
                sinks=with_wandb,
            ),
            "chat": Command(
                cfg=GPTChatConfig,
                fn=_lazy("gradling.studies.gpt2.chat", "chat"),
            ),
            "eval": Command(
                cfg=EvalConfig,
                fn=_lazy("gradling.studies.gpt2.eval", "eval"),
            ),
        },
    ),
}


__all__ = [
    "Command",
    "RunIdentity",
    "Study",
    "STUDIES",
]
