from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Protocol, cast

import jax

from gradling import logger
from gradling.dir import ROOT

log = logger.get(__name__)


class MetricSink(Protocol):
    def track(self, metrics: dict[str, Any], step: int) -> None: ...
    def close(self) -> None: ...


def is_loggable(x):
    return isinstance(x, int | float | str | jax.Array)


class LogSink:
    def track(self, metrics: dict[str, Any], step: int) -> None:
        log.info(
            " ".join([f"{k}={v:.2f}" for k, v in metrics.items() if is_loggable(v)])
        )

    def close(self) -> None:
        pass


class WandbSink:
    def __init__(self, config: dict[str, Any]) -> None:
        import wandb

        self.run = wandb.init(entity="tdfirth", project="Gradling", config=config)

    def track(self, metrics: dict[str, Any], step: int) -> None:
        self.run.log(metrics, step=step)

    def close(self) -> None:
        self.run.finish()


def _load_dotenv() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


class Metrics:
    def __init__(self, config: dict[str, Any], *, enable_wandb: bool = True) -> None:
        self.sinks: list[MetricSink] = [LogSink()]
        self._wandb_sink: WandbSink | None = None
        if enable_wandb:
            _load_dotenv()
            if os.environ.get("WANDB_API_KEY"):
                try:
                    self._wandb_sink = WandbSink(config)
                    self.sinks.append(self._wandb_sink)
                except Exception:
                    log.warning("Failed to initialize wandb sink", exc_info=True)

    @property
    def name(self) -> str:
        if self._wandb_sink is not None:
            return cast(str, self._wandb_sink.run.name)
        fallback_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        return fallback_name

    def track(self, metrics: dict[str, Any], step: int) -> None:
        for sink in self.sinks:
            sink.track(metrics, step)

    def close(self) -> None:
        for sink in self.sinks:
            sink.close()
