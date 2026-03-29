from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from gradling import logger
from gradling.context import Context, load_dotenv

log = logger.get(__name__)


@dataclass(frozen=True)
class RunIdentity:
    study: str
    experiment: str
    run_id: str
    notes: str = ""


class MetricSink(Protocol):
    def track(self, metrics: dict[str, Any], step: int) -> None: ...
    def close(self) -> None: ...


SinkFactory = Callable[["Context", "RunIdentity", dict[str, Any]], list[MetricSink]]


def is_loggable(x):
    import jax

    return isinstance(x, int | float | str | jax.Array)


class LogSink:
    def track(self, metrics: dict[str, Any], step: int) -> None:
        log.info(
            " ".join([f"{k}={v:.2f}" for k, v in metrics.items() if is_loggable(v)])
        )

    def close(self) -> None:
        pass


class WandbSink:
    def __init__(self, identity: RunIdentity, config: dict[str, Any]) -> None:
        import wandb

        self.run = wandb.init(
            entity="tdfirth",
            project="Gradling",
            config=config,
            name=f"{identity.study}/{identity.experiment}/{identity.run_id}",
            group=f"{identity.study}/{identity.experiment}",
            tags=[identity.study, identity.experiment],
            notes=identity.notes,
        )

    def track(self, metrics: dict[str, Any], step: int) -> None:
        self.run.log(metrics, step=step)

    def close(self) -> None:
        self.run.finish()


def log_only(
    ctx: Context, _identity: RunIdentity, _config: dict[str, Any]
) -> list[MetricSink]:
    return [LogSink()]


def with_wandb(
    ctx: Context, identity: RunIdentity, config: dict[str, Any]
) -> list[MetricSink]:
    sinks: list[MetricSink] = [LogSink()]
    load_dotenv()
    if ctx.wandb_api_key:
        try:
            sinks.append(WandbSink(identity, config))
        except Exception:
            log.warning("Failed to initialize wandb sink", exc_info=True)
    return sinks


class Metrics:
    def __init__(self, sinks: list[MetricSink]) -> None:
        self.sinks = list(sinks)

    def track(self, metrics: dict[str, Any], step: int) -> None:
        for sink in self.sinks:
            sink.track(metrics, step)

    def close(self) -> None:
        for sink in self.sinks:
            sink.close()
