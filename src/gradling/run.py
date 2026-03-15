from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, Self, cast

import orbax.checkpoint as ocp
import tomlkit
from flax import nnx

from gradling.context import Context
from gradling.metrics import Metrics

RUNS = Path("runs")
CHECKPOINTS = Path("checkpoints")
EXPERIMENT_FILE = "experiment.toml"
RUN_FILE = "run.toml"


class MetricSink(Protocol):
    def track(self, metrics: dict[str, Any], step: int) -> None: ...
    def close(self) -> None: ...


MetricsFactory = Callable[..., MetricSink]


def _experiment_file(path: Path) -> Path:
    return path / EXPERIMENT_FILE


def _run_file(path: Path) -> Path:
    return path / RUN_FILE


def _next_run_dir(experiment_path: Path) -> Path:
    runs_path = experiment_path / RUNS
    runs_path.mkdir(parents=True, exist_ok=True)
    existing = [int(path.name) for path in runs_path.iterdir() if path.name.isdigit()]
    return runs_path / f"{max(existing, default=0) + 1:04d}"


class Experiment:
    def __init__(
        self,
        ctx: Context,
        path: Path,
        *,
        study: str,
        name: str,
        notes: str,
        cfg: dict[str, Any],
    ) -> None:
        self.ctx = ctx
        self.path = path
        self.study = study
        self.name = name
        self.notes = notes
        self.cfg = cfg
        self.runs = path / RUNS
        self.runs.mkdir(parents=True, exist_ok=True)

    @classmethod
    def create(
        cls,
        ctx: Context,
        study: str,
        name: str,
        notes: str,
        cfg: dict[str, Any],
    ) -> Self:
        path = ctx.experiments / study / name
        if path.exists():
            raise RuntimeError(f"Experiment {ctx.display_path(path)} already exists.")
        path.mkdir(parents=True)

        experiment_cfg = dict(cfg)
        experiment_cfg["experiment_name"] = name
        experiment_cfg["run_path"] = ""
        _experiment_file(path).write_text(
            tomlkit.dumps(
                {
                    "experiment": {
                        "study": study,
                        "name": name,
                        "notes": notes,
                    },
                    "config": experiment_cfg,
                }
            )
        )
        return cls(
            ctx,
            path,
            study=study,
            name=name,
            notes=notes,
            cfg=experiment_cfg,
        )

    @classmethod
    def from_path(cls, ctx: Context, path: Path) -> Self:
        path = ctx.resolve_path(path)
        doc = tomlkit.parse(_experiment_file(path).read_text()).unwrap()
        experiment = doc["experiment"]
        cfg = doc["config"]
        return cls(
            ctx,
            path,
            study=cast(str, experiment["study"]),
            name=cast(str, experiment["name"]),
            notes=cast(str, experiment.get("notes", "")),
            cfg=cfg,
        )

    def create_run(
        self,
        notes: str,
        cfg: dict[str, Any],
        *,
        metrics_factory: MetricsFactory = Metrics,
    ) -> Run:
        path = _next_run_dir(self.path)
        path.mkdir(parents=True, exist_ok=True)

        run_cfg = dict(cfg)
        run_cfg["experiment_name"] = self.name
        run_cfg["run_path"] = self.ctx.display_path(path)
        _run_file(path).write_text(
            tomlkit.dumps(
                {
                    "run": {
                        "id": path.name,
                        "notes": notes,
                    },
                    "config": run_cfg,
                }
            )
        )
        return Run.from_path(self.ctx, path, metrics_factory=metrics_factory)

    def list_runs(self, *, metrics_factory: MetricsFactory = Metrics) -> list[Run]:
        if not self.runs.exists():
            return []
        runs = [path for path in self.runs.iterdir() if path.is_dir()]
        return [
            Run.from_path(self.ctx, path, metrics_factory=metrics_factory)
            for path in sorted(runs, key=lambda path: path.name)
        ]


class Run:
    def __init__(
        self,
        ctx: Context,
        path: Path,
        cfg: dict[str, Any],
        metrics: MetricSink,
        *,
        run_id: str = "",
        notes: str = "",
    ) -> None:
        self.ctx = ctx
        self.path = path
        self.cfg = cfg
        self.metrics = metrics
        self.id = run_id or path.name
        self.notes = notes
        self.checkpoints = path / CHECKPOINTS
        self.checkpoints.mkdir(parents=True, exist_ok=True)
        self.checkpointer = ocp.StandardCheckpointer()

    @classmethod
    def from_config(
        cls,
        ctx: Context,
        study: str,
        experiment: str,
        cfg: dict[str, Any],
        *,
        metrics_factory: MetricsFactory = Metrics,
    ) -> Self:
        experiment_path = ctx.experiments / study / experiment
        spec = (
            Experiment.from_path(ctx, experiment_path)
            if experiment_path.exists()
            else Experiment.create(ctx, study, experiment, "", cfg)
        )
        run = spec.create_run("", cfg, metrics_factory=metrics_factory)
        return cls.from_path(
            ctx,
            run.path,
            metrics_factory=metrics_factory,
            enable_wandb=True,
        )

    @classmethod
    def from_path(
        cls,
        ctx: Context,
        path: Path,
        *,
        metrics_factory: MetricsFactory = Metrics,
        enable_wandb: bool = False,
    ) -> Self:
        path = ctx.resolve_path(path)
        doc = tomlkit.parse(_run_file(path).read_text()).unwrap()
        run = doc["run"]
        cfg = doc["config"]
        return cls(
            ctx,
            path,
            cfg,
            metrics_factory(cfg, enable_wandb=enable_wandb),
            run_id=cast(str, run.get("id", path.name)),
            notes=cast(str, run.get("notes", "")),
        )

    def checkpoint(self, label: str, model: nnx.Module):
        state = nnx.state(model)
        self.checkpointer.save(self.checkpoints / label, state)
        self.checkpointer.wait_until_finished()

    def load_checkpoint(self, label: str, model: nnx.Module):
        state = nnx.state(model)
        checkpoint_path = self.checkpoints / label
        to_restore = self.checkpointer.restore(checkpoint_path.absolute(), target=state)
        nnx.update(model, to_restore)

    def track(self, metrics: dict[str, Any], step: int) -> None:
        self.metrics.track(metrics, step)

    def finalize(self):
        self.checkpointer.wait_until_finished()
        self.metrics.close()
