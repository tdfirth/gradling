from __future__ import annotations

from pathlib import Path
from typing import Any, Self, cast

import orbax.checkpoint as ocp
import tomlkit
from flax import nnx

from gradling.config import Config
from gradling.context import Context
from gradling.metrics import Metrics, RunIdentity, SinkFactory, log_only

RUNS = Path("runs")
CHECKPOINTS = Path("checkpoints")
EXPERIMENT_FILE = "experiment.toml"
RUN_FILE = "run.toml"


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

    def create_run(self, notes: str, cfg: dict[str, Any]) -> Path:
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
                        "study": self.study,
                        "experiment": self.name,
                        "notes": notes,
                    },
                    "config": run_cfg,
                }
            )
        )
        return path

    def create_debug_run(self, cfg: dict[str, Any]) -> Path:
        path = self.path / RUNS / "debug"
        if path.exists():
            import shutil

            shutil.rmtree(path)
        path.mkdir(parents=True)

        run_cfg = dict(cfg)
        run_cfg["experiment_name"] = self.name
        run_cfg["run_path"] = self.ctx.display_path(path)
        _run_file(path).write_text(
            tomlkit.dumps(
                {
                    "run": {
                        "id": "debug",
                        "study": self.study,
                        "experiment": self.name,
                        "notes": "debug run",
                    },
                    "config": run_cfg,
                }
            )
        )
        return path

    def list_runs(self) -> list[Path]:
        if not self.runs.exists():
            return []
        runs = [path for path in self.runs.iterdir() if path.is_dir()]
        return sorted(runs, key=lambda path: path.name)


class Run[Cfg: Config]:
    def __init__(
        self,
        ctx: Context,
        path: Path,
        cfg: Cfg,
        metrics: Metrics,
        *,
        identity: RunIdentity | None = None,
        raw_cfg: dict[str, Any] | None = None,
    ) -> None:
        self.ctx = ctx
        self.path = path
        self.cfg = cfg
        self.raw_cfg = raw_cfg or {}
        self.metrics = metrics
        self.identity = identity or RunIdentity(
            study="",
            experiment="",
            run_id=path.name,
        )
        self.id = self.identity.run_id
        self.notes = self.identity.notes
        self.checkpoints = path / CHECKPOINTS
        self.checkpoints.mkdir(parents=True, exist_ok=True)
        self.checkpointer = ocp.StandardCheckpointer()

    @classmethod
    def from_config(
        cls,
        ctx: Context,
        study: str,
        experiment: str,
        cfg_cls: type[Cfg],
        cfg: dict[str, Any],
        *,
        sink_factory: SinkFactory = log_only,
    ) -> Self:
        experiment_path = ctx.experiments / study / experiment
        spec = (
            Experiment.from_path(ctx, experiment_path)
            if experiment_path.exists()
            else Experiment.create(ctx, study, experiment, "", cfg)
        )
        path = spec.create_run("", cfg)
        return cls.from_path(
            ctx,
            path,
            cfg_cls=cfg_cls,
            sink_factory=sink_factory,
        )

    @classmethod
    def from_path(
        cls,
        ctx: Context,
        path: Path,
        *,
        cfg_cls: type[Cfg],
        overrides: dict[str, Any] | None = None,
        sink_factory: SinkFactory = log_only,
    ) -> Self:
        path = ctx.resolve_path(path)
        doc = tomlkit.parse(_run_file(path).read_text()).unwrap()
        run_meta = doc["run"]
        raw_cfg = doc["config"]
        if overrides:
            raw_cfg.update(overrides)
        cfg = cfg_cls.from_dict(raw_cfg)
        identity = RunIdentity(
            study=cast(str, run_meta.get("study", "")),
            experiment=cast(str, run_meta.get("experiment", "")),
            run_id=cast(str, run_meta.get("id", path.name)),
            notes=cast(str, run_meta.get("notes", "")),
        )
        return cls(
            ctx,
            path,
            cfg,
            Metrics(sink_factory(identity, raw_cfg)),
            identity=identity,
            raw_cfg=raw_cfg,
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
