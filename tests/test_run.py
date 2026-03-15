from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import tomlkit

from gradling.context import Context
from gradling.run import CHECKPOINTS, RUNS, Experiment, Run


class FakeMetrics:
    def __init__(self, cfg: dict, *, enable_wandb: bool = True) -> None:
        self.cfg = cfg
        self.enable_wandb = enable_wandb

    def track(self, metrics: dict, step: int) -> None:
        return None

    def close(self) -> None:
        return None


def test_experiment_create_writes_toml():
    with TemporaryDirectory() as tmp:
        ctx = Context(root=Path(tmp))
        experiment = Experiment.create(
            ctx,
            "aiayn",
            "baseline",
            "baseline notes",
            {
                "seed": 42,
                "batch_size": 32,
                "experiment_name": "",
                "run_path": "",
                "checkpoint_label": "final",
            },
        )

        assert experiment.path == ctx.experiments / "aiayn" / "baseline"
        doc = tomlkit.parse((experiment.path / "experiment.toml").read_text()).unwrap()
        assert doc["experiment"]["study"] == "aiayn"
        assert doc["experiment"]["name"] == "baseline"
        assert doc["experiment"]["notes"] == "baseline notes"
        assert doc["config"]["experiment_name"] == "baseline"
        assert doc["config"]["run_path"] == ""


def test_experiment_create_run_writes_run_toml_and_increments():
    with TemporaryDirectory() as tmp:
        ctx = Context(root=Path(tmp))
        experiment = Experiment.create(
            ctx,
            "aiayn",
            "baseline",
            "",
            {
                "seed": 42,
                "batch_size": 32,
                "experiment_name": "",
                "run_path": "",
                "checkpoint_label": "final",
            },
        )

        first = experiment.create_run(
            "first run",
            experiment.cfg,
            metrics_factory=FakeMetrics,
        )
        second = experiment.create_run(
            "second run",
            experiment.cfg,
            metrics_factory=FakeMetrics,
        )

        assert first.path == experiment.path / RUNS / "0001"
        assert second.path == experiment.path / RUNS / "0002"
        assert first.checkpoints == first.path / CHECKPOINTS
        doc = tomlkit.parse((first.path / "run.toml").read_text()).unwrap()
        assert doc["run"]["id"] == "0001"
        assert doc["run"]["notes"] == "first run"
        assert doc["config"]["run_path"] == "experiments/aiayn/baseline/runs/0001"


def test_run_from_path_loads_run_metadata():
    with TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "experiments" / "aiayn" / "baseline" / "runs" / "0001"
        run_dir.mkdir(parents=True)
        (run_dir / "run.toml").write_text(
            "\n".join(
                [
                    "[run]",
                    'id = "0001"',
                    'notes = "hello"',
                    "",
                    "[config]",
                    "seed = 42",
                    'experiment_name = "baseline"',
                    'run_path = "experiments/aiayn/baseline/runs/0001"',
                    'checkpoint_label = "final"',
                ]
            )
            + "\n"
        )

        run = Run.from_path(
            Context(root=Path(tmp)), run_dir, metrics_factory=FakeMetrics
        )

        assert run.id == "0001"
        assert run.notes == "hello"
        assert run.cfg["seed"] == 42
        assert run.checkpoints == run_dir / CHECKPOINTS
