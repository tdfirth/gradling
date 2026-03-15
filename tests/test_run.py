from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import tomlkit

from gradling.config import Config
from gradling.context import Context
from gradling.metrics import MetricSink, RunIdentity
from gradling.run import CHECKPOINTS, RUNS, Experiment, Run


@dataclass
class FakeConfig(Config):
    seed: int = 0
    batch_size: int = 32
    experiment_name: str = ""
    run_path: str = ""
    checkpoint_label: str = "final"


class FakeSink:
    def __init__(self) -> None:
        self.tracked: list[tuple[dict[str, Any], int]] = []
        self.closed = False

    def track(self, metrics: dict[str, Any], step: int) -> None:
        self.tracked.append((metrics, step))

    def close(self) -> None:
        self.closed = True


def _fake_sink_factory(
    _identity: RunIdentity, _config: dict[str, Any]
) -> list[MetricSink]:
    return [FakeSink()]


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

        first = experiment.create_run("first run", experiment.cfg)
        second = experiment.create_run("second run", experiment.cfg)

        assert first == experiment.path / RUNS / "0001"
        assert second == experiment.path / RUNS / "0002"
        assert (first / CHECKPOINTS).exists() is False
        doc = tomlkit.parse((first / "run.toml").read_text()).unwrap()
        assert doc["run"]["id"] == "0001"
        assert doc["run"]["study"] == "aiayn"
        assert doc["run"]["experiment"] == "baseline"
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
                    'study = "aiayn"',
                    'experiment = "baseline"',
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
            Context(root=Path(tmp)),
            run_dir,
            cfg_cls=FakeConfig,
            sink_factory=_fake_sink_factory,
        )

        assert run.id == "0001"
        assert run.notes == "hello"
        assert run.cfg.seed == 42
        assert run.identity.study == "aiayn"
        assert run.identity.experiment == "baseline"
        assert run.checkpoints == run_dir / CHECKPOINTS
