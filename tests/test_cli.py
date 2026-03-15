from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from gradling import cli, storage
from gradling.config import Config
from gradling.context import Context
from gradling.run import Run
from gradling.studies import Command, Study


@dataclass
class CliConfigFixture(Config):
    n: int = 1
    ratio: float = 0.1
    title: str = "base"
    enabled: bool = False
    experiment_name: str = ""
    run_path: str = ""


@dataclass
class ChatConfigFixture(Config):
    run_path: str = ""
    max_tokens: int = 32


def _noop_train(_: Run[CliConfigFixture]) -> None:
    return None


def _noop_chat(_: Run[ChatConfigFixture]) -> None:
    return None


TEST_REGISTRY: dict[str, Study] = {
    "test_model": Study(
        cfg=CliConfigFixture,
        commands={
            "train": Command(cfg=CliConfigFixture, fn=_noop_train),
            "chat": Command(cfg=ChatConfigFixture, fn=_noop_chat),
        },
        description="A test model.",
    ),
}


class FakeTransport:
    repo_id = "fake/repo"

    def __init__(self) -> None:
        self.push_calls: list[tuple[Path, Path]] = []
        self.pull_calls: list[tuple[Path, Path]] = []

    def push(self, path: Path, *, root: Path) -> None:
        self.push_calls.append((path, root))

    def pull(self, path: Path, *, root: Path) -> None:
        self.pull_calls.append((path, root))


def _run(
    ctx: Context,
    registry: dict[str, Study],
    argv: list[str],
    *,
    store: storage.RunStorage | None = None,
    dataset_store: storage.DatasetTransport | None = None,
) -> int:
    try:
        ns = cli.parse_args(
            ctx,
            registry,
            argv,
            store=store,
            dataset_store=dataset_store,
        )
        ns.func(ns)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1
    except cli.HANDLER_ERRORS:
        return 2
    return 0


def _write_experiment(path: Path, *, study: str = "test_model") -> None:
    path.mkdir(parents=True)
    (path / "experiment.toml").write_text(
        "\n".join(
            [
                "[experiment]",
                f'study = "{study}"',
                'name = "baseline"',
                'notes = ""',
                "",
                "[config]",
                "n = 1",
                "ratio = 0.1",
                'title = "base"',
                "enabled = false",
                'experiment_name = "baseline"',
                'run_path = ""',
            ]
        )
        + "\n"
    )


def _write_run(
    path: Path,
    *,
    run_id: str = "0001",
    run_path: str = "",
    study: str = "test_model",
    experiment: str = "baseline",
) -> None:
    path.mkdir(parents=True)
    (path / "run.toml").write_text(
        "\n".join(
            [
                "[run]",
                f'id = "{run_id}"',
                f'study = "{study}"',
                f'experiment = "{experiment}"',
                'notes = ""',
                "",
                "[config]",
                "n = 1",
                "ratio = 0.1",
                'title = "base"',
                "enabled = false",
                'experiment_name = "baseline"',
                f'run_path = "{run_path}"',
            ]
        )
        + "\n"
    )


def test_study_list(capsys):
    code = _run(Context(), TEST_REGISTRY, ["study", "list"])
    out = capsys.readouterr().out
    assert code == 0
    assert "test_model" in out


def test_experiment_create_help_shows_config_fields(capsys):
    code = _run(
        Context(), TEST_REGISTRY, ["experiment", "test_model", "create", "--help"]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "--n" in out
    assert "--ratio" in out
    assert "--title" in out
    assert "--enabled" in out
    assert "--name" in out
    assert "--notes" in out


def test_experiment_list(tmp_path, capsys):
    experiment_path = tmp_path / "experiments" / "test_model" / "baseline"
    _write_experiment(experiment_path)

    code = _run(
        Context(root=tmp_path),
        TEST_REGISTRY,
        ["experiment", "test_model", "list"],
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "baseline" in out


def test_run_create_help_shows_model_config_fields(capsys):
    code = _run(
        Context(),
        TEST_REGISTRY,
        ["run", "test_model", "create", "--help"],
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "--n" in out
    assert "--ratio" in out
    assert "--title" in out
    assert "--enabled" in out
    assert "--notes" in out


def test_run_chat_help_shows_chat_config_fields(capsys):
    code = _run(
        Context(),
        TEST_REGISTRY,
        ["run", "test_model", "chat", "--help"],
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "--max-tokens" in out


def test_run_start_dispatches_to_train_command(tmp_path):
    captured: list[Run[CliConfigFixture]] = []

    def train(run: Run[CliConfigFixture]) -> None:
        captured.append(run)

    registry = {
        "test_model": Study(
            cfg=CliConfigFixture,
            commands={
                "train": Command(cfg=CliConfigFixture, fn=train),
                "chat": Command(cfg=ChatConfigFixture, fn=_noop_chat),
            },
        )
    }
    experiment_path = tmp_path / "experiments" / "test_model" / "baseline"
    run_path = experiment_path / "runs" / "0001"
    _write_experiment(experiment_path)
    _write_run(run_path, run_path="experiments/test_model/baseline/runs/0001")

    code = _run(
        Context(root=tmp_path),
        registry,
        ["run", "test_model", "start", "baseline", "0001"],
    )

    assert code == 0
    assert len(captured) == 1
    assert captured[0].cfg.run_path == "experiments/test_model/baseline/runs/0001"


def test_run_chat_dispatches_to_chat_command(tmp_path):
    captured: list[Run[ChatConfigFixture]] = []

    def chat(run: Run[ChatConfigFixture]) -> None:
        captured.append(run)

    registry = {
        "test_model": Study(
            cfg=CliConfigFixture,
            commands={
                "train": Command(cfg=CliConfigFixture, fn=_noop_train),
                "chat": Command(cfg=ChatConfigFixture, fn=chat),
            },
        )
    }
    experiment_path = tmp_path / "experiments" / "test_model" / "baseline"
    run_path = experiment_path / "runs" / "0001"
    _write_experiment(experiment_path)
    _write_run(run_path, run_path="experiments/test_model/baseline/runs/0001")

    code = _run(
        Context(root=tmp_path),
        registry,
        ["run", "test_model", "chat", "baseline", "0001", "--max-tokens", "99"],
    )

    assert code == 0
    assert len(captured) == 1
    assert captured[0].cfg.run_path == "experiments/test_model/baseline/runs/0001"
    assert captured[0].cfg.max_tokens == 99


def test_debug_dispatches_command(tmp_path):
    captured: list[Run[CliConfigFixture]] = []

    def train(run: Run[CliConfigFixture]) -> None:
        captured.append(run)

    registry = {
        "test_model": Study(
            cfg=CliConfigFixture,
            commands={
                "train": Command(cfg=CliConfigFixture, fn=train),
                "chat": Command(cfg=ChatConfigFixture, fn=_noop_chat),
            },
        )
    }
    experiment_path = tmp_path / "experiments" / "test_model" / "baseline"
    _write_experiment(experiment_path)

    code = _run(
        Context(root=tmp_path),
        registry,
        ["debug", "test_model", "train", "baseline"],
    )

    assert code == 0
    assert len(captured) == 1
    assert captured[0].id == "debug"


def test_debug_overwrites_previous_debug_run(tmp_path):
    captured: list[Run[CliConfigFixture]] = []

    def train(run: Run[CliConfigFixture]) -> None:
        captured.append(run)

    registry = {
        "test_model": Study(
            cfg=CliConfigFixture,
            commands={
                "train": Command(cfg=CliConfigFixture, fn=train),
            },
        )
    }
    experiment_path = tmp_path / "experiments" / "test_model" / "baseline"
    _write_experiment(experiment_path)

    ctx = Context(root=tmp_path)
    _run(ctx, registry, ["debug", "test_model", "train", "baseline"])
    _run(ctx, registry, ["debug", "test_model", "train", "baseline"])

    assert len(captured) == 2
    debug_path = experiment_path / "runs" / "debug"
    assert debug_path.exists()
    runs = list((experiment_path / "runs").iterdir())
    assert len(runs) == 1


def test_unknown_command():
    with pytest.raises(SystemExit):
        cli.parse_args(Context(), TEST_REGISTRY, ["bogus"])


def test_push_invokes_storage_handler(tmp_path):
    transport = FakeTransport()
    ctx = Context(root=tmp_path)
    store = storage.RunStorage(ctx, transport)
    run_path = Path("experiments/test_model/baseline/runs/0001")
    (tmp_path / run_path / "checkpoints").mkdir(parents=True)

    code = _run(
        ctx,
        TEST_REGISTRY,
        ["run", "test_model", "push", "baseline", "0001"],
        store=store,
    )

    assert code == 0
    assert transport.push_calls == [(run_path / "checkpoints", tmp_path)]


def test_pull_invokes_storage_handler(tmp_path):
    transport = FakeTransport()
    ctx = Context(root=tmp_path)
    store = storage.RunStorage(ctx, transport)
    run_path = Path("experiments/test_model/baseline/runs/0001")

    code = _run(
        ctx,
        TEST_REGISTRY,
        ["run", "test_model", "pull", "baseline", "0001"],
        store=store,
    )

    assert code == 0
    assert transport.pull_calls == [(run_path / "checkpoints", tmp_path)]


class FakeDatasetStorage:
    def __init__(self) -> None:
        self.push_calls: list[str] = []
        self.pull_calls: list[str] = []

    def push(self, dataset_name: str) -> None:
        self.push_calls.append(dataset_name)

    def pull(self, dataset_name: str) -> None:
        self.pull_calls.append(dataset_name)


def test_datasets_prepare_dispatches(tmp_path, monkeypatch):
    prepared = []

    def fake_prepare(root, dataset_name):
        from gradling.data import DatasetMeta

        prepared.append(dataset_name)
        d = root / "data" / "roneneldan" / "TinyStories"
        d.mkdir(parents=True, exist_ok=True)
        return DatasetMeta(
            source=dataset_name,
            repo="tdfirth/TinyStories",
            tokenizer_name="gpt2",
            dtype="uint16",
            train_tokens=100,
            dev_tokens=50,
        )

    monkeypatch.setattr(cli.datalib, "prepare", fake_prepare)

    code = _run(
        Context(root=tmp_path),
        TEST_REGISTRY,
        ["datasets", "prepare", "roneneldan/TinyStories"],
        dataset_store=FakeDatasetStorage(),
    )

    assert code == 0
    assert prepared == ["roneneldan/TinyStories"]


def test_datasets_push_dispatches(tmp_path):
    ds_store = FakeDatasetStorage()

    code = _run(
        Context(root=tmp_path),
        TEST_REGISTRY,
        ["datasets", "push", "roneneldan/TinyStories"],
        dataset_store=ds_store,
    )

    assert code == 0
    assert ds_store.push_calls == ["roneneldan/TinyStories"]


def test_datasets_pull_dispatches(tmp_path):
    ds_store = FakeDatasetStorage()

    code = _run(
        Context(root=tmp_path),
        TEST_REGISTRY,
        ["datasets", "pull", "roneneldan/TinyStories"],
        dataset_store=ds_store,
    )

    assert code == 0
    assert ds_store.pull_calls == ["roneneldan/TinyStories"]


def test_main_loads_dotenv(monkeypatch):
    called = False

    def fake_load_dotenv(self):
        nonlocal called
        called = True

    monkeypatch.setattr(cli.Context, "load_dotenv", fake_load_dotenv)
    monkeypatch.setattr(cli, "STUDIES", TEST_REGISTRY)

    code = cli.main(["study", "list"])

    assert code == 0
    assert called is True
