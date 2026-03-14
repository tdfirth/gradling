from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from gradling import cli, storage
from gradling.config import Config
from gradling.context import Context
from gradling.models import Command, Model


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


def _noop_train(_: CliConfigFixture) -> None:
    return None


def _noop_chat(_: ChatConfigFixture) -> None:
    return None


TEST_REGISTRY: dict[str, Model] = {
    "test_model": Model(
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
    registry: dict[str, Model],
    argv: list[str],
    *,
    store: storage.RunStorage | None = None,
) -> int:
    try:
        ns = cli.parse_args(ctx, registry, argv, store=store)
        ns.func(ns)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1
    except cli.HANDLER_ERRORS:
        return 2
    return 0


def _write_experiment(path: Path, *, model: str = "test_model") -> None:
    path.mkdir(parents=True)
    (path / "experiment.toml").write_text(
        "\n".join(
            [
                "[experiment]",
                f'model = "{model}"',
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


def _write_run(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "run.toml").write_text(
        "\n".join(
            [
                "[run]",
                'id = "0001"',
                'notes = ""',
                "",
                "[config]",
                "n = 1",
                "ratio = 0.1",
                'title = "base"',
                "enabled = false",
                'experiment_name = "baseline"',
                'run_path = "experiments/test_model/baseline/runs/0001"',
            ]
        )
        + "\n"
    )


def test_models_list(capsys):
    code = _run(Context(), TEST_REGISTRY, ["models", "list"])
    out = capsys.readouterr().out
    assert code == 0
    assert "test_model" in out


def test_experiment_create_help_shows_config_fields(capsys):
    code = _run(
        Context(), TEST_REGISTRY, ["experiment", "create", "test_model", "--help"]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "--n" in out
    assert "--ratio" in out
    assert "--title" in out
    assert "--enabled" in out
    assert "--name" in out
    assert "--notes" in out


def test_run_create_help_shows_model_config_fields(tmp_path, capsys):
    experiment_path = tmp_path / "experiments" / "test_model" / "baseline"
    _write_experiment(experiment_path)
    ctx = Context(root=tmp_path)

    code = _run(
        ctx,
        TEST_REGISTRY,
        ["run", "create", str(experiment_path), "--help"],
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "--n" in out
    assert "--ratio" in out
    assert "--title" in out
    assert "--enabled" in out
    assert "--notes" in out


def test_run_chat_help_shows_chat_config_fields(tmp_path, capsys):
    experiment_path = tmp_path / "experiments" / "test_model" / "baseline"
    run_path = experiment_path / "runs" / "0001"
    _write_experiment(experiment_path)
    _write_run(run_path)
    ctx = Context(root=tmp_path)

    code = _run(
        ctx,
        TEST_REGISTRY,
        ["run", "chat", str(run_path), "--help"],
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "--max-tokens" in out


def test_run_start_dispatches_to_train_command(tmp_path):
    captured: list[CliConfigFixture] = []

    def train(cfg: CliConfigFixture) -> None:
        captured.append(cfg)

    registry = {
        "test_model": Model(
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
    _write_run(run_path)

    code = _run(Context(root=tmp_path), registry, ["run", "start", str(run_path)])

    assert code == 0
    assert len(captured) == 1
    assert captured[0].run_path == "experiments/test_model/baseline/runs/0001"


def test_run_chat_dispatches_to_chat_command(tmp_path):
    captured: list[ChatConfigFixture] = []

    def chat(cfg: ChatConfigFixture) -> None:
        captured.append(cfg)

    registry = {
        "test_model": Model(
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
    _write_run(run_path)

    code = _run(
        Context(root=tmp_path),
        registry,
        ["run", "chat", str(run_path), "--max-tokens", "99"],
    )

    assert code == 0
    assert len(captured) == 1
    assert captured[0].run_path == "experiments/test_model/baseline/runs/0001"
    assert captured[0].max_tokens == 99


def test_unknown_command():
    with pytest.raises(SystemExit):
        cli.parse_args(Context(), TEST_REGISTRY, ["bogus"])


def test_push_invokes_storage_handler(tmp_path):
    transport = FakeTransport()
    ctx = Context(root=tmp_path)
    store = storage.RunStorage(ctx, transport)
    run_path = Path("artifacts/run-1")
    (tmp_path / run_path / "checkpoints").mkdir(parents=True)

    code = _run(ctx, TEST_REGISTRY, ["run", "push", str(run_path)], store=store)

    assert code == 0
    assert transport.push_calls == [(run_path / "checkpoints", tmp_path)]


def test_pull_invokes_storage_handler(tmp_path):
    transport = FakeTransport()
    ctx = Context(root=tmp_path)
    store = storage.RunStorage(ctx, transport)
    run_path = Path("artifacts/run-1")

    code = _run(ctx, TEST_REGISTRY, ["run", "pull", str(run_path)], store=store)

    assert code == 0
    assert transport.pull_calls == [(run_path / "checkpoints", tmp_path)]


def test_main_loads_dotenv(monkeypatch):
    called = False

    def fake_load_dotenv(self):
        nonlocal called
        called = True

    monkeypatch.setattr(cli.Context, "load_dotenv", fake_load_dotenv)
    monkeypatch.setattr(cli, "MODELS", TEST_REGISTRY)

    code = cli.main(["models", "list"])

    assert code == 0
    assert called is True
