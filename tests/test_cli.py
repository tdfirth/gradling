from dataclasses import dataclass

import pytest

from gradling import cli
from gradling.config import Config
from gradling.models import Command, Model


@dataclass
class CliConfigFixture(Config):
    n: int = 1
    ratio: float = 0.1
    title: str = "base"
    enabled: bool = False


def _noop(_: CliConfigFixture) -> None:
    return None


TEST_REGISTRY: dict[str, Model] = {
    "test_model": Model(
        cfg=CliConfigFixture,
        commands={"noop": Command(cfg=CliConfigFixture, fn=_noop)},
        description="A test model.",
    ),
}


def _run(registry: dict[str, Model], argv: list[str]) -> int:
    try:
        ns = cli.parse_args(registry, argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1
    return ns.func(ns)


def test_models_list(capsys):
    code = _run(TEST_REGISTRY, ["models", "list"])
    out = capsys.readouterr().out
    assert code == 0
    assert "test_model" in out


def test_run_model_help_shows_commands(capsys):
    code = _run(TEST_REGISTRY, ["run", "test_model", "--help"])
    out = capsys.readouterr().out
    assert code == 0
    assert "noop" in out


def test_run_help_shows_config_fields(capsys):
    code = _run(TEST_REGISTRY, ["run", "test_model", "noop", "--help"])
    out = capsys.readouterr().out
    assert code == 0
    assert "--n" in out
    assert "--ratio" in out
    assert "--title" in out
    assert "--enabled" in out


def test_run_rejects_unknown_override(capsys):
    code = _run(TEST_REGISTRY, ["run", "test_model", "noop", "--does-not-exist", "1"])
    assert code == 2


def test_run_parses_scalar_overrides(capsys):
    def printing_noop(cfg: CliConfigFixture) -> None:
        print(f"{cfg.n}|{cfg.ratio}|{cfg.title}|{cfg.enabled}")

    registry: dict[str, Model] = {
        "test_model": Model(
            cfg=CliConfigFixture,
            commands={"noop": Command(cfg=CliConfigFixture, fn=printing_noop)},
        ),
    }

    code = _run(
        registry,
        [
            "run",
            "test_model",
            "noop",
            "--n",
            "3",
            "--ratio",
            "0.5",
            "--title",
            "demo",
            "--enabled",
        ],
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "3|0.5|demo|True" in out


def test_unknown_command():
    with pytest.raises(SystemExit):
        cli.parse_args(TEST_REGISTRY, ["bogus"])


def test_unknown_model():
    with pytest.raises(SystemExit):
        cli.parse_args(TEST_REGISTRY, ["run", "nonexistent", "train"])
