from dataclasses import dataclass

from gradling import cli
from gradling.config import Config
from gradling.models import Command, Model


@dataclass
class CliConfigFixture(Config):
    n: int = 1
    ratio: float = 0.1
    title: str = "base"
    enabled: bool = False


def _with_test_model(monkeypatch, name: str, cfg_cls: type[Config], command):
    patched = dict(cli.MODELS)
    patched[name] = Model(
        cfg=cfg_cls, commands={"noop": Command(cfg=cfg_cls, fn=command)}
    )
    monkeypatch.setattr(cli, "MODELS", patched)


def test_models_list_includes_gpt(capsys):
    code = cli.main(["models", "list"])
    out = capsys.readouterr().out
    assert code == 0
    assert "gpt" in out


def test_run_list_shows_commands(capsys):
    code = cli.main(["run", "gpt", "list"])
    out = capsys.readouterr().out
    assert code == 0
    assert "train" in out
    assert "sample" in out


def test_run_help_shows_config_fields(capsys):
    code = cli.main(["run", "gpt", "train", "--help"])
    output = capsys.readouterr().out
    assert code == 0
    assert "--batch-size" in output


def test_run_rejects_unknown_override(monkeypatch, capsys):
    def noop(_: CliConfigFixture):
        return None

    _with_test_model(monkeypatch, "test_cli_config", CliConfigFixture, noop)

    code = cli.main(["run", "test_cli_config", "noop", "--does-not-exist", "1"])
    err = capsys.readouterr().err
    assert code == 2
    assert "unrecognized arguments" in err


def test_run_parses_scalar_overrides(monkeypatch, capsys):
    def noop(cfg: CliConfigFixture):
        print(f"{cfg.n}|{cfg.ratio}|{cfg.title}|{cfg.enabled}")

    _with_test_model(monkeypatch, "test_cli_config", CliConfigFixture, noop)

    code = cli.main(
        [
            "run",
            "test_cli_config",
            "noop",
            "--n",
            "3",
            "--ratio",
            "0.5",
            "--title",
            "demo",
            "--enabled",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "3|0.5|demo|True" in out
