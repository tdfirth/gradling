from typer.testing import CliRunner

from gradling.cli import _parse_overrides, app
from gradling.configs import Config, pipeline
from gradling.configs.transformer import Transformer

runner = CliRunner()


class CliConfigFixture(Config, name="test_cli_config"):
    n: int = 1
    ratio: float = 0.1
    title: str = "base"
    enabled: bool = False


class StringHintConfig(Config, name="test_string_hint_config"):
    n: "int" = 1
    dry_run: "bool" = False


@pipeline(CliConfigFixture, name="noop")
def cli_config_noop(cfg: CliConfigFixture):
    print(f"{cfg.n}|{cfg.ratio}|{cfg.title}|{cfg.enabled}")


@pipeline(StringHintConfig, name="noop")
def string_hint_config_noop(cfg: StringHintConfig):
    print(f"{cfg.n}|{cfg.dry_run}")


def test_configs_list_includes_transformer():
    result = runner.invoke(app, ["configs", "list"])
    assert result.exit_code == 0
    assert "transformer" in result.output


def test_run_list_shows_pipelines():
    result = runner.invoke(app, ["run", "transformer", "list"])
    assert result.exit_code == 0
    assert "train" in result.output


def test_run_help_shows_config_fields():
    result = runner.invoke(app, ["run", "transformer", "train", "--help"])
    assert result.exit_code == 0
    assert "Config Fields" in result.output
    assert "--batch-size" in result.output


def test_run_rejects_unknown_override():
    result = runner.invoke(
        app,
        ["run", "test_cli_config", "noop", "--does-not-exist", "1"],
    )
    assert result.exit_code == 2
    assert "Unknown override" in result.output


def test_run_parses_scalar_overrides():
    result = runner.invoke(
        app,
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
            "true",
        ],
    )
    assert result.exit_code == 0
    assert "3|0.5|demo|True" in result.output


def test_parse_overrides_with_transformer_types():
    parsed = _parse_overrides(
        Transformer,
        ["--train-steps", "1", "--dry-run", "true"],
    )
    assert parsed["train_steps"] == 1
    assert parsed["dry_run"] is True


def test_run_parses_string_annotated_config_end_to_end():
    result = runner.invoke(
        app,
        ["run", "test_string_hint_config", "noop", "--n", "7", "--dry-run", "true"],
    )
    assert result.exit_code == 0
    assert "7|True" in result.output


def test_transformer_child_inherits_parent_pipelines():
    class TransformerChild(Transformer, name="test_transformer_child"):
        pass

    result = runner.invoke(app, ["run", "test_transformer_child", "list"])
    assert result.exit_code == 0
    assert "train" in result.output
    assert "sample" in result.output
