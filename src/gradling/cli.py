from __future__ import annotations

from dataclasses import MISSING
from typing import Any, NoReturn, cast, get_type_hints

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from gradling.configs import (
    Config,
    config_fields,
    get_config,
    get_pipeline,
    list_configs,
    list_pipelines,
    load_config_modules,
    normalize_scalar_type,
)
from gradling.pipelines import load_pipeline_modules

console = Console()
app = typer.Typer(no_args_is_help=True)
configs_app = typer.Typer(help="List and inspect registered config classes.")
app.add_typer(configs_app, name="configs")


def _fail(message: str, *, hint: str | None = None, code: int = 2) -> NoReturn:
    console.print(f"[bold red]Error:[/bold red] {message}")
    if hint:
        console.print(f"[dim]{hint}[/dim]")
    raise typer.Exit(code=code)


def _ensure_runtime_loaded() -> None:
    load_config_modules()
    load_pipeline_modules()


def _type_name(type_hint: Any) -> str:
    normalized = _normalize_cli_scalar_type(type_hint)
    return getattr(normalized, "__name__", str(normalized))


def _default_value(field: Any) -> str:
    if field.default is not MISSING:
        return repr(field.default)
    if field.default_factory is not MISSING:
        return "<factory>"
    return "<required>"


def _parse_bool(raw: str) -> bool:
    lowered = raw.lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    msg = f"`{raw}` is not a valid bool"
    raise ValueError(msg)


def _cast_value(raw: str, type_hint: Any) -> Any:
    normalized = _normalize_cli_scalar_type(type_hint)
    if normalized is bool:
        return _parse_bool(raw)
    if normalized is int:
        return int(raw)
    if normalized is float:
        return float(raw)
    if normalized is str:
        return raw
    msg = (
        f"Unsupported field type `{_type_name(type_hint)}`. "
        "Only int, float, str, and bool are supported."
    )
    raise TypeError(msg)


def _normalize_cli_scalar_type(type_hint: Any) -> Any:
    normalized = normalize_scalar_type(type_hint)
    if isinstance(normalized, str):
        basic = normalized.replace("builtins.", "").strip()
        aliases = {
            "int": int,
            "float": float,
            "str": str,
            "bool": bool,
        }
        return aliases.get(basic, normalized)
    return normalized


def _field_type_map(cfg_cls: type[Config]) -> dict[str, Any]:
    resolved: dict[str, Any]
    try:
        resolved = get_type_hints(cfg_cls)
    except Exception:
        resolved = {}

    mapping: dict[str, Any] = {}
    for field in config_fields(cfg_cls):
        mapping[field.name] = resolved.get(field.name, field.type)
    return mapping


def _parse_overrides(cfg_cls: type[Config], args: list[str]) -> dict[str, Any]:
    field_map = {field.name: field for field in config_fields(cfg_cls)}
    field_types = _field_type_map(cfg_cls)
    overrides: dict[str, Any] = {}
    i = 0
    while i < len(args):
        token = args[i]
        if token in {"-h", "--help"}:
            i += 1
            continue

        if not token.startswith("--"):
            _fail(
                f"Unexpected argument `{token}`.",
                hint="Expected overrides in the form `--field value`.",
            )

        key = token[2:].replace("-", "_")
        field = field_map.get(key)
        if field is None:
            known = ", ".join(sorted(field_map))
            _fail(
                f"Unknown override `{token}`.",
                hint=f"Valid fields are: {known}",
            )

        if i + 1 >= len(args):
            _fail(f"Missing value for `{token}`.")

        value = args[i + 1]
        if value.startswith("--"):
            _fail(f"Missing value for `{token}`.")

        try:
            type_hint = field_types.get(key, field.type)
            overrides[key] = _cast_value(value, type_hint)
        except (TypeError, ValueError) as exc:
            _fail(
                f"Could not parse value for `{token}`: {exc}",
                hint=f"Expected type `{_type_name(type_hint)}`.",
            )

        i += 2

    return overrides


def _render_configs_table() -> None:
    table = Table(title="Registered Configs")
    table.add_column("Name", style="cyan")
    table.add_column("Class")
    table.add_column("Module", style="dim")
    for name, cls in list_configs():
        table.add_row(name, cls.__name__, cls.__module__)
    console.print(table)


def _render_pipelines(config_name: str, cfg_cls: type[Config]) -> None:
    pipelines = list_pipelines(cfg_cls)
    table = Table(title=f"Pipelines for {config_name}")
    table.add_column("Pipeline", style="cyan")
    table.add_column("Description")
    if not pipelines:
        table.add_row("-", "No pipelines registered")
    else:
        for name, fn in pipelines.items():
            doc = (fn.__doc__ or "").strip().splitlines()
            summary = doc[0] if doc else "-"
            table.add_row(name, summary)
    console.print(table)


def _render_run_usage() -> None:
    text = (
        "Usage:\n"
        "  gradling run <config> list\n"
        "  gradling run <config> <pipeline> [--field value ...]\n"
        "  gradling run <config> --help\n"
        "  gradling run <config> <pipeline> --help"
    )
    console.print(Panel(text, title="Dynamic Run CLI"))


def _render_config_help(
    config_name: str, cfg_cls: type[Config], pipeline_name: str | None
) -> None:
    target = f"{config_name}.{pipeline_name}" if pipeline_name else config_name
    console.print(Panel(f"Config help for [bold]{target}[/bold]", title="gradling run"))
    _render_pipelines(config_name, cfg_cls)

    fields_table = Table(title="Config Fields")
    fields_table.add_column("Flag", style="cyan")
    fields_table.add_column("Type")
    fields_table.add_column("Default")
    field_types = _field_type_map(cfg_cls)
    for field in config_fields(cfg_cls):
        kebab = field.name.replace("_", "-")
        type_hint = field_types.get(field.name, field.type)
        fields_table.add_row(
            f"--{kebab}",
            _type_name(type_hint),
            _default_value(field),
        )
    console.print(fields_table)

    if pipeline_name:
        console.print(
            "[dim]Example:[/dim] "
            f"gradling run {config_name} {pipeline_name} --batch-size 64"
        )


def _run_config_pipeline(config_name: str, action: str, extra_args: list[str]) -> None:
    try:
        cfg_cls = get_config(config_name)
    except KeyError:
        _fail(
            f"Unknown config `{config_name}`.",
            hint="Use `gradling configs list` to see available configs.",
        )

    pipelines = list_pipelines(cfg_cls)
    if action == "list":
        _render_pipelines(config_name, cfg_cls)
        return

    if action not in pipelines:
        known = ", ".join(pipelines) if pipelines else "(none)"
        _fail(
            f"Unknown pipeline `{action}` for config `{config_name}`.",
            hint=f"Known pipelines: {known}",
        )

    if any(token in {"-h", "--help"} for token in extra_args):
        _render_config_help(config_name, cfg_cls, action)
        return

    overrides = _parse_overrides(cfg_cls, extra_args)
    try:
        cfg = cast(Any, cfg_cls)(**overrides)
    except TypeError as exc:
        _fail(
            f"Could not instantiate config `{config_name}`: {exc}",
            hint="Use `--help` to inspect required fields.",
        )

    runner = get_pipeline(cfg_cls, action)
    try:
        runner(cfg)
    except ValueError as exc:
        _fail(str(exc))


@configs_app.command("list")
def list_configs_cmd() -> None:
    _ensure_runtime_loaded()
    _render_configs_table()


@app.command(
    context_settings={
        "allow_extra_args": True,
        "ignore_unknown_options": True,
        "help_option_names": [],
    }
)
def run(
    ctx: typer.Context,
    config_name: str | None = typer.Argument(None),
    action: str | None = typer.Argument(None),
) -> None:
    _ensure_runtime_loaded()

    extra_args = list(ctx.args)
    if config_name in {None, "help", "-h", "--help"}:
        _render_run_usage()
        _render_configs_table()
        return

    try:
        cfg_cls = get_config(config_name)
    except KeyError:
        _fail(
            f"Unknown config `{config_name}`.",
            hint="Use `gradling configs list` to see available configs.",
        )

    if action in {None, "help", "-h", "--help"}:
        _render_config_help(config_name, cfg_cls, pipeline_name=None)
        return

    _run_config_pipeline(config_name, action, extra_args)
