from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass
from types import UnionType
from typing import Any, Union, get_args, get_origin

from pydantic import ValidationError
from pydantic_core import PydanticUndefined
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from gradling.configs import Config
from gradling.models.gpt.config import GPTConfig
from gradling.models.gpt.sample import sample as gpt_sample
from gradling.models.gpt.train import train as gpt_train

console = Console()


@dataclass(frozen=True)
class ModelSpec:
    cfg: type[Config]
    commands: dict[str, Callable[..., None]]
    description: str = ""


MODELS: dict[str, ModelSpec] = {
    "gpt": ModelSpec(
        cfg=GPTConfig,
        commands={
            "train": gpt_train,
            "sample": gpt_sample,
        },
        description="Character-level GPT model and commands.",
    ),
}


def _fail(message: str, *, hint: str | None = None) -> int:
    console.print(f"[bold red]Error:[/bold red] {message}")
    if hint:
        console.print(f"[dim]{hint}[/dim]")
    return 2


def _normalize_scalar_type(type_hint: Any) -> type | None:
    origin = get_origin(type_hint)
    if origin in (Union, UnionType):
        args = [arg for arg in get_args(type_hint) if arg is not type(None)]
        if len(args) == 1:
            type_hint = args[0]

    if type_hint in (int, float, str, bool):
        return type_hint
    if isinstance(type_hint, str):
        normalized = type_hint.replace("builtins.", "").strip()
        aliases = {
            "int": int,
            "float": float,
            "str": str,
            "bool": bool,
        }
        return aliases.get(normalized)

    return None


def _render_models_table() -> None:
    table = Table(title="Registered Models")
    table.add_column("Model", style="cyan")
    table.add_column("Config")
    table.add_column("Description")
    for name, spec in sorted(MODELS.items()):
        table.add_row(name, spec.cfg.__name__, spec.description or "-")
    console.print(table)


def _render_commands_table(model_name: str, spec: ModelSpec) -> None:
    table = Table(title=f"Commands for {model_name}")
    table.add_column("Command", style="cyan")
    table.add_column("Description")
    for name, fn in sorted(spec.commands.items()):
        doc = (fn.__doc__ or "").strip().splitlines()
        summary = doc[0] if doc else "-"
        table.add_row(name, summary)
    console.print(table)


def _render_config_fields_table(cfg_cls: type[Config]) -> None:
    table = Table(title="Config Fields")
    table.add_column("Flag", style="cyan")
    table.add_column("Type")
    table.add_column("Default")
    table.add_column("Description")

    for name, field in cfg_cls.model_fields.items():
        scalar = _normalize_scalar_type(field.annotation)
        if scalar is None:
            continue
        default = "<required>"
        if field.default is not PydanticUndefined:
            default = repr(field.default)
        table.add_row(
            f"--{name.replace('_', '-')}",
            scalar.__name__,
            default,
            field.description or "-",
        )

    console.print(table)


def _render_root_help() -> None:
    text = (
        "Usage:\n"
        "  gradling models list\n"
        "  gradling run <model> list\n"
        "  gradling run <model> <command> [--field value ...]\n"
        "  gradling run <model> <command> --help"
    )
    console.print(Panel(text, title="Gradling CLI"))


def _build_command_parser(model_name: str, command: str, cfg_cls: type[Config]):
    parser = argparse.ArgumentParser(
        prog=f"gradling run {model_name} {command}",
        add_help=True,
    )
    for name, field in cfg_cls.model_fields.items():
        scalar = _normalize_scalar_type(field.annotation)
        if scalar is None:
            continue

        help_parts: list[str] = []
        if field.description:
            help_parts.append(field.description)
        if field.default is not PydanticUndefined:
            help_parts.append(f"default: {field.default!r}")

        kwargs: dict[str, Any] = {
            "dest": name,
            "default": argparse.SUPPRESS,
            "help": "; ".join(help_parts) or None,
        }
        if scalar is bool:
            kwargs["action"] = argparse.BooleanOptionalAction
        else:
            kwargs["type"] = scalar

        parser.add_argument(f"--{name.replace('_', '-')}", **kwargs)
    return parser


def _handle_models(args: list[str]) -> int:
    if not args or args[0] in {"-h", "--help", "help"}:
        console.print("Usage: gradling models list")
        return 0

    if args[0] != "list":
        return _fail(
            f"Unknown models command `{args[0]}`.",
            hint="Supported command: list",
        )

    _render_models_table()
    return 0


def _handle_run(args: list[str]) -> int:
    if not args or args[0] in {"-h", "--help", "help"}:
        _render_root_help()
        return 0

    model_name = args[0]
    spec = MODELS.get(model_name)
    if spec is None:
        known = ", ".join(sorted(MODELS))
        return _fail(
            f"Unknown model `{model_name}`.",
            hint=f"Available models: {known}",
        )

    if len(args) == 1 or args[1] in {"-h", "--help", "help"}:
        _render_commands_table(model_name, spec)
        _render_config_fields_table(spec.cfg)
        return 0

    command = args[1]
    if command == "list":
        _render_commands_table(model_name, spec)
        return 0

    runner = spec.commands.get(command)
    if runner is None:
        known = ", ".join(sorted(spec.commands))
        return _fail(
            f"Unknown command `{command}` for model `{model_name}`.",
            hint=f"Known commands: {known}",
        )

    parser = _build_command_parser(model_name, command, spec.cfg)
    try:
        parsed = parser.parse_args(args[2:])
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1

    try:
        cfg = spec.cfg.model_validate(vars(parsed))
    except ValidationError as exc:
        return _fail(
            f"Invalid config for `{model_name}`: {exc}",
            hint="Use `--help` to inspect accepted fields.",
        )

    try:
        runner(cfg)
    except ValueError as exc:
        return _fail(str(exc))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        _render_root_help()
        return 0

    command = args[0]
    if command in {"-h", "--help", "help"}:
        _render_root_help()
        return 0
    if command == "models":
        return _handle_models(args[1:])
    if command == "run":
        return _handle_run(args[1:])
    return _fail(
        f"Unknown command `{command}`.",
        hint="Supported commands: models, run",
    )
