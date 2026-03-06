from __future__ import annotations

import argparse
import sys
from dataclasses import fields
from functools import partial
from types import UnionType
from typing import Any, Union, get_args, get_origin

from rich.console import Console
from rich.table import Table
from rich_argparse import RichHelpFormatter

from gradling.config import Config
from gradling.models import MODELS, Command, Model

console = Console()
Formatter = partial(RichHelpFormatter, max_help_position=120)


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


def _models_table(registry: dict[str, Model]) -> Table:
    table = Table(title="Registered Models", highlight=True)
    table.add_column("Model", style="bold cyan")
    table.add_column("Config")
    table.add_column("Description", style="dim")
    for name, model in sorted(registry.items()):
        table.add_row(name, model.cfg.__name__, model.description or "-")
    return table


def _add_config_flags(parser: argparse.ArgumentParser, cfg_cls: type[Config]) -> None:
    for f in cfg_cls.cli_fields():
        scalar = _normalize_scalar_type(f.type)
        if scalar is None:
            continue
        kwargs: dict[str, Any] = {
            "dest": f.name,
            "default": argparse.SUPPRESS,
            "help": f"(default: {f.default})",
        }
        if scalar is bool:
            kwargs["action"] = argparse.BooleanOptionalAction
        else:
            kwargs["type"] = scalar
        parser.add_argument(f"--{f.name.replace('_', '-')}", **kwargs)


def _make_models_list_handler(
    registry: dict[str, Model],
):
    def handler(_ns: argparse.Namespace) -> int:
        console.print(_models_table(registry))
        return 0

    return handler


def _make_run_handler(cmd: Command, cfg_cls: type[Config]):
    config_keys = {f.name for f in fields(cfg_cls)}

    def handler(ns: argparse.Namespace) -> int:
        overrides = {k: v for k, v in vars(ns).items() if k in config_keys}
        cfg = cfg_cls(**overrides)
        try:
            cmd.fn(cfg)
        except ValueError as exc:
            console.print(f"[bold red]Error:[/bold red] {exc}")
            return 2
        return 0

    return handler


def _add_models_subcommand(
    sub: argparse._SubParsersAction, registry: dict[str, Model]
) -> None:
    models_parser = sub.add_parser(
        "models", help="List and inspect models", formatter_class=Formatter
    )
    models_sub = models_parser.add_subparsers(dest="models_command", required=True)
    list_parser = models_sub.add_parser(
        "list", help="List all registered models", formatter_class=Formatter
    )
    list_parser.set_defaults(func=_make_models_list_handler(registry))


def _add_run_subcommand(
    sub: argparse._SubParsersAction, registry: dict[str, Model]
) -> None:
    run_parser = sub.add_parser(
        "run", help="Run a model command", formatter_class=Formatter
    )
    run_sub = run_parser.add_subparsers(dest="model_name", required=True)

    for model_name, spec in registry.items():
        model_parser = run_sub.add_parser(
            model_name,
            help=spec.description,
            formatter_class=Formatter,
        )
        cmd_sub = model_parser.add_subparsers(dest="model_command", required=True)

        for cmd_name, cmd in spec.commands.items():
            doc = (cmd.fn.__doc__ or "").strip().splitlines()
            summary = doc[0] if doc else None
            cmd_parser = cmd_sub.add_parser(
                cmd_name,
                help=summary,
                formatter_class=Formatter,
            )
            _add_config_flags(cmd_parser, cmd.cfg)
            cmd_parser.set_defaults(func=_make_run_handler(cmd, cmd.cfg))


def parse_args(
    registry: dict[str, Model], argv: list[str] | None = None
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="gradling",
        description="Gradling CLI",
        formatter_class=Formatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    _add_models_subcommand(sub, registry)
    _add_run_subcommand(sub, registry)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    try:
        app = parse_args(MODELS, args)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else (1 if exc.code else 0)
    return app.func(app)
