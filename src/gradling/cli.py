from __future__ import annotations

import argparse
import logging
import sys
from functools import partial
from pathlib import Path
from types import UnionType
from typing import Any, Union, get_args, get_origin, get_type_hints

from rich.console import Console
from rich.table import Table
from rich_argparse import RichHelpFormatter
from tomlkit.exceptions import ParseError

from gradling import run as runlib
from gradling import storage
from gradling.config import Config
from gradling.context import Context
from gradling.models import MODELS, Command, Model

log = logging.getLogger(__name__)

HIDDEN_CONFIG_FIELDS = {"experiment_name", "run_path"}
HANDLER_ERRORS = (
    FileNotFoundError,
    KeyError,
    RuntimeError,
    ParseError,
    TypeError,
    ValueError,
)

_formatter = partial(RichHelpFormatter, max_help_position=120)


def _subparser(
    sub: argparse._SubParsersAction, name: str, **kwargs: Any
) -> argparse.ArgumentParser:
    return sub.add_parser(name, formatter_class=_formatter, **kwargs)


def _normalize_scalar_type(type_hint: Any) -> type | None:
    origin = get_origin(type_hint)
    if origin in (Union, UnionType):
        args = [arg for arg in get_args(type_hint) if arg is not type(None)]
        if len(args) == 1:
            type_hint = args[0]

    if type_hint in (int, float, str, bool):
        return type_hint

    return None


def _add_config_flags(
    parser: argparse.ArgumentParser,
    cfg_cls: type[Config],
    *,
    exclude: set[str] | None = None,
) -> None:
    excluded = exclude or set()
    hints = get_type_hints(cfg_cls)
    for f in cfg_cls.cli_fields():
        if f.name in excluded:
            continue
        scalar = _normalize_scalar_type(hints.get(f.name, f.type))
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


def _config_overrides(ns: argparse.Namespace, cfg_cls: type[Config]) -> dict[str, Any]:
    keys = cfg_cls.field_names()
    return {k: v for k, v in vars(ns).items() if k in keys}


def _models_table(registry: dict[str, Model]) -> Table:
    table = Table(title="Registered Models", highlight=True)
    table.add_column("Model", style="bold cyan")
    table.add_column("Config")
    table.add_column("Description", style="dim")
    for name, model in sorted(registry.items()):
        table.add_row(name, model.cfg.__name__, model.description or "-")
    return table


def _runs_table(
    ctx: Context, experiment: runlib.Experiment, runs: list[runlib.Run]
) -> Table:
    table = Table(
        title=f"Runs for {ctx.display_path(experiment.path)}",
        highlight=True,
    )
    table.add_column("Run", style="bold cyan")
    table.add_column("Path")
    table.add_column("Notes", style="dim")
    for run in runs:
        table.add_row(run.id, ctx.display_path(run.path), run.notes.strip() or "-")
    return table


def _pre_parse_argv(argv: list[str]) -> tuple[str | None, str | None, str | None]:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("command", nargs="?")
    bootstrap.add_argument("sub_command", nargs="?")
    bootstrap.add_argument("path", nargs="?")
    try:
        ns, _ = bootstrap.parse_known_args(argv)
    except SystemExit:
        return None, None, None
    return ns.command, ns.sub_command, ns.path


def _resolve_dynamic_cfg(
    ctx: Context,
    registry: dict[str, Model],
    argv: list[str],
    sub_command: str,
    model_command: str,
) -> type[Config] | None:
    cmd, sub, path = _pre_parse_argv(argv)
    if cmd != "run" or sub != sub_command or path is None:
        return None

    try:
        if model_command == "create":
            experiment = runlib.Experiment.from_path(ctx, Path(path))
            spec = registry.get(experiment.model)
            return spec.cfg if spec is not None else None

        _, _, command = _load_run_command(ctx, registry, Path(path), model_command)
        return command.cfg
    except Exception:
        log.debug(
            "Failed to resolve %s config at %s", model_command, path, exc_info=True
        )
        return None


def _load_run_command(
    ctx: Context,
    registry: dict[str, Model],
    run_path: Path,
    command_name: str,
) -> tuple[runlib.Run, Model, Command]:
    run = runlib.Run.from_path(ctx, run_path)
    experiment = runlib.Experiment.from_path(ctx, run.path.parent.parent)
    spec = registry.get(experiment.model)
    if spec is None:
        msg = f"Unknown model {experiment.model!r}."
        raise RuntimeError(msg)
    command = spec.commands.get(command_name)
    if command is None:
        msg = f"Model {experiment.model!r} has no {command_name!r} command."
        raise RuntimeError(msg)
    return run, spec, command


class App:
    def __init__(
        self,
        ctx: Context,
        registry: dict[str, Model],
        store: storage.RunStorage,
        console: Console | None = None,
    ) -> None:
        self.ctx = ctx
        self.registry = registry
        self.store = store
        self.console = console or Console()

    def models_list(self, _ns: argparse.Namespace) -> None:
        self.console.print(_models_table(self.registry))

    def experiment_create(self, ns: argparse.Namespace, cfg_cls: type[Config]) -> None:
        name = ns.name or self.console.input("Experiment name: ").strip()
        notes = ns.notes if ns.notes is not None else self.console.input("Notes: ")
        if not name:
            raise ValueError("Experiment name is required.")

        overrides = _config_overrides(ns, cfg_cls)
        cfg = cfg_cls(**overrides).to_dict()
        experiment = runlib.Experiment.create(self.ctx, ns.model_name, name, notes, cfg)
        self.console.print(self.ctx.display_path(experiment.path))

    def run_create(self, ns: argparse.Namespace) -> None:
        experiment = runlib.Experiment.from_path(self.ctx, Path(ns.experiment_path))
        spec = self.registry.get(experiment.model)
        if spec is None:
            raise RuntimeError(f"Unknown model {experiment.model!r}.")

        overrides = _config_overrides(ns, spec.cfg)
        resolved_cfg = spec.cfg(**{**experiment.cfg, **overrides}).to_dict()
        run = experiment.create_run(ns.notes or "", resolved_cfg)
        self.console.print(self.ctx.display_path(run.path))

    def run_list(self, ns: argparse.Namespace) -> None:
        experiment = runlib.Experiment.from_path(self.ctx, Path(ns.experiment_path))
        self.console.print(_runs_table(self.ctx, experiment, experiment.list_runs()))

    def run_start(self, ns: argparse.Namespace) -> None:
        run, _, command = _load_run_command(
            self.ctx, self.registry, Path(ns.path), "train"
        )
        cfg = command.cfg(**run.cfg)
        command.fn(cfg)

    def run_chat(self, ns: argparse.Namespace) -> None:
        run, _, command = _load_run_command(
            self.ctx, self.registry, Path(ns.path), "chat"
        )
        overrides = _config_overrides(ns, command.cfg)
        cfg = command.cfg(
            run_path=run.cfg.get("run_path", self.ctx.display_path(run.path)),
            **overrides,
        )
        command.fn(cfg)

    def run_push(self, ns: argparse.Namespace) -> None:
        self.store.push(Path(ns.path))

    def run_pull(self, ns: argparse.Namespace) -> None:
        self.store.pull(Path(ns.path))

    def build_parser(self, argv: list[str]) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            prog="gradling",
            description="Gradling CLI",
            formatter_class=_formatter,
        )
        sub = parser.add_subparsers(dest="command", required=True)

        self._add_models(sub)
        self._add_experiment(sub)
        self._add_run(sub, argv)

        return parser

    def _add_models(self, sub: argparse._SubParsersAction) -> None:
        models = _subparser(sub, "models", help="List and inspect models")
        models_sub = models.add_subparsers(dest="models_command", required=True)

        p = _subparser(models_sub, "list", help="List all registered models")
        p.set_defaults(func=self.models_list)

    def _add_experiment(self, sub: argparse._SubParsersAction) -> None:
        experiment = _subparser(
            sub, "experiment", help="Create and inspect experiments"
        )
        experiment_sub = experiment.add_subparsers(
            dest="experiment_command", required=True
        )

        create = _subparser(experiment_sub, "create", help="Create an experiment")
        create_sub = create.add_subparsers(dest="model_name", required=True)

        for model_name, spec in self.registry.items():
            p = _subparser(create_sub, model_name, help=spec.description)
            p.add_argument("--name", help="Experiment name")
            p.add_argument("--notes", help="Experiment notes")
            _add_config_flags(p, spec.cfg, exclude=HIDDEN_CONFIG_FIELDS)
            p.set_defaults(
                func=partial(self.experiment_create, cfg_cls=spec.cfg),
            )

    def _add_run(self, sub: argparse._SubParsersAction, argv: list[str]) -> None:
        run = _subparser(sub, "run", help="Create, inspect, and sync runs")
        run_sub = run.add_subparsers(dest="run_command", required=True)

        create_cfg = _resolve_dynamic_cfg(
            self.ctx, self.registry, argv, "create", "create"
        )
        chat_cfg = _resolve_dynamic_cfg(self.ctx, self.registry, argv, "chat", "chat")

        p = _subparser(run_sub, "create", help="Create a run from an experiment")
        p.add_argument("experiment_path", help="Experiment directory")
        p.add_argument("--notes", default="", help="Run notes")
        if create_cfg is not None:
            _add_config_flags(p, create_cfg, exclude=HIDDEN_CONFIG_FIELDS)
        p.set_defaults(func=self.run_create)

        p = _subparser(run_sub, "list", help="List runs for an experiment")
        p.add_argument("experiment_path", help="Experiment directory")
        p.set_defaults(func=self.run_list)

        p = _subparser(run_sub, "start", help="Start training for a run")
        p.add_argument("path", help="Run directory")
        p.set_defaults(func=self.run_start)

        p = _subparser(run_sub, "chat", help="Chat with a run checkpoint")
        p.add_argument("path", help="Run directory")
        if chat_cfg is not None:
            _add_config_flags(p, chat_cfg, exclude=HIDDEN_CONFIG_FIELDS)
        p.set_defaults(func=self.run_chat)

        p = _subparser(run_sub, "push", help="Upload run checkpoints to Hugging Face")
        p.add_argument("path", help="Run directory")
        p.set_defaults(func=self.run_push)

        p = _subparser(
            run_sub, "pull", help="Download run checkpoints from Hugging Face"
        )
        p.add_argument("path", help="Run directory")
        p.set_defaults(func=self.run_pull)


def parse_args(
    ctx: Context,
    registry: dict[str, Model],
    argv: list[str] | None = None,
    *,
    store: storage.RunStorage | None = None,
) -> argparse.Namespace:
    args = argv if argv is not None else sys.argv[1:]
    if store is None:
        store = storage.RunStorage(ctx, storage.HfApiTransport(ctx))
    app = App(ctx, registry, store)
    parser = app.build_parser(args)
    return parser.parse_args(args)


def main(argv: list[str] | None = None) -> int:
    ctx = Context()
    ctx.load_dotenv()
    console = Console()
    try:
        ns = parse_args(ctx, MODELS, argv)
        ns.func(ns)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else (1 if exc.code else 0)
    except HANDLER_ERRORS as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        return 2
    return 0
