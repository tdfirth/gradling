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

from gradling import data as datalib
from gradling import run as runlib
from gradling import storage
from gradling.config import Config
from gradling.context import Context
from gradling.metrics import SinkFactory, log_only, with_wandb
from gradling.studies import STUDIES, Command, Study

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


def _studies_table(registry: dict[str, Study]) -> Table:
    table = Table(title="Registered Studies", highlight=True)
    table.add_column("Study", style="bold cyan")
    table.add_column("Config")
    table.add_column("Description", style="dim")
    for name, study in sorted(registry.items()):
        table.add_row(name, study.cfg.__name__, study.description or "-")
    return table


def _experiments_table(
    ctx: Context, study_name: str, experiments: list[runlib.Experiment]
) -> Table:
    table = Table(title=f"Experiments for {study_name}", highlight=True)
    table.add_column("Experiment", style="bold cyan")
    table.add_column("Path")
    table.add_column("Notes", style="dim")
    for exp in experiments:
        table.add_row(exp.name, ctx.display_path(exp.path), exp.notes.strip() or "-")
    return table


def _runs_table(
    ctx: Context, experiment: runlib.Experiment, run_paths: list[Path]
) -> Table:
    table = Table(
        title=f"Runs for {ctx.display_path(experiment.path)}",
        highlight=True,
    )
    table.add_column("Run", style="bold cyan")
    table.add_column("Path")
    for run_path in run_paths:
        table.add_row(run_path.name, ctx.display_path(run_path))
    return table


class App:
    def __init__(
        self,
        ctx: Context,
        registry: dict[str, Study],
        store: storage.RunStorage,
        dataset_store: storage.DatasetTransport,
        console: Console | None = None,
    ) -> None:
        self.ctx = ctx
        self.registry = registry
        self.store = store
        self.dataset_store = dataset_store
        self.console = console or Console()

    def _get_study(self, study_name: str) -> Study:
        spec = self.registry.get(study_name)
        if spec is None:
            raise RuntimeError(f"Unknown study {study_name!r}.")
        return spec

    def _experiment_path(self, study_name: str, experiment: str) -> Path:
        return self.ctx.experiments / study_name / experiment

    def _run_rel_path(self, study_name: str, experiment: str, run_id: str) -> Path:
        return Path("experiments") / study_name / experiment / "runs" / run_id

    def _resolve_experiment(
        self, study_name: str, experiment: str
    ) -> runlib.Experiment:
        return runlib.Experiment.from_path(
            self.ctx, self._experiment_path(study_name, experiment)
        )

    def _resolve_run(
        self,
        study_name: str,
        experiment: str,
        run_id: str,
        *,
        cfg_cls: type[Config],
        overrides: dict[str, Any] | None = None,
        sink_factory: SinkFactory = log_only,
    ) -> runlib.Run:
        path = self._experiment_path(study_name, experiment) / "runs" / run_id
        return runlib.Run.from_path(
            self.ctx,
            path,
            cfg_cls=cfg_cls,
            overrides=overrides,
            sink_factory=sink_factory,
        )

    def study_list(self, _ns: argparse.Namespace) -> None:
        self.console.print(_studies_table(self.registry))

    def experiment_create(self, ns: argparse.Namespace, cfg_cls: type[Config]) -> None:
        name = ns.name or self.console.input("Experiment name: ").strip()
        notes = ns.notes if ns.notes is not None else self.console.input("Notes: ")
        if not name:
            raise ValueError("Experiment name is required.")

        overrides = _config_overrides(ns, cfg_cls)
        cfg = cfg_cls(**overrides).to_dict()
        experiment = runlib.Experiment.create(self.ctx, ns.study_name, name, notes, cfg)
        self.console.print(self.ctx.display_path(experiment.path))

    def experiment_list(self, ns: argparse.Namespace) -> None:
        study_name = ns.study_name
        study_dir = self.ctx.experiments / study_name
        if not study_dir.exists():
            self.console.print(_experiments_table(self.ctx, study_name, []))
            return
        experiments = []
        for p in sorted(study_dir.iterdir()):
            if p.is_dir() and (p / "experiment.toml").exists():
                experiments.append(runlib.Experiment.from_path(self.ctx, p))
        self.console.print(_experiments_table(self.ctx, study_name, experiments))

    def run_create(self, ns: argparse.Namespace, cfg_cls: type[Config]) -> None:
        experiment = self._resolve_experiment(ns.study_name, ns.experiment)
        overrides = _config_overrides(ns, cfg_cls)
        resolved_cfg = cfg_cls(**{**experiment.cfg, **overrides}).to_dict()
        path = experiment.create_run(ns.notes or "", resolved_cfg)
        self.console.print(self.ctx.display_path(path))

    def run_list(self, ns: argparse.Namespace) -> None:
        experiment = self._resolve_experiment(ns.study_name, ns.experiment)
        self.console.print(_runs_table(self.ctx, experiment, experiment.list_runs()))

    def run_start(self, ns: argparse.Namespace) -> None:
        spec = self._get_study(ns.study_name)
        command = spec.commands.get("train")
        if command is None:
            raise RuntimeError(f"Study {ns.study_name!r} has no 'train' command.")
        run = self._resolve_run(
            ns.study_name,
            ns.experiment,
            ns.run_id,
            cfg_cls=command.cfg,
            sink_factory=command.sinks,
        )
        command.fn(run)

    def run_chat(self, ns: argparse.Namespace, command: Command) -> None:
        overrides = _config_overrides(ns, command.cfg)
        run = self._resolve_run(
            ns.study_name,
            ns.experiment,
            ns.run_id,
            cfg_cls=command.cfg,
            overrides=overrides,
            sink_factory=command.sinks,
        )
        command.fn(run)

    def debug(self, ns: argparse.Namespace, command: Command) -> None:
        logging.getLogger("gradling").setLevel(logging.DEBUG)
        experiment = self._resolve_experiment(ns.study_name, ns.experiment)
        overrides = _config_overrides(ns, command.cfg)
        resolved_cfg = command.cfg(**{**experiment.cfg, **overrides}).to_dict()
        path = experiment.create_debug_run(resolved_cfg)
        sink_factory: SinkFactory = with_wandb if ns.wandb else log_only
        run = runlib.Run.from_path(
            self.ctx,
            path,
            cfg_cls=command.cfg,
            sink_factory=sink_factory,
        )
        command.fn(run)

    def run_push(self, ns: argparse.Namespace) -> None:
        self.store.push(self._run_rel_path(ns.study_name, ns.experiment, ns.run_id))

    def run_pull(self, ns: argparse.Namespace) -> None:
        self.store.pull(self._run_rel_path(ns.study_name, ns.experiment, ns.run_id))

    def dataset_prepare(self, ns: argparse.Namespace) -> None:
        meta = datalib.prepare(self.ctx.root, ns.dataset_name)
        d = datalib.dataset_dir(self.ctx.root, ns.dataset_name)
        self.console.print(self.ctx.display_path(d))
        self.console.print(
            f"train: {meta.train_tokens:,} tokens, dev: {meta.dev_tokens:,} tokens"
        )

    def dataset_push(self, ns: argparse.Namespace) -> None:
        self.dataset_store.push(ns.dataset_name)

    def dataset_pull(self, ns: argparse.Namespace) -> None:
        self.dataset_store.pull(ns.dataset_name)

    def build_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            prog="gradling",
            description="Gradling CLI",
            formatter_class=_formatter,
        )
        sub = parser.add_subparsers(dest="command", required=True)

        self._add_study(sub)
        self._add_experiment(sub)
        self._add_run(sub)
        self._add_debug(sub)
        self._add_datasets(sub)

        return parser

    def _add_study(self, sub: argparse._SubParsersAction) -> None:
        study = _subparser(sub, "study", help="List and inspect studies")
        study_sub = study.add_subparsers(dest="study_command", required=True)

        p = _subparser(study_sub, "list", help="List all registered studies")
        p.set_defaults(func=self.study_list)

    def _add_experiment(self, sub: argparse._SubParsersAction) -> None:
        experiment = _subparser(
            sub, "experiment", help="Create and inspect experiments"
        )
        experiment_sub = experiment.add_subparsers(dest="study_name", required=True)

        for name, spec in self.registry.items():
            study_parser = _subparser(experiment_sub, name, help=spec.description)
            study_cmd_sub = study_parser.add_subparsers(
                dest="experiment_command", required=True
            )

            create = _subparser(study_cmd_sub, "create", help="Create an experiment")
            create.add_argument("--name", help="Experiment name")
            create.add_argument("--notes", help="Experiment notes")
            _add_config_flags(create, spec.cfg, exclude=HIDDEN_CONFIG_FIELDS)
            create.set_defaults(
                func=partial(self.experiment_create, cfg_cls=spec.cfg),
            )

            lst = _subparser(study_cmd_sub, "list", help="List experiments")
            lst.set_defaults(func=self.experiment_list)

    def _add_run(self, sub: argparse._SubParsersAction) -> None:
        run = _subparser(sub, "run", help="Create, inspect, and sync runs")
        run_sub = run.add_subparsers(dest="study_name", required=True)

        for name, spec in self.registry.items():
            study_parser = _subparser(run_sub, name, help=spec.description)
            study_cmd_sub = study_parser.add_subparsers(
                dest="run_command", required=True
            )

            self._add_run_create(study_cmd_sub, spec)
            self._add_run_list(study_cmd_sub)
            self._add_run_start(study_cmd_sub)
            self._add_run_chat(study_cmd_sub, spec)
            self._add_run_push(study_cmd_sub)
            self._add_run_pull(study_cmd_sub)

    def _add_run_create(self, run_sub: argparse._SubParsersAction, spec: Study) -> None:
        create = _subparser(run_sub, "create", help="Create a run from an experiment")
        create.add_argument("experiment", help="Experiment name")
        create.add_argument("--notes", default="", help="Run notes")
        _add_config_flags(create, spec.cfg, exclude=HIDDEN_CONFIG_FIELDS)
        create.set_defaults(func=partial(self.run_create, cfg_cls=spec.cfg))

    def _add_run_list(self, run_sub: argparse._SubParsersAction) -> None:
        p = _subparser(run_sub, "list", help="List runs for an experiment")
        p.add_argument("experiment", help="Experiment name")
        p.set_defaults(func=self.run_list)

    def _add_run_start(self, run_sub: argparse._SubParsersAction) -> None:
        p = _subparser(run_sub, "start", help="Start training for a run")
        p.add_argument("experiment", help="Experiment name")
        p.add_argument("run_id", help="Run ID")
        p.set_defaults(func=self.run_start)

    def _add_run_chat(self, run_sub: argparse._SubParsersAction, spec: Study) -> None:
        chat_cmd = spec.commands.get("chat")
        if chat_cmd is None:
            return
        p = _subparser(run_sub, "chat", help="Chat with a run checkpoint")
        p.add_argument("experiment", help="Experiment name")
        p.add_argument("run_id", help="Run ID")
        _add_config_flags(p, chat_cmd.cfg, exclude=HIDDEN_CONFIG_FIELDS)
        p.set_defaults(func=partial(self.run_chat, command=chat_cmd))

    def _add_run_push(self, run_sub: argparse._SubParsersAction) -> None:
        p = _subparser(run_sub, "push", help="Upload run checkpoints to Hugging Face")
        p.add_argument("experiment", help="Experiment name")
        p.add_argument("run_id", help="Run ID")
        p.set_defaults(func=self.run_push)

    def _add_debug(self, sub: argparse._SubParsersAction) -> None:
        debug = _subparser(
            sub, "debug", help="Run a command in debug mode (no wandb, overwrites run)"
        )
        debug_sub = debug.add_subparsers(dest="study_name", required=True)

        for name, spec in self.registry.items():
            study_parser = _subparser(debug_sub, name, help=spec.description)
            cmd_sub = study_parser.add_subparsers(dest="debug_command", required=True)

            for cmd_name, cmd in spec.commands.items():
                p = _subparser(cmd_sub, cmd_name, help=f"Debug {cmd_name}")
                p.add_argument("experiment", help="Experiment name")
                p.add_argument(
                    "--wandb", action="store_true", default=False, help="Enable wandb"
                )
                _add_config_flags(p, cmd.cfg, exclude=HIDDEN_CONFIG_FIELDS)
                p.set_defaults(func=partial(self.debug, command=cmd))

    def _add_datasets(self, sub: argparse._SubParsersAction) -> None:
        datasets = _subparser(sub, "datasets", help="Prepare and sync datasets")
        ds_sub = datasets.add_subparsers(dest="datasets_command", required=True)

        ds_name_help = "HF dataset name (e.g. roneneldan/TinyStories)"

        p = _subparser(ds_sub, "prepare", help="Download and tokenize a dataset")
        p.add_argument("dataset_name", help=ds_name_help)
        p.set_defaults(func=self.dataset_prepare)

        p = _subparser(ds_sub, "push", help="Upload a prepared dataset to HF")
        p.add_argument("dataset_name", help=ds_name_help)
        p.set_defaults(func=self.dataset_push)

        p = _subparser(ds_sub, "pull", help="Download a prepared dataset from HF")
        p.add_argument("dataset_name", help=ds_name_help)
        p.set_defaults(func=self.dataset_pull)

    def _add_run_pull(self, run_sub: argparse._SubParsersAction) -> None:
        p = _subparser(
            run_sub, "pull", help="Download run checkpoints from Hugging Face"
        )
        p.add_argument("experiment", help="Experiment name")
        p.add_argument("run_id", help="Run ID")
        p.set_defaults(func=self.run_pull)


def parse_args(
    ctx: Context,
    registry: dict[str, Study],
    argv: list[str] | None = None,
    *,
    store: storage.RunStorage | None = None,
    dataset_store: storage.DatasetTransport | None = None,
) -> argparse.Namespace:
    args = argv if argv is not None else sys.argv[1:]
    if store is None:
        store = storage.RunStorage(ctx, storage.HfApiTransport(ctx))
    if dataset_store is None:
        dataset_store = storage.HfDatasetStorage(ctx)
    app = App(ctx, registry, store, dataset_store)
    parser = app.build_parser()
    return parser.parse_args(args)


def main(argv: list[str] | None = None) -> int:
    ctx = Context()
    ctx.load_dotenv()
    console = Console()
    try:
        ns = parse_args(ctx, STUDIES, argv)
        ns.func(ns)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else (1 if exc.code else 0)
    except HANDLER_ERRORS as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        return 2
    return 0
