from __future__ import annotations

import argparse
import logging
import sys
from functools import partial
from pathlib import Path
from types import UnionType
from typing import TYPE_CHECKING, Any, Union, get_args, get_origin, get_type_hints

import argcomplete

if TYPE_CHECKING:
    from gradling import run as runlib
    from gradling import storage
from rich.console import Console
from rich.table import Table
from rich_argparse import RichHelpFormatter
from tomlkit.exceptions import ParseError

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


def _parse_target(
    target: str, *, expect_run: bool = False
) -> tuple[str, str | None, str | None]:
    parts = target.split("/")
    study = parts[0]
    experiment = parts[1] if len(parts) > 1 else None
    run_id = parts[2] if len(parts) > 2 else None
    if expect_run and run_id is None:
        raise ValueError(f"Expected study/experiment/run_id, got {target!r}")
    return study, experiment, run_id


def _get_study(registry: dict[str, Study], name: str) -> Study:
    spec = registry.get(name)
    if spec is None:
        raise RuntimeError(f"Unknown study {name!r}.")
    return spec


def _experiment_path(ctx: Context, study_name: str, experiment: str) -> Path:
    return ctx.experiments / study_name / experiment


def _resolve_experiment(
    ctx: Context, study_name: str, experiment: str
) -> runlib.Experiment:
    from gradling import run as runlib

    return runlib.Experiment.from_path(
        ctx, _experiment_path(ctx, study_name, experiment)
    )


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


def _complete_study(experiments_dir: Path, prefix: str, **kwargs: Any) -> list[str]:
    if not experiments_dir.exists():
        return []
    return [
        d.name + "/"
        for d in sorted(experiments_dir.iterdir())
        if d.is_dir() and d.name.startswith(prefix)
    ]


def _complete_experiment_target(
    experiments_dir: Path, prefix: str, **kwargs: Any
) -> list[str]:
    parts = prefix.split("/")
    if len(parts) <= 1:
        return _complete_study(experiments_dir, prefix)

    study_dir = experiments_dir / parts[0]
    if not study_dir.exists():
        return []
    exp_prefix = parts[1]
    return [
        f"{parts[0]}/{d.name}"
        for d in sorted(study_dir.iterdir())
        if d.is_dir()
        and (d / "experiment.toml").exists()
        and d.name.startswith(exp_prefix)
    ]


def _complete_run_target(
    experiments_dir: Path, prefix: str, **kwargs: Any
) -> list[str]:
    parts = prefix.split("/")
    if len(parts) <= 1:
        return _complete_study(experiments_dir, prefix)

    if len(parts) == 2:
        study_dir = experiments_dir / parts[0]
        if not study_dir.exists():
            return []
        exp_prefix = parts[1]
        return [
            f"{parts[0]}/{d.name}/"
            for d in sorted(study_dir.iterdir())
            if d.is_dir()
            and (d / "experiment.toml").exists()
            and d.name.startswith(exp_prefix)
        ]

    runs_dir = experiments_dir / parts[0] / parts[1] / "runs"
    if not runs_dir.exists():
        return []
    run_prefix = parts[2]
    return [
        f"{parts[0]}/{parts[1]}/{d.name}"
        for d in sorted(runs_dir.iterdir())
        if d.is_dir() and d.name.startswith(run_prefix)
    ]


class StudySubcommand:
    def __init__(self, registry: dict[str, Study], console: Console) -> None:
        self.registry = registry
        self.console = console

    def register(self, sub: argparse._SubParsersAction) -> None:
        study = _subparser(sub, "study", help="List and inspect studies", aliases=["s"])
        study_sub = study.add_subparsers(dest="study_command", required=True)
        p = _subparser(study_sub, "list", help="List all registered studies")
        p.set_defaults(func=self.list)

    def list(self, _ns: argparse.Namespace) -> None:
        self.console.print(_studies_table(self.registry))


class ExperimentSubcommand:
    def __init__(
        self, ctx: Context, registry: dict[str, Study], console: Console
    ) -> None:
        self.ctx = ctx
        self.registry = registry
        self.console = console

    def register(self, sub: argparse._SubParsersAction) -> None:
        experiment = _subparser(
            sub,
            "experiment",
            help="Create and inspect experiments",
            aliases=["x"],
            add_help=False,
        )
        completer = partial(_complete_study, self.ctx.experiments)
        experiment.add_argument("target", help="Study name").completer = completer  # type: ignore[attr-defined]
        experiment.set_defaults(func=self.dispatch)

    def _build_command_parser(
        self, target: str, spec: Study
    ) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            prog=f"gradling experiment {target}",
            formatter_class=_formatter,
        )
        cmd_sub = parser.add_subparsers(dest="experiment_command", required=True)

        create = _subparser(cmd_sub, "create", help="Create an experiment")
        create.add_argument("--name", help="Experiment name")
        create.add_argument("--notes", help="Experiment notes")
        _add_config_flags(create, spec.cfg, exclude=HIDDEN_CONFIG_FIELDS)

        _subparser(cmd_sub, "list", help="List experiments")

        return parser

    def dispatch(self, ns: argparse.Namespace) -> None:
        study_name = ns.target.rstrip("/")
        spec = _get_study(self.registry, study_name)
        remaining: list[str] = getattr(ns, "_remaining", [])

        parser = self._build_command_parser(study_name, spec)
        cmd_ns = parser.parse_args(remaining)

        match cmd_ns.experiment_command:
            case "create":
                self._create(study_name, spec, cmd_ns)
            case "list":
                self._list(study_name)

    def _create(
        self,
        study_name: str,
        spec: Study,
        cmd_ns: argparse.Namespace,
    ) -> None:
        overrides = _config_overrides(cmd_ns, spec.cfg)
        name = cmd_ns.name or self.console.input("Experiment name: ").strip()
        notes = (
            cmd_ns.notes if cmd_ns.notes is not None else self.console.input("Notes: ")
        )
        if not name:
            raise ValueError("Experiment name is required.")

        from gradling import run as runlib

        cfg = spec.cfg(**overrides).to_dict()
        experiment = runlib.Experiment.create(self.ctx, study_name, name, notes, cfg)
        self.console.print(self.ctx.display_path(experiment.path))

    def _list(self, study_name: str) -> None:
        from gradling import run as runlib

        study_dir = self.ctx.experiments / study_name
        if not study_dir.exists():
            self.console.print(_experiments_table(self.ctx, study_name, []))
            return
        experiments = []
        for p in sorted(study_dir.iterdir()):
            if p.is_dir() and (p / "experiment.toml").exists():
                experiments.append(runlib.Experiment.from_path(self.ctx, p))
        self.console.print(_experiments_table(self.ctx, study_name, experiments))


class RunSubcommand:
    def __init__(
        self,
        ctx: Context,
        registry: dict[str, Study],
        store: storage.RunStorage,
        console: Console,
    ) -> None:
        self.ctx = ctx
        self.registry = registry
        self.store = store
        self.console = console

    def register(self, sub: argparse._SubParsersAction) -> None:
        run = _subparser(
            sub,
            "run",
            help="Create, inspect, and sync runs",
            aliases=["r"],
            add_help=False,
        )
        completer = partial(_complete_run_target, self.ctx.experiments)
        run.add_argument(
            "target", help="study/experiment[/run_id]"
        ).completer = completer  # type: ignore[attr-defined]
        run.set_defaults(func=self.dispatch)

    def _build_command_parser(
        self, target: str, spec: Study
    ) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            prog=f"gradling run {target}",
            formatter_class=_formatter,
        )
        cmd_sub = parser.add_subparsers(dest="run_command", required=True)

        create = _subparser(cmd_sub, "create", help="Create a run from an experiment")
        create.add_argument("--notes", default="", help="Run notes")
        _add_config_flags(create, spec.cfg, exclude=HIDDEN_CONFIG_FIELDS)

        _subparser(cmd_sub, "list", help="List runs for an experiment")

        for cmd_name, cmd in spec.commands.items():
            p = _subparser(cmd_sub, cmd_name, help=f"Run `{cmd_name}`")
            _add_config_flags(p, cmd.cfg, exclude=HIDDEN_CONFIG_FIELDS)

        _subparser(cmd_sub, "push", help="Upload run checkpoints to Hugging Face")
        _subparser(
            cmd_sub,
            "pull",
            help="Download run checkpoints from Hugging Face",
        )

        return parser

    def dispatch(self, ns: argparse.Namespace) -> None:
        target = ns.target.rstrip("/")
        study_name, experiment_name, run_id = _parse_target(target)
        spec = _get_study(self.registry, study_name)
        remaining: list[str] = getattr(ns, "_remaining", [])

        parser = self._build_command_parser(target, spec)
        cmd_ns = parser.parse_args(remaining)

        match cmd_ns.run_command:
            case "create":
                self._create(study_name, experiment_name, spec, cmd_ns)
            case "list":
                self._list(study_name, experiment_name)
            case "push":
                self._push(study_name, experiment_name, run_id)
            case "pull":
                self._pull(study_name, experiment_name, run_id)
            case cmd_name:
                cmd = spec.commands[cmd_name]
                overrides = _config_overrides(cmd_ns, cmd.cfg)
                self._command(study_name, experiment_name, run_id, cmd, overrides)

    def _run_rel_path(self, study_name: str, experiment: str, run_id: str) -> Path:
        return Path("experiments") / study_name / experiment / "runs" / run_id

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
        from gradling import run as runlib

        path = _experiment_path(self.ctx, study_name, experiment) / "runs" / run_id
        return runlib.Run.from_path(
            self.ctx,
            path,
            cfg_cls=cfg_cls,
            overrides=overrides,
            sink_factory=sink_factory,
        )

    def _create(
        self,
        study_name: str,
        experiment_name: str | None,
        spec: Study,
        cmd_ns: argparse.Namespace,
    ) -> None:
        if experiment_name is None:
            raise ValueError("Expected study/experiment for create")
        overrides = _config_overrides(cmd_ns, spec.cfg)
        experiment = _resolve_experiment(self.ctx, study_name, experiment_name)
        resolved_cfg = spec.cfg(**{**experiment.cfg, **overrides}).to_dict()
        path = experiment.create_run(cmd_ns.notes or "", resolved_cfg)
        self.console.print(self.ctx.display_path(path))

    def _list(self, study_name: str, experiment_name: str | None) -> None:
        if experiment_name is None:
            raise ValueError("Expected study/experiment for list")
        experiment = _resolve_experiment(self.ctx, study_name, experiment_name)
        self.console.print(_runs_table(self.ctx, experiment, experiment.list_runs()))

    def _command(
        self,
        study_name: str,
        experiment_name: str | None,
        run_id: str | None,
        cmd: Command,
        overrides: dict[str, Any],
    ) -> None:
        if experiment_name is None or run_id is None:
            raise ValueError("Expected study/experiment/run_id for commands")
        run = self._resolve_run(
            study_name,
            experiment_name,
            run_id,
            cfg_cls=cmd.cfg,
            overrides=overrides,
            sink_factory=cmd.sinks,
        )
        cmd.fn(run)

    def _push(
        self,
        study_name: str,
        experiment_name: str | None,
        run_id: str | None,
    ) -> None:
        if experiment_name is None or run_id is None:
            raise ValueError("Expected study/experiment/run_id for push")
        self.store.push(self._run_rel_path(study_name, experiment_name, run_id))

    def _pull(
        self,
        study_name: str,
        experiment_name: str | None,
        run_id: str | None,
    ) -> None:
        if experiment_name is None or run_id is None:
            raise ValueError("Expected study/experiment/run_id for pull")
        self.store.pull(self._run_rel_path(study_name, experiment_name, run_id))


class DebugSubcommand:
    def __init__(
        self, ctx: Context, registry: dict[str, Study], console: Console
    ) -> None:
        self.ctx = ctx
        self.registry = registry
        self.console = console

    def register(self, sub: argparse._SubParsersAction) -> None:
        debug = _subparser(
            sub,
            "debug",
            help="Run a command in debug mode (no wandb, overwrites run)",
            aliases=["d"],
            add_help=False,
        )
        completer = partial(_complete_experiment_target, self.ctx.experiments)
        debug.add_argument("target", help="study/experiment").completer = completer  # type: ignore[attr-defined]
        debug.set_defaults(func=self.dispatch)

    def _build_command_parser(
        self, target: str, spec: Study
    ) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            prog=f"gradling debug {target}",
            formatter_class=_formatter,
        )
        cmd_sub = parser.add_subparsers(dest="debug_command", required=True)
        for cmd_name, cmd in spec.commands.items():
            p = _subparser(cmd_sub, cmd_name, help=f"Debug {cmd_name}")
            p.add_argument(
                "--wandb",
                action="store_true",
                default=False,
                help="Enable wandb",
            )
            _add_config_flags(p, cmd.cfg, exclude=HIDDEN_CONFIG_FIELDS)
        return parser

    def dispatch(self, ns: argparse.Namespace) -> None:
        target = ns.target.rstrip("/")
        study_name, experiment_name, _ = _parse_target(target)
        if experiment_name is None:
            raise ValueError(f"Expected study/experiment, got {target!r}")
        spec = _get_study(self.registry, study_name)
        remaining: list[str] = getattr(ns, "_remaining", [])

        parser = self._build_command_parser(target, spec)
        cmd_ns = parser.parse_args(remaining)

        from gradling import run as runlib

        logging.getLogger("gradling").setLevel(logging.DEBUG)
        command = spec.commands[cmd_ns.debug_command]
        overrides = _config_overrides(cmd_ns, command.cfg)
        experiment = _resolve_experiment(self.ctx, study_name, experiment_name)
        resolved_cfg = command.cfg(**{**experiment.cfg, **overrides}).to_dict()
        path = experiment.create_debug_run(resolved_cfg)
        sink_factory: SinkFactory = with_wandb if cmd_ns.wandb else log_only
        run = runlib.Run.from_path(
            self.ctx,
            path,
            cfg_cls=command.cfg,
            sink_factory=sink_factory,
        )
        command.fn(run)


class DatasetSubcommand:
    def __init__(
        self,
        ctx: Context,
        dataset_store: storage.DatasetTransport,
        console: Console,
    ) -> None:
        self.ctx = ctx
        self.dataset_store = dataset_store
        self.console = console

    def register(self, sub: argparse._SubParsersAction) -> None:
        datasets = _subparser(
            sub, "datasets", help="Prepare and sync datasets", aliases=["ds"]
        )
        ds_sub = datasets.add_subparsers(dest="datasets_command", required=True)

        ds_name_help = "HF dataset name (e.g. roneneldan/TinyStories)"
        config_help = "HF dataset config/subset (e.g. sample-10BT)"

        p = _subparser(ds_sub, "prepare", help="Download and tokenize a dataset")
        p.add_argument("dataset_name", help=ds_name_help)
        p.add_argument("--config", default=None, help=config_help)
        p.set_defaults(func=self.prepare)

        p = _subparser(ds_sub, "push", help="Upload a prepared dataset to HF")
        p.add_argument("dataset_name", help=ds_name_help)
        p.add_argument("--config", default=None, help=config_help)
        p.set_defaults(func=self.push)

        p = _subparser(ds_sub, "pull", help="Download a prepared dataset from HF")
        p.add_argument("dataset_name", help=ds_name_help)
        p.add_argument("--config", default=None, help=config_help)
        p.set_defaults(func=self.pull)

    def prepare(self, ns: argparse.Namespace) -> None:
        from gradling import data as datalib

        meta = datalib.prepare(self.ctx.root, ns.dataset_name, ns.config)
        d = datalib.dataset_dir(self.ctx.root, ns.dataset_name, ns.config)
        self.console.print(self.ctx.display_path(d))
        self.console.print(
            f"train: {meta.train_tokens:,} tokens, dev: {meta.dev_tokens:,} tokens"
        )

    def push(self, ns: argparse.Namespace) -> None:
        self.dataset_store.push(ns.dataset_name, ns.config)

    def pull(self, ns: argparse.Namespace) -> None:
        self.dataset_store.pull(ns.dataset_name, ns.config)


class App:
    def __init__(
        self,
        ctx: Context,
        registry: dict[str, Study],
        store: storage.RunStorage,
        dataset_store: storage.DatasetTransport,
        console: Console | None = None,
    ) -> None:
        console = console or Console()
        self._subcommands = [
            StudySubcommand(registry, console),
            ExperimentSubcommand(ctx, registry, console),
            RunSubcommand(ctx, registry, store, console),
            DebugSubcommand(ctx, registry, console),
            DatasetSubcommand(ctx, dataset_store, console),
        ]

    def build_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            prog="gradling",
            description="Gradling CLI",
            formatter_class=_formatter,
        )
        sub = parser.add_subparsers(dest="command", required=True)
        for sc in self._subcommands:
            sc.register(sub)
        return parser


def parse_args(
    ctx: Context,
    registry: dict[str, Study],
    argv: list[str] | None = None,
    *,
    store: storage.RunStorage | None = None,
    dataset_store: storage.DatasetTransport | None = None,
) -> argparse.Namespace:
    from gradling import storage

    args = argv if argv is not None else sys.argv[1:]
    if store is None:
        store = storage.RunStorage(ctx, storage.HfApiTransport(ctx))
    if dataset_store is None:
        dataset_store = storage.HfDatasetStorage(ctx)
    app = App(ctx, registry, store, dataset_store)
    parser = app.build_parser()
    argcomplete.autocomplete(parser)
    ns, remaining = parser.parse_known_args(args)
    ns._remaining = remaining
    return ns


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
