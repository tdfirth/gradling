from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _is_root(dir: Path):
    return (dir / "pyproject.toml").exists()


_here = Path(__file__).resolve()
ROOT = next(p for p in _here.parents if _is_root(p))


def load_dotenv(path: Path | None = None) -> None:
    env_path = path or (ROOT / ".env")
    if not env_path.exists():
        return

    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


@dataclass(frozen=True)
class Context:
    root: Path = ROOT

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def experiments(self) -> Path:
        return self.root / "experiments"

    @property
    def env_path(self) -> Path:
        return self.root / ".env"

    @property
    def wandb_api_key(self) -> str | None:
        self.load_dotenv()
        return os.environ.get("WANDB_API_KEY")

    def load_dotenv(self) -> None:
        load_dotenv(self.env_path)

    def resolve_path(self, path: Path) -> Path:
        return path if path.is_absolute() else self.root / path

    def display_path(self, path: Path) -> str:
        try:
            return path.relative_to(self.root).as_posix()
        except ValueError:
            return path.as_posix()
