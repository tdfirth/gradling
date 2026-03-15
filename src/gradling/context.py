from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from gradling.dir import ROOT
from gradling.env import load_dotenv


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

    def load_dotenv(self) -> None:
        load_dotenv(self.env_path)

    def resolve_path(self, path: Path) -> Path:
        return path if path.is_absolute() else self.root / path

    def display_path(self, path: Path) -> str:
        try:
            return path.relative_to(self.root).as_posix()
        except ValueError:
            return path.as_posix()
