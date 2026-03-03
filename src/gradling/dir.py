from pathlib import Path

__all__ = ["ROOT"]


def _is_root(dir: Path):
    return (dir / "pyproject.toml").exists()


_here = Path(__file__).resolve()
ROOT = next(p for p in _here.parents if _is_root(p))
