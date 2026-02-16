from pathlib import Path


def is_root(dir: Path):
    return (dir / "pyproject.toml").exists()


HERE = Path(__file__).resolve()
ROOT = next(p for p in HERE.parents if is_root(p))
DATA = ROOT / "data"
NAMES = DATA / "names.txt"
SHAKESPEARE = DATA / "shakespeare.txt"
