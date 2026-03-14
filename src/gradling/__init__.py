import os

from gradling import logger

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

logger.setup()

from gradling import cli  # noqa: E402


def main() -> None:
    raise SystemExit(cli.main())
