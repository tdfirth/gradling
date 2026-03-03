from gradling import logger

logger.setup()

from gradling import cli  # noqa: E402


def main() -> None:
    raise SystemExit(cli.main())
