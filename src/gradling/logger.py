import logging

from rich.logging import RichHandler


def setup(level: str = "INFO", *, jax_logging: bool = False) -> None:
    logging.basicConfig(
        level=logging.WARNING if not jax_logging else level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                rich_tracebacks=True,
                tracebacks_show_locals=True,
                markup=True,
            )
        ],
    )
    logging.getLogger("gradling").setLevel(level)


def get(name: str) -> logging.Logger:
    """Convenience wrapper around getLogger."""
    return logging.getLogger(name)
