from gradling import logger
from gradling.data import SHAKESPEARE

log = logger.get(__name__)


def load_corpus() -> str:
    with open(SHAKESPEARE) as f:
        return f.read()
