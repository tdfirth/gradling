from gradling.run import Run
from gradling.studies.mlp.config import MLPConfig
from gradling.studies.mlp.model import train as _train_impl


def train(run: Run[MLPConfig]) -> None:
    _train_impl(run.cfg)
