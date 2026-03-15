from gradling.config import Config
from gradling.data import create_dataset
from gradling.run import Run


class DataConfig(Config):
    dir: str = "data/cache"


def data(run: Run[DataConfig]) -> None:
    tok, train, _ = create_dataset("roneneldan/TinyStories")
    print(f"Vocab size: {len(tok.vocab)}")
    print(train.shape)
