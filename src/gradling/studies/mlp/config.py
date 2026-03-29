from dataclasses import dataclass

from gradling.config import Config, runtime_field


@dataclass
class MLPConfig(Config):
    rng_seed: int = 42
    ctx_length: int = 8
    emb_size: int = 24
    hidden_size: int = 200
    vocab_size: int = runtime_field(0)
