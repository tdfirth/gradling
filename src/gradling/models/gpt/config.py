from dataclasses import dataclass

from gradling.config import Config


@dataclass
class GPTConfig(Config):
    seed: int = 42
    batch_size: int = 32
    n_ctx: int = 8
    n_emb: int = 32
    head_size: int = 32
    num_heads: int = 4
    num_blocks: int = 3
    dropout: float = 0.1
    learning_rate: float = 1e-3
    momentum: float = 0.9
    train_steps: int = 10_000
    dry_run: bool = False
    run_path: str = ""
    checkpoint_label: str = "final"
