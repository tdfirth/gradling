from dataclasses import dataclass

from gradling.config import Config, runtime_field


@dataclass
class GPTConfig(Config):
    seed: int = 42
    n_layers: int = 12
    d_model: int = 768
    n_heads: int = 12
    d_head: int = 64
    n_ctx: int = 1024
    dropout: float = 0.0
    batch_size: int = 524_288
    micro_batch_size: int = 64
    learning_rate: float = 6.0e-4
    momentum: float = 0.9
    train_steps: int = 600_000
    dry_run: bool = False
    experiment_name: str = "baseline"
    run_path: str = runtime_field("")
    checkpoint_label: str = runtime_field("final")
