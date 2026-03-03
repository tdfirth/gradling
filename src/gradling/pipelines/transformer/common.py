from __future__ import annotations

from gradling import logger
from gradling.configs.transformer import Transformer
from gradling.data import SHAKESPEARE
from gradling.models.gpt import Config as GPTConfig

log = logger.get(__name__)


def load_corpus() -> str:
    with open(SHAKESPEARE) as f:
        return f.read()


def build_model_config(cfg: Transformer, n_vocab: int) -> GPTConfig:
    return GPTConfig(
        seed=cfg.seed,
        batch_size=cfg.batch_size,
        n_vocab=n_vocab,
        n_ctx=cfg.n_ctx,
        n_emb=cfg.n_emb,
        head_size=cfg.head_size,
        num_heads=cfg.num_heads,
        num_blocks=cfg.num_blocks,
        dropout=cfg.dropout,
        learning_rate=cfg.learning_rate,
        momentum=cfg.momentum,
        train_steps=cfg.train_steps,
    )
