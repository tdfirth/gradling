from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import optax
from flax import nnx

from gradling import logger
from gradling.configs import pipeline
from gradling.configs.transformer import Transformer
from gradling.data import SHAKESPEARE, make_loader, prepare_training_data
from gradling.models.gpt import GPT
from gradling.models.gpt import Config as GPTConfig
from gradling.run import Run
from gradling.sample import sample as sample_gpt
from gradling.tokenizers import CharacterTokenizer
from gradling.train import train as train_gpt

log = logger.get(__name__)


def _load_corpus() -> str:
    with open(SHAKESPEARE) as f:
        return f.read()


def _build_model_config(cfg: Transformer, n_vocab: int) -> GPTConfig:
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


@pipeline(Transformer)
def train(cfg: Transformer) -> None:
    rngs = nnx.Rngs(cfg.seed)

    log.info("Loading data")
    corpus = _load_corpus()

    log.info("Training tokenizer")
    tok = CharacterTokenizer.train(corpus)
    log.info("Creating test, dev split")
    train_data, dev_data = prepare_training_data(tok, corpus)

    model_cfg = _build_model_config(cfg, len(tok.vocab))
    log.info("Starting training run with config %s", model_cfg)

    log.info("Initializing model")
    model = GPT(model_cfg)

    log.info("Initializing optimizer")
    optimizer = nnx.Optimizer(
        model,
        optax.adamw(model_cfg.learning_rate, model_cfg.momentum),
        wrt=nnx.Param,
    )

    log.info("Initializing metrics")
    metrics = nnx.MultiMetric(
        accuracy=nnx.metrics.Accuracy(), loss=nnx.metrics.Average("loss")
    )

    log.info("Preparing to train")
    if cfg.dry_run:
        log.info("Dry run, exiting before training")
        return

    run_cfg = asdict(cast(Any, cfg))
    run_cfg["n_vocab"] = len(tok.vocab)
    run = Run.from_config(run_cfg, family=cfg.config_name())

    def log_loss(metric, value):
        log.info(f"{metric}: {value:.4f}")

    train_gpt(
        run,
        model_cfg,
        model,
        optimizer,
        metrics,
        rngs,
        train_data,
        dev_data,
        [log_loss],
    )
    run.finalize()


@pipeline(Transformer)
def sample(cfg: Transformer) -> None:
    if not cfg.run_path:
        msg = "run_path is required for sample pipeline."
        raise ValueError(msg)

    log.info("Loading run")
    run = Run.from_path(Path(cfg.run_path))
    saved = run.cfg

    seed = int(saved.get("seed", cfg.seed))
    batch_size = int(saved.get("batch_size", cfg.batch_size))
    n_ctx = int(saved.get("n_ctx", cfg.n_ctx))
    n_emb = int(saved.get("n_emb", cfg.n_emb))
    head_size = int(saved.get("head_size", cfg.head_size))
    num_heads = int(saved.get("num_heads", cfg.num_heads))
    num_blocks = int(saved.get("num_blocks", cfg.num_blocks))
    dropout = float(saved.get("dropout", cfg.dropout))
    learning_rate = float(saved.get("learning_rate", cfg.learning_rate))
    momentum = float(saved.get("momentum", cfg.momentum))
    train_steps = int(saved.get("train_steps", cfg.train_steps))

    rngs = nnx.Rngs(seed)

    log.info("Loading data")
    corpus = _load_corpus()

    log.info("Training tokenizer")
    tok = CharacterTokenizer.train(corpus)
    log.info("Preparing data loader")
    _, dev_data = prepare_training_data(tok, corpus)
    loader = make_loader(rngs, batch_size, n_ctx, dev_data)

    log.info("Initializing model")
    model_cfg = GPTConfig(
        seed=seed,
        batch_size=batch_size,
        n_vocab=len(tok.vocab),
        n_ctx=n_ctx,
        n_emb=n_emb,
        head_size=head_size,
        num_heads=num_heads,
        num_blocks=num_blocks,
        dropout=dropout,
        learning_rate=learning_rate,
        momentum=momentum,
        train_steps=train_steps,
    )
    model = GPT(model_cfg)

    log.info("Restoring weights")
    run.load_checkpoint(cfg.checkpoint_label, model)

    log.info("Preparing inputs")
    xs, _ = next(loader)
    model.eval()
    sample_gpt(model, tok, xs)
    run.finalize()
