from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax
from flax import nnx
from jax import numpy as jnp
from jax import random

from gradling.data import make_loader, prepare_training_data
from gradling.models.gpt.common import load_corpus, log
from gradling.models.gpt.config import GPTConfig
from gradling.models.gpt.model import GPT, ModelConfig
from gradling.run import Run
from gradling.tokenizers import CharacterTokenizer, Tokenizer


class _SampleState(NamedTuple):
    key: jax.Array
    output: jax.Array
    ctx_buf: jax.Array


def _sample_tokens(
    model: nnx.Module,
    tok: Tokenizer,
    inputs: jax.Array,
    max_tokens: int = 1024,
) -> None:
    @nnx.jit
    def _sample(ctx: jax.Array):
        key = random.key(42)
        B, T = ctx.shape
        output = jnp.empty((B, max_tokens))
        ctx_buf = ctx.copy()

        def fn(i: int, state: _SampleState):
            logits = model(state.ctx_buf)[:, -1, :]
            key, sk = random.split(state.key)
            next_tokens = random.categorical(sk, logits)
            output = state.output.at[:, i].set(next_tokens)
            ctx_buf = jnp.concat([state.ctx_buf[:, 1:T], next_tokens[:, None]], axis=1)
            return _SampleState(key=key, output=output, ctx_buf=ctx_buf)

        state = nnx.fori_loop(
            0, max_tokens, fn, _SampleState(key=key, output=output, ctx_buf=ctx_buf)
        )
        return state.output

    output = _sample(inputs)
    for i, sample in enumerate(output):
        print(f"Sample {i}")
        print(tok.decode(sample.tolist()))
        print("")


def sample(cfg: GPTConfig) -> None:
    """Sample from a GPT."""

    if not cfg.run_path:
        msg = "run_path is required for sample command."
        raise ValueError(msg)

    log.info("Loading run")
    run = Run.from_path(Path(cfg.run_path))
    saved = run.cfg

    model_cfg = ModelConfig(
        seed=int(saved["seed"]),
        batch_size=int(saved["batch_size"]),
        n_vocab=int(saved["n_vocab"]),
        n_ctx=int(saved["n_ctx"]),
        n_emb=int(saved["n_emb"]),
        head_size=int(saved["head_size"]),
        num_heads=int(saved["num_heads"]),
        num_blocks=int(saved["num_blocks"]),
        dropout=float(saved["dropout"]),
        learning_rate=float(saved["learning_rate"]),
        momentum=float(saved["momentum"]),
        train_steps=int(saved["train_steps"]),
    )
    rngs = nnx.Rngs(model_cfg.seed)

    log.info("Loading data")
    corpus = load_corpus()

    log.info("Training tokenizer")
    tok = CharacterTokenizer.train(corpus)
    log.info("Preparing data loader")
    _, dev_data = prepare_training_data(tok, corpus)
    loader = make_loader(rngs, model_cfg.batch_size, model_cfg.n_ctx, dev_data)

    log.info("Initializing model")
    model = GPT(model_cfg)

    log.info("Restoring weights")
    run.load_checkpoint(cfg.checkpoint_label, model)

    log.info("Preparing inputs")
    xs, _ = next(loader)
    model.eval()
    _sample_tokens(model, tok, xs)
    run.finalize()
