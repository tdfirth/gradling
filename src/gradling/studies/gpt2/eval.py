from dataclasses import dataclass
from typing import cast

import optax
from flax import nnx
from jax import Array
from jax import numpy as jnp
from transformers import AutoTokenizer, TokenizersBackend

from gradling import logger
from gradling.config import Config
from gradling.evals import hellaswag
from gradling.run import Run
from gradling.studies.gpt2.config import GPTConfig
from gradling.studies.gpt2.model import GPT2

log = logger.get(__name__)


@dataclass
class EvalConfig(Config):
    max_tokens: int = 1024


def right_pad(input: list[int], n: int, tok: int) -> list[int]:
    pad_length = n - len(input)
    return input + ([tok] * pad_length)


def eval(run: Run[EvalConfig]) -> None:
    log.info("Initializing model")
    model_cfg = GPTConfig.from_dict(run.raw_cfg)
    tok = cast(TokenizersBackend, AutoTokenizer.from_pretrained("gpt2"))
    model = GPT2(model_cfg, len(tok.vocab))
    model.eval()

    log.info("Restoring weights")
    run.load_checkpoint(model_cfg.checkpoint_label, model)

    @nnx.jit
    def forward(ctx: Array, mask: Array, end_lens: Array):
        xs = ctx[:, :-1]
        ys = ctx[:, 1:]
        logits = model(xs)
        losses = optax.softmax_cross_entropy_with_integer_labels(logits, ys)
        masked = losses * mask
        mean = jnp.sum(masked, axis=-1) / end_lens
        return jnp.argmin(mean)

    log.info("Running examples")
    eot_token = tok.encode("<|endoftext|>")[0]
    n_losses = model_cfg.n_ctx - 1
    correct = 0
    total = 0
    for example in hellaswag.loader():
        prompt_toks = tok.encode(example.ctx)
        end_toks = [tok.encode(e) for e in example.endings]
        prompts = [
            right_pad(prompt_toks + e_toks, model_cfg.n_ctx, eot_token)
            for e_toks in end_toks
        ]
        mask = jnp.zeros((len(end_toks), n_losses), dtype=jnp.float32)
        for i, e_toks in enumerate(end_toks):
            mask = mask.at[
                i, len(prompt_toks) - 1 : len(prompt_toks) - 1 + len(e_toks)
            ].set(1.0)
        ctx = jnp.array(prompts).astype(jnp.int32)
        prediction = forward(ctx, mask, jnp.array([len(e) for e in end_toks]))
        total += 1
        correct += int(prediction) == int(example.label)
        if total % 100 == 0:
            log.info(f"Accuracy: {correct}/{total} ({correct / total:.2%})")
    log.info(f"Final accuracy: {correct}/{total} ({correct / total:.2%})")
