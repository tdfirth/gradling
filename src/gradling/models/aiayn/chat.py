import queue
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, cast

import jax
from flax import nnx
from jax import lax, random
from jax import numpy as jnp
from jax.experimental import io_callback
from transformers import AutoTokenizer, TokenizersBackend

from gradling import logger
from gradling.config import Config
from gradling.models.aiayn.model import AIAYN, AIAYNConfig
from gradling.run import Run

log = logger.get(__name__)


@dataclass
class ChatConfig(Config):
    run_path: str
    max_tokens: int = 1024


def right_pad(input: list[int], n: int, tok: int) -> list[int]:
    pad_length = n - len(input)
    return input + ([tok] * pad_length)


def chat(cfg: ChatConfig) -> None:
    """Chat with a model"""
    if not cfg.run_path:
        msg = "run_path is required for sample command."
        raise ValueError(msg)

    log.info("Loading run")
    run = Run.from_path(Path(cfg.run_path))
    model_cfg = AIAYNConfig(**run.cfg)

    log.info("Initializing model")
    tok = cast(TokenizersBackend, AutoTokenizer.from_pretrained("gpt2"))
    model = AIAYN(model_cfg, len(tok.vocab))
    model.eval()

    log.info("Restoring weights")
    run.load_checkpoint(model_cfg.checkpoint_label, model)

    class SampleState(NamedTuple):
        key: jax.Array
        i: jax.Array
        pos: jax.Array
        stop: jax.Array
        output: jax.Array
        ctx_buf: jax.Array

    eot_token = tok.encode("<|endoftext|>")[0]
    q = queue.Queue()

    def put_token(token: jnp.ndarray):
        q.put(int(token))
        return ()  # io_callback must return an array. Empty tumple means no output.

    @nnx.jit
    def sample_loop(ctx: jax.Array, prompt_len: int) -> None:
        B, T = ctx.shape  # For now we're assuming that B is always 1.
        output = jnp.empty((B, cfg.max_tokens), dtype=jnp.int32)
        ctx_buf = ctx.copy()

        def should_continue(state: SampleState):
            end_of_text = state.stop
            max_tokens = state.i >= cfg.max_tokens
            return ~(end_of_text | max_tokens)

        def loop_body(state: SampleState):
            key, sub_key = random.split(state.key)
            logits = model(state.ctx_buf)[:, state.pos, :]
            predictions = random.categorical(sub_key, logits)
            next_token = predictions[0]
            io_callback(put_token, (), next_token, ordered=True)
            output = state.output.at[:, state.i].set(predictions)
            next_pos = state.pos + 1
            # If we're still within the context window, write into the next slot.
            # Otherwise, shift left and append at the end.
            ctx_in_bounds = state.ctx_buf.at[:, next_pos].set(predictions)
            ctx_shifted = jnp.concat(
                [state.ctx_buf[:, 1:], predictions[:, None]], axis=1
            )
            at_end = next_pos >= T
            ctx_buf = jnp.where(at_end, ctx_shifted, ctx_in_bounds)
            pos = jnp.where(at_end, state.pos, next_pos)

            return SampleState(
                key=key,
                i=state.i + 1,
                pos=pos,
                stop=next_token == eot_token,
                output=output,
                ctx_buf=ctx_buf,
            )

        lax.while_loop(
            should_continue,
            loop_body,
            SampleState(
                key=random.key(42),
                i=jnp.int32(0),
                pos=jnp.int32(prompt_len - 1),
                stop=jnp.bool(False),
                output=output,
                ctx_buf=ctx_buf,
            ),
        )

    while True:
        prompt = input("> ")
        encoded = tok.encode(prompt)
        tokens = right_pad(encoded, model_cfg.n_ctx, eot_token)
        ctx = jnp.array([tokens]).astype(jnp.int32)

        done = threading.Event()

        def run_sample():
            sample_loop(ctx, len(encoded))
            jax.effects_barrier()
            done.set()

        threading.Thread(target=run_sample).start()

        decoded_so_far = ""
        token_buffer = []
        while not done.is_set() or not q.empty():
            try:
                next_token = q.get(timeout=0.05)
            except queue.Empty:
                continue
            if next_token == eot_token:
                break
            token_buffer.append(next_token)
            new_decoded = str(tok.decode(token_buffer))
            new_text = new_decoded[len(decoded_so_far) :]
            if new_text:
                sys.stdout.write(new_text)
                sys.stdout.flush()
                decoded_so_far = new_decoded
        sys.stdout.write("\n")
        sys.stdout.flush()
