import marimo

__generated_with = "0.20.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import re

    import jax
    from jax import numpy as jnp

    from gradling.models.gpt import GPT, GPTConfig

    cfg = GPTConfig()

    model = GPT(cfg, 65)

    def format_path(path):
        return "".join(format(p) for p in path)

    def path_matches(path, regex):
        result = re.search(regex, format_path(path))
        return result is not None

    attn = [
        [format_path(path), value]
        for path, value in jax.tree.leaves_with_path(model)
        if path_matches(path, r"sa_heads\.attn")
    ]

    for p, value in attn:
        print(p, jnp.linalg.norm(value))

    # model
    return


if __name__ == "__main__":
    app.run()
