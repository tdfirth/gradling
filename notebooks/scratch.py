import marimo

__generated_with = "0.20.1"
app = marimo.App(width="medium")


@app.cell
def _():
    from gradling.train import train_step, eval_step, train
    from flax import nnx
    import jax.numpy as jnp
    import jax
    import optax
    import marimo as mo
    import matplotlib.pyplot as plt

    from gradling.data import make_loader, prepare_training_data, SHAKESPEARE
    from gradling.tokenizers import CharacterTokenizer

    jnp.set_printoptions(precision=4, linewidth=200)
    return (
        CharacterTokenizer,
        SHAKESPEARE,
        make_loader,
        mo,
        nnx,
        optax,
        plt,
        prepare_training_data,
        train,
    )


@app.cell
def _(CharacterTokenizer, SHAKESPEARE, prepare_training_data):
    with open(SHAKESPEARE, "r") as f:
        CORPUS = f.read()

    TOK = CharacterTokenizer.train(CORPUS)
    TRAIN, DEV = prepare_training_data(TOK, CORPUS)
    return DEV, TOK, TRAIN


@app.cell(disabled=True, hide_code=True)
def _():
    # from gradling.train_gpt import Bigram

    # jnp.set_printoptions(precision=4, linewidth=200)

    # learning_rate = 0.005
    # momentum = 0.9
    # train_steps = 10

    # model = Bigram(CFG)
    # optimizer = nnx.Optimizer(
    #     model, optax.adamw(learning_rate, momentum), wrt=nnx.Param
    # )
    # metrics = nnx.MultiMetric(
    #     accuracy=nnx.metrics.Accuracy(), loss=nnx.metrics.Average("loss")
    # )

    # metric_history = {
    #     "train_loss": [],
    #     "train_accuracy": [],
    #     "dev_loss": [],
    #     "dev_accuracy": [],
    # }

    # rngs = nnx.Rngs(42)

    # def render_metrics(metric_history):
    #     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    #     ax1.set_title("Loss")
    #     ax2.set_title("Accuracy")
    #     for dataset in ("train", "dev"):
    #         l, a = f"{dataset}_loss", f"{dataset}_accuracy"
    #         ax1.plot(metric_history[l], label=l)
    #         ax2.plot(metric_history[a], label=a)
    #     ax1.legend()
    #     ax2.legend()
    #     mo.output.replace(fig)
    #     plt.close(fig)

    # train(
    #     CFG,
    #     model,
    #     optimizer,
    #     metrics,
    #     metric_history,
    #     rngs,
    #     loader,
    #     dev_loader,
    #     steps=train_steps,
    #     metric_sinks=[render_metrics],
    # )
    return


@app.cell
def _(DEV, TOK, TRAIN, make_loader, mo, nnx, optax, plt, train):
    from gradling.models.gpt import GPT, Config

    CFG = Config(
        seed=42,
        n_ctx=8,
        n_vocab=len(TOK.vocab),
        n_emb=32,
        head_size=32,
        num_heads=4,
        num_blocks=3,
    )

    rngs = nnx.Rngs(CFG.seed)
    loader = make_loader(rngs, 32, 8, TRAIN)
    dev_loader = make_loader(rngs, 32, 8, DEV)

    learning_rate = 1e-3
    momentum = 0.9
    train_steps = 2_000

    model = GPT(CFG)
    optimizer = nnx.Optimizer(
        model, optax.adamw(learning_rate, momentum), wrt=nnx.Param
    )
    metrics = nnx.MultiMetric(
        accuracy=nnx.metrics.Accuracy(), loss=nnx.metrics.Average("loss")
    )

    metric_history = {
        "train_loss": [],
        "train_accuracy": [],
        "dev_loss": [],
        "dev_accuracy": [],
    }

    rngs = nnx.Rngs(42)

    def render_metrics(metric_history):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        ax1.set_title("Loss")
        ax2.set_title("Accuracy")
        for dataset in ("train", "dev"):
            l, a = f"{dataset}_loss", f"{dataset}_accuracy"
            ax1.plot(metric_history[l], label=l)
            ax2.plot(metric_history[a], label=a)
        ax1.legend()
        ax2.legend()
        mo.output.replace(fig)
        plt.close(fig)

    train(
        CFG,
        model,
        optimizer,
        metrics,
        metric_history,
        rngs,
        loader,
        dev_loader,
        steps=train_steps,
        metric_sinks=[render_metrics],
    )

    print(f"Loss: {metric_history['dev_loss'][-1]}")
    return dev_loader, model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Journal
    - Attention: 2.37
    - MultiHead: 2.25
    - FeedForward: 2.18
    - MultipleBlocks:2.19 (worse... optimization issues)
    - Residual Connections: 2.09
    - 4x size in FFN: 2.05
    - LayerNorm: 2.03
    """)
    return


@app.cell
def _(TOK, dev_loader, model):
    from gradling.sample import sample

    xs, _ = next(dev_loader)
    sample(model, TOK, xs)
    return


@app.cell
def _(mo, model, nnx):
    from pathlib import Path
    import treescope
    import tempfile
    # jax.tree_util.

    with treescope.active_autovisualizer.set_scoped(treescope.ArrayAutovisualizer()):
        html = treescope.render_to_html(nnx.state(model))
        p = Path(tempfile.mktemp(suffix=".html"))
        p.write_text(html)

    print(p)
    mo.md(f"[Open Treescope View]({p})")
    return (treescope,)


@app.cell
def _(TOK, model, nnx, treescope):
    treescope.render_array(
        nnx.softmax(nnx.state(model).linear.kernel.get_value()),
        axis_item_labels={0: TOK.vocab, 1: TOK.vocab},
    )
    return


if __name__ == "__main__":
    app.run()
