import marimo

__generated_with = "0.20.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import numpy as np
    from datasets import load_dataset
    from transformers import GPT2TokenizerFast

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    dataset = load_dataset("roneneldan/TinyStories")
    return dataset, np, tokenizer


@app.cell
def _(dataset):
    sample = dataset["train"].select(range(1024 * 8))
    sample
    return (sample,)


@app.cell
def _(sample, tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    def tokenize(batch):
        return tokenizer(batch["text"], return_attention_mask=False)

    tokenized = sample.map(
        tokenize,
        batched=True,
        remove_columns=["text"],
        num_proc=4,
    )
    return (tokenized,)


@app.cell
def _(np, tokenized, tokenizer):
    eot = tokenizer("<|endoftext|>")["input_ids"][0]
    tokens = tokenized["input_ids"]
    total_size = sum(len(s) + 1 for s in tokens)
    arr = np.memmap("train_tokens.npy", dtype=np.uint16, mode="w+", shape=(total_size,))
    idx = 0
    for t in tokens:
        arr[idx : idx + len(t)] = t
        idx += len(t)
        arr[idx : idx + 1] = eot
        idx += 1
    # np.array(tokenized[0]['input_ids']).size
    return arr, tokens


@app.cell
def _(arr, tokenizer, tokens):
    tokenizer.decode(arr[: len(tokens[0]) + len(tokens[1]) + 2])
    return


@app.cell
def _(tokenizer):
    tokenizer("<|endoftext|>")
    return


if __name__ == "__main__":
    app.run()
