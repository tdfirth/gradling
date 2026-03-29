import json
from collections.abc import Generator
from dataclasses import dataclass

from gradling.data import DATA


@dataclass
class Example:
    ctx: str
    endings: list[str]
    label: int


def loader() -> Generator[Example]:
    with open(DATA / "rowanz" / "hellaswag" / "hellaswag_val.jsonl") as f:
        lines = f.readlines()

    for line in lines:
        example = json.loads(line)
        yield Example(
            ctx=example["ctx"],
            # Add a space to each one so they form more natural sentences.
            endings=[f" {e}" for e in example["endings"]],
            label=example["label"],
        )
