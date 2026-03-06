import json
from pathlib import Path
from typing import Any, Self

import orbax.checkpoint as ocp
from flax import nnx

from gradling.data import ROOT
from gradling.metrics import Metrics

EXPERIMENTS = ROOT / "experiments"


def _cfg_json(path: Path) -> Path:
    return path / "config.json"


class Run:
    def __init__(self, path: Path, cfg: dict, metrics: Metrics):
        self.cfg = cfg
        self.metrics = metrics
        self.checkpoints = path / "checkpoints"
        self.checkpoints.mkdir(parents=True, exist_ok=True)
        self.checkpointer = ocp.StandardCheckpointer()

    @classmethod
    def from_config(cls, cfg: dict, family: str) -> Self:
        metrics = Metrics(cfg)
        safe_family = family.strip().replace("/", "_") or "default"
        path = EXPERIMENTS / safe_family / metrics.name
        path.mkdir(parents=True, exist_ok=True)
        _cfg_json(path).write_text(json.dumps(cfg, indent=2))
        return cls(path, cfg, metrics)

    @classmethod
    def from_path(cls, path: Path) -> Self:
        cfg = json.loads(_cfg_json(path).read_text())
        return cls(path, cfg, Metrics(cfg))

    # TODO handle optimizer state as well as weights.
    def checkpoint(self, label: str, model: nnx.Module):
        state = nnx.state(model)
        self.checkpointer.save(self.checkpoints / label, state)
        self.checkpointer.wait_until_finished()

    def load_checkpoint(self, label: str, model: nnx.Module):
        state = nnx.state(model)
        checkpoint_path = self.checkpoints / label
        to_restore = self.checkpointer.restore(checkpoint_path.absolute(), target=state)
        nnx.update(model, to_restore)

    def track(self, metrics: dict[str, Any], step: int) -> None:
        self.metrics.track(metrics, step)

    def finalize(self):
        self.checkpointer.wait_until_finished()
        self.metrics.close()
