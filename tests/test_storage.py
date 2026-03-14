from __future__ import annotations

import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from gradling.context import Context
from gradling.storage import CHECKPOINTS, RunStorage


class FakeTransport:
    repo_id = "fake/repo"

    def __init__(self, remote_root: Path) -> None:
        self.remote_root = remote_root
        self.push_calls: list[Path] = []
        self.pull_calls: list[Path] = []

    def push(self, path: Path, *, root: Path) -> None:
        self.push_calls.append(path)
        source = root / path
        target = self.remote_root / path
        if target.exists():
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target)

    def pull(self, path: Path, *, root: Path) -> None:
        self.pull_calls.append(path)
        source = self.remote_root / path
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target)


def test_push_uploads_only_run_checkpoints():
    with TemporaryDirectory() as local, TemporaryDirectory() as remote:
        root = Path(local)
        run_path = Path("experiments/aiayn/baseline/runs/001")
        run_dir = root / run_path
        checkpoints = run_dir / CHECKPOINTS
        checkpoints.mkdir(parents=True)
        (run_dir / "config.toml").write_text('model = "aiayn"\n')
        (checkpoints / "final").mkdir()
        (checkpoints / "final" / "weights.bin").write_bytes(b"123")

        transport = FakeTransport(Path(remote))
        storage = RunStorage(Context(root=root), transport)

        storage.push(run_path)

        assert transport.push_calls == [run_path / CHECKPOINTS]
        remote_weights = Path(remote) / run_path / CHECKPOINTS / "final" / "weights.bin"
        assert remote_weights.read_bytes() == b"123"
        assert not (Path(remote) / run_path / "config.toml").exists()


def test_pull_rehydrates_only_run_checkpoints():
    with TemporaryDirectory() as local, TemporaryDirectory() as remote:
        root = Path(local)
        run_path = Path("experiments/aiayn/baseline/runs/001")
        run_dir = root / run_path
        checkpoints = run_dir / CHECKPOINTS
        run_dir.mkdir(parents=True)
        checkpoints.mkdir()
        (run_dir / "config.toml").write_text('model = "aiayn"\n')
        (checkpoints / "stale.txt").write_text("old")

        remote_checkpoints = Path(remote) / run_path / CHECKPOINTS
        (remote_checkpoints / "final").mkdir(parents=True)
        (remote_checkpoints / "final" / "weights.bin").write_bytes(b"abc")

        transport = FakeTransport(Path(remote))
        storage = RunStorage(Context(root=root), transport)

        storage.pull(run_path)

        assert transport.pull_calls == [run_path / CHECKPOINTS]
        assert (run_dir / "config.toml").read_text() == 'model = "aiayn"\n'
        assert (checkpoints / "final" / "weights.bin").read_bytes() == b"abc"
        assert not (checkpoints / "stale.txt").exists()


def test_push_rejects_absolute_paths():
    with TemporaryDirectory() as local, TemporaryDirectory() as remote:
        storage = RunStorage(Context(root=Path(local)), FakeTransport(Path(remote)))

        with pytest.raises(RuntimeError, match="relative to the project root"):
            storage.push(Path(local) / "artifact")
