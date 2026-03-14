"""
Sync run checkpoints with the project's Hugging Face repo.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Protocol

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import get_token

from gradling import logger
from gradling.context import Context

log = logger.get(__name__)

CHECKPOINTS = Path("checkpoints")


class Transport(Protocol):
    repo_id: str

    def push(self, path: Path, *, root: Path) -> None: ...
    def pull(self, path: Path, *, root: Path) -> None: ...


class HfApiTransport:
    repo_id = "tdfirth/gradling"
    repo_type = "model"

    def __init__(self, ctx: Context) -> None:
        self.ctx = ctx

    def _token(self) -> str:
        self.ctx.load_dotenv()
        token = get_token()
        if token is None:
            raise RuntimeError("Missing Hugging Face token. Set HF_TOKEN in .env.")
        return token

    def push(self, path: Path, *, root: Path) -> None:
        HfApi(token=self._token()).upload_folder(
            repo_id=self.repo_id,
            repo_type=self.repo_type,
            folder_path=root / path,
            path_in_repo=path.as_posix(),
            delete_patterns="**",
            commit_message=f"gradling push {path.parent.as_posix()}",
        )

    def pull(self, path: Path, *, root: Path) -> None:
        snapshot_download(
            repo_id=self.repo_id,
            repo_type=self.repo_type,
            token=self._token(),
            local_dir=root,
            allow_patterns=[f"{path.as_posix()}/*", f"{path.as_posix()}/**"],
        )


class RunStorage:
    def __init__(self, ctx: Context, transport: Transport) -> None:
        self.ctx = ctx
        self.transport = transport

    def push(self, run_path: Path) -> None:
        checkpoints_path = self._checkpoints_path(run_path)
        if not checkpoints_path.exists():
            raise RuntimeError(f"Path {checkpoints_path} does not exist.")
        if not checkpoints_path.is_dir():
            raise RuntimeError(f"Path {checkpoints_path} is not a directory.")

        log.info(
            "Uploading %s to hf://%s/%s",
            checkpoints_path.relative_to(self.ctx.root),
            self.transport.repo_id,
            checkpoints_path.relative_to(self.ctx.root).as_posix(),
        )
        self.transport.push(self._artifact_path(run_path), root=self.ctx.root)

    def pull(self, run_path: Path) -> None:
        run_dir = self._run_dir(run_path)
        if run_dir.exists() and not run_dir.is_dir():
            raise RuntimeError(f"Path {run_dir} is not a directory.")

        checkpoints_path = self._checkpoints_path(run_path)
        if checkpoints_path.exists() and not checkpoints_path.is_dir():
            raise RuntimeError(f"Path {checkpoints_path} is not a directory.")
        log.info(
            "Downloading hf://%s/%s to %s",
            self.transport.repo_id,
            self._artifact_path(run_path).as_posix(),
            checkpoints_path.relative_to(self.ctx.root),
        )
        if checkpoints_path.exists():
            shutil.rmtree(checkpoints_path)
        self.transport.pull(self._artifact_path(run_path), root=self.ctx.root)

    def _run_dir(self, run_path: Path) -> Path:
        if run_path.is_absolute():
            raise RuntimeError("Path must be relative to the project root.")
        return self.ctx.root / run_path

    def _artifact_path(self, run_path: Path) -> Path:
        return run_path / CHECKPOINTS

    def _checkpoints_path(self, run_path: Path) -> Path:
        return self._run_dir(run_path) / CHECKPOINTS
