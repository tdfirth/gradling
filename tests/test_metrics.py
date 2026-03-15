import logging
import os
from typing import Any
from unittest.mock import patch

from gradling.context import Context
from gradling.metrics import (
    LogSink,
    Metrics,
    RunIdentity,
    log_only,
    with_wandb,
)
from gradling.run import Run


class FakeSink:
    def __init__(self):
        self.tracked: list[tuple[dict[str, Any], int]] = []
        self.closed = False

    def track(self, metrics: dict[str, Any], step: int) -> None:
        self.tracked.append((metrics, step))

    def close(self) -> None:
        self.closed = True


FAKE_IDENTITY = RunIdentity(study="test", experiment="baseline", run_id="0001")


class TestLogSink:
    def test_track_logs_each_metric(self, caplog):
        sink = LogSink()
        with caplog.at_level(logging.INFO, logger="gradling.metrics"):
            sink.track({"loss": 0.1234, "accuracy": 0.9876}, step=42)

        assert "loss=0.12" in caplog.text
        assert "accuracy=0.99" in caplog.text

    def test_close_is_noop(self):
        LogSink().close()


class TestMetrics:
    def test_track_fans_out_to_all_sinks(self):
        a, b = FakeSink(), FakeSink()
        m = Metrics([a, b])

        m.track({"loss": 1.0}, step=5)

        for sink in (a, b):
            assert sink.tracked == [({"loss": 1.0}, 5)]

    def test_close_closes_all_sinks(self):
        a, b = FakeSink(), FakeSink()
        m = Metrics([a, b])

        m.close()

        for sink in (a, b):
            assert sink.closed


class TestRunTrack:
    def _make_run(self, tmp_path):
        from dataclasses import dataclass

        from gradling.config import Config

        @dataclass
        class EmptyConfig(Config):
            pass

        path = tmp_path / "run"
        path.mkdir()
        fake = FakeSink()
        m = Metrics([fake])
        run = Run(Context(root=tmp_path), path, EmptyConfig(), m)
        return run, fake

    def test_track_delegates_to_metrics(self, tmp_path):
        run, fake = self._make_run(tmp_path)

        run.track({"loss": 1.0}, step=5)

        assert fake.tracked == [({"loss": 1.0}, 5)]

    def test_finalize_closes_metrics(self, tmp_path):
        run, fake = self._make_run(tmp_path)

        run.finalize()

        assert fake.closed


class TestLogOnly:
    def test_returns_log_sink(self):
        sinks = log_only(FAKE_IDENTITY, {})
        assert len(sinks) == 1
        assert isinstance(sinks[0], LogSink)


class TestWithWandb:
    def test_returns_log_sink_without_key(self):
        env = {k: v for k, v in os.environ.items() if k != "WANDB_API_KEY"}
        with (
            patch("gradling.metrics._load_dotenv"),
            patch.dict(os.environ, env, clear=True),
        ):
            sinks = with_wandb(FAKE_IDENTITY, {})
        assert len(sinks) == 1
        assert isinstance(sinks[0], LogSink)

    def test_always_includes_log_sink(self):
        with (
            patch("gradling.metrics._load_dotenv"),
            patch.dict(os.environ, {}, clear=True),
        ):
            sinks = with_wandb(FAKE_IDENTITY, {})
        assert any(isinstance(s, LogSink) for s in sinks)
