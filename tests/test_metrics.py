import logging
import os
from typing import Any
from unittest.mock import patch

from gradling.metrics import LogSink, Metrics
from gradling.run import Run


class FakeSink:
    def __init__(self):
        self.tracked: list[tuple[dict[str, Any], int]] = []
        self.closed = False

    def track(self, metrics: dict[str, Any], step: int) -> None:
        self.tracked.append((metrics, step))

    def close(self) -> None:
        self.closed = True


class TestLogSink:
    def test_track_logs_each_metric(self, caplog):
        sink = LogSink()
        with caplog.at_level(logging.INFO, logger="gradling.metrics"):
            sink.track({"loss": 0.1234, "accuracy": 0.9876}, step=42)

        assert "[step 42] loss: 0.1234" in caplog.text
        assert "[step 42] accuracy: 0.9876" in caplog.text

    def test_close_is_noop(self):
        LogSink().close()


class TestMetrics:
    def test_track_fans_out_to_all_sinks(self):
        m = Metrics.__new__(Metrics)
        m.sinks = [FakeSink(), FakeSink()]

        m.track({"loss": 1.0}, step=5)

        for sink in m.sinks:
            assert sink.tracked == [({"loss": 1.0}, 5)]

    def test_close_closes_all_sinks(self):
        m = Metrics.__new__(Metrics)
        m.sinks = [FakeSink(), FakeSink()]

        m.close()

        for sink in m.sinks:
            assert sink.closed


class TestRunTrack:
    def _make_run(self, tmp_path):

        path = tmp_path / "run"
        path.mkdir()
        with (
            patch("gradling.metrics._load_dotenv"),
            patch.dict(os.environ, {}, clear=True),
        ):
            run = Run(path, {})
        fake = FakeSink()
        run.metrics.sinks = [fake]
        return run, fake

    def test_track_delegates_to_metrics(self, tmp_path):
        run, fake = self._make_run(tmp_path)

        run.track({"loss": 1.0}, step=5)

        assert fake.tracked == [({"loss": 1.0}, 5)]

    def test_finalize_closes_metrics(self, tmp_path):
        run, fake = self._make_run(tmp_path)

        run.finalize()

        assert fake.closed


class TestMetricsInit:
    def test_always_includes_log_sink(self):
        with (
            patch("gradling.metrics._load_dotenv"),
            patch.dict(os.environ, {}, clear=True),
        ):
            m = Metrics({})
        assert any(isinstance(s, LogSink) for s in m.sinks)

    def test_no_wandb_without_key(self):
        env = {k: v for k, v in os.environ.items() if k != "WANDB_API_KEY"}
        with (
            patch("gradling.metrics._load_dotenv"),
            patch.dict(os.environ, env, clear=True),
        ):
            m = Metrics({})
        assert len(m.sinks) == 1
        assert isinstance(m.sinks[0], LogSink)
