from __future__ import annotations

from app.metrics import HISTORY_MAX, RuntimeMetrics

SUCCESSFUL_TRANSCRIPTION_LATENCY_MS = 120
FAILED_TRANSCRIPTION_LATENCY_MS = 280
AVERAGE_TRANSCRIPTION_LATENCY_MS = 200
METRICS_SAMPLE_INTERVAL_SECONDS = 5.0


class _FakeClock:
    def __init__(self, start: float) -> None:
        self.current = start

    def __call__(self) -> float:
        return self.current

    def advance(self, seconds: float) -> None:
        self.current += seconds


def _simulate_transcription(metrics: RuntimeMetrics, latency: int, *, success: bool) -> None:
    metrics.queued()
    metrics.started()
    metrics.record_result(latency, success=success)
    metrics.finished()


def test_runtime_metrics_track_work_and_outcomes() -> None:
    metrics = RuntimeMetrics(concurrency_limit=2)

    metrics.queued()
    assert metrics.snapshot().queue_depth == 1
    metrics.started()
    metrics.finished()

    _simulate_transcription(metrics, SUCCESSFUL_TRANSCRIPTION_LATENCY_MS, success=True)
    _simulate_transcription(metrics, FAILED_TRANSCRIPTION_LATENCY_MS, success=False)

    metrics.queued()
    metrics.dequeued(rejected=True)
    metrics.queued()
    metrics.dequeued()
    snapshot = metrics.snapshot()

    assert (
        snapshot.queue_depth,
        snapshot.active_transcriptions,
        snapshot.concurrency_limit,
        snapshot.successful_transcriptions,
        snapshot.failed_transcriptions,
        snapshot.rejected_transcriptions,
        snapshot.average_latency_ms,
        snapshot.last_latency_ms,
        snapshot.history,
    ) == (
        0,
        0,
        2,
        1,
        1,
        1,
        AVERAGE_TRANSCRIPTION_LATENCY_MS,
        FAILED_TRANSCRIPTION_LATENCY_MS,
        (),
    )


def test_runtime_metrics_history_samples_wh_aa(monkeypatch) -> None:
    metrics = RuntimeMetrics(concurrency_limit=1)
    clock = _FakeClock(1000.0)
    monkeypatch.setattr("app.metrics.time.monotonic", clock)

    history = metrics.snapshot(sample=True).history
    assert (len(history), history[0].queue_depth) == (1, 0)

    # Inside the min interval: no second sample.
    clock.advance(1.0)
    assert len(metrics.snapshot(sample=True).history) == 1

    metrics.queued()
    clock.advance(METRICS_SAMPLE_INTERVAL_SECONDS)
    history = metrics.snapshot(sample=True).history
    assert (len(history), history[-1].queue_depth) == (2, 1)

    # Cap the ring buffer.
    for _ in range(HISTORY_MAX + 5):
        clock.advance(METRICS_SAMPLE_INTERVAL_SECONDS)
        metrics.snapshot(sample=True)
    assert len(metrics.snapshot().history) == HISTORY_MAX
