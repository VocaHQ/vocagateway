from __future__ import annotations

from app.metrics import HISTORY_MAX, RuntimeMetrics


def test_runtime_metrics_track_work_and_outcomes() -> None:
    metrics = RuntimeMetrics(concurrency_limit=2)

    metrics.queued()
    assert metrics.snapshot().queue_depth == 1

    metrics.started()
    metrics.succeeded(120)
    metrics.finished()

    metrics.queued()
    metrics.started()
    metrics.failed(280)
    metrics.finished()

    metrics.queued()
    metrics.rejected()
    metrics.queued()
    metrics.cancelled()
    snapshot = metrics.snapshot()

    assert snapshot.queue_depth == 0
    assert snapshot.active_transcriptions == 0
    assert snapshot.concurrency_limit == 2
    assert snapshot.successful_transcriptions == 1
    assert snapshot.failed_transcriptions == 1
    assert snapshot.rejected_transcriptions == 1
    assert snapshot.average_latency_ms == 200
    assert snapshot.last_latency_ms == 280
    assert snapshot.history == ()


def test_runtime_metrics_history_samples_wh_aa(monkeypatch) -> None:
    metrics = RuntimeMetrics(concurrency_limit=1)
    clock = {"t": 1000.0}

    def fake_monotonic() -> float:
        return clock["t"]

    monkeypatch.setattr("app.metrics.time.monotonic", fake_monotonic)

    first = metrics.snapshot(sample=True)
    assert len(first.history) == 1
    assert first.history[0].queue_depth == 0

    # Inside the min interval: no second sample.
    clock["t"] += 1.0
    second = metrics.snapshot(sample=True)
    assert len(second.history) == 1

    metrics.queued()
    clock["t"] += 5.0
    third = metrics.snapshot(sample=True)
    assert len(third.history) == 2
    assert third.history[-1].queue_depth == 1

    # Cap the ring buffer.
    for _ in range(HISTORY_MAX + 5):
        clock["t"] += 5.0
        metrics.snapshot(sample=True)
    assert len(metrics.snapshot().history) == HISTORY_MAX
