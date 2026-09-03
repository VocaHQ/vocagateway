from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass

# ~5 minutes of history when the WebUI polls Live operations every 5s.
HISTORY_MAX = 60
SAMPLE_MIN_INTERVAL_S = 4.0
# A deliberately old value guarantees the first requested sample is retained.
INITIAL_SAMPLE_TIMESTAMP = -SAMPLE_MIN_INTERVAL_S


@dataclass(frozen=True, slots=True)
class PipelineTiming:
    total_ms: int
    normalization_ms: int
    model_load_ms: int
    inference_ms: int
    audio_duration_ms: int
    real_time_factor: float | None
    engine: str
    peak_memory_mb: float | None


@dataclass(frozen=True, slots=True)
class MetricsSample:
    """One point on the Live operations sparklines (process-local, not persisted)."""

    uptime_seconds: int
    queue_depth: int
    active_transcriptions: int
    last_latency_ms: int | None
    successful_transcriptions: int
    failed_transcriptions: int


@dataclass(frozen=True, slots=True)
class MetricsSnapshot:
    uptime_seconds: int
    queue_depth: int
    active_transcriptions: int
    concurrency_limit: int
    successful_transcriptions: int
    failed_transcriptions: int
    rejected_transcriptions: int
    average_latency_ms: int | None
    last_latency_ms: int | None
    last_pipeline: PipelineTiming | None
    history: tuple[MetricsSample, ...]


class RuntimeMetrics:
    """Small, privacy-safe in-memory counters scoped to one server process."""

    def __init__(self, concurrency_limit: int) -> None:
        self._started_at = time.monotonic()
        self._concurrency_limit = concurrency_limit
        self._lock = threading.Lock()
        self._queue_depth = 0
        self._active_transcriptions = 0
        self._successful_transcriptions = 0
        self._failed_transcriptions = 0
        self._rejected_transcriptions = 0
        self._total_latency_ms = 0
        self._last_latency_ms: int | None = None
        self._last_pipeline: PipelineTiming | None = None
        self._history: deque[MetricsSample] = deque(maxlen=HISTORY_MAX)
        self._last_sample_at = INITIAL_SAMPLE_TIMESTAMP

    def queued(self) -> None:
        with self._lock:
            self._queue_depth += 1

    def dequeued(self, *, rejected: bool = False) -> None:
        with self._lock:
            self._queue_depth = max(0, self._queue_depth - 1)
            if rejected:
                self._rejected_transcriptions += 1

    def started(self) -> None:
        with self._lock:
            self._queue_depth = max(0, self._queue_depth - 1)
            self._active_transcriptions += 1

    def finished(self) -> None:
        with self._lock:
            self._active_transcriptions = max(0, self._active_transcriptions - 1)

    def record_result(
        self, latency_ms: int, *, success: bool, timing: PipelineTiming | None = None
    ) -> None:
        with self._lock:
            if success:
                self._successful_transcriptions += 1
                self._last_pipeline = timing
            else:
                self._failed_transcriptions += 1
            _record_latency(self, latency_ms)

    def snapshot(self, *, sample: bool = False) -> MetricsSnapshot:
        with self._lock:
            uptime = max(0, int(time.monotonic() - self._started_at))
            completed = self._successful_transcriptions + self._failed_transcriptions
            average = round(self._total_latency_ms / completed) if completed else None
            if sample:
                _maybe_sample_locked(self, uptime)
            return MetricsSnapshot(
                uptime_seconds=uptime,
                queue_depth=self._queue_depth,
                active_transcriptions=self._active_transcriptions,
                concurrency_limit=self._concurrency_limit,
                successful_transcriptions=self._successful_transcriptions,
                failed_transcriptions=self._failed_transcriptions,
                rejected_transcriptions=self._rejected_transcriptions,
                average_latency_ms=average,
                last_latency_ms=self._last_latency_ms,
                last_pipeline=self._last_pipeline,
                history=tuple(self._history),
            )


def _maybe_sample_locked(metrics: RuntimeMetrics, uptime_seconds: int) -> None:
    now = time.monotonic()
    if metrics._history and now - metrics._last_sample_at < SAMPLE_MIN_INTERVAL_S:
        return
    metrics._history.append(
        MetricsSample(
            uptime_seconds=uptime_seconds,
            queue_depth=metrics._queue_depth,
            active_transcriptions=metrics._active_transcriptions,
            last_latency_ms=metrics._last_latency_ms,
            successful_transcriptions=metrics._successful_transcriptions,
            failed_transcriptions=metrics._failed_transcriptions,
        )
    )
    metrics._last_sample_at = now


def _record_latency(metrics: RuntimeMetrics, latency_ms: int) -> None:
    safe_latency = max(0, latency_ms)
    metrics._total_latency_ms += safe_latency
    metrics._last_latency_ms = safe_latency
