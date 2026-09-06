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
# The cleanup reason vocabulary is a closed enum, so this cap is a belt-and-
# braces guard rather than a real bound: a counter map must never be able to
# grow with request content.
MAXIMUM_CLEANUP_REASONS = 24
CLEANUP_APPLIED = "applied"
CLEANUP_UNCHANGED = "unchanged"
CLEANUP_FALLBACK = "fallback"


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
class CleanupCounters:
    """Text-free cleanup tallies: how often each outcome happened, never what."""

    applied: int = 0
    unchanged: int = 0
    fallback: int = 0
    disabled: int = 0
    skipped: int = 0
    last_ms: int | None = None
    reasons: tuple[tuple[str, int], ...] = ()


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
    cleanup: CleanupCounters = CleanupCounters()


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
        self._cleanup_totals: dict[str, int] = {}
        self._cleanup_reasons: dict[str, int] = {}
        self._cleanup_last_ms: int | None = None

    def queued(self) -> None:
        with self._lock:
            self._queue_depth += 1

    def dequeued(self, *, rejected: bool = False) -> None:
        with self._lock:
            self._queue_depth = max(0, self._queue_depth - 1)
            if rejected:
                self._rejected_transcriptions += 1

    def started(self, *, queued: bool = True) -> None:
        with self._lock:
            if queued:
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
            else:
                self._failed_transcriptions += 1
            self._last_pipeline = timing if success else None
            _record_latency(self, latency_ms)

    def record_cleanup(self, status: str, reason: str | None, duration_ms: int) -> None:
        """Count one cleanup decision.

        Called for every outcome including `disabled`, so the ratio of applied
        to fallback is measured against real traffic rather than against the
        subset that happened to reach a model.
        """
        with self._lock:
            self._cleanup_totals[status] = self._cleanup_totals.get(status, 0) + 1
            if reason is not None and len(self._cleanup_reasons) < MAXIMUM_CLEANUP_REASONS:
                self._cleanup_reasons[reason] = self._cleanup_reasons.get(reason, 0) + 1
            elif reason is not None and reason in self._cleanup_reasons:
                self._cleanup_reasons[reason] += 1
            self._cleanup_last_ms = max(0, duration_ms)

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
                cleanup=self._cleanup_counters(),
            )

    def _cleanup_counters(self) -> CleanupCounters:
        totals = self._cleanup_totals
        return CleanupCounters(
            applied=totals.get(CLEANUP_APPLIED, 0),
            unchanged=totals.get(CLEANUP_UNCHANGED, 0),
            fallback=totals.get(CLEANUP_FALLBACK, 0),
            disabled=totals.get("disabled", 0),
            skipped=totals.get("skipped", 0),
            last_ms=self._cleanup_last_ms,
            reasons=tuple(sorted(self._cleanup_reasons.items())),
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
