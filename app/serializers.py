from __future__ import annotations

from typing import Any, cast

from app.cleanup.base import CleanupOutcome
from app.metrics import MetricsSnapshot, PipelineTiming
from app.schemas import (
    AdminModelEntry,
    CleanupResult,
    MetricsHistoryPoint,
    OperationalMetricsStatus,
    ResolvedCleanupMode,
    SessionResponse,
)
from app.storage import StoredSession


def session_response(stored: StoredSession) -> SessionResponse:
    return SessionResponse(
        session_id=stored.session_id,
        job_id=stored.job_id,
        state=stored.state,
        language=stored.language,
        style=stored.style,
        transcript=stored.transcript,
        original_transcript=stored.original_transcript,
        cleanup=stored_cleanup(stored),
        error_code=stored.error_code,
        created_at=stored.created_at,
        updated_at=stored.updated_at,
    )


def stored_cleanup(stored: StoredSession) -> CleanupResult | None:
    """The cleanup block for a session, or None when there is nothing to report.

    A legacy row, a cleanup-off session, and an unfinished one all answer None
    rather than an empty object. Inventing `disabled` for a pending conservative
    run would let a client confuse "not yet finished" with "turned off". An old
    row with no stored original reports null — never an original manufactured
    by treating an already styled transcript as raw.
    """
    record = stored.cleanup_result
    if record.status is None:
        return None
    return CleanupResult(
        requested=cast(ResolvedCleanupMode, stored.cleanup.mode),
        status=cast(Any, record.status),
        reason=record.reason,
        model_id=stored.cleanup.model_id,
        prompt_version=stored.cleanup.prompt_version,
        duration_ms=record.duration_ms or 0,
    )


def cleanup_result(outcome: CleanupOutcome) -> CleanupResult:
    """Serialize a live outcome for a one-shot or streaming response."""
    return CleanupResult(
        requested=cast(ResolvedCleanupMode, outcome.requested),
        status=cast(Any, str(outcome.status)),
        reason=outcome.reason_name,
        model_id=outcome.model_id,
        prompt_version=outcome.prompt_version,
        duration_ms=outcome.duration_ms,
    )


def metrics_status(metrics: MetricsSnapshot) -> OperationalMetricsStatus:
    pipeline = metrics.last_pipeline
    return OperationalMetricsStatus(
        uptime_seconds=metrics.uptime_seconds,
        queue_depth=metrics.queue_depth,
        active_transcriptions=metrics.active_transcriptions,
        concurrency_limit=metrics.concurrency_limit,
        successful_transcriptions=metrics.successful_transcriptions,
        failed_transcriptions=metrics.failed_transcriptions,
        rejected_transcriptions=metrics.rejected_transcriptions,
        average_latency_ms=metrics.average_latency_ms,
        last_latency_ms=metrics.last_latency_ms,
        cleanup_applied=metrics.cleanup.applied,
        cleanup_unchanged=metrics.cleanup.unchanged,
        cleanup_fallback=metrics.cleanup.fallback,
        cleanup_disabled=metrics.cleanup.disabled,
        cleanup_skipped=metrics.cleanup.skipped,
        cleanup_last_ms=metrics.cleanup.last_ms,
        cleanup_reasons=dict(metrics.cleanup.reasons),
        **_pipeline_metrics(pipeline),
        history=[
            MetricsHistoryPoint(
                uptime_seconds=point.uptime_seconds,
                queue_depth=point.queue_depth,
                active_transcriptions=point.active_transcriptions,
                last_latency_ms=point.last_latency_ms,
                successful_transcriptions=point.successful_transcriptions,
                failed_transcriptions=point.failed_transcriptions,
            )
            for point in metrics.history
        ],
    )


def _pipeline_metrics(pipeline: PipelineTiming | None) -> dict[str, int | float | None]:
    """Serialize optional pipeline timing fields without repeating its guard."""
    metric_names = (
        "normalization_ms",
        "model_load_ms",
        "inference_ms",
        "audio_duration_ms",
        "real_time_factor",
        "peak_memory_mb",
    )
    if pipeline is None:
        return dict.fromkeys(metric_names)
    return {metric_name: getattr(pipeline, metric_name) for metric_name in metric_names}


def joined_stream_lines(lines: dict[int, str]) -> str:
    ordered_lines = (line for _, line in sorted(lines.items()) if line)
    return " ".join(ordered_lines).strip()


def model_covers(entry: AdminModelEntry, language: str) -> bool:
    """Whether a model can transcribe `language`.

    A model with no declared languages matches everything rather than nothing.
    No catalog entry is in that position today, but a user's own imported model
    is, and hiding an unlabelled model from every filter would make it look like
    the import had failed.
    """
    if not entry.language_codes:
        return True
    return language in entry.language_codes
