from __future__ import annotations

from app.metrics import MetricsSnapshot, PipelineTiming
from app.schemas import (
    AdminModelEntry,
    MetricsHistoryPoint,
    OperationalMetricsStatus,
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
        error_code=stored.error_code,
        created_at=stored.created_at,
        updated_at=stored.updated_at,
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
