from __future__ import annotations

from datetime import datetime
from typing import Literal, cast
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.runtime_config import AUTO_ENGINE, DEFAULT_IDLE_OFFLOAD_MINUTES

MAXIMUM_LANGUAGE_TAG_LENGTH = 20
MINIMUM_CUSTOM_MODEL_URL_LENGTH = 12
MAXIMUM_CUSTOM_MODEL_URL_LENGTH = 2_000
MAXIMUM_CPU_THREADS = 256
FORBID_EXTRA_FIELDS: Literal["forbid"] = "forbid"


class CreateSessionRequest(BaseModel):
    model_config = ConfigDict(extra=FORBID_EXTRA_FIELDS)

    client_session_id: UUID
    language: str = Field(
        default=AUTO_ENGINE,
        max_length=MAXIMUM_LANGUAGE_TAG_LENGTH,
        pattern=r"^(?:[A-Za-z-]+|hinglish_roman)$",
    )
    style: Literal[
        "raw",
        "clean",
        "formal",
        "casual",
        "very_casual",
        "excited",
    ] = "casual"


class SessionResponse(BaseModel):
    session_id: UUID
    job_id: str
    state: str
    language: str
    style: str
    transcript: str | None = None
    error_code: str | None = None
    created_at: datetime
    updated_at: datetime


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    engine_ready: bool
    engine: str
    streaming_supported: bool
    # What the loaded model can actually do with the `language` field, so a client
    # can stop offering choices it cannot honour. An empty list means "unknown" —
    # an older gateway, no model selected, or a user's own imported model — and
    # clients must keep every language available rather than locking the picker.
    languages: list[str] = []
    # True when the model picks the language itself. `languages` then describes
    # what it transcribes well, not what a client may ask for.
    detects_language_automatically: bool = False


class LivenessResponse(BaseModel):
    status: Literal["ok"] = "ok"
    uptime_seconds: int


class ReadinessResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    engine_ready: bool
    engine: str
    probe_age_seconds: float
    warmup_state: Literal[
        "pending", "warming", "complete", "unsupported", "unavailable", "failed", "offloaded"
    ]


class ModelResponse(BaseModel):
    id: str
    ready: bool
    local: Literal[True] = True


class DeleteResponse(BaseModel):
    deleted: bool


class DependencyStatus(BaseModel):
    name: str
    available: bool
    path: str | None = None
    install_hint: str | None = None
    # Set when a hint alone cannot carry the setup, as for a tool the project
    # does not ship and the operator has to build.
    docs_url: str | None = None


class SystemStatus(BaseModel):
    os: str
    arch: str
    chip: str
    ram_gb: float
    is_apple_silicon: bool
    logical_cpus: int
    effective_cpus: float
    containerized: bool
    accelerators: list[str]
    cpu_features: list[str]


class EngineStatus(BaseModel):
    id: str
    name: str
    ready: bool


class PathStatus(BaseModel):
    data_dir: str
    models_dir: str
    config_file: str
    token_file: str


class SetupChecklist(BaseModel):
    token_configured: bool
    ffmpeg_available: bool
    engine_binary_available: bool
    model_installed: bool
    engine_ready: bool


class MetricsHistoryPoint(BaseModel):
    """One Live-operations sample for sparklines (in-process ring buffer)."""

    uptime_seconds: int
    queue_depth: int
    active_transcriptions: int
    last_latency_ms: int | None = None
    successful_transcriptions: int = 0
    failed_transcriptions: int = 0


class OperationalMetricsStatus(BaseModel):
    uptime_seconds: int
    queue_depth: int
    active_transcriptions: int
    concurrency_limit: int
    successful_transcriptions: int
    failed_transcriptions: int
    rejected_transcriptions: int
    average_latency_ms: int | None
    last_latency_ms: int | None
    normalization_ms: int | None = None
    model_load_ms: int | None = None
    inference_ms: int | None = None
    audio_duration_ms: int | None = None
    real_time_factor: float | None = None
    peak_memory_mb: float | None = None
    history: list[MetricsHistoryPoint] = []


class ReadinessStatus(BaseModel):
    probe_age_seconds: float
    warmup_state: Literal[
        "pending", "warming", "complete", "unsupported", "unavailable", "failed", "offloaded"
    ]
    warmed_bytes: int


class CommitStatus(BaseModel):
    """The source commit the running gateway was built from."""

    sha: str
    short_sha: str
    subject: str
    committed_at: datetime | None = None


class AdminStatusResponse(BaseModel):
    status: Literal["ok"] = "ok"
    version: str
    commit: CommitStatus | None = None
    engine: EngineStatus
    system: SystemStatus
    dependencies: list[DependencyStatus]
    paths: PathStatus
    bind_host: str
    port: int
    setup: SetupChecklist
    metrics: OperationalMetricsStatus
    readiness: ReadinessStatus


class AdminModelEntry(BaseModel):
    id: str
    engine: str
    label: str
    size_bytes: int
    languages: str
    quality: str
    family: str
    description: str
    source: str
    source_url: str | None = None
    supports_streaming: bool = False
    license_name: str = "See model source"
    commercial_use: bool = True
    detects_language_automatically: bool = False
    # Named languages behind the `languages` summary, and the codes the filter
    # matches on. Empty codes mean "matches any language" rather than none.
    language_names: list[str] = []
    language_codes: list[str] = []
    state: Literal["installed", "downloading", "not_installed"]
    active: bool
    offloaded: bool = False
    recommended: bool
    progress: float | None = None
    downloaded_bytes: int | None = None
    total_bytes: int | None = None
    error: str | None = None
    retired: bool = False
    replacement_id: str | None = None
    retirement_reason: str | None = None
    # Set when this host cannot run the model's engine yet, naming the runtime
    # that is missing and how to install it. Downloading stays allowed - the
    # weights are still correct, so the card warns rather than hiding the button.
    runtime_requirement: str | None = None
    runtime_hint: str | None = None
    runtime_docs_url: str | None = None


class CustomDownloadRequest(BaseModel):
    model_config = ConfigDict(extra=FORBID_EXTRA_FIELDS)

    url: str = Field(
        min_length=MINIMUM_CUSTOM_MODEL_URL_LENGTH,
        max_length=MAXIMUM_CUSTOM_MODEL_URL_LENGTH,
    )
    # Optional: a model card's published SHA-256. When given, the download is
    # discarded unless it matches, which is the only integrity guarantee
    # available for a URL the catalog does not vouch for.
    sha256: str | None = Field(default=None, max_length=100)


class DeviceTokenEntry(BaseModel):
    id: str
    label: str
    created_at: datetime | None
    revocable: bool


class DeviceTokenCreateRequest(BaseModel):
    model_config = ConfigDict(extra=FORBID_EXTRA_FIELDS)

    label: str = Field(min_length=1, max_length=100)


class DeviceTokenCreateResponse(BaseModel):
    id: str
    label: str
    token: str
    created_at: datetime


class DeviceTokenRevokeResponse(BaseModel):
    revoked: bool


class DownloadResponse(BaseModel):
    model_id: str
    status: str


class ConfigResponse(BaseModel):
    engine: str
    available_engines: list[str]
    whisper_model: str | None = None
    transcribe_model: str | None = None
    whisperkit_model: str | None = None
    faster_whisper_model: str | None = None
    moonshine_model: str = "moonshine:en"
    moonshine_language: str = "en"
    sherpa_model: str | None = None
    mlx_audio_model: str | None = None
    compute_device: str = AUTO_ENGINE
    compute_type: str = AUTO_ENGINE
    cpu_threads: int = 0
    idle_offload_enabled: bool = False
    idle_offload_minutes: int = DEFAULT_IDLE_OFFLOAD_MINUTES


class ConfigUpdateRequest(BaseModel):
    model_config = ConfigDict(extra=FORBID_EXTRA_FIELDS)

    engine: Literal[
        "auto",
        "vocamac",
        "handy",
        "whisper.cpp",
        "transcribe.cpp",
        "whisperkit",
        "faster-whisper",
        "moonshine",
        "sherpa-onnx",
        "mlx-audio",
    ]
    compute_device: Literal["auto", "cpu", "cuda"] = cast(Literal["auto"], AUTO_ENGINE)
    compute_type: Literal["auto", "int8", "int8_float16", "float16", "float32"] = cast(
        Literal["auto"], AUTO_ENGINE
    )
    cpu_threads: int = Field(default=0, ge=0, le=MAXIMUM_CPU_THREADS)
    idle_offload_enabled: bool = False
    idle_offload_minutes: Literal[10, 15, 30, 60, 120] = cast(
        Literal[10, 15, 30, 60, 120], DEFAULT_IDLE_OFFLOAD_MINUTES
    )


class SelectModelResponse(BaseModel):
    engine: EngineStatus


class TestTranscriptionResponse(BaseModel):
    transcript: str
    engine: str
    duration_ms: int
    normalization_ms: int
    model_load_ms: int
    inference_ms: int
    audio_duration_ms: int
    real_time_factor: float | None
    peak_memory_mb: float | None


class OpenAITranscriptionResponse(BaseModel):
    text: str


class ErrorDetail(BaseModel):
    code: str
    message: str
    recoverable: bool


class ErrorEnvelope(BaseModel):
    error: ErrorDetail


class DiagnosticsBundle(BaseModel):
    """A redacted operational snapshot, safe to attach to a bug report.

    Never contains the bearer token, recording audio, transcript text, or
    session identifiers — only setup, dependency, hardware, and counter data
    already shown in the authenticated WebUI.
    """

    generated_at: datetime
    version: str
    commit: CommitStatus | None = None
    engine: EngineStatus
    system: SystemStatus
    dependencies: list[DependencyStatus]
    paths: PathStatus
    bind_host: str
    port: int
    setup: SetupChecklist
    metrics: OperationalMetricsStatus
    readiness: ReadinessStatus
    config: ConfigResponse
    never_included: list[str]
