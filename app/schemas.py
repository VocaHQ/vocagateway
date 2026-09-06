from __future__ import annotations

from datetime import datetime
from typing import Literal, cast
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.cleanup.base import (
    DEFAULT_TIMEOUT_SECONDS,
    MAXIMUM_TIMEOUT_SECONDS,
    MINIMUM_TIMEOUT_SECONDS,
    PROMPT_VERSION,
)
from app.runtime_config import (
    AUTO_ENGINE,
    DEFAULT_CLEANUP_IDLE_UNLOAD_MINUTES,
    DEFAULT_IDLE_OFFLOAD_MINUTES,
)

MAXIMUM_LANGUAGE_TAG_LENGTH = 20
MINIMUM_CUSTOM_MODEL_URL_LENGTH = 12
MAXIMUM_CUSTOM_MODEL_URL_LENGTH = 2_000
MAXIMUM_CPU_THREADS = 256
MAXIMUM_CLEANUP_MODEL_ID_LENGTH = 200
# Long enough for a tag like `hinglish_roman`, short enough to be a language code.
MAXIMUM_LANGUAGE_LENGTH = 32
# A preview is a sentence or two to see the feature work, not a transcript
# ceiling; the service applies its own input limit either way.
MAXIMUM_PREVIEW_CHARACTERS = 2_000
FORBID_EXTRA_FIELDS: Literal["forbid"] = "forbid"
CleanupMode = Literal["off", "conservative", "inherit"]
ResolvedCleanupMode = Literal["off", "conservative"]
CleanupState = Literal["disabled", "unavailable", "loading", "ready", "offloaded", "error"]
IdleUnloadMinutes = Literal[5, 15, 30, 60, 120]
WritingStyle = Literal["raw", "clean", "formal", "casual", "very_casual", "excited"]


class CreateSessionRequest(BaseModel):
    model_config = ConfigDict(extra=FORBID_EXTRA_FIELDS)

    client_session_id: UUID
    language: str = Field(
        default=AUTO_ENGINE,
        max_length=MAXIMUM_LANGUAGE_TAG_LENGTH,
        pattern=r"^(?:[A-Za-z-]+|hinglish_roman)$",
    )
    style: WritingStyle = "casual"
    # Resolved once, here, against the gateway default. `inherit` keeps an
    # older client's request meaning exactly what it means today; an explicit
    # value neither installs a model nor overrides an operator who turned
    # cleanup off. An unsupported value is a validation error, never a silent
    # reinterpretation.
    cleanup: CleanupMode = "inherit"


class CleanupResult(BaseModel):
    """Redacted metadata about one cleanup decision.

    Carries no prompt, no backend message, and no runtime address — `reason` is
    a bounded enum precisely so an operator's diagnosis never depends on
    forwarding text a model produced.
    """

    requested: ResolvedCleanupMode
    status: Literal["disabled", "skipped", "unchanged", "applied", "fallback"]
    reason: str | None = None
    model_id: str | None = None
    prompt_version: str | None = None
    duration_ms: int = 0


class SessionResponse(BaseModel):
    session_id: UUID
    job_id: str
    state: str
    language: str
    style: str
    transcript: str | None = None
    # The recognised text before correction, for a session that opted in —
    # including one where cleanup fell back. Null for legacy and cleanup-off
    # sessions, which have no second copy and never had one.
    original_transcript: str | None = None
    cleanup: CleanupResult | None = None
    error_code: str | None = None
    created_at: datetime
    updated_at: datetime


class CleanupCapability(BaseModel):
    """What a client may ask this gateway for, without describing its insides."""

    supported: bool
    enabled: bool
    default_mode: ResolvedCleanupMode
    modes: list[str]
    model_id: str | None = None
    prompt_version: str = PROMPT_VERSION
    # Languages cleanup will run for. `evaluated_languages` is the subset that
    # has actually passed the published release gates; the two are reported
    # apart so a tested language is never confused with an offered one.
    languages: list[str] = []
    evaluated_languages: list[str] = []
    auto_language: str = ""
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS


class CapabilitiesResponse(BaseModel):
    version: str
    styles: list[str]
    streaming_supported: bool
    cleanup: CleanupCapability


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
    # True when the model refuses to run without being told the language. The
    # opposite of the flag above: a client must drop "detect language" from its
    # picker, because sending it fails the request rather than guessing.
    requires_explicit_language: bool = False


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
    # Cleanup counters, bounded and text-free: how often each outcome happened
    # and how long the last pass took, never what was corrected.
    cleanup_applied: int = 0
    cleanup_unchanged: int = 0
    cleanup_fallback: int = 0
    cleanup_disabled: int = 0
    cleanup_skipped: int = 0
    cleanup_last_ms: int | None = None
    cleanup_reasons: dict[str, int] = {}
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
    # Relative 1-5 ratings derived in app.model_ratings from the catalog's own
    # `quality` wording and size. Not measured WER — the UI says so too. 0 means
    # "not rated", which is what a user-supplied custom model gets.
    speed_rating: int = 0
    accuracy_rating: int = 0
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


class CleanupConfigResponse(BaseModel):
    """The cleanup block of the gateway configuration, safe for diagnostics.

    Deliberately omits the runtime executable path and endpoint: those are
    operator-only settings, and a redacted bundle attached to a bug report has
    no reason to carry the deployment's internal topology.
    """

    enabled: bool = False
    mode: ResolvedCleanupMode = "off"
    model_id: str | None = None
    model_label: str | None = None
    model_installed: bool = False
    runtime_available: bool = False
    managed: bool = True
    state: CleanupState = "disabled"
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    languages: list[str] = []
    evaluated_languages: list[str] = []
    # Which language a transcript left on `auto` is corrected as. Empty means
    # the gateway will not guess, and `auto` falls back unless the writing
    # system names a language on its own.
    auto_language: str = ""
    idle_unload_enabled: bool = False
    idle_unload_minutes: int = DEFAULT_CLEANUP_IDLE_UNLOAD_MINUTES
    # Fields an environment variable has taken away from the UI. Shown as
    # locked rather than pretending a save changed them.
    locked_settings: list[str] = []
    detail: str = ""


class CleanupConfigUpdateRequest(BaseModel):
    """A partial update: every field is optional and only what is sent changes.

    Partial by design. Changing whether transcripts are corrected must not reset
    the speech engine, the hardware options, or the ASR idle-offload policy,
    which is exactly what a whole-object save of the shared config would do.
    """

    model_config = ConfigDict(extra=FORBID_EXTRA_FIELDS)

    enabled: bool | None = None
    mode: ResolvedCleanupMode | None = None
    model_id: str | None = Field(default=None, max_length=MAXIMUM_CLEANUP_MODEL_ID_LENGTH)
    timeout_seconds: float | None = Field(
        default=None, ge=MINIMUM_TIMEOUT_SECONDS, le=MAXIMUM_TIMEOUT_SECONDS
    )
    idle_unload_enabled: bool | None = None
    idle_unload_minutes: IdleUnloadMinutes | None = None
    # An empty string is a real value here — "stop guessing" — so this is not
    # the same as the field being absent, which means "leave it alone".
    auto_language: str | None = Field(default=None, max_length=MAXIMUM_LANGUAGE_LENGTH)


class CleanupPreviewRequest(BaseModel):
    """Text an operator wants corrected, to see what cleanup does to it."""

    model_config = ConfigDict(extra=FORBID_EXTRA_FIELDS)

    text: str = Field(min_length=1, max_length=MAXIMUM_PREVIEW_CHARACTERS)
    language: str = Field(default="en", max_length=MAXIMUM_LANGUAGE_LENGTH)


class CleanupPreviewResponse(BaseModel):
    """Three versions of one preview, plus the outcome a dictation would report.

    Three rather than two, because the interesting comparison is not "typed text
    versus final text". The gateway already fixes spacing and sentence case
    without any model at all, and crediting the model for that would overstate
    what it does. `without_cleanup` is what this text becomes with the feature
    switched off, so the difference between it and `transcript` is the model's
    contribution and nothing else.

    Held only for the length of the response: a preview is never stored, never
    logged, and never reaches a diagnostics bundle.
    """

    original: str
    without_cleanup: str
    transcript: str
    cleanup: CleanupResult


class CleanupModelEntry(BaseModel):
    """One installable cleanup artifact, with the provenance behind it."""

    id: str
    label: str
    description: str
    runtime: str
    size_bytes: int
    minimum_ram_gb: float
    upstream_model: str
    quantization: str
    conversion_source: str
    chat_template_source: str
    license_name: str
    license_notice: str
    source_url: str
    revision: str | None = None
    sha256: str | None = None
    installable: bool = False
    languages: list[str] = []
    evaluated_languages: list[str] = []
    state: Literal["installed", "downloading", "not_installed"]
    active: bool = False
    progress: float | None = None
    downloaded_bytes: int | None = None
    total_bytes: int | None = None
    error: str | None = None


class ConfigResponse(BaseModel):
    engine: str
    available_engines: list[str]
    whisper_model: str | None = None
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
    cleanup: CleanupConfigResponse = CleanupConfigResponse()


class ConfigUpdateRequest(BaseModel):
    model_config = ConfigDict(extra=FORBID_EXTRA_FIELDS)

    engine: Literal[
        "auto",
        "vocamac",
        "handy",
        "whisper.cpp",
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
    # The mic test shows both texts side by side so an operator can see what
    # cleanup changed. Null whenever cleanup did not run.
    original_transcript: str | None = None
    cleanup: CleanupResult | None = None
    cleanup_ms: int = 0


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
