"""Shared vocabulary for optional local transcript cleanup.

Cleanup is a bounded, text-only pass that runs *after* speech recognition has
produced a final transcript. It never sees audio, never rewrites a streaming
partial, and never turns a successful recognition into a failure: every path
out of this package either improves the text or hands back exactly what the
existing deterministic styling would have produced on its own.

The types here are deliberately small and immutable so the same decision can be
made identically by the session, one-shot, and streaming entry points.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

# What a client may ask for. `inherit` is resolved once, against the gateway
# default, before any of it reaches the service.
MODE_OFF = "off"
MODE_CONSERVATIVE = "conservative"
MODE_INHERIT = "inherit"
REQUESTABLE_MODES = (MODE_OFF, MODE_CONSERVATIVE, MODE_INHERIT)
RESOLVED_MODES = (MODE_OFF, MODE_CONSERVATIVE)
DEFAULT_MODE = MODE_OFF

# Bumped whenever the system instruction or the output contract changes, and
# snapshotted per session so a stored result always names the prompt that made
# it. Never derived from the model.
PROMPT_VERSION = "cleanup-v1"

# Engineering budgets. A 120 s dictation is a few hundred tokens; these
# ceilings leave headroom for that without allocating an 8k f16 KV cache
# that dwarfs the 0.6B weights on a low-end host.
MILLISECONDS_PER_SECOND = 1000
DEFAULT_TIMEOUT_SECONDS = 5.0
MINIMUM_TIMEOUT_SECONDS = 1.0
MAXIMUM_TIMEOUT_SECONDS = 30.0
ADMISSION_WAIT_SECONDS = 0.1
MAXIMUM_INPUT_BYTES = 16_384
MAXIMUM_OUTPUT_BYTES = 32_768
MAXIMUM_INPUT_TOKENS = 1_536
MAXIMUM_OUTPUT_TOKENS = 1_920
MINIMUM_CONTEXT_TOKENS = 4_096
# A character cannot be more than one token. Packing to this many characters
# keeps each piece inside the token ceiling without a tokenize round-trip.
# The margin is for BOS / template tokens a tokenizer might add on top.
TOKENIZER_MARGIN_TOKENS = 32
CHUNK_CHAR_LIMIT = MAXIMUM_INPUT_TOKENS - TOKENIZER_MARGIN_TOKENS
MINIMUM_OUTPUT_TOKENS = 64
JSON_WRAPPER_TOKENS = 24


class CleanupStatus(StrEnum):
    """What happened to one transcript, in the order of increasing involvement."""

    DISABLED = "disabled"
    SKIPPED = "skipped"
    UNCHANGED = "unchanged"
    APPLIED = "applied"
    FALLBACK = "fallback"


class CleanupReason(StrEnum):
    """Why cleanup did not apply. A bounded enum: never a backend message."""

    RAW_STYLE = "raw_style"
    MODEL_UNAVAILABLE = "model_unavailable"
    MODEL_LOADING = "model_loading"
    CONTEXT_TOO_SMALL = "context_too_small"
    UNSUPPORTED_LANGUAGE = "unsupported_language"
    INPUT_TOO_LONG = "input_too_long"
    EMPTY_INPUT = "empty_input"
    BUSY = "busy"
    TIMEOUT = "timeout"
    INVALID_OUTPUT = "invalid_output"
    UNSAFE_EDIT = "unsafe_edit"
    RUNTIME_ERROR = "runtime_error"


@dataclass(frozen=True, slots=True)
class CleanupOptions:
    """The resolved processing decision for one unit of work.

    Snapshotted at session creation and pinned for the request, so a settings
    change mid-flight cannot silently move a transcript to another model.
    """

    mode: str = DEFAULT_MODE
    model_id: str | None = None
    prompt_version: str = PROMPT_VERSION
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS

    @property
    def enabled(self) -> bool:
        return self.mode == MODE_CONSERVATIVE


@dataclass(frozen=True, slots=True)
class CleanupOutcome:
    """Redacted metadata about one cleanup decision. Never carries prompt text."""

    requested: str
    status: CleanupStatus
    reason: CleanupReason | None = None
    model_id: str | None = None
    prompt_version: str | None = None
    duration_ms: int = 0

    @property
    def reason_name(self) -> str | None:
        return None if self.reason is None else str(self.reason)


@dataclass(frozen=True, slots=True)
class FinalTranscript:
    """The insertion result plus the original ASR text it came from."""

    transcript: str
    original_transcript: str | None
    cleanup: CleanupOutcome


class CleanupUnavailable(Exception):
    """The cleanup runtime cannot be reached, started, or trusted right now."""


class CleanupRejected(Exception):
    """The runtime answered, but not with something this package will accept."""

    def __init__(self, reason: CleanupReason, message: str = "") -> None:
        super().__init__(message or str(reason))
        self.reason = reason


class CleanupRuntime(Protocol):
    """A local text model that returns a corrected transcript, and nothing else.

    Implementations must be cancellable: an abandoned request has to stop the
    backend's compute rather than leave it running behind the caller's back.
    """

    @property
    def model_id(self) -> str | None: ...

    async def available(self) -> bool: ...

    async def clean(self, transcript: str, language: str, *, budget_seconds: float) -> str: ...
