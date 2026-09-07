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
MINIMUM_CONTEXT_TOKENS = 4_096
# Tokens one request spends on something other than the transcript: the system
# instruction, the chat template's own turn markers, and the JSON envelope
# around the transcript and the answer. Measured against the shipped
# instruction with room to spare, so a prompt revision cannot quietly overrun
# the window it was sized for.
PROMPT_OVERHEAD_TOKENS = 640
# A correction is the transcript said again, so the decode budget has to be at
# least as large as the transcript. `app/cleanup/validation.py` refuses an
# answer longer than 1.6x the original, and a quarter more than the input
# covers ordinary repair without reserving window for a rewrite that would be
# thrown away anyway.
OUTPUT_TO_INPUT_RATIO = 1.25
# Slack between an estimated token count and the budget it is measured
# against, for the variation between one paragraph of a transcript and the
# next. The window keeps a reserve besides, so a bad estimate costs headroom
# rather than a dropped system instruction.
TOKEN_ESTIMATE_MARGIN = 0.85
# Room for the BOS and template tokens a tokenizer adds on top of the text.
TOKENIZER_MARGIN_TOKENS = 32
MINIMUM_OUTPUT_TOKENS = 64
JSON_WRAPPER_TOKENS = 24


@dataclass(frozen=True, slots=True)
class TokenBudget:
    """How one `llama-server` window is divided between transcript and answer.

    Derived from the window a worker was actually launched with rather than
    fixed, so a high-end host's larger `--ctx-size` buys longer one-pass
    corrections instead of a KV cache that is never filled.
    """

    context_tokens: int
    input_tokens: int
    output_tokens: int

    @property
    def certain_bytes(self) -> int:
        """Encoded size below which a transcript cannot exceed `input_tokens`.

        Bytes, not characters. A byte-fallback BPE never emits more than one
        token per *byte*, but it emits several per character for any script
        outside its vocabulary: Runic "\u16a0\u16a2\u16a6" is three characters and six
        tokens. Counting characters here would let a Devanagari, CJK, or
        emoji-heavy transcript skip the tokenize round trip and then overrun
        the window, which drops the front of the context - the system
        instruction - and corrects the text under no rules at all.
        """
        return max(1, self.input_tokens - TOKENIZER_MARGIN_TOKENS)


def budget_for_context(context_tokens: int) -> TokenBudget:
    """Split a context window between the transcript and its correction."""
    window = max(MINIMUM_CONTEXT_TOKENS, context_tokens)
    usable = window - PROMPT_OVERHEAD_TOKENS
    input_tokens = int(usable / (1 + OUTPUT_TO_INPUT_RATIO))
    return TokenBudget(
        context_tokens=window,
        input_tokens=input_tokens,
        output_tokens=usable - input_tokens,
    )


# What a runtime assumes when nobody has told it which window it is talking to.
DEFAULT_TOKEN_BUDGET = budget_for_context(MINIMUM_CONTEXT_TOKENS)


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
