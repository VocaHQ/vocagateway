"""The one place a final transcript becomes the text a client inserts.

Sessions, one-shot requests, and streaming finals all come through here, so the
three cannot drift into making different decisions about the same transcript.
The contract is narrow and never negotiable:

* The deterministic result is computed first and independently. Whenever
  cleanup is off, skipped, or rejected, *that exact string* is what comes back —
  cleanup can only ever be an improvement on it, never a regression from it.
* A recognition that succeeded stays a success. Every cleanup failure becomes
  metadata on a successful result, not an error the caller has to handle.
* Cancellation propagates. It is never swallowed as a tidy fallback, and it
  never leaves inference running behind an abandoned request.
"""

from __future__ import annotations

import asyncio
import time
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from types import MappingProxyType

from app import text_styles
from app.cleanup import validation
from app.cleanup.base import (
    CleanupOptions,
    CleanupOutcome,
    CleanupReason,
    CleanupRejected,
    CleanupRuntime,
    CleanupStatus,
    CleanupUnavailable,
    FinalTranscript,
)
from app.cleanup.manager import CleanupManager
from app.cleanup.transport import TransportError

RAW_STYLE = "raw"
MILLISECONDS_PER_SECOND = 1000
# Scripts that identify a language on their own well enough to resolve `auto`.
# Latin is deliberately absent: most of the world's languages are written in it,
# and treating every Latin transcript as English is exactly the mistake that
# would send a French dictation to an English-tuned corrector.
_SCRIPT_LANGUAGES: MappingProxyType[str, str] = MappingProxyType({"DEVANAGARI": "hi"})

# How a runtime failure is reported. Order matters only in that the first match
# wins; every entry is a bounded reason, never a backend message.
_FAILURE_REASONS: tuple[tuple[type[BaseException], CleanupReason], ...] = (
    (TimeoutError, CleanupReason.TIMEOUT),
    (CleanupUnavailable, CleanupReason.MODEL_UNAVAILABLE),
    (TransportError, CleanupReason.MODEL_UNAVAILABLE),
    (ValueError, CleanupReason.RUNTIME_ERROR),
    (OSError, CleanupReason.RUNTIME_ERROR),
)
_RUNTIME_FAILURES = tuple(failure for failure, _ in _FAILURE_REASONS)
# A second copy of the text is stored only where it is actually recoverable
# information. `disabled` and `skipped` transcripts were never sent to a model,
# and their "original" would be the same string twice under the same retention.
_ORIGINAL_KEEPING_STATUSES = frozenset(
    (CleanupStatus.APPLIED, CleanupStatus.UNCHANGED, CleanupStatus.FALLBACK)
)

OutcomeRecorder = Callable[[CleanupOutcome], None]


@dataclass(frozen=True, slots=True)
class _Work:
    """One transcript and everything already decided about how to finish it."""

    transcript: str
    style: str
    language: str
    options: CleanupOptions
    # The exact string the gateway would have returned with cleanup switched
    # off, computed before anything else happens so it is always available.
    legacy: str
    # The language cleanup will run as, or None when it must not run at all.
    resolved: str | None


class CleanupService:
    """Applies optional cleanup and always returns something safe to insert."""

    def __init__(
        self,
        manager: CleanupManager | None,
        *,
        record: OutcomeRecorder | None = None,
    ) -> None:
        self.manager = manager
        self._record = record

    async def finalize(
        self,
        transcript: str,
        *,
        style: str,
        language: str,
        options: CleanupOptions,
    ) -> FinalTranscript:
        work = _Work(
            transcript=transcript,
            style=style,
            language=language,
            options=options,
            legacy=text_styles.apply_writing_style(transcript, style, language),
            resolved=resolve_language(language, transcript, self._supported()),
        )
        bypass = self._bypass(work)
        if bypass is not None:
            return self._finish(work.legacy, transcript, bypass)
        started = time.monotonic()
        cleaned, reason = await self._attempt(work)
        if cleaned is None:
            refused = reason or CleanupReason.RUNTIME_ERROR
            return self._finish(work.legacy, transcript, self._fallback(options, refused, started))
        return self._accept(work, cleaned, started)

    def _bypass(self, work: _Work) -> CleanupOutcome | None:
        """Decisions that need no runtime, in the order they must be made."""
        options = work.options
        if not options.enabled:
            return CleanupOutcome(requested=options.mode, status=CleanupStatus.DISABLED)
        if work.style == RAW_STYLE:
            # Raw means raw. An explicit cleanup request does not get to redefine
            # a style whose entire promise is that nothing was done to the text.
            return self._skip(options, CleanupReason.RAW_STYLE)
        if not work.transcript.strip():
            return self._skip(options, CleanupReason.EMPTY_INPUT)
        blocked = self._blocked(work)
        return None if blocked is None else self._fallback(options, blocked, None)

    def _blocked(self, work: _Work) -> CleanupReason | None:
        """Why this transcript must not reach a model, or None to go ahead."""
        manager = self.manager
        if manager is None or manager.model_id != work.options.model_id:
            # Includes a session pinned to a model that is no longer selected.
            # Falling back is right; quietly substituting another model is not.
            return CleanupReason.MODEL_UNAVAILABLE
        if work.resolved is None:
            return CleanupReason.UNSUPPORTED_LANGUAGE
        if validation.exceeds_input_ceiling(work.transcript):
            return CleanupReason.INPUT_TOO_LONG
        return None

    async def _attempt(self, work: _Work) -> tuple[str | None, CleanupReason | None]:
        """The candidate text, or None with the single reason it was refused.

        `asyncio.CancelledError` deliberately escapes: the caller went away, and
        reporting a tidy fallback for work nobody is waiting for would hide that
        the request was abandoned rather than completed.
        """
        manager = self.manager
        if manager is None:
            return None, CleanupReason.MODEL_UNAVAILABLE
        async with manager.lease() as lease:
            if lease.runtime is None:
                return None, lease.reason or CleanupReason.MODEL_UNAVAILABLE
            candidate = await self._infer(lease.runtime, work)
        if isinstance(candidate, CleanupReason):
            return None, candidate
        rejected = validation.rejection(work.transcript, candidate, work.resolved or work.language)
        if rejected is not None:
            return None, rejected
        return candidate, None

    async def _infer(self, runtime: CleanupRuntime, work: _Work) -> str | CleanupReason:
        budget = work.options.timeout_seconds
        try:
            return await asyncio.wait_for(
                runtime.clean(
                    work.transcript, work.resolved or work.language, budget_seconds=budget
                ),
                timeout=budget,
            )
        except CleanupRejected as rejection:
            return rejection.reason
        except _RUNTIME_FAILURES as failure:
            return _reason_for(failure)

    def _accept(self, work: _Work, cleaned: str, started: float) -> FinalTranscript:
        final = text_styles.apply_writing_style_paragraphs(cleaned, work.style, work.language)
        if self._styling_broke_it(work, cleaned, final):
            outcome = self._fallback(work.options, CleanupReason.UNSAFE_EDIT, started)
            return self._finish(work.legacy, work.transcript, outcome)
        status = CleanupStatus.UNCHANGED if final == work.legacy else CleanupStatus.APPLIED
        return self._finish(
            final,
            work.transcript,
            CleanupOutcome(
                requested=work.options.mode,
                status=status,
                model_id=work.options.model_id,
                prompt_version=work.options.prompt_version,
                duration_ms=_elapsed_ms(started),
            ),
        )

    def _styling_broke_it(self, work: _Work, cleaned: str, final: str) -> bool:
        """Whether *formatting* introduced a change the checks refuse.

        Only additional damage counts. A style that already trips a check on the
        legacy path — very casual lowercases an identifier, for instance — is
        the behaviour the gateway has always had, and holding the cleanup path
        to a stricter standard than the fallback it would return instead would
        reject good corrections for no gain.
        """
        language = work.resolved or work.language
        if validation.rejection(cleaned, final, language) is None:
            return False
        return validation.rejection(work.transcript, work.legacy, language) is None

    def _supported(self) -> tuple[str, ...]:
        return self.manager.supported_languages() if self.manager else ()

    def _skip(self, options: CleanupOptions, reason: CleanupReason) -> CleanupOutcome:
        return CleanupOutcome(requested=options.mode, status=CleanupStatus.SKIPPED, reason=reason)

    def _fallback(
        self, options: CleanupOptions, reason: CleanupReason, started: float | None
    ) -> CleanupOutcome:
        return CleanupOutcome(
            requested=options.mode,
            status=CleanupStatus.FALLBACK,
            reason=reason,
            model_id=options.model_id,
            prompt_version=options.prompt_version,
            duration_ms=0 if started is None else _elapsed_ms(started),
        )

    def _finish(self, transcript: str, original: str, outcome: CleanupOutcome) -> FinalTranscript:
        if self._record is not None:
            self._record(outcome)
        keep_original = outcome.status in _ORIGINAL_KEEPING_STATUSES
        return FinalTranscript(
            transcript=transcript,
            original_transcript=original if keep_original else None,
            cleanup=outcome,
        )


def _reason_for(failure: BaseException) -> CleanupReason:
    for failure_type, reason in _FAILURE_REASONS:
        if isinstance(failure, failure_type):
            return reason
    return CleanupReason.RUNTIME_ERROR


def resolve_language(language: str, transcript: str, supported: tuple[str, ...]) -> str | None:
    """The language cleanup will run as, or None when it must not run at all.

    A requested language is honoured when it is on the allowlist. `auto` is only
    resolved when the writing system narrows it to exactly one supported
    language — never by assuming that Latin script means English.
    """
    code = language.lower().split("-", maxsplit=1)[0]
    allowed = {name.lower() for name in supported}
    if language.lower() in allowed:
        return language.lower()
    if code in allowed:
        return code
    if code not in {"auto", ""}:
        return None
    return _from_script(transcript, allowed)


def _from_script(transcript: str, allowed: set[str]) -> str | None:
    seen = {_script_of(character) for character in transcript if character.isalpha()} - {""}
    candidates = {_SCRIPT_LANGUAGES[name] for name in seen if name in _SCRIPT_LANGUAGES}
    if len(candidates) != 1 or len(seen) != 1:
        return None
    only = candidates.pop()
    return only if only in allowed else None


def _script_of(character: str) -> str:
    try:
        return unicodedata.name(character).split(" ", maxsplit=1)[0]
    except ValueError:
        return ""


def _elapsed_ms(started: float) -> int:
    return max(0, int((time.monotonic() - started) * MILLISECONDS_PER_SECOND))
