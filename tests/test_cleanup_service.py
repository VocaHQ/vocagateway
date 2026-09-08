# Copyright (c) 2025-present VocaHQ, Inc.
# SPDX-License-Identifier: AGPL-3.0-only
# See the LICENSE file for details.

"""The shared finalization contract every entry point depends on.

The single most important property under test: with cleanup off, skipped, or
failed for any reason at all, the returned string is byte-for-byte what
`apply_writing_style` would have produced on its own. If that ever stops
holding, enabling the feature becomes a way to lose a good transcript.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from app import text_styles
from app.cleanup.base import (
    MODE_OFF,
    CleanupOptions,
    CleanupOutcome,
    CleanupReason,
    CleanupRejected,
    CleanupRuntime,
    CleanupStatus,
    CleanupUnavailable,
)
from app.cleanup.manager import Lease
from app.cleanup.service import CleanupService, resolve_language
from app.cleanup.transport import TransportError
from tests.conftest import CLEANUP_MODEL_ID, FakeCleanupRuntime, cleanup_options

SPOKEN = "we was going to leave early but the train was late"
LANGUAGES = ("en", "hi", "hinglish_roman")


class StubManager:
    """The slice of `CleanupManager` the finalization service actually uses."""

    def __init__(
        self,
        runtime: CleanupRuntime | None,
        *,
        model_id: str | None = CLEANUP_MODEL_ID,
        installed: bool = True,
        admit: bool = True,
        auto: str = "",
    ) -> None:
        self.auto = auto
        self.runtime = runtime
        self.model_id = model_id
        self.installed = installed
        self.admit = admit
        self.leases = 0

    def model_path(self) -> Path | None:
        return Path("model.gguf") if self.installed else None

    def supported_languages(self) -> tuple[str, ...]:
        return LANGUAGES

    def auto_language(self) -> str:
        return self.auto

    def lease(self) -> object:
        return _StubLease(self)


class _StubLease:
    def __init__(self, manager: StubManager) -> None:
        self.manager = manager

    async def __aenter__(self) -> Lease:
        if not self.manager.admit:
            return Lease(reason=CleanupReason.BUSY)
        if self.manager.runtime is None or not self.manager.installed:
            return Lease(reason=CleanupReason.MODEL_UNAVAILABLE)
        self.manager.leases += 1
        return Lease(runtime=self.manager.runtime)

    async def __aexit__(self, *_exc: object) -> bool:
        return False


def service_for(*answers: str | BaseException, **kwargs: object) -> CleanupService:
    runtime = FakeCleanupRuntime(*answers)
    return CleanupService(StubManager(runtime, **kwargs))  # type: ignore[arg-type]


async def finalize(
    service: CleanupService,
    transcript: str = SPOKEN,
    *,
    style: str = "casual",
    language: str = "en",
    options: CleanupOptions | None = None,
) -> object:
    return await service.finalize(
        transcript,
        style=style,
        language=language,
        options=options or cleanup_options(),
    )


def legacy(transcript: str = SPOKEN, style: str = "casual", language: str = "en") -> str:
    return text_styles.apply_writing_style(transcript, style, language)


async def test_disabled_returns_the_legacy_result_and_no_original() -> None:
    service = service_for("Anything at all.")
    final = await finalize(service, options=CleanupOptions(mode=MODE_OFF))
    assert final.transcript == legacy()
    assert final.original_transcript is None
    assert final.cleanup.status is CleanupStatus.DISABLED


async def test_raw_style_is_never_corrected_even_when_asked() -> None:
    """Raw's entire promise is that nothing was done to the text."""
    runtime = FakeCleanupRuntime("Corrected!")
    service = CleanupService(StubManager(runtime))  # type: ignore[arg-type]
    final = await finalize(service, style="raw")
    assert final.transcript == SPOKEN
    assert final.cleanup.status is CleanupStatus.SKIPPED
    assert final.cleanup.reason is CleanupReason.RAW_STYLE
    assert runtime.calls == []


async def test_empty_input_is_skipped_without_a_model_call() -> None:
    runtime = FakeCleanupRuntime("something")
    service = CleanupService(StubManager(runtime))  # type: ignore[arg-type]
    final = await finalize(service, "   ")
    assert final.cleanup.reason is CleanupReason.EMPTY_INPUT
    assert runtime.calls == []


async def test_a_valid_repair_is_applied_and_keeps_the_original() -> None:
    corrected = "We were going to leave early, but the train was late."
    service = service_for(corrected)
    final = await finalize(service)
    assert final.transcript == text_styles.apply_writing_style(corrected, "casual", "en")
    assert final.original_transcript == SPOKEN
    assert final.cleanup.status is CleanupStatus.APPLIED
    assert final.cleanup.model_id == CLEANUP_MODEL_ID
    assert final.cleanup.prompt_version == "cleanup-v1"


async def test_text_the_model_leaves_alone_reports_unchanged() -> None:
    already = "This is already correct."
    service = service_for(already)
    final = await finalize(service, already, style="clean")
    assert final.transcript == legacy(already, "clean")
    assert final.cleanup.status is CleanupStatus.UNCHANGED


async def test_paragraph_breaks_survive_the_writing_style() -> None:
    """Styling normally collapses a blank line; accepted cleanup must not lose it."""
    spoken = "first we ship the fix then we write the notes"
    corrected = "First we ship the fix.\n\nThen we write the notes."
    service = service_for(corrected)
    final = await finalize(service, spoken, style="formal")
    assert "\n\n" in final.transcript
    assert final.transcript.startswith("First we ship the fix.")


FALLBACKS = (
    (TimeoutError(), CleanupReason.TIMEOUT),
    (CleanupUnavailable("gone"), CleanupReason.MODEL_UNAVAILABLE),
    (TransportError("closed"), CleanupReason.MODEL_UNAVAILABLE),
    (CleanupRejected(CleanupReason.INVALID_OUTPUT), CleanupReason.INVALID_OUTPUT),
    (CleanupRejected(CleanupReason.INPUT_TOO_LONG), CleanupReason.INPUT_TOO_LONG),
    (ValueError("bad"), CleanupReason.RUNTIME_ERROR),
    (OSError("io"), CleanupReason.RUNTIME_ERROR),
)


@pytest.mark.parametrize(("failure", "reason"), FALLBACKS)
async def test_every_runtime_failure_falls_back_to_the_legacy_result(
    failure: BaseException, reason: CleanupReason
) -> None:
    service = service_for(failure)
    final = await finalize(service)
    assert final.transcript == legacy()
    assert final.cleanup.status is CleanupStatus.FALLBACK
    assert final.cleanup.reason is reason
    # Fallback still exposes the original, so the text is recoverable.
    assert final.original_transcript == SPOKEN


async def test_an_unsafe_edit_falls_back_rather_than_being_inserted() -> None:
    service = service_for("We were going to leave early but the train was 40 minutes late.")
    final = await finalize(service)
    assert final.transcript == legacy()
    assert final.cleanup.reason is CleanupReason.UNSAFE_EDIT


async def test_a_full_runtime_reports_busy_not_a_missing_model() -> None:
    service = CleanupService(StubManager(None, admit=False))  # type: ignore[arg-type]
    final = await finalize(service)
    assert final.cleanup.reason is CleanupReason.BUSY


async def test_a_missing_model_is_reported_as_unavailable() -> None:
    service = CleanupService(StubManager(None, installed=False))  # type: ignore[arg-type]
    final = await finalize(service)
    assert final.cleanup.reason is CleanupReason.MODEL_UNAVAILABLE


async def test_no_manager_at_all_still_returns_the_legacy_result() -> None:
    final = await finalize(CleanupService(None))
    assert final.transcript == legacy()
    assert final.cleanup.reason is CleanupReason.MODEL_UNAVAILABLE


async def test_a_session_pinned_to_another_model_falls_back() -> None:
    """A pinned model that is no longer selected must not be silently substituted."""
    service = service_for("We were going to leave early, but the train was late.")
    final = await finalize(service, options=cleanup_options(model_id="cleanup:something-else"))
    assert final.transcript == legacy()
    assert final.cleanup.reason is CleanupReason.MODEL_UNAVAILABLE


async def test_an_unsupported_language_is_not_sent_to_the_model() -> None:
    runtime = FakeCleanupRuntime("Bonjour.")
    service = CleanupService(StubManager(runtime))  # type: ignore[arg-type]
    final = await finalize(service, "bonjour tout le monde", language="fr")
    assert final.cleanup.reason is CleanupReason.UNSUPPORTED_LANGUAGE
    assert runtime.calls == []


async def test_an_oversized_transcript_is_returned_whole_not_truncated() -> None:
    runtime = FakeCleanupRuntime("short")
    service = CleanupService(StubManager(runtime))  # type: ignore[arg-type]
    long_text = "we was going to leave early " * 800
    final = await finalize(service, long_text)
    assert final.cleanup.reason is CleanupReason.INPUT_TOO_LONG
    assert final.transcript == legacy(long_text)
    assert runtime.calls == []


async def test_cancellation_propagates_instead_of_becoming_a_tidy_fallback() -> None:
    class Hanging:
        model_id = CLEANUP_MODEL_ID

        async def available(self) -> bool:
            return True

        async def clean(self, transcript: str, language: str, *, budget_seconds: float) -> str:
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    service = CleanupService(StubManager(Hanging()))  # type: ignore[arg-type]
    task = asyncio.create_task(finalize(service, options=cleanup_options(timeout_seconds=30)))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_the_deadline_is_enforced_by_the_service_itself() -> None:
    class Slow:
        model_id = CLEANUP_MODEL_ID

        async def available(self) -> bool:
            return True

        async def clean(self, transcript: str, language: str, *, budget_seconds: float) -> str:
            await asyncio.sleep(5)
            return transcript

    service = CleanupService(StubManager(Slow()))  # type: ignore[arg-type]
    final = await finalize(service, options=cleanup_options(timeout_seconds=0.05))
    assert final.cleanup.reason is CleanupReason.TIMEOUT
    assert final.transcript == legacy()


async def test_every_outcome_is_recorded_including_disabled() -> None:
    recorded: list[CleanupOutcome] = []
    service = CleanupService(
        StubManager(FakeCleanupRuntime()),  # type: ignore[arg-type]
        record=recorded.append,
    )
    await finalize(service, options=CleanupOptions(mode=MODE_OFF))
    await finalize(service)
    assert [str(outcome.status) for outcome in recorded] == ["disabled", "unchanged"]


LANGUAGE_CASES = (
    ("en", "hello there", "en"),
    ("EN", "hello there", "en"),
    ("en-GB", "hello there", "en"),
    ("hi", "मैं ठीक हूँ", "hi"),
    ("fr", "bonjour", None),
    # `auto` over Latin script stays unresolved: most languages are written in
    # it, and guessing English is exactly the mistake to avoid.
    ("auto", "hello there", None),
    ("auto", "मैं कल जाऊंगा", "hi"),
    # Mixed scripts under `auto` are ambiguous, so cleanup does not run.
    ("auto", "मैं report भेजूंगा", None),
)


@pytest.mark.parametrize(("requested", "text", "expected"), LANGUAGE_CASES)
def test_language_resolution(requested: str, text: str, expected: str | None) -> None:
    assert resolve_language(requested, text, LANGUAGES) == expected


async def test_auto_falls_back_when_the_operator_has_not_named_a_language() -> None:
    """The default stays "do not guess": Latin script does not name a language."""
    service = service_for("We were going to leave early, but the train was late.")
    final = await finalize(service, language="auto")
    assert final.transcript == legacy(language="auto")
    assert final.cleanup.reason is CleanupReason.UNSUPPORTED_LANGUAGE


async def test_auto_uses_the_language_the_operator_configured() -> None:
    """The one place that knows: nothing in the pipeline detects a language."""
    service = service_for("We were going to leave early, but the train was late.", auto="en")
    final = await finalize(service, language="auto")
    assert final.cleanup.status is CleanupStatus.APPLIED


async def test_the_writing_system_still_wins_over_the_configured_default() -> None:
    """Evidence about this transcript beats a standing preference about all of them."""
    runtime = FakeCleanupRuntime("यह ठीक है।")
    service = CleanupService(StubManager(runtime, auto="en"))  # type: ignore[arg-type]
    await finalize(service, "यह ठीक है", language="auto")
    assert [language for _, language in runtime.calls] == ["hi"]


async def test_a_configured_default_off_the_allowlist_is_ignored_not_honoured() -> None:
    """Narrowing the allowlist cannot leave this pointing somewhere unsupported."""
    service = service_for("We were going to leave early, but the train was late.", auto="fr")
    final = await finalize(service, language="auto")
    assert final.cleanup.reason is CleanupReason.UNSUPPORTED_LANGUAGE
