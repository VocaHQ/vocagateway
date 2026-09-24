from __future__ import annotations

import asyncio
import platform
import resource
import time
import wave
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID, uuid4

from starlette import status

from app import config, engines, errors, metrics, scripts, storage
from app.cleanup.base import (
    DEFAULT_TIMEOUT_SECONDS,
    MODE_OFF,
    PROMPT_VERSION,
    CleanupOptions,
    CleanupOutcome,
    CleanupStatus,
    FinalTranscript,
)
from app.cleanup.manager import CleanupManager
from app.cleanup.service import CleanupService
from app.models.base import (
    AudioNormalizer,
    EngineTranscription,
    TranscriptionEngine,
    TranscriptionOptions,
)

ENGINE_OVERLOADED = "engine_overloaded"
CLIENT_DISCONNECTED = "client_disconnected"
TRANSCRIPTION_CANCELLED = "transcription_cancelled"
# Not a failed decode: the gateway refused or dropped the request before one ran.
_UNATTEMPTED = frozenset((ENGINE_OVERLOADED, CLIENT_DISCONNECTED))
# nginx's "client closed request". Nobody reads it; it only labels the log line.
CLIENT_CLOSED_REQUEST = 499
DISCONNECT_POLL_SECONDS = 0.5
FAILED_SESSION_STATE = "failed"
COMPLETED_SESSION_STATE = "completed"
RAW_STYLE = "raw"
# The formatting an explicit `/v1/audio/transcriptions` cleanup request gets.
# That endpoint has no writing-style field, and routing an opt-in through the
# unconditional raw bypass would make the option do nothing at all.
#
# `formal` is a tone-neutral name for a tone-neutral transform: it capitalises
# sentence starts and ensures a terminator, and does nothing else. The styles
# that actually change register are `very_casual` and `excited`, and neither
# belongs in an endpoint whose caller never asked for a voice.
ADHOC_CLEANUP_STYLE = "formal"
_MILLISECONDS_PER_SECOND = 1000
_DARWIN_RSS_DIVISOR = 1024 * 1024
_LINUX_RSS_DIVISOR = 1024


@dataclass(frozen=True, slots=True)
class AdhocTranscription:
    transcript: str
    engine: str
    timing: metrics.PipelineTiming
    original_transcript: str | None = None
    cleanup: CleanupOutcome | None = None


def _busy_engine() -> errors.APIProblem:
    return errors.APIProblem(
        status.HTTP_503_SERVICE_UNAVAILABLE,
        ENGINE_OVERLOADED,
        "The local transcription engine is busy.",
        recoverable=True,
    )


def _cancelled() -> errors.APIProblem:
    return errors.APIProblem(
        status.HTTP_503_SERVICE_UNAVAILABLE,
        TRANSCRIPTION_CANCELLED,
        "The transcription was cancelled before it finished.",
        recoverable=True,
    )


def _client_gone() -> errors.APIProblem:
    return errors.APIProblem(
        CLIENT_CLOSED_REQUEST,
        CLIENT_DISCONNECTED,
        "The client disconnected while waiting for a decode slot.",
        recoverable=True,
    )


Disconnected = Callable[[], Awaitable[bool]]


def snapshot_options(snapshot: storage.CleanupSnapshot) -> CleanupOptions:
    """Rebuild the pinned decision a session was created under.

    Read back from the row rather than from current settings, so finishing a
    session recorded weeks ago uses the options it was promised — and falls back
    if the model it named is no longer the selected one.
    """
    timeout_ms = snapshot.timeout_ms
    return CleanupOptions(
        mode=snapshot.mode,
        model_id=snapshot.model_id,
        prompt_version=snapshot.prompt_version or PROMPT_VERSION,
        timeout_seconds=(
            timeout_ms / _MILLISECONDS_PER_SECOND if timeout_ms else DEFAULT_TIMEOUT_SECONDS
        ),
    )


def cleanup_record(outcome: CleanupOutcome) -> storage.CleanupRecord:
    """What gets stored beside the transcript.

    A session that never opted in stores nothing, so its row and its response
    are indistinguishable from one written before this feature existed. The
    counters still see the decision — they are fed from the service, not from
    the database.
    """
    if outcome.status is CleanupStatus.DISABLED:
        return storage.CleanupRecord()
    return storage.CleanupRecord(
        status=str(outcome.status),
        reason=outcome.reason_name,
        duration_ms=outcome.duration_ms,
    )


class TranscriptionService:
    def __init__(
        self,
        settings: config.Settings,
        repository: storage.SessionRepository,
        engine_provider: engines.EngineProvider,
        normalizer: AudioNormalizer,
        cleanup_manager: CleanupManager | None = None,
    ) -> None:
        self.settings = settings
        self.repository = repository
        self.upload_dir = settings.data_dir / "audio"
        self.normalized_dir = settings.data_dir / "normalized"
        self._engine_provider = engine_provider
        # Both read the limit of whichever engine is loaded when they ask, so a
        # model swap takes effect for waiters and for the status page at once.
        self.metrics = metrics.RuntimeMetrics(self.parallel_limit)
        self._normalizer = normalizer
        self._decode_limiter = _DecodeLimiter(self.parallel_limit)
        # One shared finalization service for sessions, one-shot requests and
        # streaming finals, so the three cannot drift into different decisions.
        # Independent of the speech pipeline on purpose: cleanup has its own
        # admission bound, and a busy corrector must never look like a busy
        # transcriber to a client.
        self.cleanup = CleanupService(cleanup_manager, record=self._record_cleanup)

    async def finish(
        self, session_id: UUID, *, disconnected: Disconnected | None = None
    ) -> storage.StoredSession:
        """Transcribe a session exactly once, however many callers ask.

        A completed session answers from storage — including its original text
        and cleanup metadata — without another model call, after a settings
        change, a fallback, or a restart. There is no implicit re-clean here:
        reprocessing an existing result would need its own contract.
        """
        stored = self.require(session_id)
        if stored.state == COMPLETED_SESSION_STATE:
            return stored
        if stored.audio_name is None:
            raise errors.APIProblem(
                status.HTTP_409_CONFLICT,
                "audio_missing",
                "Upload audio before finishing the session.",
            )
        if stored.state == "transcribing":
            raise self._in_progress()
        # Take a place in line before the claim, so a full line leaves the
        # session exactly as the client left it rather than failed.
        with _Ticket(self, disconnected=disconnected) as ticket:
            claimed = self.repository.claim_transcribing(session_id)
            if claimed is None:
                return self._resolve_lost_claim(session_id)
            return await _SessionJob(self, ticket, claimed).run()

    async def transcribe_adhoc(
        self,
        source: Path,
        language: str,
        *,
        style: str = RAW_STYLE,
        cleanup: CleanupOptions | None = None,
        disconnected: Disconnected | None = None,
    ) -> AdhocTranscription:
        """One-shot transcription with no session stored.

        The default is unchanged and deliberately raw: the OpenAI-compatible
        endpoint and the WebUI benchmark both depend on getting the model's own
        text back. Callers opt into anything else explicitly.
        """
        with _Ticket(self, disconnected=disconnected) as ticket:
            return await _AdhocJob(self, ticket, source, language, style, cleanup).run()

    def require(self, session_id: UUID) -> storage.StoredSession:
        session = self.repository.get(session_id)
        if session is None:
            raise errors.APIProblem(
                status.HTTP_404_NOT_FOUND, "session_not_found", "The session does not exist."
            )
        return session

    def delete(self, session_id: UUID) -> bool:
        session = self.repository.delete(session_id)
        if session is None:
            return False
        if session.audio_name:
            _Pipeline.safe_audio_path(self.upload_dir, session.audio_name).unlink(missing_ok=True)
        (self.normalized_dir / f"{session_id}.wav").unlink(missing_ok=True)
        return True

    def cleanup_expired(self) -> int:
        expired = self.repository.expired(self.settings.retention_hours)
        for session in expired:
            self.delete(session.session_id)
        return len(expired)

    @asynccontextmanager
    async def occupy_slot(self, *, wait: bool = True) -> AsyncIterator[None]:
        with _Ticket(self, wait=wait) as ticket:
            async with ticket.decode_slot():
                yield

    def release_transcription_slot(self) -> None:
        self._decode_limiter.release()
        self.metrics.finished()

    async def acquire_transcription_slot(self, *, wait: bool = True) -> None:
        """Take a decode slot, queueing for one unless `wait` is false.

        Batch jobs do not come through here: they take a `_Ticket` on arrival so
        FFmpeg and cleanup run outside the slot. This is the one-step form for
        callers that decode as soon as they are admitted, like a live stream.
        `wait=False` refuses unless a decode slot is free now; a job still in
        FFmpeg does not occupy one.
        """
        with _Ticket(self, wait=wait) as ticket:
            await ticket.take_slot()

    def parallel_limit(self) -> int:
        """How many clips the loaded engine may decode at once, right now."""
        advertised = getattr(self._engine_provider.current(), "max_parallel_decodes", 1)
        return max(1, min(self.settings.maximum_concurrent_transcriptions, int(advertised)))

    def _record_cleanup(self, outcome: CleanupOutcome) -> None:
        self.metrics.record_cleanup(str(outcome.status), outcome.reason_name, outcome.duration_ms)

    def _in_progress(self) -> errors.APIProblem:
        return errors.APIProblem(
            status.HTTP_409_CONFLICT,
            "transcription_in_progress",
            "Transcription is already in progress.",
        )

    def _resolve_lost_claim(self, session_id: UUID) -> storage.StoredSession:
        """What a caller that lost the race is told.

        Losing means somebody else already owns this session's transcription, so
        this call never starts a second one: it returns the finished result if
        the winner is done, and otherwise reports the same in-progress conflict
        a plain second `finish` would have.
        """
        current = self.require(session_id)
        if current.state == COMPLETED_SESSION_STATE:
            return current
        raise self._in_progress()


class _Ticket:
    """One request's place in line, from arrival until it holds a decode slot.

    Taken before FFmpeg runs and before a session is claimed, so a full line is
    refused before any work is done. The queue timeout runs from arrival, so a
    slow normalisation spends the same budget as waiting would. Clips enter the
    decode line when their audio is ready: holding a place for one still in
    FFmpeg would leave the engine idle while ready clips wait behind it.
    """

    def __init__(
        self,
        service: TranscriptionService,
        *,
        wait: bool = True,
        disconnected: Disconnected | None = None,
    ) -> None:
        if wait:
            waiting = service.settings.maximum_queued_transcriptions
            if not service.metrics.offer_queue(waiting, slots=service.parallel_limit()):
                raise _busy_engine()
            budget = service.settings.transcription_queue_timeout_seconds
            in_line = True
        else:
            budget = 0
            in_line = False
        self._service = service
        self._deadline = asyncio.get_running_loop().time() + budget
        self._disconnected = disconnected
        self._abandoned = False
        self._in_line = in_line

    def __enter__(self) -> _Ticket:
        return self

    def __exit__(self, *_exc: object) -> None:
        self._leave()

    @asynccontextmanager
    async def before_decode(self) -> AsyncIterator[None]:
        """Bound pre-decode work, such as FFmpeg, by this ticket's deadline."""
        try:
            async with asyncio.timeout_at(self._deadline):
                await self._abandon_if_gone()
                async with self._guard_disconnect():
                    yield
                await self._abandon_if_gone()
        except TimeoutError as error:
            self._leave(rejected=True)
            raise _busy_engine() from error

    async def take_slot(self) -> None:
        service = self._service
        remaining = self._deadline - asyncio.get_running_loop().time()
        try:
            await self._abandon_if_gone()
            await self._unless_abandoned(service._decode_limiter.acquire(remaining))
            await self._abandon_if_gone(held=True)
        except TimeoutError as error:
            self._leave(rejected=True)
            raise _busy_engine() from error
        queued = self._in_line
        self._in_line = False
        service.metrics.started(queued=queued)

    @asynccontextmanager
    async def decode_slot(self) -> AsyncIterator[None]:
        held = False
        try:
            await self.take_slot()
            held = True
            yield
        except BaseException:
            raise
        finally:
            if held:
                self._service.release_transcription_slot()

    async def _unless_abandoned(self, acquire: Awaitable[None]) -> None:
        async with self._guard_disconnect():
            await acquire

    @asynccontextmanager
    async def _guard_disconnect(self) -> AsyncIterator[None]:
        """Give up the place in line if the client hangs up while this work runs.

        Starlette does not cancel a handler when its client goes away, so
        without this a phone that timed out would keep its place and later be
        decoded for nobody.
        """
        if self._disconnected is None:
            yield
            return
        waiter = asyncio.current_task()
        assert waiter is not None
        watcher = asyncio.create_task(self._watch(waiter))
        try:
            yield
        except asyncio.CancelledError:
            if not self._abandoned:
                raise
            waiter.uncancel()
            raise _client_gone() from None
        finally:
            watcher.cancel()

    async def _watch(self, waiter: asyncio.Task[object]) -> None:
        disconnected = self._disconnected
        assert disconnected is not None
        # Starlette offers no disconnect event to wait on, only a probe.
        while not await disconnected():  # noqa: ASYNC110
            await asyncio.sleep(DISCONNECT_POLL_SECONDS)
        self._abandoned = True
        waiter.cancel()

    async def _abandon_if_gone(self, *, held: bool = False) -> None:
        disconnected = self._disconnected
        if disconnected is None:
            return
        release_grant = held
        try:
            if not await disconnected():
                release_grant = False
                return
            self._leave()
            raise _client_gone()
        except BaseException:
            raise
        finally:
            if release_grant:
                self._service._decode_limiter.release()

    def _leave(self, *, rejected: bool = False) -> None:
        if self._in_line:
            self._in_line = False
            self._service.metrics.dequeued(rejected=rejected)
            return
        if rejected:
            self._service.metrics.reject()


class _DecodeLimiter:
    """First-come, first-served decode slots under the loaded engine's limit.

    A plain `asyncio.Condition` lets a newcomer take a freed slot before the
    waiter it was freed for wakes up. Here a release hands the slot straight to
    the head of the line, and nobody passes a waiter that is still in it. The
    limit is read at every grant, so a swap to a single-decode engine stops
    further grants until the decodes already running drain below it.
    """

    def __init__(self, limit: Callable[[], int]) -> None:
        self._limit = limit
        self._active = 0
        self._waiters: deque[asyncio.Future[None]] = deque()

    async def acquire(self, wait_seconds: float) -> None:
        # A swap to a roomier engine raises the limit without a release.
        self._wake()
        if not self._waiters and self._active < self._limit():
            self._active += 1
            try:
                # Checkpoint after the grant: a cancelling awaiter would otherwise
                # discard this return and leak `_active` forever.
                await asyncio.sleep(0)
            except BaseException:
                self.release()
                raise
            return
        if wait_seconds <= 0:
            raise TimeoutError
        seat: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._waiters.append(seat)
        try:
            async with asyncio.timeout(wait_seconds):
                await seat
        except BaseException:
            if seat.done() and not seat.cancelled():
                # Handed a slot in the same tick this waiter gave up.
                self.release()
            else:
                with suppress(ValueError):
                    self._waiters.remove(seat)
            raise

    def release(self) -> None:
        self._active = max(0, self._active - 1)
        self._wake()

    def _wake(self) -> None:
        limit = self._limit()
        while self._waiters and self._active < limit:
            seat = self._waiters.popleft()
            if not seat.done():
                self._active += 1
                seat.set_result(None)


def _require_matching_script(text: str, language: str) -> None:
    """Refuse a transcript written in the wrong alphabet.

    Models that detect the language themselves can return fluent text in a
    language nobody asked for — Dolphin turns a short Hindi phrase into Cyrillic.
    Inserting that at the cursor is worse than failing, because it looks like a
    real transcript. Raised as `LanguageUnsupportedError` so it carries the same
    non-retryable `language_unsupported` code the clients already explain.
    """
    if scripts.transcript_matches_language(text, language):
        return
    raise errors.LanguageUnsupportedError(
        f"The model transcribed this as a different language than {language}. "
        "It detects the language itself and misread a short recording; try "
        "speaking a full sentence, or choose a model that supports this language."
    )


class _Pipeline:
    @classmethod
    def elapsed_ms(cls, started: float) -> int:
        return max(0, int((time.monotonic() - started) * _MILLISECONDS_PER_SECOND))

    @classmethod
    def safe_audio_path(cls, upload_dir: Path, audio_name: str) -> Path:
        if Path(audio_name).name != audio_name:
            raise errors.APIProblem(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                "invalid_storage_reference",
                "Stored audio reference is invalid.",
            )
        return upload_dir / audio_name

    @classmethod
    def engine_outcome(
        cls,
        engine_result: str | EngineTranscription,
        inference_started: float,
    ) -> EngineTranscription:
        if isinstance(engine_result, EngineTranscription):
            return engine_result
        return EngineTranscription(
            text=engine_result, inference_ms=cls.elapsed_ms(inference_started)
        )

    @classmethod
    def wav_duration_ms(cls, path: Path) -> int:
        try:
            with wave.open(str(path), "rb") as source:
                frames = source.getnframes()
                frame_rate = source.getframerate()
        except (OSError, EOFError, wave.Error):
            return 0
        return round(frames * _MILLISECONDS_PER_SECOND / frame_rate) if frame_rate else 0

    @classmethod
    def timing(
        cls,
        total_ms: int,
        normalization_ms: int,
        outcome: EngineTranscription,
        audio_duration_ms: int,
        engine: str,
    ) -> metrics.PipelineTiming:
        inference_ms = outcome.inference_ms
        rtf = round(inference_ms / audio_duration_ms, 3) if audio_duration_ms else None
        return metrics.PipelineTiming(
            total_ms=total_ms,
            normalization_ms=normalization_ms,
            model_load_ms=outcome.model_load_ms,
            inference_ms=inference_ms,
            audio_duration_ms=audio_duration_ms,
            real_time_factor=rtf,
            engine=engine,
            peak_memory_mb=cls.peak_memory_mb(),
        )

    @classmethod
    def peak_memory_mb(cls) -> float | None:
        try:
            usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        except (OSError, ValueError):
            return None
        divisor = _DARWIN_RSS_DIVISOR if platform.system() == "Darwin" else _LINUX_RSS_DIVISOR
        return round(usage / divisor, 1)

    @classmethod
    def mapped_failure(
        cls,
        error: Exception,
        runtime_metrics: metrics.RuntimeMetrics,
        started: float,
    ) -> Exception:
        # A refusal is already counted as rejected; it is not a failed decode.
        if not (isinstance(error, errors.APIProblem) and error.code in _UNATTEMPTED):
            runtime_metrics.record_result(cls.elapsed_ms(started), success=False)
        for error_type, code, status_code, recoverable in _KNOWN_FAILURES:
            if isinstance(error, error_type):
                problem = errors.APIProblem(status_code, code, str(error), recoverable=recoverable)
                problem.__cause__ = error
                return problem
        return error


# Subclass-first so SilentAudioError is not swallowed by InvalidAudioError,
# and LanguageUnsupportedError is not swallowed by TranscriptionProcessError.
_FailureRow = tuple[type[Exception], str, int, bool]
_KNOWN_FAILURES: tuple[_FailureRow, ...] = (
    (errors.SilentAudioError, "silent_audio", status.HTTP_422_UNPROCESSABLE_CONTENT, False),
    (errors.InvalidAudioError, "invalid_audio", status.HTTP_422_UNPROCESSABLE_CONTENT, False),
    (
        errors.EngineUnavailableError,
        "engine_unavailable",
        status.HTTP_503_SERVICE_UNAVAILABLE,
        True,
    ),
    (
        errors.LanguageUnsupportedError,
        "language_unsupported",
        status.HTTP_422_UNPROCESSABLE_CONTENT,
        False,
    ),
    (errors.TranscriptionProcessError, "transcription_failed", status.HTTP_502_BAD_GATEWAY, True),
)


class _EnginePass:
    def __init__(
        self, service: TranscriptionService, ticket: _Ticket, language: str, style: str
    ) -> None:
        self.service = service
        self.ticket = ticket
        self.language = language
        self.style = style

    async def run(
        self, source: Path, normalized: Path
    ) -> tuple[EngineTranscription, int, TranscriptionEngine]:
        normalization_started = time.monotonic()
        async with self.ticket.before_decode():
            await self.service._normalizer.normalize(
                source, normalized, self.service.settings.maximum_duration_seconds
            )
        normalization_ms = _Pipeline.elapsed_ms(normalization_started)
        outcome, engine = await self._infer(normalized)
        return outcome, normalization_ms, engine

    async def _infer(self, normalized: Path) -> tuple[EngineTranscription, TranscriptionEngine]:
        # Slot before lease: a request still waiting must not hold a lease, or
        # a model swap would keep the retired model resident until it ran.
        async with (
            self.ticket.decode_slot(),
            self.service._engine_provider.lease() as engine,
        ):
            inference_started = time.monotonic()
            raw_result = await engine.transcribe(
                normalized,
                TranscriptionOptions(language=self.language, style=self.style),
            )
            return _Pipeline.engine_outcome(raw_result, inference_started), engine


class _TranscriptionJob[JobResult](ABC):
    def __init__(self, service: TranscriptionService, ticket: _Ticket) -> None:
        self.service = service
        self.ticket = ticket

    async def run(self) -> JobResult:
        normalized = self._normalized_path()
        started = time.monotonic()
        try:
            return await self._complete(normalized, started)
        except Exception as error:
            mapped = _Pipeline.mapped_failure(error, self.service.metrics, started)
            mapped = self._record_failure(mapped)
            if mapped is error:
                raise
            raise mapped from error
        except asyncio.CancelledError:
            # Shutdown, or a cancelled handler. A session left `transcribing`
            # would refuse every later finish and retry, so free it first.
            self._record_failure(_cancelled())
            raise
        finally:
            self._release(normalized)

    @abstractmethod
    def _normalized_path(self) -> Path: ...

    @abstractmethod
    async def _complete(self, normalized: Path, started: float) -> JobResult: ...

    def _record_failure(self, mapped: Exception) -> Exception:
        return mapped

    def _release(self, normalized: Path) -> None:
        normalized.unlink(missing_ok=True)


class _SessionJob(_TranscriptionJob[storage.StoredSession]):
    def __init__(
        self, service: TranscriptionService, ticket: _Ticket, stored: storage.StoredSession
    ) -> None:
        super().__init__(service, ticket)
        self.stored = stored

    def _normalized_path(self) -> Path:
        session_id = str(self.stored.session_id)
        return self.service.normalized_dir / f"{session_id}.wav"

    async def _complete(self, normalized: Path, started: float) -> storage.StoredSession:
        stored = self.stored
        source = _Pipeline.safe_audio_path(self.service.upload_dir, stored.audio_name or "")
        outcome, normalization_ms, engine = await _EnginePass(
            self.service, self.ticket, stored.language, stored.style
        ).run(source, normalized)
        _require_matching_script(outcome.text, stored.language)
        # The engine lease is already released here, so the optional text model
        # never runs while a speech model is held open behind it.
        final = await self.service.cleanup.finalize(
            outcome.text,
            style=stored.style,
            language=stored.language,
            options=snapshot_options(stored.cleanup),
        )
        completed = self._persist(source, final)
        await self._record_success(engine, started, normalization_ms, outcome, normalized)
        return completed

    def _persist(self, source: Path, final: FinalTranscript) -> storage.StoredSession:
        """Store the final text, the original, and the cleanup result in one write.

        The state test lives inside the statement, so a session deleted or
        re-uploaded while this job was running is neither resurrected nor
        overwritten — the write matches no row and this reports a conflict.
        """
        stored = self.stored
        delete_audio = self.service.settings.delete_successful_audio
        completed = self.service.repository.complete(
            stored.session_id,
            transcript=final.transcript,
            original_transcript=final.original_transcript,
            cleanup=cleanup_record(final.cleanup),
            audio_name=None if delete_audio else stored.audio_name,
        )
        if completed is None:
            raise errors.APIProblem(
                status.HTTP_409_CONFLICT,
                "session_superseded",
                "The session was deleted or replaced while it was being transcribed.",
            )
        if delete_audio:
            source.unlink(missing_ok=True)
        return completed

    async def _record_success(
        self,
        engine: TranscriptionEngine,
        started: float,
        normalization_ms: int,
        outcome: EngineTranscription,
        normalized: Path,
    ) -> None:
        total_ms = _Pipeline.elapsed_ms(started)
        engine_name = (await engine.health()).name
        self.service.metrics.record_result(
            total_ms,
            success=True,
            timing=_Pipeline.timing(
                total_ms,
                normalization_ms,
                outcome,
                _Pipeline.wav_duration_ms(normalized),
                engine_name,
            ),
        )

    def _record_failure(self, mapped: Exception) -> Exception:
        # A refusal before decode is not a failed transcription: restore the
        # session to `uploaded` so the client can retry as it would after overflow.
        if isinstance(mapped, errors.APIProblem) and mapped.code in _UNATTEMPTED:
            self.service.repository.restore_uploaded(self.stored.session_id)
            return mapped
        code = mapped.code if isinstance(mapped, errors.APIProblem) else "internal_error"
        # Leave unknown failures retryable: stuck "transcribing" rejects finish
        # and is not in the retry allow-list (failed/uploaded/completed).
        if isinstance(mapped, errors.APIProblem) and mapped.code == "language_unsupported":
            # Retrying replays the same language against the same model.
            code = "language_unsupported"
        # Conditional on the job still owning the session: a delete or a fresh
        # upload during transcription must not be turned into a failed state by
        # the job it superseded.
        self.service.repository.fail_transcribing(self.stored.session_id, code)
        return mapped


class _AdhocJob(_TranscriptionJob[AdhocTranscription]):
    def __init__(
        self,
        service: TranscriptionService,
        ticket: _Ticket,
        source: Path,
        language: str,
        style: str = RAW_STYLE,
        cleanup: CleanupOptions | None = None,
    ) -> None:
        super().__init__(service, ticket)
        self.source = source
        self.language = language
        self.style = style
        self.options = cleanup or CleanupOptions(mode=MODE_OFF)

    def _normalized_path(self) -> Path:
        return self.service.normalized_dir / f"adhoc-{uuid4()}.wav"

    async def _complete(self, normalized: Path, started: float) -> AdhocTranscription:
        # The engine is always asked for raw text; the writing style is applied
        # afterwards, so opting into cleanup cannot change what the model is
        # asked to produce.
        outcome, normalization_ms, engine = await _EnginePass(
            self.service, self.ticket, self.language, RAW_STYLE
        ).run(self.source, normalized)
        _require_matching_script(outcome.text, self.language)
        final = await self.service.cleanup.finalize(
            outcome.text,
            style=self.style,
            language=self.language,
            options=self.options,
        )
        return await self._success(engine, outcome, normalization_ms, normalized, started, final)

    async def _success(
        self,
        engine: TranscriptionEngine,
        outcome: EngineTranscription,
        normalization_ms: int,
        normalized: Path,
        started: float,
        final: FinalTranscript,
    ) -> AdhocTranscription:
        name = (await engine.health()).name
        duration_ms = _Pipeline.elapsed_ms(started)
        timing = _Pipeline.timing(
            duration_ms,
            normalization_ms,
            outcome,
            _Pipeline.wav_duration_ms(normalized),
            name,
        )
        self.service.metrics.record_result(duration_ms, success=True, timing=timing)
        return AdhocTranscription(
            final.transcript.strip(),
            name,
            timing,
            original_transcript=final.original_transcript,
            cleanup=final.cleanup,
        )
