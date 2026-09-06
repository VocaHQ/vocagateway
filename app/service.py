from __future__ import annotations

import asyncio
import platform
import resource
import time
import wave
from abc import ABC, abstractmethod
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

TRANSCRIPTION_SLOT_TIMEOUT_SECONDS = 0.05
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
        self.metrics = metrics.RuntimeMetrics(settings.maximum_concurrent_transcriptions)
        self._engine_provider = engine_provider
        self._normalizer = normalizer
        self._transcription_slots = asyncio.Semaphore(settings.maximum_concurrent_transcriptions)
        # One shared finalization service for sessions, one-shot requests and
        # streaming finals, so the three cannot drift into different decisions.
        # Independent of the speech pipeline on purpose: cleanup has its own
        # admission bound, and a busy corrector must never look like a busy
        # transcriber to a client.
        self.cleanup = CleanupService(cleanup_manager, record=self._record_cleanup)

    async def finish(self, session_id: UUID) -> storage.StoredSession:
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
        await self._acquire_transcription_slot()
        claimed = self.repository.claim_transcribing(session_id)
        if claimed is None:
            self._release_slot()
            return self._resolve_lost_claim(session_id)
        return await _SessionJob(self, claimed).run()

    async def transcribe_adhoc(
        self,
        source: Path,
        language: str,
        *,
        style: str = RAW_STYLE,
        cleanup: CleanupOptions | None = None,
    ) -> AdhocTranscription:
        """One-shot transcription with no session stored.

        The default is unchanged and deliberately raw: the OpenAI-compatible
        endpoint and the WebUI benchmark both depend on getting the model's own
        text back. Callers opt into anything else explicitly.
        """
        await self._acquire_transcription_slot()
        return await _AdhocJob(self, source, language, style, cleanup).run()

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

    def _release_slot(self) -> None:
        self._transcription_slots.release()
        self.metrics.finished()

    async def _acquire_transcription_slot(self) -> None:
        self.metrics.queued()
        try:
            await asyncio.wait_for(
                self._transcription_slots.acquire(), timeout=TRANSCRIPTION_SLOT_TIMEOUT_SECONDS
            )
        except TimeoutError as error:
            self.metrics.dequeued(rejected=True)
            raise errors.APIProblem(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "engine_overloaded",
                "The local transcription engine is busy.",
                recoverable=True,
            ) from error
        except BaseException:
            self.metrics.dequeued()
            raise
        self.metrics.started()


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
    def __init__(self, service: TranscriptionService, language: str, style: str) -> None:
        self.service = service
        self.language = language
        self.style = style

    async def run(
        self, source: Path, normalized: Path
    ) -> tuple[EngineTranscription, int, TranscriptionEngine]:
        normalization_started = time.monotonic()
        await self.service._normalizer.normalize(
            source, normalized, self.service.settings.maximum_duration_seconds
        )
        normalization_ms = _Pipeline.elapsed_ms(normalization_started)
        outcome, engine = await self._infer(normalized)
        return outcome, normalization_ms, engine

    async def _infer(self, normalized: Path) -> tuple[EngineTranscription, TranscriptionEngine]:
        async with self.service._engine_provider.lease() as engine:
            inference_started = time.monotonic()
            raw_result = await engine.transcribe(
                normalized,
                TranscriptionOptions(language=self.language, style=self.style),
            )
            return _Pipeline.engine_outcome(raw_result, inference_started), engine


class _TranscriptionJob[JobResult](ABC):
    def __init__(self, service: TranscriptionService) -> None:
        self.service = service

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
        self.service._transcription_slots.release()
        self.service.metrics.finished()


class _SessionJob(_TranscriptionJob[storage.StoredSession]):
    def __init__(self, service: TranscriptionService, stored: storage.StoredSession) -> None:
        super().__init__(service)
        self.stored = stored

    def _normalized_path(self) -> Path:
        session_id = str(self.stored.session_id)
        return self.service.normalized_dir / f"{session_id}.wav"

    async def _complete(self, normalized: Path, started: float) -> storage.StoredSession:
        stored = self.stored
        source = _Pipeline.safe_audio_path(self.service.upload_dir, stored.audio_name or "")
        outcome, normalization_ms, engine = await _EnginePass(
            self.service, stored.language, stored.style
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
        source: Path,
        language: str,
        style: str = RAW_STYLE,
        cleanup: CleanupOptions | None = None,
    ) -> None:
        super().__init__(service)
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
            self.service, self.language, RAW_STYLE
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
