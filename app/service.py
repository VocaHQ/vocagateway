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

from app import config, engines, errors, metrics, scripts, storage, text_styles
from app.models.base import (
    AudioNormalizer,
    EngineTranscription,
    TranscriptionEngine,
    TranscriptionOptions,
)

TRANSCRIPTION_SLOT_TIMEOUT_SECONDS = 0.05
FAILED_SESSION_STATE = "failed"
_MILLISECONDS_PER_SECOND = 1000
_DARWIN_RSS_DIVISOR = 1024 * 1024
_LINUX_RSS_DIVISOR = 1024


@dataclass(frozen=True, slots=True)
class AdhocTranscription:
    transcript: str
    engine: str
    timing: metrics.PipelineTiming


class TranscriptionService:
    def __init__(
        self,
        settings: config.Settings,
        repository: storage.SessionRepository,
        engine_provider: engines.EngineProvider,
        normalizer: AudioNormalizer,
    ) -> None:
        self.settings = settings
        self.repository = repository
        self.upload_dir = settings.data_dir / "audio"
        self.normalized_dir = settings.data_dir / "normalized"
        self.metrics = metrics.RuntimeMetrics(settings.maximum_concurrent_transcriptions)
        self._engine_provider = engine_provider
        self._normalizer = normalizer
        self._transcription_slots = asyncio.Semaphore(settings.maximum_concurrent_transcriptions)

    async def finish(self, session_id: UUID) -> storage.StoredSession:
        stored = self.require(session_id)
        if stored.state == "completed":
            return stored
        if stored.audio_name is None:
            raise errors.APIProblem(
                status.HTTP_409_CONFLICT,
                "audio_missing",
                "Upload audio before finishing the session.",
            )
        if stored.state == "transcribing":
            raise errors.APIProblem(
                status.HTTP_409_CONFLICT,
                "transcription_in_progress",
                "Transcription is already in progress.",
            )
        await self._acquire_transcription_slot()
        return await _SessionJob(self, stored).run()

    async def transcribe_adhoc(self, source: Path, language: str) -> AdhocTranscription:
        """One-shot transcription for the WebUI test recorder (no session stored)."""
        await self._acquire_transcription_slot()
        return await _AdhocJob(self, source, language).run()

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
        engine = self.service._engine_provider.current()
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
        self.service.repository.update(self.stored.session_id, state="transcribing")
        source = _Pipeline.safe_audio_path(self.service.upload_dir, self.stored.audio_name or "")
        outcome, normalization_ms, engine = await _EnginePass(
            self.service, self.stored.language, self.stored.style
        ).run(source, normalized)
        _require_matching_script(outcome.text, self.stored.language)
        completed = self._persist(source, outcome.text)
        await self._record_success(engine, started, normalization_ms, outcome, normalized)
        return completed

    def _persist(self, source: Path, raw: str) -> storage.StoredSession:
        stored = self.stored
        transcript = text_styles.apply_writing_style(raw, stored.style, stored.language)
        keep_audio = stored.audio_name
        if self.service.settings.delete_successful_audio:
            keep_audio = None
        completed = self.service.repository.update(
            stored.session_id,
            state="completed",
            transcript=transcript,
            error_code=None,
            audio_name=keep_audio,
        )
        if self.service.settings.delete_successful_audio:
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
        self.service.repository.update(
            self.stored.session_id, state=FAILED_SESSION_STATE, error_code=code
        )
        return mapped


class _AdhocJob(_TranscriptionJob[AdhocTranscription]):
    def __init__(self, service: TranscriptionService, source: Path, language: str) -> None:
        super().__init__(service)
        self.source = source
        self.language = language

    def _normalized_path(self) -> Path:
        return self.service.normalized_dir / f"adhoc-{uuid4()}.wav"

    async def _complete(self, normalized: Path, started: float) -> AdhocTranscription:
        outcome, normalization_ms, engine = await _EnginePass(
            self.service, self.language, "raw"
        ).run(self.source, normalized)
        _require_matching_script(outcome.text, self.language)
        return await self._success(engine, outcome, normalization_ms, normalized, started)

    async def _success(
        self,
        engine: TranscriptionEngine,
        outcome: EngineTranscription,
        normalization_ms: int,
        normalized: Path,
        started: float,
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
        return AdhocTranscription(outcome.text.strip(), name, timing)
