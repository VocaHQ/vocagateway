from __future__ import annotations

import asyncio
import tempfile
import time
from pathlib import Path

from app import scripts, system
from app.catalog import CatalogModel
from app.errors import EngineUnavailableError, LanguageUnsupportedError, TranscriptionProcessError
from app.models.base import EngineHealth, EngineTranscription, TranscriptionOptions
from app.models.warmup import prefetch_model_paths
from app.models.whisper_server import (
    WhisperServerUnavailable,
    WhisperServerWorker,
    elapsed_ms,
    resolve_server_binary,
)

TRANSCRIPTION_TIMEOUT_SECONDS = 75
MAXIMUM_ERROR_MESSAGE_LENGTH = 200
# whisper-cli defaults to a beam of 5 with 5 greedy candidates, which is tuned for
# batch transcription of long recordings. Dictation clips are short and the decoder
# dominates on a CPU-only host, so the gateway narrows the search: measured on a
# 17 s clip with ggml-tiny.en and the GPU disabled, `-t <cores> -bs 2 -bo 2` cut the
# run from 1.83 s to 0.77 s with no change to the transcript. Temperature fallback
# stays on — it is what rescues a degenerate segment from a repetition loop.
DECODER_BEAM_SIZE = 2
DECODER_BEST_OF = 2


class WhisperCppEngine:
    """whisper.cpp adapter backed by a resident worker, with a one-shot CLI fallback.

    When the build ships `whisper-server` next to `whisper-cli`, the model and
    the accelerator context stay loaded in a private loopback process between
    requests instead of being rebuilt per clip. Hosts without that binary keep
    the original behavior: one `whisper-cli` run per transcription.
    """

    def __init__(
        self,
        binary: Path,
        model: Path,
        catalog_model: CatalogModel | None = None,
        *,
        cpu_threads: int = 0,
        server_binary: Path | None = None,
    ) -> None:
        self.binary = binary
        self.model = model
        self.catalog_model = catalog_model
        self.cpu_threads = system.inference_thread_count(cpu_threads)
        self._worker = self._build_worker(resolve_server_binary(binary, server_binary))

    async def health(self) -> EngineHealth:
        ready = self.binary.is_file() and self.model.is_file()
        model_name = self.model.name
        return EngineHealth(ready=ready, name=f"whisper.cpp:{model_name}")

    async def warmup(self) -> int:
        if not (await self.health()).ready:
            return 0
        advised = await asyncio.to_thread(prefetch_model_paths, [self.model])
        await self._start_worker()
        # A resident worker holds the whole model, which is what the readiness
        # report means by warmed bytes. The advised page count only stands in
        # for it when the transcription path is still the one-shot CLI.
        return self.model.stat().st_size if self.model_is_resident else advised

    @property
    def model_is_resident(self) -> bool:
        return self._worker is not None and self._worker.is_running

    def unload(self) -> None:
        if self._worker is not None:
            self._worker.stop()

    async def transcribe(
        self, audio_path: Path, options: TranscriptionOptions
    ) -> EngineTranscription:
        await self._require_ready()
        language = self._decoder_language(options.language)
        outcome = await self._resident_transcription(audio_path, language)
        if outcome is None:
            outcome = await _CliRun(self, audio_path, language).transcribe()
        self._require_fixed_output_script(outcome.text)
        return outcome

    def _build_worker(self, server_binary: Path | None) -> WhisperServerWorker | None:
        if server_binary is None:
            return None
        return WhisperServerWorker(
            server_binary,
            self.model,
            cpu_threads=self.cpu_threads,
            beam_size=DECODER_BEAM_SIZE,
            best_of=DECODER_BEST_OF,
        )

    async def _start_worker(self) -> bool:
        """Bring the resident worker up, reporting whether this call loaded it."""
        worker = self._worker
        if worker is None:
            return False
        try:
            return await worker.ensure_started()
        except WhisperServerUnavailable:
            self._retire_worker()
            return False

    def _retire_worker(self) -> None:
        """A build that cannot serve will not serve later either.

        Dropping the worker keeps every following request from paying the
        start timeout again before falling back to the CLI. Selecting another
        model or engine rebuilds the adapter and tries once more.
        """
        self.unload()
        self._worker = None

    async def _resident_transcription(
        self, audio_path: Path, language: str
    ) -> EngineTranscription | None:
        load_started = time.monotonic()
        loaded_now = await self._start_worker()
        worker = self._worker
        if worker is None:
            return None
        load_ms = elapsed_ms(load_started) if loaded_now else 0
        inference_started = time.monotonic()
        try:
            transcript = await worker.transcribe(audio_path, language)
        except WhisperServerUnavailable:
            worker.stop()
            return None
        return EngineTranscription(
            text=_collapse_whitespace(transcript),
            model_load_ms=load_ms,
            inference_ms=elapsed_ms(inference_started),
        )

    def _decoder_language(self, requested: str) -> str:
        model = self.catalog_model
        if model is None or model.decoder_language_code is None:
            return requested
        if requested != "auto" and requested not in model.language_codes:
            supported = ", ".join(model.language_codes)
            raise LanguageUnsupportedError(
                f"The selected model supports only {supported}; choose that output mode or Auto."
            )
        return model.decoder_language_code

    def _require_fixed_output_script(self, transcript: str) -> None:
        model = self.catalog_model
        if model is None or model.decoder_language_code is None or len(model.language_codes) != 1:
            return
        output_language = model.language_codes[0]
        if not scripts.transcript_matches_language(transcript, output_language):
            raise LanguageUnsupportedError(
                f"The model did not produce the required {output_language} writing system."
            )

    async def _require_ready(self) -> None:
        health = await self.health()
        if not health.ready:
            raise EngineUnavailableError("The whisper.cpp binary or selected model is unavailable.")


class _CliRun:
    """One `whisper-cli` process: the fallback when no resident worker is available."""

    def __init__(self, engine: WhisperCppEngine, audio_path: Path, language: str) -> None:
        self.engine = engine
        self.audio_path = audio_path
        self.language = language

    async def transcribe(self) -> EngineTranscription:
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="vocagateway-transcript-") as temporary:
            output_stem = Path(temporary) / "result"
            await _execute_whisper_cpp(self._arguments(output_stem))
            transcript = _read_output_text(output_stem.with_suffix(".txt"))
        return EngineTranscription(text=transcript, inference_ms=elapsed_ms(started))

    def _arguments(self, output_stem: Path) -> list[str]:
        engine = self.engine
        return _build_arguments(
            engine.binary,
            engine.model,
            self.audio_path,
            output_stem,
            self.language,
            engine.cpu_threads,
        )


def _collapse_whitespace(transcript: str) -> str:
    """One dictation line out of whisper.cpp's per-segment output.

    Both paths print a segment per line — the server also indents each with a
    leading space — and a client pastes the transcript into a single field.
    """
    return " ".join(transcript.split())


def _build_arguments(
    binary: Path, model: Path, audio: Path, output: Path, language: str, threads: int
) -> list[str]:
    arguments = [str(binary), "-m", str(model), "-f", str(audio)]
    arguments.extend(["-otxt", "-of", str(output), "-np", "-nt"])
    arguments.extend(["-t", str(threads)])
    arguments.extend(["-bs", str(DECODER_BEAM_SIZE), "-bo", str(DECODER_BEST_OF)])
    if language != "auto":
        arguments.extend(["-l", language])
    return arguments


async def _execute_whisper_cpp(arguments: list[str]) -> None:
    process = await asyncio.create_subprocess_exec(
        *arguments,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        _, stderr = await asyncio.wait_for(
            process.communicate(), timeout=TRANSCRIPTION_TIMEOUT_SECONDS
        )
    except TimeoutError as error:
        process.kill()
        await process.wait()
        raise TranscriptionProcessError("Transcription timed out.") from error
    if process.returncode != 0:
        message = (stderr or b"").decode("utf-8", errors="replace").strip()
        detail = message[-MAXIMUM_ERROR_MESSAGE_LENGTH:]
        raise TranscriptionProcessError(f"whisper.cpp exited unsuccessfully: {detail}")


def _read_output_text(output_path: Path) -> str:
    if not output_path.is_file():
        raise TranscriptionProcessError("whisper.cpp did not produce a transcript.")
    transcript = _collapse_whitespace(output_path.read_text(encoding="utf-8"))
    if not transcript:
        raise TranscriptionProcessError("The transcription result was empty.")
    return transcript
