from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from app import scripts, system
from app.catalog import CatalogModel
from app.errors import EngineUnavailableError, LanguageUnsupportedError, TranscriptionProcessError
from app.models.base import EngineHealth, TranscriptionOptions
from app.models.warmup import prefetch_model_paths

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
    def __init__(
        self,
        binary: Path,
        model: Path,
        catalog_model: CatalogModel | None = None,
        *,
        cpu_threads: int = 0,
    ) -> None:
        self.binary = binary
        self.model = model
        self.catalog_model = catalog_model
        self.cpu_threads = system.inference_thread_count(cpu_threads)

    async def health(self) -> EngineHealth:
        ready = self.binary.is_file() and self.model.is_file()
        model_name = self.model.name
        return EngineHealth(ready=ready, name=f"whisper.cpp:{model_name}")

    async def warmup(self) -> int:
        if not (await self.health()).ready:
            return 0
        return await asyncio.to_thread(prefetch_model_paths, [self.model])

    async def transcribe(self, audio_path: Path, options: TranscriptionOptions) -> str:
        await self._require_ready()
        with tempfile.TemporaryDirectory(prefix="vocagateway-transcript-") as temporary:
            output_stem = Path(temporary) / "result"
            arguments = _build_arguments(
                self.binary,
                self.model,
                audio_path,
                output_stem,
                self._decoder_language(options.language),
                self.cpu_threads,
            )
            await _execute_whisper_cpp(arguments)
            transcript = _read_output_text(output_stem.with_suffix(".txt"))
            self._require_fixed_output_script(transcript)
            return transcript

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
    transcript = output_path.read_text(encoding="utf-8").strip()
    if not transcript:
        raise TranscriptionProcessError("The transcription result was empty.")
    return transcript
