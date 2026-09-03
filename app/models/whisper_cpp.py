from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from app.errors import EngineUnavailableError, TranscriptionProcessError
from app.models.base import EngineHealth, TranscriptionOptions
from app.models.warmup import prefetch_model_paths

TRANSCRIPTION_TIMEOUT_SECONDS = 75
MAXIMUM_ERROR_MESSAGE_LENGTH = 200


class WhisperCppEngine:
    def __init__(self, binary: Path, model: Path) -> None:
        self.binary = binary
        self.model = model

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
                self.binary, self.model, audio_path, output_stem, options.language
            )
            await _execute_whisper_cpp(arguments)
            return _read_output_text(output_stem.with_suffix(".txt"))

    async def _require_ready(self) -> None:
        health = await self.health()
        if not health.ready:
            raise EngineUnavailableError("The whisper.cpp binary or selected model is unavailable.")


def _build_arguments(
    binary: Path, model: Path, audio: Path, output: Path, language: str
) -> list[str]:
    arguments = [str(binary), "-m", str(model), "-f", str(audio)]
    arguments.extend(["-otxt", "-of", str(output), "-np", "-nt"])
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
