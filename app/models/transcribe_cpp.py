from __future__ import annotations

import asyncio
import tempfile
import wave
from pathlib import Path

from app import system
from app.catalog import CatalogModel
from app.errors import EngineUnavailableError, LanguageUnsupportedError, TranscriptionProcessError
from app.models.base import EngineHealth, TranscriptionOptions
from app.models.transcribe_chunks import prepare_recordings, read_batch_output
from app.models.warmup import prefetch_model_paths

TRANSCRIPTION_TIMEOUT_SECONDS = 180
MAXIMUM_ERROR_MESSAGE_LENGTH = 240


class TranscribeCppEngine:
    """Optional native GGUF adapter; each invocation owns and releases its model."""

    def __init__(
        self,
        binary: str,
        model: Path | None,
        catalog_model: CatalogModel | None,
        *,
        cpu_threads: int = 0,
    ) -> None:
        self.binary = binary
        self.model = model
        self.catalog_model = catalog_model
        self.threads = system.inference_thread_count(cpu_threads)

    async def health(self) -> EngineHealth:
        name = self.model.name if self.model else "no-model-selected"
        return EngineHealth(
            ready=system.resolve_binary(self.binary) is not None
            and self.model is not None
            and self.model.is_file(),
            name=f"transcribe.cpp:{name}",
        )

    async def warmup(self) -> int:
        if not (await self.health()).ready or self.model is None:
            return 0
        return await asyncio.to_thread(prefetch_model_paths, [self.model])

    async def transcribe(self, audio_path: Path, options: TranscriptionOptions) -> str:
        # Readiness first: with no model selected there is no catalog entry, so
        # `_language` would reject even "auto" and hide the real problem.
        if not (await self.health()).ready or self.model is None:
            raise EngineUnavailableError(
                "Install transcribe-cli, set VOCAGATEWAY_TRANSCRIBE_BINARY if needed, "
                "and download a transcribe.cpp model."
            )
        language = self._language(options.language)
        with tempfile.TemporaryDirectory(prefix="vocagateway-transcribe-") as temporary:
            root = Path(temporary)
            recordings = await _prepared_recordings(audio_path, root)
            output = root / "transcript.txt"
            arguments = [
                self.binary,
                "-m",
                str(self.model),
                "-q",
                "--threads",
                str(self.threads),
                "-l",
                language,
            ]
            if len(recordings) > 1:
                text = await _transcribe_batch(arguments, root, recordings)
            else:
                await _execute([*arguments, "-o", str(output), str(audio_path)])
                text = _read_transcript(output)
            if not text:
                raise TranscriptionProcessError("transcribe.cpp returned an empty transcript.")
            return text

    def _language(self, requested: str) -> str:
        supported = self.catalog_model.language_codes if self.catalog_model else ()
        normalized = requested.lower().split("-", maxsplit=1)[0]
        if requested == "auto":
            if len(supported) == 1:
                return supported[0]
            raise LanguageUnsupportedError(
                "Choose the spoken language for this transcribe.cpp model."
            )
        if supported and normalized not in supported:
            raise LanguageUnsupportedError(f"The selected model does not support {requested}.")
        return normalized


def _read_transcript(output: Path) -> str:
    if not output.is_file():
        raise TranscriptionProcessError("transcribe.cpp did not produce a transcript.")
    return output.read_text(encoding="utf-8").strip()


async def _transcribe_batch(arguments: list[str], root: Path, recordings: list[Path]) -> str:
    manifest = root / "recordings.list"
    manifest.write_text("".join(f"{path}\n" for path in recordings), encoding="utf-8")
    output = root / "results.jsonl"
    with output.open("wb") as stream:
        await _execute(
            [*arguments, "--batch", str(manifest), "--batch-jsonl"], stdout=stream.fileno()
        )
    return read_batch_output(output, recordings)


async def _prepared_recordings(audio_path: Path, root: Path) -> list[Path]:
    """Split the recording, reporting an unreadable WAV as a transcription failure.

    The normalizer hands over a RIFF file, so `wave.Error` here means it was
    truncated or written in another container - an engine failure, not a crash.
    """
    try:
        return await asyncio.to_thread(prepare_recordings, audio_path, root)
    except (wave.Error, OSError, ValueError) as error:
        raise TranscriptionProcessError(
            f"transcribe.cpp could not read the recording: {error}"
        ) from error


async def _execute(arguments: list[str], *, stdout: int = asyncio.subprocess.DEVNULL) -> None:
    try:
        process = await asyncio.create_subprocess_exec(
            *arguments,
            stdout=stdout,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as error:
        # The binary passed `health()` but is gone, or is not executable.
        raise EngineUnavailableError(f"Could not run {arguments[0]}: {error}") from error
    try:
        _, stderr = await asyncio.wait_for(
            process.communicate(), timeout=TRANSCRIPTION_TIMEOUT_SECONDS
        )
    except (TimeoutError, asyncio.CancelledError) as error:
        if process.returncode is None:
            process.kill()
        await process.wait()
        if isinstance(error, asyncio.CancelledError):
            raise
        raise TranscriptionProcessError("transcribe.cpp transcription timed out.") from error
    if process.returncode != 0:
        # Without the tail, "unsupported GGUF", "out of memory" and "unknown
        # flag on an older build" are the same opaque exit code to an operator.
        detail = (stderr or b"").decode("utf-8", errors="replace").strip()
        suffix = f": {detail[-MAXIMUM_ERROR_MESSAGE_LENGTH:]}" if detail else "."
        raise TranscriptionProcessError(
            f"transcribe.cpp exited with code {process.returncode}{suffix}"
        )
