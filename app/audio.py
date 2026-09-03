from __future__ import annotations

import asyncio
import math
import os
import wave
from array import array
from pathlib import Path
from types import MappingProxyType
from typing import Any

from starlette.status import (
    HTTP_413_CONTENT_TOO_LARGE,
    HTTP_415_UNSUPPORTED_MEDIA_TYPE,
    HTTP_422_UNPROCESSABLE_CONTENT,
)

from app.errors import APIProblem, InvalidAudioError, SilentAudioError

MONO_CHANNEL_COUNT = 1
PCM_SAMPLE_WIDTH_BYTES = 2
NORMALIZED_SAMPLE_RATE_HZ = 16_000
MINIMUM_RECORDING_SECONDS = 0.15
DURATION_LIMIT_TOLERANCE_SECONDS = 0.25
SILENCE_RMS_THRESHOLD = 20
MAXIMUM_FFMPEG_ERROR_LENGTH = 160
MINIMUM_AUDIO_UPLOAD_BYTES = 128

ALLOWED_AUDIO_TYPES: MappingProxyType[str, str] = MappingProxyType(
    {
        "audio/mp4": ".m4a",
        "audio/m4a": ".m4a",
        "audio/x-m4a": ".m4a",
        "audio/x-caf": ".caf",
        "audio/wav": ".wav",
        "audio/wave": ".wav",
        "audio/webm": ".webm",
        "audio/ogg": ".ogg",
    }
)


class FFmpegNormalizer:
    def __init__(self, ffmpeg_binary: str = "ffmpeg") -> None:
        self.ffmpeg_binary = ffmpeg_binary

    async def normalize(self, source: Path, destination: Path, maximum_seconds: int) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        process = await asyncio.create_subprocess_exec(
            self.ffmpeg_binary,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-t",
            str(maximum_seconds + 1),
            "-ac",
            str(MONO_CHANNEL_COUNT),
            "-ar",
            str(NORMALIZED_SAMPLE_RATE_HZ),
            "-c:a",
            "pcm_s16le",
            "-y",
            str(destination),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await process.communicate()
        if process.returncode != 0:
            destination.unlink(missing_ok=True)
            raise InvalidAudioError(_safe_ffmpeg_message(stderr))
        try:
            # Reading the whole PCM file and summing its squares is tens of
            # milliseconds of pure CPU for a minute of audio. On the event loop
            # that stalls every other request in flight, so it runs on a worker.
            await asyncio.to_thread(self._validate_wave, destination, maximum_seconds)
        except Exception:
            destination.unlink(missing_ok=True)
            raise
        return destination

    def _validate_wave(self, path: Path, maximum_seconds: int) -> None:
        try:
            with wave.open(str(path), "rb") as recording:
                self._check_wave_properties(recording, maximum_seconds)
                frames = recording.getnframes()
                samples = array("h", recording.readframes(frames))
        except (wave.Error, EOFError) as error:
            raise InvalidAudioError("Recording could not be decoded.") from error
        self._check_silence(samples)

    def _check_wave_properties(self, recording: wave.Wave_read, maximum_seconds: int) -> None:
        rate = recording.getframerate()
        if (
            recording.getsampwidth() != PCM_SAMPLE_WIDTH_BYTES
            or recording.getnchannels() != MONO_CHANNEL_COUNT
            or rate != NORMALIZED_SAMPLE_RATE_HZ
        ):
            raise InvalidAudioError("Normalized audio has an unexpected format.")
        duration = recording.getnframes() / rate if rate else 0
        if duration <= MINIMUM_RECORDING_SECONDS:
            raise InvalidAudioError("Recording is empty or too short.")
        if duration > maximum_seconds + DURATION_LIMIT_TOLERANCE_SECONDS:
            raise InvalidAudioError("Recording exceeds the duration limit.")

    def _check_silence(self, samples: array[int]) -> None:
        if not samples:
            raise InvalidAudioError("Recording contains no audio frames.")
        if _root_mean_square(samples) < SILENCE_RMS_THRESHOLD:
            raise SilentAudioError("Recording appears to be silent.")


def _root_mean_square(samples: array[int]) -> float:
    """RMS amplitude, vectorized when numpy is present.

    numpy arrives with every engine extra but the core install does without it,
    so the pure-Python sum stays as the fallback rather than becoming a new
    hard dependency of the gateway.
    """
    try:
        import numpy
    except ImportError:
        energy = sum(sample * sample for sample in samples)
        return math.sqrt(energy / len(samples))
    block = numpy.frombuffer(samples, dtype=numpy.int16).astype(numpy.float64)
    return float(numpy.sqrt(numpy.square(block).mean()))


def validate_audio_upload_headers(
    content_type: str | None, content_length: int | None, max_bytes: int
) -> str:
    normalized = (content_type or "").split(";", maxsplit=1)[0].lower()
    suffix = ALLOWED_AUDIO_TYPES.get(normalized)
    if suffix is None:
        raise APIProblem(
            HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            "unsupported_audio_type",
            "This audio type is not supported.",
        )
    if content_length is not None and content_length > max_bytes:
        raise APIProblem(
            HTTP_413_CONTENT_TOO_LARGE,
            "audio_too_large",
            "The recording exceeds the upload limit.",
        )
    return suffix


async def save_streamed_upload(
    stream: Any,
    directory: Path,
    session_id: str,
    suffix: str,
    maximum_bytes: int,
) -> Path:
    temporary, final = atomic_upload_path(directory, session_id, suffix)
    try:
        await _write_stream_to_path(stream, temporary, maximum_bytes)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    complete_atomic_upload(temporary, final)
    return final


async def _write_stream_to_path(stream: Any, path: Path, maximum_bytes: int) -> None:
    received = 0
    with path.open("wb") as output:
        async for chunk in stream:
            received += len(chunk)
            if received > maximum_bytes:
                raise APIProblem(
                    HTTP_413_CONTENT_TOO_LARGE,
                    "audio_too_large",
                    "The recording exceeds the upload limit.",
                )
            output.write(chunk)
    if received < MINIMUM_AUDIO_UPLOAD_BYTES:
        raise APIProblem(HTTP_422_UNPROCESSABLE_CONTENT, "audio_empty", "The recording is empty.")


def atomic_upload_path(directory: Path, session_id: str, suffix: str) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    final = directory / f"{session_id}{suffix}"
    temporary = directory / f".{session_id}.upload"
    return temporary, final


def complete_atomic_upload(temporary: Path, final: Path) -> None:
    os.replace(temporary, final)


def _safe_ffmpeg_message(stderr: bytes | None) -> str:
    if not stderr:
        return "FFmpeg could not decode the recording."
    lines = stderr.decode("utf-8", errors="replace").strip().splitlines()
    detail = lines[-1][:MAXIMUM_FFMPEG_ERROR_LENGTH]
    return f"FFmpeg rejected the recording: {detail}"
