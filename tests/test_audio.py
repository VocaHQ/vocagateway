from __future__ import annotations

import wave
from pathlib import Path

import pytest

from app.audio import FFmpegNormalizer
from app.errors import InvalidAudioError, SilentAudioError

MAXIMUM_RECORDING_DURATION_SECONDS = 120
NORMALIZED_SAMPLE_RATE_HZ = 16_000
SILENT_AUDIO_FRAME_COUNT = 8_000


async def test_ffmpeg_normalizes_real_audio(
    tmp_path: Path,
    audio_bytes: bytes,
) -> None:
    source = tmp_path / "source.wav"
    destination = tmp_path / "normalized.wav"
    source.write_bytes(audio_bytes)
    operation_result = await FFmpegNormalizer().normalize(
        source, destination, MAXIMUM_RECORDING_DURATION_SECONDS
    )
    with wave.open(str(operation_result), "rb") as normalized:
        assert normalized.getframerate() == NORMALIZED_SAMPLE_RATE_HZ
        assert normalized.getnchannels() == 1


async def test_ffmpeg_rejects_invalid_audio(tmp_path: Path) -> None:
    source = tmp_path / "invalid.m4a"
    source.write_bytes(b"not audio" * 100)
    with pytest.raises(InvalidAudioError):
        await FFmpegNormalizer().normalize(
            source, tmp_path / "output.wav", MAXIMUM_RECORDING_DURATION_SECONDS
        )


async def test_silence_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "silent.wav"
    with wave.open(str(source), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(NORMALIZED_SAMPLE_RATE_HZ)
        output.writeframes(b"\0\0" * SILENT_AUDIO_FRAME_COUNT)
    with pytest.raises(SilentAudioError):
        await FFmpegNormalizer().normalize(
            source, tmp_path / "output.wav", MAXIMUM_RECORDING_DURATION_SECONDS
        )
