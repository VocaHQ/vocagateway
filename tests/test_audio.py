from __future__ import annotations

import builtins
import math
import random
import wave
from array import array
from pathlib import Path

import pytest
from pytest import MonkeyPatch

from app.audio import FFmpegNormalizer, _root_mean_square
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


def _pure_python_rms(samples: array[int]) -> float:
    return math.sqrt(sum(sample * sample for sample in samples) / len(samples))


@pytest.mark.parametrize("sample_count", [1, 5, 1000, 48_000])
def test_both_rms_paths_agree_exactly(sample_count: int, monkeypatch: MonkeyPatch) -> None:
    """The vectorized path must not shift the silence threshold.

    numpy comes in with the engine extras and is missing from a core install,
    so the two branches decide the same recordings are silent or they disagree
    about which uploads the gateway rejects.
    """
    random.seed(sample_count)
    samples = array("h", [random.randint(-32768, 32767) for _ in range(sample_count)])
    expected = _pure_python_rms(samples)

    assert _root_mean_square(samples) == pytest.approx(expected)

    real_import = builtins.__import__

    def without_numpy(name: str, *arguments: object, **keywords: object) -> object:
        if name == "numpy":
            raise ImportError(name)
        return real_import(name, *arguments, **keywords)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", without_numpy)
    assert _root_mean_square(samples) == pytest.approx(expected)
