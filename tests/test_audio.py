from __future__ import annotations

import asyncio
import builtins
import math
import os
import random
import signal
import time
import wave
from array import array
from contextlib import suppress
from pathlib import Path

import pytest

from app.audio import FFmpegNormalizer, _root_mean_square
from app.errors import InvalidAudioError, SilentAudioError

MAXIMUM_RECORDING_DURATION_SECONDS = 120
NORMALIZED_SAMPLE_RATE_HZ = 16_000
SILENT_AUDIO_FRAME_COUNT = 8_000
ABANDON_AFTER_SECONDS = 0.05
REAP_BOUND_SECONDS = 0.2
ABANDON_BOUND_SECONDS = 3
PID_POLL_ATTEMPTS = 100


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


async def test_an_abandoned_ffmpeg_never_holds_its_caller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A wrapper whose child keeps stderr open after the wrapper is killed, the
    # case in which waiting for the exit alone never returns.
    wrapper = tmp_path / "ffmpeg-wrapper"
    holder = tmp_path / "holder.pid"
    wrapper.write_text(
        f'#!/bin/sh\nsleep 30 &\necho $! > "{holder}"\nexec sleep 30\n',
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    monkeypatch.setattr("app.audio.FFMPEG_REAP_SECONDS", REAP_BOUND_SECONDS)
    source = tmp_path / "input.wav"
    source.write_bytes(b"not audio")
    destination = tmp_path / "normalized.wav"
    holder_pid: int | None = None
    normalize = asyncio.create_task(
        FFmpegNormalizer(str(wrapper)).normalize(
            source, destination, MAXIMUM_RECORDING_DURATION_SECONDS
        )
    )
    try:
        holder_pid = await asyncio.to_thread(_read_pid, holder)

        started = time.monotonic()
        normalize.cancel()
        with pytest.raises(asyncio.CancelledError):
            await normalize
        elapsed = time.monotonic() - started
        assert elapsed < ABANDON_BOUND_SECONDS
        assert not destination.exists()
        assert holder_pid is not None
        # Production `_reap` must have killed the process group, including the
        # grandchild that held stderr open. This runs before the safety-net
        # SIGKILL in `finally`.
        await asyncio.to_thread(_wait_until_dead, holder_pid)
    except BaseException:
        raise
    finally:
        if not normalize.done():
            normalize.cancel()
            with suppress(asyncio.CancelledError):
                await normalize
        if holder_pid is not None:
            with suppress(ProcessLookupError):
                os.kill(holder_pid, signal.SIGKILL)


def _read_pid(pid_file: Path) -> int:
    for _ in range(PID_POLL_ATTEMPTS):
        if pid_file.exists() and pid_file.read_text(encoding="utf-8").strip():
            return int(pid_file.read_text(encoding="utf-8"))
        time.sleep(ABANDON_AFTER_SECONDS)
    raise AssertionError("the wrapper never started its child")


def _wait_until_dead(pid: int) -> None:
    for _ in range(PID_POLL_ATTEMPTS):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(ABANDON_AFTER_SECONDS)
    raise AssertionError("production reap left the FFmpeg grandchild running")


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
def test_both_rms_paths_agree_exactly(sample_count: int, monkeypatch: pytest.MonkeyPatch) -> None:
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
