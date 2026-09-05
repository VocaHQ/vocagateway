from __future__ import annotations

import json
import wave
from array import array
from functools import partial
from pathlib import Path

from app.errors import TranscriptionProcessError

CHUNK_SECONDS = 20
PAUSE_SEARCH_SECONDS = 2
PAUSE_WINDOW_HZ = 50


def prepare_recordings(audio: Path, temporary: Path) -> list[Path]:
    """Bound native decoder output; prefer quiet boundaries and retain every sample."""
    with wave.open(str(audio), "rb") as recording:
        properties = recording.getparams()
        maximum = properties.framerate * CHUNK_SECONDS
        if properties.nframes <= maximum:
            return [audio]
        pcm = recording.readframes(properties.nframes)
    frame_bytes = properties.nchannels * properties.sampwidth
    paths: list[Path] = []
    start = 0
    while start < properties.nframes:
        end = min(start + maximum, properties.nframes)
        if end < properties.nframes and frame_bytes == 2:
            end = _quiet_boundary(pcm, end, properties.framerate)
        path = temporary / f"chunk-{len(paths)}.wav"
        with wave.open(str(path), "wb") as chunk:
            chunk.setparams(properties)
            chunk.writeframes(pcm[start * frame_bytes : end * frame_bytes])
        paths.append(path)
        start = end
    return paths


def _quiet_boundary(pcm: bytes, end: int, rate: int) -> int:
    window = max(1, rate // PAUSE_WINDOW_HZ)
    candidates = range(end - rate * PAUSE_SEARCH_SECONDS, end, window)

    return min(candidates, key=partial(_window_energy, pcm, window)) + window // 2


def _window_energy(pcm: bytes, window: int, frame: int) -> int:
    samples = array("h", pcm[frame * 2 : (frame + window) * 2])
    return sum(sample * sample for sample in samples)


def read_batch_output(output: Path, recordings: list[Path]) -> str:
    """Do not accept partial/truncated native batch results as successful dictation."""
    try:
        records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
        if records and isinstance(records[0], dict) and records[0].get("type") == "batch_header":
            records = records[1:]
        if len(records) != len(recordings):
            raise ValueError("Missing batch results")
        texts = []
        for record, path in zip(records, recordings, strict=True):
            if (
                not isinstance(record, dict)
                or record.get("error")
                or record.get("file") != str(path)
                or not isinstance(record.get("text"), str)
            ):
                raise ValueError("Incomplete batch result")
            texts.append(record["text"].strip())
        return " ".join(text for text in texts if text)
    except (OSError, ValueError) as error:
        raise TranscriptionProcessError(
            "transcribe.cpp could not fully transcribe this recording. Try a shorter recording."
        ) from error
