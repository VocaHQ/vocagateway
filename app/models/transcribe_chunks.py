from __future__ import annotations

import json
import wave
from array import array
from functools import partial
from pathlib import Path
from typing import Any

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
    """Reassemble the batch transcript, refusing partial or truncated results.

    Results are matched to chunks by file name rather than by position or by an
    identical path string: a CLI is free to canonicalise the paths it was given
    (on macOS the temporary directory is `/var/...`, a symlink to `/private/var/...`)
    or to finish them out of order, and neither means the transcript is wrong.
    A chunk that is missing, errored, or reported twice still fails the recording.
    """
    try:
        records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
        results = _batch_results(records)
        texts = []
        for path in recordings:
            record = results.get(path.name)
            if record is None or record.get("error") or not isinstance(record.get("text"), str):
                raise ValueError(f"Incomplete batch result for {path.name}")
            texts.append(record["text"].strip())
        return " ".join(text for text in texts if text)
    except (OSError, ValueError) as error:
        raise TranscriptionProcessError(
            "transcribe.cpp could not fully transcribe this recording."
        ) from error


def _batch_results(records: list[Any]) -> dict[str, dict[str, Any]]:
    """Index the per-chunk records by file name, ignoring the batch envelope."""
    results: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict) or "file" not in record:
            # A leading `batch_header` and any trailing summary carry no chunk.
            continue
        name = Path(str(record["file"])).name
        if name in results:
            raise ValueError(f"Duplicate batch result for {name}")
        results[name] = record
    return results
