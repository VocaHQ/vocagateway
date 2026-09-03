from __future__ import annotations

import sys
import types
from importlib import machinery
from pathlib import Path

from pytest import MonkeyPatch

from app.models.base import EngineTranscription, TranscriptionOptions
from app.models.faster_whisper import FasterWhisperEngine, _decode, _extract_text


class _Segment:
    text = " persistent result "


class _FakeWhisperModel:
    options_records: list[dict[str, object]] = []

    def __init__(self, _: str, **options: object) -> None:
        self.options_records.append(options)

    def transcribe(self, _: str, **options: object) -> tuple[list[_Segment], object]:
        assert options["beam_size"] == 1
        return [_Segment()], object()


async def test_faster_whisper_keeps_one_model_loaded(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    model_path = tmp_path / "tiny.en"
    model_path.mkdir()
    (model_path / "model.bin").write_bytes(b"model")
    (tmp_path / "audio.wav").write_bytes(b"audio")
    _FakeWhisperModel.options_records = []

    module = types.ModuleType("faster_whisper")
    module.WhisperModel = _FakeWhisperModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    monkeypatch.setattr(
        "app.models.faster_whisper.importlib_util.find_spec",
        lambda _: machinery.ModuleSpec("faster_whisper", loader=None),
    )

    engine = FasterWhisperEngine(model_path, device="cpu", compute_type="int8", cpu_threads=2)
    first = await engine.transcribe(tmp_path / "audio.wav", TranscriptionOptions("en", "raw"))

    assert isinstance(first, EngineTranscription)
    assert first.text == "persistent result"
    assert (
        await engine.transcribe(tmp_path / "audio.wav", TranscriptionOptions("en", "raw"))
    ).model_load_ms == 0
    assert _FakeWhisperModel.options_records == [
        {
            "device": "cpu",
            "compute_type": "int8",
            "cpu_threads": 2,
            "num_workers": 1,
            "local_files_only": True,
        }
    ]


class _VadOnlyModel:
    """Stands in for a model whose VAD pass finds nothing to decode."""

    def __init__(self, _: str, **options: object) -> None:
        self.vad_flags: list[object] = []

    def transcribe(self, _: str, **options: object) -> tuple[list[_Segment], object]:
        self.vad_flags.append(options["vad_filter"])
        if options["vad_filter"]:
            return [], object()
        return [_Segment()], object()


def test_the_vad_pass_runs_first_and_skips_the_silence() -> None:
    model = _VadOnlyModel("model")
    _decode(model, Path("audio.wav"), TranscriptionOptions("en", "raw"), use_vad=True)

    assert model.vad_flags == [True]


def test_a_quiet_clip_that_vad_rejects_is_decoded_in_full_instead() -> None:
    """The RMS gate upstream already refused true silence.

    An empty VAD pass therefore means quiet speech Silero was unsure about, and
    a slower full decode beats returning a transcription failure for it.
    """
    model = _VadOnlyModel("model")

    text = _extract_text(model, Path("audio.wav"), TranscriptionOptions("en", "raw"))

    assert text == "persistent result"
    assert model.vad_flags == [True, False]
