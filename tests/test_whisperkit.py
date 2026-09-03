from __future__ import annotations

from pathlib import Path

from pytest import MonkeyPatch

from app.models import whisperkit
from app.models.base import EngineTranscription, TranscriptionOptions

FAKE_SERVER_PORT = 50123


class FakeProcess:
    def __init__(self) -> None:
        self.return_code: int | None = None
        self.terminated = False

    def poll(self) -> int | None:
        return self.return_code

    def terminate(self) -> None:
        self.terminated = True
        self.return_code = 0

    def wait(self, timeout: int) -> int:
        assert timeout == 3
        return self.return_code or 0

    def kill(self) -> None:
        self.return_code = -9


class _FakeServerSpawner:
    def __init__(self, process: FakeProcess) -> None:
        self.process = process
        self.starts = 0

    def start_server(
        self, _: str, _model: Path, _config: Path | None = None
    ) -> whisperkit._WhisperKitServer:
        self.starts += 1
        return whisperkit._WhisperKitServer(process=self.process, port=FAKE_SERVER_PORT)  # type: ignore[arg-type]


def _fake_transcribe(
    _: whisperkit._WhisperKitServer,
    _audio: Path,
    _model: Path,
    _style: str,
) -> str:
    return "persistent native result"


async def test_whisperkit_keeps_one_native_server_aa(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    model_path = tmp_path / "openai_whisper-small"
    model_path.mkdir()
    (model_path / "config.json").write_text("{}")
    (tmp_path / "audio.wav").write_bytes(b"audio")
    process = FakeProcess()
    spawner = _FakeServerSpawner(process)

    engine = whisperkit.WhisperKitEngine("whisperkit-cli", model_path)
    monkeypatch.setattr(engine, "_resolved_binary", lambda: "/fake/whisperkit-cli")
    monkeypatch.setattr(whisperkit, "_start_server", spawner.start_server)
    monkeypatch.setattr(whisperkit, "_server_transcription", _fake_transcribe)

    first = await engine.transcribe(tmp_path / "audio.wav", TranscriptionOptions("en", "raw"))

    assert isinstance(first, EngineTranscription)
    assert first.text == "persistent native result"
    assert (
        await engine.transcribe(tmp_path / "audio.wav", TranscriptionOptions("en", "raw"))
    ).model_load_ms == 0
    assert spawner.starts == 1

    engine.close()
    assert process.terminated is True


def test_whisperkit_multipart_contains_mode_aaa(tmp_path: Path) -> None:
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"wave-bytes")

    body = whisperkit._multipart_body(
        "boundary",
        [("model", "whisper-small"), ("language", "en")],
        audio_path,
    )

    assert b'name="model"\r\n\r\nwhisper-small' in body
    assert b'name="language"\r\n\r\nen' in body
    assert b'name="file"; filename="sample.wav"' in body
    assert b"wave-bytes" in body
    assert body.endswith(b"--boundary--\r\n")
