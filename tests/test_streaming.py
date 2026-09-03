from __future__ import annotations

import asyncio
from array import array
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from app.config import Settings
from app.main import create_app
from app.models.base import EngineHealth, TranscriptionOptions
from app.models.moonshine import MoonshineEngine

TOKEN_PADDING_LENGTH = 48
NORMALIZED_SAMPLE_RATE_HZ = 16_000
WHISPER_BINARY_NAME = "whisper-cli"
MODEL_FILE_NAME = "model.bin"
STREAM_PATH = "/v1/stream"
AUTHORIZATION_HEADER = "Authorization"
MESSAGE_TYPE_KEY = "type"
ENGINE_KEY = "engine"
TRANSCRIPT_KEY = "transcript"

TOKEN = "stream-" + ("x" * TOKEN_PADDING_LENGTH)


def moonshine_engine(tmp_path: Path, model_arch: int = 5) -> MoonshineEngine:
    model_root = tmp_path / f"moonshine-{model_arch}"
    model_root.mkdir()
    (model_root / ".vocagateway-model.json").write_text(
        '{"language":"en","model_path":"model","model_arch":' + str(model_arch) + "}",
        encoding="utf-8",
    )
    return MoonshineEngine(model_root)


class BatchOnlyEngine:
    async def health(self) -> EngineHealth:
        return EngineHealth(ready=True, name="whisperkit:test-model")

    async def transcribe(self, audio_path: Path, options: TranscriptionOptions) -> str:
        return "batch result"


class FakeStream:
    def __init__(self) -> None:
        self.listener: object | None = None
        self.closed = False

    def add_listener(self, listener: object) -> None:
        self.listener = listener

    def add_audio(self, samples: list[float], sample_rate: int) -> None:
        assert samples
        assert sample_rate == NORMALIZED_SAMPLE_RATE_HZ
        line = SimpleNamespace(text="hello", line_id=1)
        assert callable(self.listener)
        self.listener(SimpleNamespace(line=line))

    def stop(self) -> object:
        return SimpleNamespace(lines=[SimpleNamespace(text="hello world", line_id=1)])

    def close(self) -> None:
        self.closed = True


def test_authenticated_moonshine_stream_ret_df9ab(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    settings = Settings(
        token=TOKEN,
        data_dir=tmp_path,
        whisper_binary=tmp_path / WHISPER_BINARY_NAME,
        whisper_model=tmp_path / MODEL_FILE_NAME,
    )
    engine = moonshine_engine(tmp_path)

    async def create_stream() -> FakeStream:
        return FakeStream()

    async def health() -> EngineHealth:
        return EngineHealth(ready=True, name="moonshine:en")

    monkeypatch.setattr(engine, "create_stream", create_stream)
    monkeypatch.setattr(engine, "health", health)
    app = create_app(settings, engine=engine)

    with (
        TestClient(app) as client,
        client.websocket_connect(
            STREAM_PATH, headers={AUTHORIZATION_HEADER: f"Bearer {TOKEN}"}
        ) as websocket,
    ):
        websocket.send_json({MESSAGE_TYPE_KEY: "start", "sample_rate": 16_000, "style": "formal"})
        assert websocket.receive_json() == {MESSAGE_TYPE_KEY: "ready", ENGINE_KEY: "moonshine"}
        websocket.send_bytes(array("f", [0.1, -0.1]).tobytes())
        assert websocket.receive_json() == {MESSAGE_TYPE_KEY: "partial", TRANSCRIPT_KEY: "hello"}
        websocket.send_json({MESSAGE_TYPE_KEY: "finish"})
        assert websocket.receive_json() == {
            MESSAGE_TYPE_KEY: "complete",
            TRANSCRIPT_KEY: "Hello world.",
        }


def test_health_advertises_ready_moonshine_dbde8(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    settings = Settings(
        token=TOKEN,
        data_dir=tmp_path,
        whisper_binary=tmp_path / WHISPER_BINARY_NAME,
        whisper_model=tmp_path / MODEL_FILE_NAME,
    )
    engine = moonshine_engine(tmp_path)

    async def health() -> EngineHealth:
        return EngineHealth(ready=True, name="moonshine:en")

    monkeypatch.setattr(engine, "health", health)
    app = create_app(settings, engine=engine)

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.json()["streaming_supported"] is True


def test_batch_moonshine_gets_structured_st_f78fb(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    settings = Settings(
        token=TOKEN,
        data_dir=tmp_path,
        whisper_binary=tmp_path / WHISPER_BINARY_NAME,
        whisper_model=tmp_path / MODEL_FILE_NAME,
    )
    engine = moonshine_engine(tmp_path, model_arch=1)

    async def health() -> EngineHealth:
        return EngineHealth(ready=True, name="moonshine:es")

    monkeypatch.setattr(engine, "health", health)
    app = create_app(settings, engine=engine)

    with (
        TestClient(app) as client,
        client.websocket_connect(
            STREAM_PATH, headers={AUTHORIZATION_HEADER: f"Bearer {TOKEN}"}
        ) as websocket,
    ):
        assert websocket.receive_json() == {
            MESSAGE_TYPE_KEY: "unsupported",
            "reason": "active_engine",
            ENGINE_KEY: "moonshine:es",
        }

    with TestClient(app) as client:
        assert client.get("/health").json()["streaming_supported"] is False


class FakeStreamingEngine:
    """A non-Moonshine engine implementing the StreamingEngine protocol.

    Proves the /v1/stream handler generalized past isinstance(MoonshineEngine)
    to any engine exposing supports_streaming/streaming_lock/create_stream.
    """

    def __init__(self, stream: FakeStream) -> None:
        self._stream = stream
        self.supports_streaming = True
        self.streaming_lock = asyncio.Lock()

    async def health(self) -> EngineHealth:
        return EngineHealth(ready=True, name="sherpa-onnx:streaming-zipformer-en-20m-int8")

    async def create_stream(self) -> FakeStream:
        return self._stream


def test_authenticated_sherpa_onnx_style_st_b5a1f(tmp_path: Path) -> None:
    settings = Settings(
        token=TOKEN,
        data_dir=tmp_path,
        whisper_binary=tmp_path / WHISPER_BINARY_NAME,
        whisper_model=tmp_path / MODEL_FILE_NAME,
    )
    engine = FakeStreamingEngine(FakeStream())
    app = create_app(settings, engine=engine)

    with (
        TestClient(app) as client,
        client.websocket_connect(
            STREAM_PATH, headers={AUTHORIZATION_HEADER: f"Bearer {TOKEN}"}
        ) as websocket,
    ):
        websocket.send_json({MESSAGE_TYPE_KEY: "start", "sample_rate": 16_000, "style": "formal"})
        assert websocket.receive_json() == {MESSAGE_TYPE_KEY: "ready", ENGINE_KEY: "sherpa-onnx"}
        websocket.send_bytes(array("f", [0.1, -0.1]).tobytes())
        assert websocket.receive_json() == {MESSAGE_TYPE_KEY: "partial", TRANSCRIPT_KEY: "hello"}
        websocket.send_json({MESSAGE_TYPE_KEY: "finish"})
        assert websocket.receive_json() == {
            MESSAGE_TYPE_KEY: "complete",
            TRANSCRIPT_KEY: "Hello world.",
        }

    assert engine._stream.closed is True


def test_authenticated_batch_engine_gets_st_aa(tmp_path: Path) -> None:
    settings = Settings(
        token=TOKEN,
        data_dir=tmp_path,
        whisper_binary=tmp_path / WHISPER_BINARY_NAME,
        whisper_model=tmp_path / MODEL_FILE_NAME,
    )
    app = create_app(settings, engine=BatchOnlyEngine())

    with (
        TestClient(app) as client,
        client.websocket_connect(
            STREAM_PATH, headers={AUTHORIZATION_HEADER: f"Bearer {TOKEN}"}
        ) as websocket,
    ):
        assert websocket.receive_json() == {
            MESSAGE_TYPE_KEY: "unsupported",
            "reason": "active_engine",
            ENGINE_KEY: "whisperkit:test-model",
        }
