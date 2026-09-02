from __future__ import annotations

import math
import wave
from array import array
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest

from app.config import Settings
from app.main import create_app
from app.models.base import EngineHealth, TranscriptionOptions

TEST_TOKEN_PADDING_LENGTH = 48
TEST_MAXIMUM_UPLOAD_BYTES = 20_000
TEST_AUDIO_AMPLITUDE = 3_000
TEST_TONE_FREQUENCY_HZ = 440
TEST_SAMPLE_RATE_HZ = 16_000
TEST_SAMPLE_COUNT = 8_000

TOKEN = "test-" + ("x" * TEST_TOKEN_PADDING_LENGTH)


class FakeEngine:
    def __init__(self, transcript: str = "hello from the local model") -> None:
        self.transcript = transcript
        self.calls = 0
        self.health_calls = 0
        self.last_options: TranscriptionOptions | None = None

    async def health(self) -> EngineHealth:
        self.health_calls += 1
        return EngineHealth(ready=True, name="fake-local-model")

    async def transcribe(self, audio_path: Path, options: TranscriptionOptions) -> str:
        assert audio_path.is_file()
        self.calls += 1
        self.last_options = options
        return self.transcript


class FakeNormalizer:
    async def normalize(self, source: Path, destination: Path, maximum_seconds: int) -> Path:
        assert source.is_file()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"normalized")
        return destination


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        token=TOKEN,
        data_dir=tmp_path,
        whisper_binary=tmp_path / "whisper-cli",
        whisper_model=tmp_path / "model.bin",
        maximum_upload_bytes=TEST_MAXIMUM_UPLOAD_BYTES,
    )


@pytest.fixture
def fake_engine() -> FakeEngine:
    return FakeEngine()


@pytest.fixture
async def client(settings: Settings, fake_engine: FakeEngine) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(settings, engine=fake_engine, normalizer=FakeNormalizer())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as test_client:
        yield test_client


@pytest.fixture
def authorization() -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def audio_bytes(tmp_path: Path) -> bytes:
    path = tmp_path / "speech.wav"
    samples = array(
        "h",
        (
            int(
                TEST_AUDIO_AMPLITUDE
                * math.sin(2 * math.pi * TEST_TONE_FREQUENCY_HZ * index / TEST_SAMPLE_RATE_HZ)
            )
            for index in range(TEST_SAMPLE_COUNT)
        ),
    )
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(TEST_SAMPLE_RATE_HZ)
        output.writeframes(samples.tobytes())
    return path.read_bytes()
