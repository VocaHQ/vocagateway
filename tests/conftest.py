# Copyright (c) 2025-present VocaHQ, Inc.
# SPDX-License-Identifier: AGPL-3.0-only
# See the LICENSE file for details.

from __future__ import annotations

import math
import wave
from array import array
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from app.cleanup.base import (
    MODE_CONSERVATIVE,
    CleanupOptions,
)
from app.cleanup.profile import COMPACT_PROFILE
from app.config import Settings
from app.main import create_app
from app.models.base import EngineHealth, TranscriptionOptions

TEST_TOKEN_PADDING_LENGTH = 48
TEST_MAXIMUM_UPLOAD_BYTES = 20_000
TEST_AUDIO_AMPLITUDE = 3_000
TEST_TONE_FREQUENCY_HZ = 440
TEST_SAMPLE_RATE_HZ = 16_000
TEST_SAMPLE_COUNT = 8_000

TEST_TOKEN_PADDING = "x" * TEST_TOKEN_PADDING_LENGTH
TOKEN = f"test-{TEST_TOKEN_PADDING}"


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
        # Never the operator's real file: a settings save in a test must not be
        # able to reach ~/.config.
        config_path=tmp_path / "config.json",
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


CLEANUP_MODEL_ID = "cleanup:qwen3-0.6b"


class FakeCleanupRuntime:
    """A cleanup runtime that answers from a script instead of from a model.

    `answers` may hold strings (returned as the corrected text) or exceptions
    (raised), which is what lets one fake cover the whole failure vocabulary
    without a real `llama-server` anywhere in CI.
    """

    def __init__(self, *answers: str | BaseException) -> None:
        self.answers: list[str | BaseException] = list(answers)
        self.calls: list[tuple[str, str]] = []
        self.model_id = CLEANUP_MODEL_ID

    async def available(self) -> bool:
        return True

    async def clean(self, transcript: str, language: str, *, budget_seconds: float) -> str:
        self.calls.append((transcript, language))
        answer = self.answers.pop(0) if self.answers else transcript
        if isinstance(answer, BaseException):
            raise answer
        return answer


class FakeWorkerHost:
    """Stands in for the managed `llama-server` process the gateway owns."""

    def __init__(self, runtime: FakeCleanupRuntime | None) -> None:
        self.runtime_value = runtime
        self.failure = ""
        self.offloaded = False
        self.is_running = runtime is not None
        self.stops = 0
        self.is_loading = False
        self.profile = COMPACT_PROFILE

    def profile_detail(self) -> str:
        return self.profile.summary

    def runtime_available(self) -> bool:
        return True

    async def runtime(self, model_id: str, model_file: Path) -> FakeCleanupRuntime | None:
        return self.runtime_value

    def stop(self, *, offloaded: bool = False) -> None:
        self.stops += 1
        self.is_running = False
        self.offloaded = offloaded

    async def aclose(self) -> None:
        self.stop()


def enable_cleanup(
    app: Any,
    runtime: FakeCleanupRuntime | None,
    *,
    installed: bool = True,
    mode: str = MODE_CONSERVATIVE,
) -> Any:
    """Turn cleanup on for one app, with a scripted runtime behind it."""
    manager = app.state.ctx.cleanup
    manager.runtime_config.cleanup_enabled = True
    manager.runtime_config.cleanup_mode = mode
    manager.runtime_config.cleanup_model = CLEANUP_MODEL_ID
    manager.host = FakeWorkerHost(runtime)
    if installed:
        installed_file = manager.models.models_dir / "llama.cpp" / "model.gguf"
        installed_file.parent.mkdir(parents=True, exist_ok=True)
        installed_file.write_bytes(b"gguf")
        manager.model_path = lambda: installed_file  # type: ignore[method-assign]
    else:
        manager.model_path = lambda: None  # type: ignore[method-assign]
    return manager


def cleanup_options(**overrides: Any) -> CleanupOptions:
    fields: dict[str, Any] = {"mode": MODE_CONSERVATIVE, "model_id": CLEANUP_MODEL_ID}
    fields.update(overrides)
    return CleanupOptions(**fields)
