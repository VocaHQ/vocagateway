from __future__ import annotations

import sqlite3
from pathlib import Path

import httpx
from conftest import FakeEngine, FakeNormalizer

from app.config import Settings
from app.errors import EngineUnavailableError
from app.main import create_app
from app.models.base import EngineHealth, TranscriptionOptions

TRANSCRIPTIONS = "/v1/audio/transcriptions"


def _wav_file(
    audio_bytes: bytes, content_type: str = "audio/wav"
) -> dict[str, tuple[str, bytes, str]]:
    return {"file": ("audio.wav", audio_bytes, content_type)}


class UnavailableEngine:
    async def health(self) -> EngineHealth:
        return EngineHealth(ready=True, name="unavailable-model")

    async def transcribe(self, audio_path: Path, options: TranscriptionOptions) -> str:
        raise EngineUnavailableError("The local engine is not available.")


async def test_transcriptions_require_a_bearer_token(
    client: httpx.AsyncClient, audio_bytes: bytes
) -> None:
    missing = await client.post(TRANSCRIPTIONS, files=_wav_file(audio_bytes))
    assert missing.status_code == 401
    assert missing.json()["error"]["code"] == "unauthorized"

    garbage = await client.post(
        TRANSCRIPTIONS,
        headers={"Authorization": "Bearer not-a-real-token"},
        files=_wav_file(audio_bytes),
    )
    assert garbage.status_code == 401
    assert garbage.json()["error"]["code"] == "unauthorized"


async def test_whisper_1_returns_text_and_creates_no_session(
    client: httpx.AsyncClient,
    authorization: dict[str, str],
    audio_bytes: bytes,
    fake_engine: FakeEngine,
    settings: Settings,
) -> None:
    response = await client.post(
        TRANSCRIPTIONS,
        headers=authorization,
        files=_wav_file(audio_bytes),
        data={"model": "whisper-1", "language": "en"},
    )
    assert response.status_code == 200
    assert response.json() == {"text": "hello from the local model"}
    assert fake_engine.calls == 1
    assert fake_engine.last_options is not None
    assert fake_engine.last_options.language == "en"

    status = await client.get("/v1/admin/status", headers=authorization)
    assert status.json()["metrics"]["successful_transcriptions"] == 1

    with sqlite3.connect(settings.data_dir / "sessions.sqlite3") as connection:
        count = connection.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    assert count == 0

    leftover = list((settings.data_dir / "transcriptions").glob("*"))
    assert leftover == []


async def test_missing_language_is_auto(
    client: httpx.AsyncClient,
    authorization: dict[str, str],
    audio_bytes: bytes,
    fake_engine: FakeEngine,
) -> None:
    response = await client.post(
        TRANSCRIPTIONS,
        headers=authorization,
        files=_wav_file(audio_bytes),
    )
    assert response.status_code == 200
    assert fake_engine.last_options is not None
    assert fake_engine.last_options.language == "auto"


async def test_sensevoice_model_field_is_ignored(
    client: httpx.AsyncClient,
    authorization: dict[str, str],
    audio_bytes: bytes,
    fake_engine: FakeEngine,
) -> None:
    response = await client.post(
        TRANSCRIPTIONS,
        headers=authorization,
        files=_wav_file(audio_bytes),
        data={"model": "sensevoice"},
    )
    assert response.status_code == 200
    assert response.json() == {"text": "hello from the local model"}
    assert fake_engine.calls == 1


async def test_unsupported_audio_type(
    client: httpx.AsyncClient, authorization: dict[str, str], audio_bytes: bytes
) -> None:
    response = await client.post(
        TRANSCRIPTIONS,
        headers=authorization,
        files=_wav_file(audio_bytes, "text/plain"),
    )
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "unsupported_audio_type"


async def test_oversized_upload(client: httpx.AsyncClient, authorization: dict[str, str]) -> None:
    response = await client.post(
        TRANSCRIPTIONS,
        headers=authorization,
        files=_wav_file(b"x" * 20_001),
    )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "audio_too_large"


async def test_empty_upload(client: httpx.AsyncClient, authorization: dict[str, str]) -> None:
    response = await client.post(
        TRANSCRIPTIONS,
        headers=authorization,
        files=_wav_file(b"x"),
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "audio_empty"


async def test_engine_unavailable_is_503(
    settings: Settings, authorization: dict[str, str], audio_bytes: bytes
) -> None:
    app = create_app(settings, engine=UnavailableEngine(), normalizer=FakeNormalizer())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            TRANSCRIPTIONS,
            headers=authorization,
            files=_wav_file(audio_bytes),
        )
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "engine_unavailable"


async def test_verbose_json_is_rejected(
    client: httpx.AsyncClient, authorization: dict[str, str], audio_bytes: bytes
) -> None:
    response = await client.post(
        TRANSCRIPTIONS,
        headers=authorization,
        files=_wav_file(audio_bytes),
        data={"response_format": "verbose_json"},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_response_format"


async def test_stream_true_is_rejected(
    client: httpx.AsyncClient, authorization: dict[str, str], audio_bytes: bytes
) -> None:
    response = await client.post(
        TRANSCRIPTIONS,
        headers=authorization,
        files=_wav_file(audio_bytes),
        data={"stream": "true"},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "streaming_not_supported"


async def test_unknown_form_fields_are_ignored(
    client: httpx.AsyncClient, authorization: dict[str, str], audio_bytes: bytes
) -> None:
    response = await client.post(
        TRANSCRIPTIONS,
        headers=authorization,
        files=_wav_file(audio_bytes),
        data={"prompt": "ignore me", "temperature": "0.2"},
    )
    assert response.status_code == 200
    assert response.json() == {"text": "hello from the local model"}
