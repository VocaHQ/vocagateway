from __future__ import annotations

import sqlite3
from pathlib import Path

import httpx
from conftest import FakeEngine, FakeNormalizer
from starlette.status import (
    HTTP_200_OK,
    HTTP_400_BAD_REQUEST,
    HTTP_401_UNAUTHORIZED,
    HTTP_411_LENGTH_REQUIRED,
    HTTP_413_CONTENT_TOO_LARGE,
    HTTP_415_UNSUPPORTED_MEDIA_TYPE,
    HTTP_422_UNPROCESSABLE_CONTENT,
    HTTP_503_SERVICE_UNAVAILABLE,
)

from app.config import Settings
from app.errors import EngineUnavailableError
from app.main import create_app
from app.models.base import EngineHealth, TranscriptionOptions
from app.routes.transcriptions import _AudioForm

TRANSCRIPTIONS = "/v1/audio/transcriptions"
MULTIPART_CONTENT_TYPE = "multipart/form-data; boundary=----x"
CONTENT_TYPE_HEADER = "Content-Type"
CONTENT_LENGTH_HEADER = "content-length"
OVERSIZED_CONTENT_LENGTH = "1000000"
OVERSIZED_AUDIO_SIZE = 20_001
OVERSIZED_AUDIO_BYTES = b"x" * OVERSIZED_AUDIO_SIZE
POST_METHOD = "POST"
UNAUTHORIZED_ERROR = "unauthorized"


def _wav_file(
    audio_bytes: bytes, content_type: str = "audio/wav"
) -> dict[str, tuple[str, bytes, str]]:
    return {"file": ("audio.wav", audio_bytes, content_type)}


def _error_code(response: httpx.Response) -> str:
    return str(response.json()["error"]["code"])


class UnavailableEngine:
    async def health(self) -> EngineHealth:
        return EngineHealth(ready=True, name="unavailable-model")

    async def transcribe(self, audio_path: Path, options: TranscriptionOptions) -> str:
        raise EngineUnavailableError("The local engine is not available.")


def test_audio_form_accepts_roman_hinglish_output_contract() -> None:
    assert _AudioForm.language("hinglish_roman") == "hinglish_roman"


async def test_transcriptions_require_a_bearer_token(
    client: httpx.AsyncClient, audio_bytes: bytes
) -> None:
    missing = await client.post(TRANSCRIPTIONS, files=_wav_file(audio_bytes))
    assert missing.status_code == HTTP_401_UNAUTHORIZED
    assert _error_code(missing) == UNAUTHORIZED_ERROR

    garbage = await client.post(
        TRANSCRIPTIONS,
        headers={"Authorization": "Bearer not-a-real-token"},
        files=_wav_file(audio_bytes),
    )
    assert garbage.status_code == HTTP_401_UNAUTHORIZED
    assert _error_code(garbage) == UNAUTHORIZED_ERROR


async def test_transcription_creates_no_session(
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
    assert response.status_code == HTTP_200_OK
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
    assert response.status_code == HTTP_200_OK
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
    assert response.status_code == HTTP_200_OK
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
    assert response.status_code == HTTP_415_UNSUPPORTED_MEDIA_TYPE
    assert _error_code(response) == "unsupported_audio_type"


async def test_oversized_upload(client: httpx.AsyncClient, authorization: dict[str, str]) -> None:
    response = await client.post(
        TRANSCRIPTIONS,
        headers=authorization,
        files=_wav_file(OVERSIZED_AUDIO_BYTES),
    )
    assert response.status_code == HTTP_413_CONTENT_TOO_LARGE
    assert _error_code(response) == "audio_too_large"


async def test_empty_upload(client: httpx.AsyncClient, authorization: dict[str, str]) -> None:
    response = await client.post(
        TRANSCRIPTIONS,
        headers=authorization,
        files=_wav_file(b"x"),
    )
    assert response.status_code == HTTP_422_UNPROCESSABLE_CONTENT
    assert _error_code(response) == "audio_empty"


async def test_unavailable_engine_is_reported(
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
    assert response.status_code == HTTP_503_SERVICE_UNAVAILABLE
    assert _error_code(response) == "engine_unavailable"


async def test_verbose_json_is_rejected(
    client: httpx.AsyncClient, authorization: dict[str, str], audio_bytes: bytes
) -> None:
    response = await client.post(
        TRANSCRIPTIONS,
        headers=authorization,
        files=_wav_file(audio_bytes),
        data={"response_format": "verbose_json"},
    )
    assert response.status_code == HTTP_400_BAD_REQUEST
    assert _error_code(response) == "unsupported_response_format"


async def test_stream_true_is_rejected(
    client: httpx.AsyncClient, authorization: dict[str, str], audio_bytes: bytes
) -> None:
    response = await client.post(
        TRANSCRIPTIONS,
        headers=authorization,
        files=_wav_file(audio_bytes),
        data={"stream": "true"},
    )
    assert response.status_code == HTTP_400_BAD_REQUEST
    assert _error_code(response) == "streaming_not_supported"


async def test_unknown_form_fields_are_ignored(
    client: httpx.AsyncClient, authorization: dict[str, str], audio_bytes: bytes
) -> None:
    response = await client.post(
        TRANSCRIPTIONS,
        headers=authorization,
        files=_wav_file(audio_bytes),
        data={"prompt": "ignore me", "temperature": "0.2"},
    )
    assert response.status_code == HTTP_200_OK
    assert response.json() == {"text": "hello from the local model"}


async def test_rejects_oversized_length_before_body(
    client: httpx.AsyncClient, authorization: dict[str, str]
) -> None:
    """The header check 413s before FastAPI parses multipart."""
    request = client.build_request(
        POST_METHOD,
        TRANSCRIPTIONS,
        headers={
            **authorization,
            CONTENT_TYPE_HEADER: MULTIPART_CONTENT_TYPE,
        },
        content=b"",
    )
    request.headers[CONTENT_LENGTH_HEADER] = OVERSIZED_CONTENT_LENGTH
    response = await client.send(request)
    assert response.status_code == HTTP_413_CONTENT_TOO_LARGE
    assert _error_code(response) == "audio_too_large"


async def test_unauthenticated_oversized_length(
    client: httpx.AsyncClient,
) -> None:
    request = client.build_request(
        POST_METHOD,
        TRANSCRIPTIONS,
        headers={CONTENT_TYPE_HEADER: MULTIPART_CONTENT_TYPE},
        content=b"",
    )
    request.headers[CONTENT_LENGTH_HEADER] = OVERSIZED_CONTENT_LENGTH
    response = await client.send(request)
    assert response.status_code == HTTP_401_UNAUTHORIZED
    assert _error_code(response) == UNAUTHORIZED_ERROR


async def test_missing_content_length_is_rejected(
    client: httpx.AsyncClient, authorization: dict[str, str]
) -> None:
    """A missing Content-Length 411s before FastAPI parses multipart."""
    request = client.build_request(
        POST_METHOD,
        TRANSCRIPTIONS,
        headers={
            **authorization,
            CONTENT_TYPE_HEADER: MULTIPART_CONTENT_TYPE,
        },
        content=b"xxxxxxxx",
    )
    request.headers.pop(CONTENT_LENGTH_HEADER, None)
    response = await client.send(request)
    assert response.status_code == HTTP_411_LENGTH_REQUIRED
    assert _error_code(response) == "length_required"


async def test_unauthenticated_missing_length(
    client: httpx.AsyncClient,
) -> None:
    request = client.build_request(
        POST_METHOD,
        TRANSCRIPTIONS,
        headers={CONTENT_TYPE_HEADER: MULTIPART_CONTENT_TYPE},
        content=b"xxxxxxxx",
    )
    request.headers.pop(CONTENT_LENGTH_HEADER, None)
    response = await client.send(request)
    assert response.status_code == HTTP_401_UNAUTHORIZED
    assert _error_code(response) == UNAUTHORIZED_ERROR


async def test_invalid_content_length_is_rejected(
    client: httpx.AsyncClient, authorization: dict[str, str]
) -> None:
    request = client.build_request(
        POST_METHOD,
        TRANSCRIPTIONS,
        headers={
            **authorization,
            CONTENT_TYPE_HEADER: MULTIPART_CONTENT_TYPE,
        },
        content=b"xxxxxxxx",
    )
    request.headers[CONTENT_LENGTH_HEADER] = "nope"
    response = await client.send(request)
    assert response.status_code == HTTP_400_BAD_REQUEST
    assert _error_code(response) == "invalid_content_length"
