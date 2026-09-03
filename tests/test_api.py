from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import httpx
import pytest
from conftest import TOKEN, FakeEngine, FakeNormalizer
from starlette.status import (
    HTTP_200_OK,
    HTTP_401_UNAUTHORIZED,
    HTTP_404_NOT_FOUND,
    HTTP_413_CONTENT_TOO_LARGE,
    HTTP_415_UNSUPPORTED_MEDIA_TYPE,
    HTTP_422_UNPROCESSABLE_CONTENT,
    HTTP_503_SERVICE_UNAVAILABLE,
)

from app.config import Settings
from app.errors import LanguageUnsupportedError
from app.main import create_app
from app.models.base import EngineHealth, TranscriptionOptions

TEST_AUDIO_SIZE = 200
TEST_AUDIO_BYTES = b"x" * TEST_AUDIO_SIZE
OVERSIZED_AUDIO_SIZE = 20_001
OVERSIZED_AUDIO_BYTES = b"x" * OVERSIZED_AUDIO_SIZE
SESSIONS_API_PATH = "/v1/sessions"
CLIENT_SESSION_ID_KEY = "client_session_id"
STYLE_KEY = "style"
CONTENT_TYPE_HEADER = "Content-Type"
WAV_CONTENT_TYPE = "audio/wav"
STATUS_KEY = "status"
LANGUAGES_KEY = "languages"
JOB_ID_KEY = "job_id"
TRANSCRIPT_KEY = "transcript"


class UnreadyEngine:
    async def health(self) -> EngineHealth:
        return EngineHealth(ready=False, name="missing-model")


class WrongLanguageEngine:
    """Stands in for a loaded model whose language list excludes the request."""

    async def health(self) -> EngineHealth:
        return EngineHealth(ready=True, name="english-only-model")

    async def transcribe(self, audio_path: Path, options: TranscriptionOptions) -> str:
        raise LanguageUnsupportedError(
            "The selected model does not support hi. Choose Auto, en, or another model."
        )


async def test_unsupported_language_is_reported_a_d68a4(
    settings: Settings, authorization: dict[str, str], audio_bytes: bytes
) -> None:
    """A language the loaded model cannot serve must not look like a transient fault.

    The clients decide whether to keep audio for Retry from this code, and no
    number of retries will make an English-only model transcribe Hindi.
    """
    app = create_app(settings, engine=WrongLanguageEngine(), normalizer=FakeNormalizer())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://gateway") as client:
        session_id = uuid4()
        await client.post(
            SESSIONS_API_PATH,
            headers=authorization,
            json={CLIENT_SESSION_ID_KEY: str(session_id), "language": "hi", STYLE_KEY: "raw"},
        )
        await client.put(
            f"/v1/sessions/{session_id}/audio",
            headers={**authorization, CONTENT_TYPE_HEADER: WAV_CONTENT_TYPE},
            content=audio_bytes,
        )
        finished = await client.post(f"/v1/sessions/{session_id}/finish", headers=authorization)

        assert finished.status_code == HTTP_422_UNPROCESSABLE_CONTENT
        error = finished.json()["error"]
        assert error["code"] == "language_unsupported"
        assert error["recoverable"] is False
        assert "does not support hi" in error["message"]

        session = await client.get(f"/v1/sessions/{session_id}", headers=authorization)
        assert session.json()["error_code"] == "language_unsupported"


async def test_health_is_public_and_separates_eng_aa(
    client: httpx.AsyncClient, fake_engine: FakeEngine
) -> None:
    response = await client.get("/health")
    assert response.status_code == HTTP_200_OK
    assert response.json() == {
        STATUS_KEY: "ok",
        "engine_ready": True,
        "engine": "fake-local-model",
        "streaming_supported": False,
        # A stub engine has no catalog entry, so the gateway makes no claim and
        # clients keep every language selectable.
        LANGUAGES_KEY: [],
        "detects_language_automatically": False,
    }

    liveness = await client.get("/health/live")
    readiness = await client.get("/health/ready")
    repeated = await client.get("/health")

    assert liveness.status_code == HTTP_200_OK
    assert liveness.json()[STATUS_KEY] == "ok"
    assert liveness.json()["uptime_seconds"] >= 0
    assert readiness.status_code == HTTP_200_OK
    assert readiness.json()[STATUS_KEY] == "ready"
    assert readiness.json()["engine"] == "fake-local-model"
    assert repeated.status_code == HTTP_200_OK
    assert fake_engine.health_calls == 1


async def test_private_endpoints_require_bearer_token(client: httpx.AsyncClient) -> None:
    response = await client.get("/v1/models")
    assert response.status_code == HTTP_401_UNAUTHORIZED
    assert response.json()["error"]["code"] == "unauthorized"
    assert TOKEN not in response.text


async def test_readiness_can_fail_without_failing_ca0a7(tmp_path) -> None:
    settings = Settings(
        token=TOKEN,
        data_dir=tmp_path,
        whisper_binary=tmp_path / "missing-whisper",
        whisper_model=tmp_path / "missing-model",
    )
    app = create_app(settings, engine=UnreadyEngine(), normalizer=FakeNormalizer())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as test_client:
        liveness = await test_client.get("/health/live")
        readiness = await test_client.get("/health/ready")

    assert liveness.status_code == HTTP_200_OK
    assert readiness.status_code == HTTP_503_SERVICE_UNAVAILABLE
    assert readiness.json()[STATUS_KEY] == "not_ready"
    assert readiness.json()["engine"] == "missing-model"


async def test_complete_flow_is_idempotent_and_de_aaa(
    client: httpx.AsyncClient,
    authorization: dict[str, str],
    audio_bytes: bytes,
) -> None:
    session_id = uuid4()
    created = await client.post(
        SESSIONS_API_PATH,
        headers=authorization,
        json={CLIENT_SESSION_ID_KEY: str(session_id), "language": "auto", STYLE_KEY: "raw"},
    )
    repeated = await client.post(
        SESSIONS_API_PATH,
        headers=authorization,
        json={CLIENT_SESSION_ID_KEY: str(session_id), "language": "auto", STYLE_KEY: "raw"},
    )
    assert created.status_code == HTTP_200_OK
    assert repeated.json()[JOB_ID_KEY] == created.json()[JOB_ID_KEY]

    assert (
        await client.put(
            f"/v1/sessions/{session_id}/audio",
            headers={**authorization, CONTENT_TYPE_HEADER: WAV_CONTENT_TYPE},
            content=audio_bytes,
        )
    ).status_code == HTTP_200_OK

    finished = await client.post(f"/v1/sessions/{session_id}/finish", headers=authorization)
    finished_again = await client.post(f"/v1/sessions/{session_id}/finish", headers=authorization)
    assert finished.status_code == HTTP_200_OK
    assert finished.json()[TRANSCRIPT_KEY] == "hello from the local model"
    assert finished_again.json()[TRANSCRIPT_KEY] == finished.json()[TRANSCRIPT_KEY]
    assert finished_again.json()[JOB_ID_KEY] == finished.json()[JOB_ID_KEY]


async def test_session_accepts_writing_styles_and_aaaa(
    client: httpx.AsyncClient,
    authorization: dict[str, str],
) -> None:
    for style in ("formal", "casual", "very_casual", "excited"):
        response = await client.post(
            SESSIONS_API_PATH,
            headers=authorization,
            json={CLIENT_SESSION_ID_KEY: str(uuid4()), STYLE_KEY: style},
        )
        assert response.status_code == HTTP_200_OK
        assert response.json()[STYLE_KEY] == style

    invalid = await client.post(
        SESSIONS_API_PATH,
        headers=authorization,
        json={CLIENT_SESSION_ID_KEY: str(uuid4()), STYLE_KEY: "pirate"},
    )
    assert invalid.status_code == HTTP_422_UNPROCESSABLE_CONTENT


async def test_writing_style_is_applied_to_the_lo_aaaaa(
    client: httpx.AsyncClient,
    authorization: dict[str, str],
    audio_bytes: bytes,
) -> None:
    session_id = uuid4()
    await client.post(
        SESSIONS_API_PATH,
        headers=authorization,
        json={CLIENT_SESSION_ID_KEY: str(session_id), STYLE_KEY: "formal"},
    )
    await client.put(
        f"/v1/sessions/{session_id}/audio",
        headers={**authorization, CONTENT_TYPE_HEADER: WAV_CONTENT_TYPE},
        content=audio_bytes,
    )

    finished = await client.post(f"/v1/sessions/{session_id}/finish", headers=authorization)

    assert finished.status_code == HTTP_200_OK
    assert finished.json()[TRANSCRIPT_KEY] == "Hello from the local model."


async def test_upload_rejects_unsupported_empty_a_f2c1d(
    client: httpx.AsyncClient,
    authorization: dict[str, str],
) -> None:
    async def create() -> str:
        session_id = str(uuid4())
        await client.post(
            SESSIONS_API_PATH,
            headers=authorization,
            json={CLIENT_SESSION_ID_KEY: session_id},
        )
        return session_id

    unsupported = await client.put(
        f"/v1/sessions/{await create()}/audio",
        headers={**authorization, CONTENT_TYPE_HEADER: "text/plain"},
        content=TEST_AUDIO_BYTES,
    )
    empty = await client.put(
        f"/v1/sessions/{await create()}/audio",
        headers={**authorization, CONTENT_TYPE_HEADER: WAV_CONTENT_TYPE},
        content=b"x",
    )
    oversized = await client.put(
        f"/v1/sessions/{await create()}/audio",
        headers={**authorization, CONTENT_TYPE_HEADER: WAV_CONTENT_TYPE},
        content=OVERSIZED_AUDIO_BYTES,
    )
    assert unsupported.status_code == HTTP_415_UNSUPPORTED_MEDIA_TYPE
    assert empty.status_code == HTTP_422_UNPROCESSABLE_CONTENT
    assert oversized.status_code == HTTP_413_CONTENT_TOO_LARGE


async def test_delete_is_idempotent(
    client: httpx.AsyncClient,
    authorization: dict[str, str],
) -> None:
    session_id = uuid4()
    await client.post(
        SESSIONS_API_PATH,
        headers=authorization,
        json={CLIENT_SESSION_ID_KEY: str(session_id)},
    )
    first = await client.delete(f"/v1/sessions/{session_id}", headers=authorization)
    second = await client.delete(f"/v1/sessions/{session_id}", headers=authorization)
    assert first.status_code == HTTP_200_OK
    assert first.json() == {"deleted": True}
    assert second.status_code == HTTP_404_NOT_FOUND
    assert second.json() == {"deleted": False}


async def test_health_reports_what_the_loaded_mod_a(settings: Settings, audio_bytes: bytes) -> None:
    """Clients cannot offer a sensible language picker without knowing whether the
    loaded model can be pinned at all. An engine holding a catalog entry reports
    that entry's languages; one that picks its own language says so."""
    from app.catalog import DEFAULT_CATALOG
    from app.models.base import EngineHealth

    dolphin = next(
        model for model in DEFAULT_CATALOG if model.id == "sherpa-onnx:dolphin-small-ctc-int8"
    )

    class DolphinLikeEngine:
        catalog_model = dolphin

        async def health(self) -> EngineHealth:
            return EngineHealth(ready=True, name="sherpa-onnx:dolphin")

        async def transcribe(self, audio_path: Path, options: TranscriptionOptions) -> str:
            return "unused"

    app = create_app(settings, engine=DolphinLikeEngine(), normalizer=FakeNormalizer())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://gateway") as client:
        payload = (await client.get("/health")).json()

    assert payload["detects_language_automatically"] is True
    assert "hi" in payload[LANGUAGES_KEY] and "bn" in payload[LANGUAGES_KEY]
    assert "en" not in payload[LANGUAGES_KEY]  # Dolphin is not trained on English


class BoomEngine:
    """Unexpected failure during transcription (not a typed engine error)."""

    async def health(self) -> EngineHealth:
        return EngineHealth(ready=True, name="boom-model")

    async def transcribe(self, audio_path: Path, options: TranscriptionOptions) -> str:
        raise RuntimeError("engine exploded")


async def test_unexpected_finish_error_leaves_ses_f5519(
    settings: Settings, authorization: dict[str, str], audio_bytes: bytes
) -> None:
    """Bare exceptions must not leave the session stuck in 'transcribing'.

    Retry only accepts failed/uploaded/completed, and finish rejects
    transcription_in_progress, so a stuck transcribing state blocks recovery.
    """
    app = create_app(settings, engine=BoomEngine(), normalizer=FakeNormalizer())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://gateway"
    ) as client:
        session_id = uuid4()
        await client.post(
            SESSIONS_API_PATH,
            headers=authorization,
            json={CLIENT_SESSION_ID_KEY: str(session_id), "language": "en", STYLE_KEY: "raw"},
        )
        await client.put(
            f"/v1/sessions/{session_id}/audio",
            headers={**authorization, CONTENT_TYPE_HEADER: WAV_CONTENT_TYPE},
            content=audio_bytes,
        )
        with pytest.raises(RuntimeError, match="engine exploded"):
            # ASGI client surfaces the unhandled exception from the route.
            await client.post(f"/v1/sessions/{session_id}/finish", headers=authorization)

        session = await client.get(f"/v1/sessions/{session_id}", headers=authorization)
        assert session.json()["state"] == "failed"
        assert session.json()["error_code"] == "internal_error"
