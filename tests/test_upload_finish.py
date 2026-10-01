from __future__ import annotations

import asyncio
import struct
from collections.abc import AsyncIterator
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
from conftest import FakeEngine, FakeNormalizer
from starlette.status import (
    HTTP_200_OK,
    HTTP_401_UNAUTHORIZED,
    HTTP_413_CONTENT_TOO_LARGE,
    HTTP_415_UNSUPPORTED_MEDIA_TYPE,
    HTTP_422_UNPROCESSABLE_CONTENT,
)
from starlette.types import ASGIApp, Message, Scope

from app.audio import FFmpegNormalizer
from app.config import Settings
from app.main import create_app
from app.models.base import TranscriptionOptions

SESSION_PATH = "/v1/sessions"
STATE_KEY = "state"
TRANSCRIPT_KEY = "transcript"
WAV_HEADERS = {"Content-Type": "audio/wav"}


class PausedEngine(FakeEngine):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def transcribe(self, audio_path: Path, options: TranscriptionOptions) -> str:
        self.calls += 1
        self.started.set()
        await self.release.wait()
        return self.transcript


class PausedNormalizer(FakeNormalizer):
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def normalize(self, source: Path, destination: Path, maximum_seconds: int) -> Path:
        await super().normalize(source, destination, maximum_seconds)
        self.started.set()
        await self.release.wait()
        return destination


class DisconnectingUpload:
    """Drive the HTTP route with a real ASGI disconnect, without cancelling it."""

    def __init__(self, path: str, headers: dict[str, str], body: bytes) -> None:
        self.scope: Scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "PUT",
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"finish=true",
            "root_path": "",
            "headers": [(key.lower().encode(), value.encode()) for key, value in headers.items()],
            "client": ("127.0.0.1", 12345),
            "server": ("test", 80),
        }
        self.incoming: asyncio.Queue[Message] = asyncio.Queue()
        self.incoming.put_nowait({"type": "http.request", "body": body, "more_body": False})
        self.outgoing: list[Message] = []

    async def send(self, message: Message) -> None:
        self.outgoing.append(message)

    def disconnect(self) -> None:
        self.incoming.put_nowait({"type": "http.disconnect"})

    async def disconnect_when(self, app: ASGIApp, ready: asyncio.Event) -> None:
        async with asyncio.TaskGroup() as group:
            pending = group.create_task(app(self.scope, self.incoming.get, self.send))
            await asyncio.wait_for(ready.wait(), timeout=2)
            self.disconnect()
            await asyncio.wait_for(pending, timeout=2)


@pytest.mark.parametrize("finish", [None, "false", "true"])
async def test_upload_can_finish_without_a_second_request(
    client: httpx.AsyncClient,
    fake_engine: FakeEngine,
    authorization: dict[str, str],
    audio_bytes: bytes,
    finish: str | None,
) -> None:
    session_id = str(uuid4())
    created = await client.post(
        SESSION_PATH,
        headers=authorization,
        json={"client_session_id": session_id, "style": "raw"},
    )
    path = f"{SESSION_PATH}/{session_id}/audio"
    response = await client.put(
        path,
        params={"finish": finish} if finish else {},
        headers=authorization | WAV_HEADERS,
        content=audio_bytes,
    )
    finishing = finish == "true"
    expected_state = "completed" if finishing else "uploaded"
    assert response.status_code == HTTP_200_OK
    assert response.json()[STATE_KEY] == expected_state
    assert fake_engine.calls == int(finishing)
    assert response.json()["job_id"] == created.json()["job_id"]
    if finishing:
        assert response.json()[TRANSCRIPT_KEY] == fake_engine.transcript
        repeated = await client.put(
            path, params={"finish": "true"}, headers=authorization | WAV_HEADERS, content=b"x"
        )
        assert repeated.json() == response.json()
        legacy = await client.post(f"{SESSION_PATH}/{session_id}/finish", headers=authorization)
        assert legacy.json() == response.json()
        assert fake_engine.calls == 1


async def test_live_unknown_length_wav_finishes_only_after_eof(
    settings: Settings,
    fake_engine: FakeEngine,
    authorization: dict[str, str],
    audio_bytes: bytes,
) -> None:
    """Exercise the real FFmpeg path with the phone's unknown RIFF lengths."""
    started = asyncio.Event()
    stopped = asyncio.Event()
    streaming_wav = bytearray(audio_bytes)
    struct.pack_into("<I", streaming_wav, 4, 0xFFFFFFFF)
    struct.pack_into("<I", streaming_wav, 40, 0xFFFFFFFF)

    async def recording() -> AsyncIterator[bytes]:
        yield bytes(streaming_wav[:100])
        started.set()
        await stopped.wait()
        yield bytes(streaming_wav[100:])

    app = create_app(settings, engine=fake_engine, normalizer=FFmpegNormalizer())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        session_id = str(uuid4())
        await client.post(
            SESSION_PATH,
            headers=authorization,
            json={"client_session_id": session_id, "style": "raw"},
        )
        pending = asyncio.create_task(
            client.put(
                f"{SESSION_PATH}/{session_id}/audio?finish=true",
                headers=authorization | WAV_HEADERS,
                content=recording(),
            )
        )
        await asyncio.wait_for(started.wait(), timeout=2)
        assert fake_engine.calls == 0
        assert not pending.done()
        stopped.set()
        response = await asyncio.wait_for(pending, timeout=5)
    assert response.status_code == HTTP_200_OK
    assert response.json()[STATE_KEY] == "completed"
    assert response.json()[TRANSCRIPT_KEY] == fake_engine.transcript
    assert fake_engine.calls == 1
    assert not list((settings.data_dir / "audio").iterdir())


async def test_cancelled_combined_request_keeps_audio_and_allows_retry(
    settings: Settings, authorization: dict[str, str], audio_bytes: bytes
) -> None:
    engine = PausedEngine()
    app = create_app(settings, engine=engine, normalizer=FakeNormalizer())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        session_id = str(uuid4())
        await client.post(
            SESSION_PATH, headers=authorization, json={"client_session_id": session_id}
        )
        pending = asyncio.create_task(
            client.put(
                f"{SESSION_PATH}/{session_id}/audio?finish=true",
                headers=authorization | WAV_HEADERS,
                content=audio_bytes,
            )
        )
        await asyncio.wait_for(engine.started.wait(), timeout=2)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        stored = await client.get(f"{SESSION_PATH}/{session_id}", headers=authorization)
        assert stored.json()[STATE_KEY] == "failed"
        assert stored.json()["error_code"] == "transcription_cancelled"
        assert list((settings.data_dir / "audio").iterdir())
        engine.release.set()
        retried = await client.post(f"{SESSION_PATH}/{session_id}/finish", headers=authorization)
    assert retried.status_code == HTTP_200_OK
    assert retried.json()[STATE_KEY] == "completed"
    assert engine.calls == 2


async def test_http_disconnect_after_upload_eof_keeps_audio_retryable(
    settings: Settings, authorization: dict[str, str], audio_bytes: bytes
) -> None:
    engine = FakeEngine()
    normalizer = PausedNormalizer()
    app = create_app(settings, engine=engine, normalizer=normalizer)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        session_id = str(uuid4())
        await client.post(
            SESSION_PATH, headers=authorization, json={"client_session_id": session_id}
        )
        upload = DisconnectingUpload(
            f"{SESSION_PATH}/{session_id}/audio", authorization | WAV_HEADERS, audio_bytes
        )
        await upload.disconnect_when(app, normalizer.started)
        stored = await client.get(f"{SESSION_PATH}/{session_id}", headers=authorization)
        assert stored.json()[STATE_KEY] == "uploaded"
        assert engine.calls == 0
        assert next((settings.data_dir / "audio").iterdir()).read_bytes() == audio_bytes
        assert not list((settings.data_dir / "normalized").iterdir())
        normalizer.release.set()
        retried = await client.post(f"{SESSION_PATH}/{session_id}/finish", headers=authorization)
    assert retried.status_code == HTTP_200_OK
    assert retried.json()[STATE_KEY] == "completed"
    assert engine.calls == 1


@pytest.mark.parametrize(
    ("headers", "body", "expected"),
    [
        ({"Content-Type": "text/plain"}, b"x" * 200, HTTP_415_UNSUPPORTED_MEDIA_TYPE),
        (WAV_HEADERS, b"x", HTTP_422_UNPROCESSABLE_CONTENT),
        (WAV_HEADERS, b"x" * 20_001, HTTP_413_CONTENT_TOO_LARGE),
    ],
)
async def test_combined_upload_keeps_validation_and_auth(
    client: httpx.AsyncClient,
    fake_engine: FakeEngine,
    authorization: dict[str, str],
    headers: dict[str, str],
    body: bytes,
    expected: int,
) -> None:
    session_id = str(uuid4())
    await client.post(SESSION_PATH, headers=authorization, json={"client_session_id": session_id})
    path = f"{SESSION_PATH}/{session_id}/audio?finish=true"
    unauthorized = await client.put(path, headers=headers, content=body)
    rejected = await client.put(path, headers=authorization | headers, content=body)
    assert unauthorized.status_code == HTTP_401_UNAUTHORIZED
    assert rejected.status_code == expected
    assert fake_engine.calls == 0
