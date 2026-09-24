from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import httpx
import pytest
from conftest import TOKEN, FakeEngine, FakeNormalizer
from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.status import HTTP_200_OK, HTTP_503_SERVICE_UNAVAILABLE

from app.config import Settings
from app.engines import EngineProvider
from app.errors import APIProblem
from app.main import create_app
from app.models.base import TranscriptionEngine, TranscriptionOptions
from app.service import _DecodeLimiter

TRANSCRIPTIONS = "/v1/audio/transcriptions"
ENGINE_OVERLOADED = "engine_overloaded"
AUTHORIZATION = {"Authorization": f"Bearer {TOKEN}"}


class SlowEngine(FakeEngine):
    def __init__(self, started: asyncio.Event, release: asyncio.Event) -> None:
        super().__init__()
        self._started = started
        self._release = release
        self.inflight = 0
        self.peak = 0

    async def transcribe(self, audio_path: Path, options: TranscriptionOptions) -> str:
        assert audio_path.is_file()
        self.calls += 1
        self.last_options = options
        self.inflight += 1
        self.peak = max(self.peak, self.inflight)
        self._started.set()
        await self._release.wait()
        self.inflight -= 1
        return self.transcript


class ParallelEngine(FakeEngine):
    max_parallel_decodes = 2

    def __init__(self, release: asyncio.Event) -> None:
        super().__init__()
        self._release = release
        self.inflight = 0
        self.peak = 0
        self.both_started = asyncio.Event()

    async def transcribe(self, audio_path: Path, options: TranscriptionOptions) -> str:
        assert audio_path.is_file()
        self.calls += 1
        self.inflight += 1
        self.peak = max(self.peak, self.inflight)
        if self.inflight == 2:
            self.both_started.set()
        await self._release.wait()
        self.inflight -= 1
        return self.transcript


def _wav_file(audio_bytes: bytes) -> dict[str, tuple[str, bytes, str]]:
    return {"file": ("audio.wav", audio_bytes, "audio/wav")}


def _error_code(response: httpx.Response) -> str:
    return str(response.json()["error"]["code"])


def _queued_settings(settings: Settings, *, queued: int, timeout: float) -> Settings:
    return replace(
        settings,
        maximum_queued_transcriptions=queued,
        transcription_queue_timeout_seconds=timeout,
    )


def _app_client(settings: Settings, engine: FakeEngine) -> tuple[FastAPI, httpx.AsyncClient]:
    app = create_app(settings, engine=engine, normalizer=FakeNormalizer())
    return app, httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    )


async def test_overlapping_transcriptions_wait_then_succeed(
    settings: Settings, audio_bytes: bytes
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    engine = SlowEngine(started, release)
    queued_settings = _queued_settings(settings, queued=4, timeout=2.0)
    _, client = _app_client(queued_settings, engine)
    async with client:
        first = asyncio.create_task(
            client.post(TRANSCRIPTIONS, headers=AUTHORIZATION, files=_wav_file(audio_bytes))
        )
        await asyncio.wait_for(started.wait(), timeout=1)
        second = asyncio.create_task(
            client.post(TRANSCRIPTIONS, headers=AUTHORIZATION, files=_wav_file(audio_bytes))
        )
        await asyncio.sleep(0.05)
        release.set()
        responses = await asyncio.gather(first, second)
    assert [response.status_code for response in responses] == [HTTP_200_OK, HTTP_200_OK]
    assert engine.calls == 2
    assert engine.peak == 1


async def test_full_queue_rejects_immediately(settings: Settings, audio_bytes: bytes) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    engine = SlowEngine(started, release)
    queued_settings = _queued_settings(settings, queued=0, timeout=2.0)
    _, client = _app_client(queued_settings, engine)
    async with client:
        first = asyncio.create_task(
            client.post(TRANSCRIPTIONS, headers=AUTHORIZATION, files=_wav_file(audio_bytes))
        )
        await asyncio.wait_for(started.wait(), timeout=1)
        rejected = await client.post(
            TRANSCRIPTIONS, headers=AUTHORIZATION, files=_wav_file(audio_bytes)
        )
        release.set()
        first_response = await first
    assert first_response.status_code == HTTP_200_OK
    assert rejected.status_code == HTTP_503_SERVICE_UNAVAILABLE
    assert _error_code(rejected) == ENGINE_OVERLOADED
    assert engine.calls == 1


async def test_queue_timeout_returns_overloaded(settings: Settings, audio_bytes: bytes) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    engine = SlowEngine(started, release)
    queued_settings = _queued_settings(settings, queued=4, timeout=0.05)
    _, client = _app_client(queued_settings, engine)
    async with client:
        first = asyncio.create_task(
            client.post(TRANSCRIPTIONS, headers=AUTHORIZATION, files=_wav_file(audio_bytes))
        )
        await asyncio.wait_for(started.wait(), timeout=1)
        rejected = await client.post(
            TRANSCRIPTIONS, headers=AUTHORIZATION, files=_wav_file(audio_bytes)
        )
        release.set()
        first_response = await first
    assert first_response.status_code == HTTP_200_OK
    assert rejected.status_code == HTTP_503_SERVICE_UNAVAILABLE
    assert _error_code(rejected) == ENGINE_OVERLOADED
    assert engine.calls == 1


async def test_cancelled_waiter_frees_the_queue_line(
    settings: Settings, audio_bytes: bytes
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    engine = SlowEngine(started, release)
    queued_settings = _queued_settings(settings, queued=1, timeout=2.0)
    app, client = _app_client(queued_settings, engine)
    files = _wav_file(audio_bytes)
    async with client:
        first = asyncio.create_task(client.post(TRANSCRIPTIONS, headers=AUTHORIZATION, files=files))
        await asyncio.wait_for(started.wait(), timeout=1)
        waiter = asyncio.create_task(
            client.post(TRANSCRIPTIONS, headers=AUTHORIZATION, files=files)
        )
        await _wait_for_queue(app, depth=1)
        await _cancel_waiter(app, waiter)
        queued = asyncio.create_task(
            client.post(TRANSCRIPTIONS, headers=AUTHORIZATION, files=files)
        )
        await _wait_for_queue(app, depth=1)
        release.set()
        first_response, queued_response = await asyncio.gather(first, queued)
    assert first_response.status_code == HTTP_200_OK
    assert queued_response.status_code == HTTP_200_OK
    assert engine.calls == 2


async def test_process_style_engines_decode_two_clips_at_once(
    settings: Settings, audio_bytes: bytes
) -> None:
    release = asyncio.Event()
    engine = ParallelEngine(release)
    parallel = replace(settings, maximum_concurrent_transcriptions=2)
    _, client = _app_client(parallel, engine)
    async with client:
        first = asyncio.create_task(
            client.post(TRANSCRIPTIONS, headers=AUTHORIZATION, files=_wav_file(audio_bytes))
        )
        second = asyncio.create_task(
            client.post(TRANSCRIPTIONS, headers=AUTHORIZATION, files=_wav_file(audio_bytes))
        )
        await asyncio.wait_for(engine.both_started.wait(), timeout=1)
        assert engine.peak == 2
        release.set()
        responses = await asyncio.gather(first, second)
    assert [response.status_code for response in responses] == [HTTP_200_OK, HTTP_200_OK]
    assert engine.calls == 2


async def _cancel_once_queued(app: FastAPI, waiter: asyncio.Task[Any]) -> None:
    await _wait_for_queue(app, depth=1)
    await _cancel_waiter(app, waiter)


async def _cancel_waiter(app: FastAPI, waiter: asyncio.Task[Any]) -> None:
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    await _wait_for_queue(app, depth=0)


async def _wait_for_queue(app: FastAPI, *, depth: int) -> None:
    service = app.state.ctx.service
    deadline = time.monotonic() + 0.5
    while service.metrics.snapshot().queue_depth != depth:
        if time.monotonic() >= deadline:
            raise AssertionError("queue did not reach expected depth")
        await asyncio.sleep(0.01)


class CountingNormalizer(FakeNormalizer):
    def __init__(self) -> None:
        self.calls = 0

    async def normalize(self, source: Path, destination: Path, maximum_seconds: int) -> Path:
        self.calls += 1
        return await super().normalize(source, destination, maximum_seconds)


async def test_a_full_line_is_refused_before_ffmpeg_runs(
    settings: Settings, audio_bytes: bytes
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    engine = SlowEngine(started, release)
    normalizer = CountingNormalizer()
    app = create_app(
        _queued_settings(settings, queued=0, timeout=2.0), engine=engine, normalizer=normalizer
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        first = asyncio.create_task(
            client.post(TRANSCRIPTIONS, headers=AUTHORIZATION, files=_wav_file(audio_bytes))
        )
        await asyncio.wait_for(started.wait(), timeout=1)
        burst = await asyncio.gather(
            *(
                client.post(TRANSCRIPTIONS, headers=AUTHORIZATION, files=_wav_file(audio_bytes))
                for _ in range(4)
            )
        )
        release.set()
        await first
    assert {response.status_code for response in burst} == {HTTP_503_SERVICE_UNAVAILABLE}
    assert normalizer.calls == 1
    service = app.state.ctx.service
    snapshot = service.metrics.snapshot()
    assert snapshot.rejected_transcriptions == 4
    assert snapshot.failed_transcriptions == 0


async def test_a_refused_session_stays_uploaded(settings: Settings, audio_bytes: bytes) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    engine = SlowEngine(started, release)
    _, client = _app_client(_queued_settings(settings, queued=0, timeout=2.0), engine)
    async with client:
        session_id = await _uploaded_session(client, audio_bytes)
        first = asyncio.create_task(
            client.post(TRANSCRIPTIONS, headers=AUTHORIZATION, files=_wav_file(audio_bytes))
        )
        await asyncio.wait_for(started.wait(), timeout=1)
        refused = await client.post(f"/v1/sessions/{session_id}/finish", headers=AUTHORIZATION)
        stored = await client.get(f"/v1/sessions/{session_id}", headers=AUTHORIZATION)
        release.set()
        await first
    assert refused.status_code == HTTP_503_SERVICE_UNAVAILABLE
    assert stored.json()["state"] == "uploaded"


async def test_a_waiting_request_does_not_pin_the_engine(
    settings: Settings, audio_bytes: bytes
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    engine = SlowEngine(started, release)
    app, client = _app_client(_queued_settings(settings, queued=2, timeout=2.0), engine)
    provider = _LeaseCounter(app.state.ctx.service._engine_provider)
    app.state.ctx.service._engine_provider = provider
    async with client:
        first = asyncio.create_task(
            client.post(TRANSCRIPTIONS, headers=AUTHORIZATION, files=_wav_file(audio_bytes))
        )
        await asyncio.wait_for(started.wait(), timeout=1)
        waiter = asyncio.create_task(
            client.post(TRANSCRIPTIONS, headers=AUTHORIZATION, files=_wav_file(audio_bytes))
        )
        await _wait_for_queue(app, depth=1)
        held_while_waiting = provider.active
        release.set()
        await asyncio.gather(first, waiter)
    # A model swap retires the engine only when its last lease ends, so a
    # waiter holding one would keep a second model resident.
    assert held_while_waiting == 1


async def _uploaded_session(client: httpx.AsyncClient, audio_bytes: bytes) -> str:
    session_id = str(uuid4())
    await client.post(
        "/v1/sessions",
        headers=AUTHORIZATION,
        json={"client_session_id": session_id, "language": "auto", "style": "raw"},
    )
    await client.put(
        f"/v1/sessions/{session_id}/audio",
        headers={**AUTHORIZATION, "Content-Type": "audio/wav"},
        content=audio_bytes,
    )
    return session_id


class _LeaseCounter:
    def __init__(self, inner: EngineProvider) -> None:
        self._inner = inner
        self.active = 0

    def current(self) -> TranscriptionEngine:
        return self._inner.current()

    @asynccontextmanager
    async def lease(self) -> AsyncIterator[TranscriptionEngine]:
        self.active += 1
        try:
            async with self._inner.lease() as engine:
                yield engine
        finally:
            self.active -= 1


async def test_limiter_serves_waiters_first_come_first_served() -> None:
    limiter = _DecodeLimiter(lambda: 1)
    await limiter.acquire(wait_seconds=1)
    order: list[str] = []

    async def take(name: str) -> None:
        await limiter.acquire(wait_seconds=1)
        order.append(name)

    early = asyncio.create_task(take("early"))
    await asyncio.sleep(0)
    late = asyncio.create_task(take("late"))
    limiter.release()
    await asyncio.sleep(0.01)
    assert order == ["early"]
    limiter.release()
    await asyncio.gather(early, late)
    assert order == ["early", "late"]


async def test_limiter_timeout_leaves_no_slot_behind() -> None:
    limiter = _DecodeLimiter(lambda: 1)
    await limiter.acquire(wait_seconds=1)
    with pytest.raises(TimeoutError):
        await limiter.acquire(wait_seconds=0.01)
    limiter.release()
    await asyncio.wait_for(limiter.acquire(wait_seconds=0.01), timeout=1)


async def test_limiter_follows_a_swap_to_a_single_decode_engine() -> None:
    limits = [2]
    limiter = _DecodeLimiter(lambda: limits[0])
    await limiter.acquire(wait_seconds=1)
    await limiter.acquire(wait_seconds=1)
    # Two clips are decoding on a two-decode engine when a single-decode model
    # is selected. The next waiter must wait for both, not just one.
    limits[0] = 1
    third = asyncio.create_task(limiter.acquire(wait_seconds=1))
    limiter.release()
    await asyncio.sleep(0.01)
    assert not third.done()
    limiter.release()
    await asyncio.wait_for(third, timeout=1)


class StallingNormalizer(FakeNormalizer):
    def __init__(self) -> None:
        self.stalled = asyncio.Event()

    async def normalize(self, source: Path, destination: Path, maximum_seconds: int) -> Path:
        self.stalled.set()
        await asyncio.Event().wait()
        return destination


async def test_a_stalled_normalisation_gives_its_place_back(
    settings: Settings, audio_bytes: bytes
) -> None:
    app = create_app(
        _queued_settings(settings, queued=0, timeout=0.1),
        engine=FakeEngine(),
        normalizer=StallingNormalizer(),
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        stalled = await client.post(
            TRANSCRIPTIONS, headers=AUTHORIZATION, files=_wav_file(audio_bytes)
        )
    assert stalled.status_code == HTTP_503_SERVICE_UNAVAILABLE
    assert _error_code(stalled) == ENGINE_OVERLOADED
    service = app.state.ctx.service
    assert service.metrics.snapshot().queue_depth == 0


async def test_a_disconnected_waiter_leaves_the_line(
    settings: Settings, audio_bytes: bytes, tmp_path: Path
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    engine = SlowEngine(started, release)
    app, client = _app_client(_queued_settings(settings, queued=2, timeout=5.0), engine)
    service = app.state.ctx.service
    source = tmp_path / "abandoned.wav"
    source.write_bytes(audio_bytes)

    async def hung_up() -> bool:
        return True

    async with client:
        first = asyncio.create_task(
            client.post(TRANSCRIPTIONS, headers=AUTHORIZATION, files=_wav_file(audio_bytes))
        )
        await asyncio.wait_for(started.wait(), timeout=1)
        with pytest.raises(APIProblem) as abandoned:
            await asyncio.wait_for(
                service.transcribe_adhoc(source, "auto", disconnected=hung_up), timeout=1
            )
        depth = service.metrics.snapshot().queue_depth
        release.set()
        await first
    assert abandoned.value.code == "client_disconnected"
    assert depth == 0
    assert engine.calls == 1
    snapshot = service.metrics.snapshot()
    assert (snapshot.failed_transcriptions, snapshot.rejected_transcriptions) == (0, 0)


async def test_a_cancelled_finish_leaves_the_session_retryable(
    settings: Settings, audio_bytes: bytes
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    engine = SlowEngine(started, release)
    app, client = _app_client(_queued_settings(settings, queued=2, timeout=5.0), engine)
    service = app.state.ctx.service
    async with client:
        session_id = await _uploaded_session(client, audio_bytes)
        first = asyncio.create_task(
            client.post(TRANSCRIPTIONS, headers=AUTHORIZATION, files=_wav_file(audio_bytes))
        )
        await asyncio.wait_for(started.wait(), timeout=1)
        waiting = asyncio.create_task(service.finish(UUID(session_id)))
        await _cancel_once_queued(app, waiting)
        stored = service.require(UUID(session_id))
        release.set()
        await first
        retried = await client.post(f"/v1/sessions/{session_id}/finish", headers=AUTHORIZATION)
    assert stored.state == "failed"
    assert retried.status_code == HTTP_200_OK


async def test_status_reports_the_engines_own_limit(settings: Settings) -> None:
    single = create_app(settings, engine=FakeEngine(), normalizer=FakeNormalizer())
    release = asyncio.Event()
    parallel = create_app(settings, engine=ParallelEngine(release), normalizer=FakeNormalizer())
    services = [app.state.ctx.service for app in (single, parallel)]
    limits = [service.metrics.snapshot().concurrency_limit for service in services]
    assert limits == [1, 2]


def test_no_middleware_hides_client_disconnects(settings: Settings) -> None:
    # BaseHTTPMiddleware (`@app.middleware("http")`) swallows the disconnect a
    # waiting request polls for, so abandoned requests would decode for nobody.
    app = create_app(settings, engine=FakeEngine(), normalizer=FakeNormalizer())
    assert all(entry.cls is not BaseHTTPMiddleware for entry in app.user_middleware)
