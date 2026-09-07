"""Sessions, storage, and the races around finishing one.

Two questions run through the whole file. Does an existing client see exactly
what it saw before the feature existed? And can a finished session ever change
its answer — after a settings change, a restart, a retry, a delete, or a second
concurrent `finish`?
"""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import httpx
import pytest
from conftest import (
    CLEANUP_MODEL_ID,
    TOKEN,
    FakeCleanupRuntime,
    FakeEngine,
    FakeNormalizer,
    enable_cleanup,
)
from starlette.status import HTTP_200_OK, HTTP_409_CONFLICT, HTTP_422_UNPROCESSABLE_CONTENT

from app import storage
from app.config import Settings
from app.main import create_app

SPOKEN = "we was going to leave early but the train was late"
CORRECTED = "We were going to leave early, but the train was late."
SESSIONS = "/v1/sessions"
AUDIO_BYTES = b"x" * 200
WAV = {"Content-Type": "audio/wav"}


class CleanupApp:
    """One gateway with a scripted cleanup runtime behind it."""

    def __init__(self, client: httpx.AsyncClient, app: Any, runtime: FakeCleanupRuntime) -> None:
        self.client = client
        self.app = app
        self.runtime = runtime
        self.auth = {"Authorization": f"Bearer {TOKEN}"}

    @property
    def repository(self) -> storage.SessionRepository:
        return self.app.state.ctx.repository

    async def create(self, **body: Any) -> httpx.Response:
        # Explicit `en`: a session left on `auto` deliberately does not resolve
        # to a language for cleanup, which `test_auto_language_is_not_assumed`
        # covers on its own.
        payload = {"client_session_id": str(uuid4()), "style": "casual", "language": "en"}
        payload.update(body)
        return await self.client.post(SESSIONS, json=payload, headers=self.auth)

    async def upload(self, session_id: str) -> httpx.Response:
        return await self.client.put(
            f"{SESSIONS}/{session_id}/audio",
            content=AUDIO_BYTES,
            headers={**self.auth, **WAV},
        )

    async def finish(self, session_id: str) -> httpx.Response:
        return await self.client.post(f"{SESSIONS}/{session_id}/finish", headers=self.auth)

    async def run(self, **body: Any) -> dict[str, Any]:
        created = (await self.create(**body)).json()
        await self.upload(created["session_id"])
        return (await self.finish(created["session_id"])).json()


@pytest.fixture
async def cleaned(settings: Settings) -> AsyncIterator[CleanupApp]:
    runtime = FakeCleanupRuntime(CORRECTED)
    app = create_app(settings, engine=FakeEngine(SPOKEN), normalizer=FakeNormalizer())
    enable_cleanup(app, runtime)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield CleanupApp(client, app, runtime)


async def test_a_corrected_session_reports_both_texts(cleaned: CleanupApp) -> None:
    payload = await cleaned.run()
    assert payload["transcript"] == "We were going to leave early, but the train was late"
    assert payload["original_transcript"] == SPOKEN
    assert payload["cleanup"]["status"] == "applied"
    assert payload["cleanup"]["model_id"] == CLEANUP_MODEL_ID
    assert payload["cleanup"]["reason"] is None


async def test_an_opted_out_session_is_untouched(cleaned: CleanupApp) -> None:
    payload = await cleaned.run(cleanup="off")
    assert payload["transcript"] == "We was going to leave early but the train was late"
    assert payload["original_transcript"] is None
    assert payload["cleanup"] is None
    assert cleaned.runtime.calls == []


async def test_raw_style_bypasses_cleanup_even_with_it_enabled(cleaned: CleanupApp) -> None:
    payload = await cleaned.run(style="raw")
    assert payload["transcript"] == SPOKEN
    assert payload["cleanup"]["status"] == "skipped"
    assert payload["cleanup"]["reason"] == "raw_style"
    assert cleaned.runtime.calls == []


async def test_a_failed_correction_still_returns_the_transcript(
    settings: Settings,
) -> None:
    runtime = FakeCleanupRuntime(TimeoutError())
    app = create_app(settings, engine=FakeEngine(SPOKEN), normalizer=FakeNormalizer())
    enable_cleanup(app, runtime)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        payload = await CleanupApp(client, app, runtime).run()
    assert payload["transcript"] == "We was going to leave early but the train was late"
    assert payload["cleanup"]["status"] == "fallback"
    assert payload["cleanup"]["reason"] == "timeout"
    # The original stays recoverable even when the correction failed.
    assert payload["original_transcript"] == SPOKEN


async def test_auto_language_is_not_assumed_to_be_english(cleaned: CleanupApp) -> None:
    """A Latin-script transcript could be any of dozens of languages.

    Resolving `auto` to English because the letters look familiar is exactly the
    mistake that would send a French dictation to an English-tuned corrector, so
    an unresolved `auto` falls back instead.
    """
    payload = await cleaned.run(language="auto")
    assert payload["cleanup"]["reason"] == "unsupported_language"
    assert payload["transcript"] == "We was going to leave early but the train was late"
    assert cleaned.runtime.calls == []


async def test_an_invalid_preference_is_a_validation_error(cleaned: CleanupApp) -> None:
    response = await cleaned.create(cleanup="aggressive")
    assert response.status_code == HTTP_422_UNPROCESSABLE_CONTENT


async def test_the_snapshot_is_pinned_at_creation(cleaned: CleanupApp) -> None:
    """A settings change between recording and finishing must not move the goalposts."""
    created = (await cleaned.create()).json()
    await cleaned.upload(created["session_id"])
    manager = cleaned.app.state.ctx.cleanup
    manager.runtime_config.cleanup_enabled = False
    payload = (await cleaned.finish(created["session_id"])).json()
    # The session was created under `conservative` and still says so, but the
    # runtime is gone, so it falls back rather than silently correcting.
    assert payload["cleanup"]["requested"] == "conservative"
    assert payload["cleanup"]["status"] == "fallback"
    assert payload["transcript"] == "We was going to leave early but the train was late"


async def test_repeating_create_keeps_the_first_decision(cleaned: CleanupApp) -> None:
    session_id = str(uuid4())
    first = (await cleaned.create(client_session_id=session_id, cleanup="off")).json()
    second = (await cleaned.create(client_session_id=session_id, cleanup="conservative")).json()
    assert second["job_id"] == first["job_id"]
    stored = cleaned.repository.get(UUID(session_id))
    assert stored is not None
    assert stored.cleanup.mode == "off"


async def test_finishing_twice_never_calls_the_model_twice(cleaned: CleanupApp) -> None:
    created = (await cleaned.create()).json()
    await cleaned.upload(created["session_id"])
    first = (await cleaned.finish(created["session_id"])).json()
    second = (await cleaned.finish(created["session_id"])).json()
    retried = await cleaned.client.post(
        f"{SESSIONS}/{created['session_id']}/retry", headers=cleaned.auth
    )
    assert second == first
    assert retried.json() == first
    assert len(cleaned.runtime.calls) == 1


async def test_a_stored_result_survives_the_model_being_switched(cleaned: CleanupApp) -> None:
    created = (await cleaned.create()).json()
    await cleaned.upload(created["session_id"])
    first = (await cleaned.finish(created["session_id"])).json()
    manager = cleaned.app.state.ctx.cleanup
    manager.runtime_config.cleanup_model = "cleanup:qwen3-1.7b"
    again = (await cleaned.finish(created["session_id"])).json()
    assert again == first


async def test_two_concurrent_finishes_start_one_transcription(cleaned: CleanupApp) -> None:
    """The state test lives inside the UPDATE, so exactly one caller can win."""
    created = (await cleaned.create()).json()
    await cleaned.upload(created["session_id"])
    provider = cleaned.app.state.ctx.engine_provider
    engine = provider.current()
    responses = await asyncio.gather(
        cleaned.finish(created["session_id"]),
        cleaned.finish(created["session_id"]),
    )
    codes = sorted(response.status_code for response in responses)
    assert engine.calls == 1
    assert codes[0] == HTTP_200_OK
    # The loser either sees the finished result or the in-progress conflict —
    # never a second job.
    assert codes[1] in {HTTP_200_OK, HTTP_409_CONFLICT, 503}


def _blocking_engine(started: asyncio.Event, release: asyncio.Event) -> FakeEngine:
    class SlowEngine(FakeEngine):
        async def transcribe(self, audio_path: Path, options: Any) -> str:
            started.set()
            await release.wait()
            return SPOKEN

    return SlowEngine(SPOKEN)


async def _create_uploaded_session(harness: CleanupApp) -> str:
    created = (await harness.create()).json()
    session_id = created["session_id"]
    await harness.upload(session_id)
    return session_id


async def _finish_after_midflight(
    started: asyncio.Event,
    release: asyncio.Event,
    finishing: asyncio.Task[Any],
    midflight: Any,
) -> None:
    await started.wait()
    await midflight
    release.set()
    await finishing


async def test_a_session_deleted_mid_flight_is_not_resurrected(
    settings: Settings,
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    app = create_app(
        settings, engine=_blocking_engine(started, release), normalizer=FakeNormalizer()
    )
    enable_cleanup(app, FakeCleanupRuntime(CORRECTED))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        harness = CleanupApp(client, app, FakeCleanupRuntime())
        session_id = await _create_uploaded_session(harness)
        finishing = asyncio.create_task(harness.finish(session_id))
        await _finish_after_midflight(
            started,
            release,
            finishing,
            client.delete(f"{SESSIONS}/{session_id}", headers=harness.auth),
        )
        lookup = await client.get(f"{SESSIONS}/{session_id}", headers=harness.auth)
    assert lookup.status_code == 404


async def test_a_new_upload_is_not_overwritten_by_a_stale_job(settings: Settings) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    app = create_app(
        settings, engine=_blocking_engine(started, release), normalizer=FakeNormalizer()
    )
    enable_cleanup(app, FakeCleanupRuntime(CORRECTED))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        harness = CleanupApp(client, app, FakeCleanupRuntime())
        session_id = await _create_uploaded_session(harness)
        finishing = asyncio.create_task(harness.finish(session_id))
        await _finish_after_midflight(
            started,
            release,
            finishing,
            harness.upload(session_id),
        )
        lookup = (await client.get(f"{SESSIONS}/{session_id}", headers=harness.auth)).json()
    # The re-upload wins: the session is waiting to be transcribed again, not
    # holding a transcript of audio that has been replaced.
    assert lookup["state"] == "uploaded"
    assert lookup["transcript"] is None


LEGACY_SCHEMA = """
    CREATE TABLE sessions (
        session_id TEXT PRIMARY KEY,
        job_id TEXT NOT NULL UNIQUE,
        state TEXT NOT NULL,
        language TEXT NOT NULL,
        style TEXT NOT NULL,
        audio_name TEXT,
        transcript TEXT,
        error_code TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
"""


def test_the_migration_is_additive_and_idempotent(tmp_path: Path) -> None:
    database = tmp_path / "sessions.sqlite3"
    session_id = str(uuid4())
    with sqlite3.connect(database) as connection:
        connection.execute(LEGACY_SCHEMA)
        connection.execute(
            "INSERT INTO sessions VALUES (?, 'job', 'completed', 'en', 'casual', NULL,"
            " 'Kept exactly.', NULL, '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00')",
            (session_id,),
        )
    repository = storage.SessionRepository(database)
    repository.initialize()
    repository.initialize()

    stored = repository.get(UUID(session_id))
    assert stored is not None
    assert stored.transcript == "Kept exactly."
    # A legacy row migrates to cleanup off, and reports no manufactured original.
    assert stored.cleanup.mode == "off"
    assert stored.original_transcript is None


def test_completing_requires_the_job_to_still_own_the_session(tmp_path: Path) -> None:
    repository = storage.SessionRepository(tmp_path / "sessions.sqlite3")
    repository.initialize()
    session_id = uuid4()
    repository.create_or_get(session_id, "en", "casual")
    repository.update(session_id, state="uploaded", audio_name="clip.wav")

    assert repository.claim_transcribing(session_id) is not None
    # A second claim finds the session already owned.
    assert repository.claim_transcribing(session_id) is None

    completed = repository.complete(
        session_id,
        transcript="Final.",
        original_transcript="final",
        cleanup=storage.CleanupRecord(status="applied", reason=None, duration_ms=12),
    )
    assert completed is not None
    assert completed.original_transcript == "final"
    assert completed.cleanup_result.status == "applied"
    # Completing again matches no row rather than overwriting the stored answer.
    assert (
        repository.complete(
            session_id,
            transcript="Other.",
            original_transcript=None,
            cleanup=storage.CleanupRecord(),
        )
        is None
    )


def test_a_failure_is_only_recorded_against_a_running_job(tmp_path: Path) -> None:
    repository = storage.SessionRepository(tmp_path / "sessions.sqlite3")
    repository.initialize()
    session_id = uuid4()
    repository.create_or_get(session_id, "en", "casual")
    repository.update(session_id, state="uploaded", audio_name="clip.wav")
    assert repository.fail_transcribing(session_id, "boom") is None
    repository.claim_transcribing(session_id)
    assert repository.fail_transcribing(session_id, "boom") is not None
