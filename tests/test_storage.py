from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest

from app.storage import SessionRepository

STALE_SESSION_AGE_HOURS = 48


@pytest.fixture
def repository(tmp_path: Path) -> SessionRepository:
    repo = SessionRepository(tmp_path / "nested" / "sessions.sqlite3")
    repo.initialize()
    return repo


def test_initialize_creates_the_database_fi_aa(
    repository: SessionRepository,
) -> None:
    assert repository.database_path.is_file()


def test_create_or_get_is_idempotent_for_th_d1296(
    repository: SessionRepository,
) -> None:
    session_id = uuid4()

    first = repository.create_or_get(session_id, "en", "raw")
    second = repository.create_or_get(session_id, "de", "casual")

    assert first.job_id == second.job_id
    assert second.language == "en"
    assert second.style == "raw"
    assert second.state == "created"


def test_create_or_get_assigns_a_unique_job_aaa(
    repository: SessionRepository,
) -> None:
    first = repository.create_or_get(uuid4(), "en", "raw")
    second = repository.create_or_get(uuid4(), "en", "raw")

    assert first.job_id != second.job_id


def test_get_returns_none_for_an_unknown_session(repository: SessionRepository) -> None:
    assert repository.get(uuid4()) is None


def test_update_preserves_audio_name_by_default(repository: SessionRepository) -> None:
    session_id = uuid4()
    repository.create_or_get(session_id, "en", "raw")
    repository.update(session_id, state="processing", audio_name="clip.wav")

    updated = repository.update(session_id, state="completed", transcript="hello")

    assert updated.audio_name == "clip.wav"
    assert updated.transcript == "hello"
    assert updated.state == "completed"


def test_update_can_overwrite_the_audio_nam_aaaa(repository: SessionRepository) -> None:
    session_id = uuid4()
    repository.create_or_get(session_id, "en", "raw")
    repository.update(session_id, state="processing", audio_name="first.wav")

    updated = repository.update(
        session_id, state="processing", audio_name="second.wav", preserve_audio_name=False
    )

    assert updated.audio_name == "second.wav"


def test_update_bumps_updated_at(repository: SessionRepository) -> None:
    session_id = uuid4()
    created = repository.create_or_get(session_id, "en", "raw")

    updated = repository.update(session_id, state="failed", error_code="engine_unavailable")

    assert updated.updated_at >= created.updated_at
    assert updated.error_code == "engine_unavailable"


def test_update_raises_key_error_for_an_unk_aaaaa(repository: SessionRepository) -> None:
    with pytest.raises(KeyError):
        repository.update(uuid4(), state="completed")


def test_delete_removes_the_session_and_ret_fdaac(
    repository: SessionRepository,
) -> None:
    session_id = uuid4()
    repository.create_or_get(session_id, "en", "raw")

    deleted = repository.delete(session_id)

    assert deleted is not None
    assert deleted.session_id == session_id
    assert repository.get(session_id) is None


def test_delete_returns_none_for_an_unknown_a(repository: SessionRepository) -> None:
    assert repository.delete(uuid4()) is None


def test_expired_only_returns_sessions_past_aa(
    repository: SessionRepository,
) -> None:
    fresh_id = uuid4()
    stale_id = uuid4()
    repository.create_or_get(fresh_id, "en", "raw")
    repository.create_or_get(stale_id, "en", "raw")

    stale_time = (datetime.now(UTC) - timedelta(hours=STALE_SESSION_AGE_HOURS)).isoformat()
    with repository._connect() as connection:
        connection.execute(
            "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
            (stale_time, str(stale_id)),
        )

    expired = repository.expired(retention_hours=24)

    assert [session.session_id for session in expired] == [stale_id]
