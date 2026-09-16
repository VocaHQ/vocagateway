# Copyright (c) 2025-present VocaHQ, Inc.
# SPDX-License-Identifier: AGPL-3.0-only
# See the LICENSE file for details.

from __future__ import annotations

import secrets
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

_KEEP_AUDIO: object = object()
TRANSCRIBING_STATE = "transcribing"
COMPLETED_STATE = "completed"
# The states a `finish` may take a session out of. Deliberately not
# `transcribing` (a job already holds it) and not `completed` (its stored result
# is the answer), which is what makes the claim below idempotent under a race.
CLAIMABLE_STATES = ("created", "uploaded", "failed")


@dataclass(frozen=True, slots=True)
class CleanupSnapshot:
    """The cleanup decision a session was created under.

    Pinned at creation so a settings change between recording and finishing
    cannot move a transcript to a different model, or turn correction on for a
    session whose client never asked for it. Legacy rows read back as `off`.
    """

    mode: str = "off"
    model_id: str | None = None
    prompt_version: str | None = None
    timeout_ms: int | None = None


@dataclass(frozen=True, slots=True)
class CleanupRecord:
    """What cleanup actually did, stored beside the transcript it produced."""

    status: str | None = None
    reason: str | None = None
    duration_ms: int | None = None


@dataclass(slots=True)
class StoredSession:
    session_id: UUID
    job_id: str
    state: str
    language: str
    style: str
    audio_name: str | None
    transcript: str | None
    error_code: str | None
    created_at: datetime
    updated_at: datetime
    # The recognised text before any correction. Present only for a session that
    # opted in, kept under exactly the same retention and deletion rules as the
    # transcript, and never manufactured by treating a styled transcript as raw.
    original_transcript: str | None = None
    cleanup: CleanupSnapshot = field(default_factory=CleanupSnapshot)
    cleanup_result: CleanupRecord = field(default_factory=CleanupRecord)


TEXT_COLUMN = "TEXT"
INTEGER_COLUMN = "INTEGER"


def _connect(database_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(database_path, timeout=10)
    connection.row_factory = sqlite3.Row
    return connection


def _from_row(row: sqlite3.Row) -> StoredSession:
    return StoredSession(
        session_id=UUID(row["session_id"]),
        job_id=row["job_id"],
        state=row["state"],
        language=row["language"],
        style=row["style"],
        audio_name=row["audio_name"],
        transcript=row["transcript"],
        error_code=row["error_code"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
        original_transcript=row["original_transcript"],
        cleanup=CleanupSnapshot(
            mode=row["cleanup_mode"] or "off",
            model_id=row["cleanup_model_id"],
            prompt_version=row["cleanup_prompt_version"],
            timeout_ms=row["cleanup_timeout_ms"],
        ),
        cleanup_result=CleanupRecord(
            status=row["cleanup_status"],
            reason=row["cleanup_reason"],
            duration_ms=row["cleanup_duration_ms"],
        ),
    )


# Added by migration, never by a table rewrite: every existing row and every
# transcript string stays exactly as it was, and a rollback to a binary that
# does not know these columns still reads the table.
_ADDED_COLUMNS: tuple[tuple[str, str], ...] = (
    ("original_transcript", TEXT_COLUMN),
    ("cleanup_mode", TEXT_COLUMN),
    ("cleanup_model_id", TEXT_COLUMN),
    ("cleanup_prompt_version", TEXT_COLUMN),
    ("cleanup_timeout_ms", INTEGER_COLUMN),
    ("cleanup_status", TEXT_COLUMN),
    ("cleanup_reason", TEXT_COLUMN),
    ("cleanup_duration_ms", INTEGER_COLUMN),
)


def _migrate(connection: sqlite3.Connection) -> None:
    """Add any missing cleanup column. Idempotent, and safe to run every start."""
    existing = {row["name"] for row in connection.execute("PRAGMA table_info(sessions)")}
    for name, column_type in _ADDED_COLUMNS:
        if name not in existing:
            connection.execute(f"ALTER TABLE sessions ADD COLUMN {name} {column_type}")


class SessionRepository:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
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
            )
            _migrate(connection)

    def create_or_get(
        self,
        session_id: UUID,
        language: str,
        style: str,
        cleanup: CleanupSnapshot | None = None,
    ) -> StoredSession:
        """First write wins, including for the cleanup snapshot.

        A repeated create with the same id returns the original session and the
        options it was created under — a client that retries with a different
        preference does not get to reopen a decision the first call settled.
        """
        existing = self.get(session_id)
        if existing is not None:
            return existing
        snapshot = cleanup or CleanupSnapshot()
        now = datetime.now(UTC)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO sessions
                (session_id, job_id, state, language, style, created_at, updated_at,
                 cleanup_mode, cleanup_model_id, cleanup_prompt_version, cleanup_timeout_ms)
                VALUES (?, ?, 'created', ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(session_id),
                    secrets.token_urlsafe(24),
                    language,
                    style,
                    now.isoformat(),
                    now.isoformat(),
                    snapshot.mode,
                    snapshot.model_id,
                    snapshot.prompt_version,
                    snapshot.timeout_ms,
                ),
            )
        created_session = self.get(session_id)
        if created_session is None:
            raise RuntimeError("Session creation failed")
        return created_session

    def get(self, session_id: UUID) -> StoredSession | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (str(session_id),)
            ).fetchone()
        return _from_row(row) if row else None

    def update(
        self,
        session_id: UUID,
        *,
        state: str,
        audio_name: str | None | object = _KEEP_AUDIO,
        transcript: str | None = None,
        error_code: str | None = None,
    ) -> StoredSession:
        current = self.get(session_id)
        if current is None:
            raise KeyError(session_id)
        chosen_audio = current.audio_name if audio_name is _KEEP_AUDIO else audio_name
        now = datetime.now(UTC)
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE sessions
                SET state = ?, audio_name = ?, transcript = ?, error_code = ?, updated_at = ?
                WHERE session_id = ?
                """,
                (
                    state,
                    chosen_audio,
                    transcript,
                    error_code,
                    now.isoformat(),
                    str(session_id),
                ),
            )
        updated_session = self.get(session_id)
        if updated_session is None:
            raise RuntimeError("Session update failed")
        return updated_session

    def claim_transcribing(self, session_id: UUID) -> StoredSession | None:
        """Move one session into `transcribing`, atomically, or return None.

        A read-then-update would let two concurrent `finish` calls both see an
        `uploaded` session and both start a job. The state test lives inside the
        statement instead, so exactly one caller wins and the loser follows the
        ordinary in-progress path rather than starting a second transcription.
        """
        placeholders = ", ".join("?" for _ in CLAIMABLE_STATES)
        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                f"""
                UPDATE sessions SET state = ?, error_code = NULL, updated_at = ?
                WHERE session_id = ? AND state IN ({placeholders})
                """,
                (TRANSCRIBING_STATE, now, str(session_id), *CLAIMABLE_STATES),
            )
            claimed = cursor.rowcount == 1
        return self.get(session_id) if claimed else None

    def complete(
        self,
        session_id: UUID,
        *,
        transcript: str,
        original_transcript: str | None,
        cleanup: CleanupRecord,
        audio_name: str | None | object = _KEEP_AUDIO,
    ) -> StoredSession | None:
        """Store the final text, the original, and the cleanup result together.

        One statement, one transaction, and only from `transcribing`. A session
        deleted or re-uploaded while its job was running therefore cannot be
        resurrected or overwritten by that job's stale result: the update simply
        matches no row and the caller is told so.
        """
        current = self.get(session_id)
        if current is None or current.state != TRANSCRIBING_STATE:
            return None
        chosen_audio = current.audio_name if audio_name is _KEEP_AUDIO else audio_name
        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE sessions
                SET state = ?, audio_name = ?, transcript = ?, original_transcript = ?,
                    error_code = NULL, updated_at = ?,
                    cleanup_status = ?, cleanup_reason = ?, cleanup_duration_ms = ?
                WHERE session_id = ? AND state = ?
                """,
                (
                    COMPLETED_STATE,
                    chosen_audio,
                    transcript,
                    original_transcript,
                    now,
                    cleanup.status,
                    cleanup.reason,
                    cleanup.duration_ms,
                    str(session_id),
                    TRANSCRIBING_STATE,
                ),
            )
            if cursor.rowcount != 1:
                return None
        return self.get(session_id)

    def fail_transcribing(self, session_id: UUID, error_code: str) -> StoredSession | None:
        """Record a failure, but only against the job that is actually running."""
        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE sessions SET state = 'failed', error_code = ?, updated_at = ?
                WHERE session_id = ? AND state = ?
                """,
                (error_code, now, str(session_id), TRANSCRIBING_STATE),
            )
            if cursor.rowcount != 1:
                return None
        return self.get(session_id)

    def delete(self, session_id: UUID) -> StoredSession | None:
        current = self.get(session_id)
        if current is None:
            return None
        with self._connect() as connection:
            connection.execute("DELETE FROM sessions WHERE session_id = ?", (str(session_id),))
        return current

    def expired(self, retention_hours: int) -> list[StoredSession]:
        cutoff = datetime.now(UTC) - timedelta(hours=retention_hours)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM sessions WHERE updated_at < ?", (cutoff.isoformat(),)
            ).fetchall()
        return [_from_row(row) for row in rows]

    def _connect(self) -> sqlite3.Connection:
        """One place the database is opened, so every statement shares its settings."""
        return _connect(self.database_path)
