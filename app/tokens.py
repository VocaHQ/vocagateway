"""Named, individually revocable bearer tokens for additional paired devices.

`Settings.token` (from `VOCAGATEWAY_TOKEN` / the token file) remains a permanent
bootstrap credential managed outside this store — whoever controls that file
or environment variable can always read/rotate it directly, so trying to
revoke it through the API would be theatre. Everything issued through
`TokenStore` sits alongside it and can be revoked independently, so losing one
phone means revoking one token rather than rotating everyone else's.

Only a SHA-256 digest of each token is ever persisted; the plaintext is
returned once, at creation, and is not recoverable afterward. To let a device
token's own QR be regenerated (for example after changing the pairing
address) without weakening that guarantee, the store also keeps an in-memory,
never-persisted cache of recently created plaintexts, cleared on revoke and
naturally gone on restart.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeGuard


@dataclass(frozen=True, slots=True)
class DeviceToken:
    id: str
    label: str
    token_hash: str
    created_at: datetime


TOKEN_SECRET_BYTES = 32


def _hash_token(token_text: str) -> str:
    return hashlib.sha256(token_text.encode("utf-8")).hexdigest()


class _TokenStorage:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._tokens: list[DeviceToken] = _load_tokens(path)
        self._plaintext_cache: dict[str, str] = {}

    def all(self) -> list[DeviceToken]:
        return list(self._tokens)

    def cached_entries(self) -> list[DeviceToken]:
        """Device tokens whose plaintext is still available for QR display."""
        return [token for token in self._tokens if token.id in self._plaintext_cache]

    def cached_plaintext(self, token_id: str) -> str | None:
        """The plaintext for *token_id*, if created (and not revoked) in this process."""
        return self._plaintext_cache.get(token_id)

    def get(self, token_id: str) -> DeviceToken | None:
        return next((token for token in self._tokens if token.id == token_id), None)

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {
                "id": token.id,
                "label": token.label,
                "token_hash": token.token_hash,
                "created_at": token.created_at.isoformat(),
            }
            for token in self._tokens
        ]
        descriptor, temporary_name = tempfile.mkstemp(
            dir=self._path.parent, prefix=".device-tokens-", suffix=".tmp"
        )
        self._write_and_replace(descriptor, temporary_name, payload)

    def _write_and_replace(
        self, descriptor: int, temp_name: str, payload: list[dict[str, str]]
    ) -> None:
        with os.fdopen(descriptor, "w", encoding="utf-8") as token_file:
            json.dump(payload, token_file, indent=2)
            token_file.write("\n")
        try:
            os.replace(temp_name, self._path)
        except BaseException:
            Path(temp_name).unlink(missing_ok=True)
            raise


class TokenStore(_TokenStorage):
    def create(self, label: str) -> tuple[DeviceToken, str]:
        plaintext = secrets.token_urlsafe(TOKEN_SECRET_BYTES)
        record = DeviceToken(
            id=secrets.token_hex(8),
            label=label.strip() or "Unnamed device",
            token_hash=_hash_token(plaintext),
            created_at=datetime.now(UTC),
        )
        self._tokens.append(record)
        self._save()
        self._plaintext_cache[record.id] = plaintext
        return record, plaintext

    def matches(self, candidate: str) -> bool:
        if not candidate:
            return False
        candidate_hash = _hash_token(candidate)
        return any(hmac.compare_digest(token.token_hash, candidate_hash) for token in self._tokens)

    def revoke(self, token_id: str) -> bool:
        remaining = [token for token in self._tokens if token.id != token_id]
        if len(remaining) == len(self._tokens):
            return False
        self._tokens = remaining
        self._save()
        self._plaintext_cache.pop(token_id, None)
        return True

    def rotate(self, token_id: str) -> tuple[DeviceToken, str] | None:
        """Replace *token_id*'s secret in place, keeping its id and label."""
        for index, token in enumerate(self._tokens):
            if token.id != token_id:
                continue
            plaintext = secrets.token_urlsafe(TOKEN_SECRET_BYTES)
            updated = DeviceToken(
                id=token.id,
                label=token.label,
                token_hash=_hash_token(plaintext),
                created_at=datetime.now(UTC),
            )
            self._tokens[index] = updated
            self._save()
            self._plaintext_cache[token.id] = plaintext
            return updated, plaintext
        return None


def _load_tokens(path: Path) -> list[DeviceToken]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, list):
        return []
    tokens = [_parse_device_token(entry) for entry in payload]
    return [token for token in tokens if token is not None]


def _parse_device_token(entry: Any) -> DeviceToken | None:
    if not isinstance(entry, dict):
        return None
    token_id = entry.get("id")
    label = entry.get("label")
    token_hash = entry.get("token_hash")
    raw_created_at = entry.get("created_at")
    if not (_valid_str(token_id) and _valid_str(label) and _valid_str(token_hash)):
        return None
    if not isinstance(raw_created_at, str):
        return None
    try:
        created_at = datetime.fromisoformat(raw_created_at)
    except ValueError:
        return None
    return DeviceToken(id=token_id, label=label, token_hash=token_hash, created_at=created_at)


def _valid_str(candidate: Any) -> TypeGuard[str]:
    return isinstance(candidate, str) and bool(candidate)
