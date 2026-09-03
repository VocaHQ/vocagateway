"""Bearer authentication: the boundary every route and the WebSocket share.

Coverage elsewhere exercises auth incidentally (`test_admin.py` for device
tokens, `test_streaming.py` for the happy-path socket). This module owns the
boundary itself: what a credential may look like, which routes demand one,
which deliberately do not, and what a rejection is allowed to say back.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from conftest import TOKEN, FakeEngine, FakeNormalizer
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.status import (
    HTTP_200_OK,
    HTTP_401_UNAUTHORIZED,
    HTTP_409_CONFLICT,
    HTTP_503_SERVICE_UNAVAILABLE,
)
from starlette.websockets import WebSocketDisconnect

from app.config import Settings
from app.main import create_app

AUTHORIZATION_HEADER = "Authorization"
ADMIN_STATUS_PATH = "/v1/admin/status"

# Reachable without a credential by design: the phone polls health before it
# has been paired, and container orchestrators probe liveness with no secret.
PUBLIC_PATHS = frozenset({"/health", "/health/live", "/health/ready"})
MINIMUM_VALID_TOKEN_BYTES = 48
ONE_BYTE_SHORT_TOKEN_BYTES = 47
MINIMUM_PROTECTED_ROUTE_COUNT = 25


async def _assert_unauthorized_route(client: httpx.AsyncClient, method: str, path: str) -> None:
    response = await client.request(method.upper(), path)
    assert response.status_code == HTTP_401_UNAUTHORIZED, f"{method.upper()} {path}"
    assert response.json()["error"]["code"] == "unauthorized"


def _auth_header(token: str) -> dict[str, str]:
    return {AUTHORIZATION_HEADER: f"Bearer {token}"}


WEBSOCKET_UNAUTHORIZED_CODE = 4401


@pytest.fixture
def auth_settings(tmp_path: Path) -> Settings:
    return Settings(
        token=TOKEN,
        data_dir=tmp_path,
        whisper_binary=tmp_path / "whisper-cli",
        whisper_model=tmp_path / "model.bin",
        models_dir=tmp_path / "models",
        config_path=tmp_path / "config.json",
    )


@pytest.fixture
def auth_app(auth_settings: Settings) -> FastAPI:
    return create_app(auth_settings, engine=FakeEngine(), normalizer=FakeNormalizer())


@pytest.fixture
async def auth_client(auth_app: FastAPI) -> httpx.AsyncClient:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=auth_app),
        base_url="http://test",
    ) as client:
        yield client


@pytest.fixture
def sync_client(auth_app: FastAPI) -> Iterator[TestClient]:
    """Sync client, because `websocket_connect` has no async equivalent."""
    with TestClient(auth_app) as client:
        yield client


def websocket_close_code(client: TestClient, header: str | None) -> int | None:
    """Close code from `/v1/stream`, or None when the socket was accepted.

    The handler authenticates before it looks at engine capability, so an
    accepted socket (which then closes 4409 for this non-streaming engine)
    means the credential passed.
    """
    headers = {AUTHORIZATION_HEADER: header} if header is not None else {}
    try:
        with client.websocket_connect("/v1/stream", headers=headers):
            return None
    except WebSocketDisconnect as disconnect:
        return disconnect.code


# --------------------------------------------------------------- header forms


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param(b"Bearer \xe9", id="latin1-e-acute"),
        pytest.param(b"Bearer \xff", id="latin1-high-byte"),
        pytest.param(
            b"Bearer " + b"x" * ONE_BYTE_SHORT_TOKEN_BYTES + b"\xe9",
            id="right-length-wrong-bytes",
        ),
        pytest.param(b"\xe9", id="no-scheme"),
    ],
)
async def test_non_ascii_credential_is_rejected_r_aa(
    auth_client: httpx.AsyncClient, raw: bytes
) -> None:
    """A high byte in the header must be a 401, never a 500.

    Headers arrive latin-1-decoded per the HTTP spec, and `hmac.compare_digest`
    raises TypeError on `str` arguments holding non-ASCII characters. Comparing
    as `str` therefore let any unauthenticated client turn every authenticated
    route into a 500 with a stack trace, just by sending one high byte.
    """
    response = await auth_client.get(ADMIN_STATUS_PATH, headers=[(b"authorization", raw)])
    assert response.status_code == HTTP_401_UNAUTHORIZED
    assert response.json()["error"]["code"] == "unauthorized"


async def test_rejection_never_echoes_the_supplie_aaa(
    auth_client: httpx.AsyncClient,
) -> None:
    """A 401 body must not reflect the attempt back into logs or proxies."""
    wrong = "wrong-" + ("y" * MINIMUM_VALID_TOKEN_BYTES)
    response = await auth_client.get(
        ADMIN_STATUS_PATH, headers={AUTHORIZATION_HEADER: f"Bearer {wrong}"}
    )
    assert response.status_code == HTTP_401_UNAUTHORIZED
    assert wrong not in response.text
    assert TOKEN not in response.text


# ------------------------------------------------------- the protected surface


async def test_no_documented_route_answers_withou_aaaa(
    auth_app: FastAPI, auth_client: httpx.AsyncClient
) -> None:
    """Guard against a new router shipping without `require_token`.

    The inventory comes from the app's own schema rather than a hand-written
    list, so a route added tomorrow is covered by this test today.
    """
    schema = auth_app.openapi()
    # An empty or truncated schema must not let this pass by checking nothing.
    assert set(schema["paths"]) >= PUBLIC_PATHS
    checked = 0
    for path, operations in schema["paths"].items():
        if path in PUBLIC_PATHS:
            continue
        # Path params are irrelevant: the security dependency is solved before
        # any path/query/body validation, so a placeholder still yields 401.
        for method in operations:
            await _assert_unauthorized_route(
                auth_client,
                method,
                path.replace("{session_id}", "x")
                .replace("{model_id}", "x")
                .replace("{token_id}", "x"),
            )
            checked += 1
    assert checked > MINIMUM_PROTECTED_ROUTE_COUNT


async def test_public_routes_answer_without_a_token(
    auth_client: httpx.AsyncClient,
) -> None:
    """The complement of the test above: these must not regress into 401."""
    for path in sorted(PUBLIC_PATHS):
        response = await auth_client.get(path)
        assert response.status_code in {HTTP_200_OK, HTTP_503_SERVICE_UNAVAILABLE}, path
        assert TOKEN not in response.text, path


# ------------------------------------------------------------------ WebSocket


@pytest.mark.parametrize(
    "header",
    [
        pytest.param(None, id="absent"),
        pytest.param("", id="empty"),
        pytest.param(TOKEN, id="bare-token-no-scheme"),
        pytest.param(f"Basic {TOKEN}", id="wrong-scheme"),
        pytest.param("Bearer", id="scheme-only"),
        pytest.param("Bearer ", id="scheme-and-space"),
        pytest.param(f"Bearer {TOKEN}x", id="wrong-token"),
    ],
)
def test_websocket_rejects_bad_credentials(sync_client: TestClient, header: str | None) -> None:
    assert websocket_close_code(sync_client, header) == WEBSOCKET_UNAUTHORIZED_CODE


@pytest.mark.parametrize(
    "header",
    [
        pytest.param(f"Bearer {TOKEN}", id="canonical"),
        # RFC 7235 makes the scheme case-insensitive and `HTTPBearer` honours
        # it, so every HTTP route accepts this. The socket parsed the prefix by
        # hand and rejected it — the same credential worked or failed depending
        # on which transport the client happened to use.
        pytest.param(f"bearer {TOKEN}", id="lowercase-scheme"),
        pytest.param(f"BEARER {TOKEN}", id="uppercase-scheme"),
    ],
)
def test_websocket_accepts_the_scheme_http_aaaaa(sync_client: TestClient, header: str) -> None:
    assert websocket_close_code(sync_client, header) is None


def test_websocket_and_http_agree_on_every_d806a(sync_client: TestClient) -> None:
    """The two code paths must never diverge on what counts as authenticated."""
    forms = [
        f"Bearer {TOKEN}",
        f"bearer {TOKEN}",
        f"BEARER {TOKEN}",
        TOKEN,
        f"Basic {TOKEN}",
        "Bearer",
        "Bearer ",
        f"Bearer {TOKEN}x",
    ]
    for header in forms:
        http_ok = (
            sync_client.get(ADMIN_STATUS_PATH, headers={AUTHORIZATION_HEADER: header}).status_code
            != HTTP_401_UNAUTHORIZED
        )
        socket_ok = websocket_close_code(sync_client, header) is None
        assert http_ok == socket_ok, header


def test_websocket_accepts_a_device_token_u_b7078(
    sync_client: TestClient,
) -> None:
    auth = {AUTHORIZATION_HEADER: f"Bearer {TOKEN}"}
    created = sync_client.post("/v1/admin/tokens", headers=auth, json={"label": "Pixel 6a"})
    assert created.status_code == HTTP_200_OK
    device = created.json()
    device_header = f"Bearer {device['token']}"

    assert websocket_close_code(sync_client, device_header) is None

    revoked = sync_client.delete(f"/v1/admin/tokens/{device['id']}", headers=auth)
    assert revoked.status_code == HTTP_200_OK
    assert websocket_close_code(sync_client, device_header) == WEBSOCKET_UNAUTHORIZED_CODE
    # Revoking a device never disturbs the bootstrap credential.
    assert websocket_close_code(sync_client, f"Bearer {TOKEN}") is None


# ------------------------------------------------- device token lifecycle


async def test_rotating_a_device_token_invalidate_a(
    auth_client: httpx.AsyncClient,
) -> None:
    created = await auth_client.post(
        "/v1/admin/tokens", headers=_auth_header(TOKEN), json={"label": "Old phone"}
    )
    original = created.json()["token"]
    token_id = created.json()["id"]

    rotated = await auth_client.post(
        f"/v1/admin/tokens/{token_id}/rotate", headers=_auth_header(TOKEN)
    )
    assert rotated.status_code == HTTP_200_OK
    replacement = rotated.json()["token"]
    assert replacement != original
    assert rotated.json()["id"] == token_id

    assert (
        await auth_client.get(ADMIN_STATUS_PATH, headers=_auth_header(original))
    ).status_code == HTTP_401_UNAUTHORIZED
    assert (
        await auth_client.get(ADMIN_STATUS_PATH, headers=_auth_header(replacement))
    ).status_code == HTTP_200_OK


async def test_bootstrap_token_cannot_be_rotated_aa(
    auth_client: httpx.AsyncClient,
) -> None:
    """It lives in a file or the environment; rotating it here would be theatre."""
    auth = {AUTHORIZATION_HEADER: f"Bearer {TOKEN}"}
    response = await auth_client.post("/v1/admin/tokens/bootstrap/rotate", headers=auth)
    assert response.status_code == HTTP_409_CONFLICT
    assert response.json()["error"]["code"] == "bootstrap_token_not_rotatable"

    still_works = await auth_client.get(ADMIN_STATUS_PATH, headers=auth)
    assert still_works.status_code == HTTP_200_OK
