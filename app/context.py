from __future__ import annotations

import hmac
from dataclasses import dataclass
from pathlib import Path

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.security.utils import get_authorization_scheme_param

from app.config import Settings
from app.engines import EngineManager, EngineProvider
from app.errors import APIProblem
from app.model_manager import ModelManager
from app.readiness import ReadinessMonitor
from app.runtime_config import RuntimeConfig
from app.service import TranscriptionService
from app.storage import SessionRepository
from app.tokens import TokenStore

VERSION = "0.1.0"
TOKEN_FILE_HINT = "~/.config/vocagateway/token"
BOOTSTRAP_TOKEN_ID = "bootstrap"

# Declaring the scheme (rather than reading the header by hand) is what puts an
# Authorize button in the /docs Swagger page and makes it attach
# `Authorization: Bearer <token>` to every Try-it-out request. auto_error=False
# keeps rejection in `require_token`, so a missing or malformed header still
# produces this API's 401 error envelope instead of Starlette's 403 default.
bearer_scheme = HTTPBearer(
    scheme_name="Bearer token",
    description=(
        "The gateway bearer token, or any device token. Paste the value alone — "
        f"Swagger adds the `Bearer ` prefix. Find it in `{TOKEN_FILE_HINT}`, or "
        "with `just token`."
    ),
    auto_error=False,
)


@dataclass
class GatewayContext:
    """The state every route needs, built once in `create_app()`.

    Stored on `app.state.ctx` and reached through the `get_context` dependency
    instead of route-module closures, so router modules can be plain,
    independently importable `APIRouter()` instances.
    """

    settings: Settings
    repository: SessionRepository
    manager: ModelManager
    token_store: TokenStore
    engine_provider: EngineProvider
    engine_manager: EngineManager | None
    service: TranscriptionService
    readiness: ReadinessMonitor
    pairing_config: RuntimeConfig
    config_path: Path

    def token_matches(self, supplied: str) -> bool:
        # Compared as bytes because `hmac.compare_digest` raises TypeError on
        # `str` arguments holding non-ASCII characters.
        if hmac.compare_digest(supplied.encode("utf-8"), self.settings.token.encode("utf-8")):
            return True
        return self.token_store.matches(supplied)

    def token_is_valid(self, authorization: str | None) -> bool:
        scheme, credentials = get_authorization_scheme_param(authorization)
        if scheme.lower() != "bearer" or not credentials:
            return False
        return self.token_matches(credentials)


def get_context(request: Request) -> GatewayContext:
    ctx: GatewayContext = request.app.state.ctx
    return ctx


def require_token(
    ctx: GatewayContext = Depends(get_context),
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> None:
    # credentials is None when the header is absent, empty, or carries a
    # non-bearer scheme; all three are the same 401 as a wrong token.
    if not ctx.token_matches(credentials.credentials if credentials else ""):
        raise APIProblem(401, "unauthorized", "A valid bearer token is required.")
