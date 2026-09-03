from __future__ import annotations

import hmac
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from fastapi import Depends, Request, security
from starlette import status

from app import (
    config,
    engines,
    errors,
    model_manager,
    readiness,
    runtime_config,
    service,
    storage,
)
from app.tokens import TokenStore

VERSION = "0.1.0"
TOKEN_FILE_HINT = "~/.config/vocagateway/token"
BOOTSTRAP_TOKEN_ID = "bootstrap"

# Declaring the scheme (rather than reading the header by hand) is what puts an
# Authorize button in the /docs Swagger page and makes it attach
# `Authorization: Bearer <token>` to every Try-it-out request. auto_error=False
# keeps rejection in `require_token`, so a missing or malformed header still
# produces this API's 401 error envelope instead of Starlette's 403 default.
bearer_scheme = security.HTTPBearer(
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

    settings: config.Settings
    repository: storage.SessionRepository
    manager: model_manager.ModelManager
    token_store: TokenStore
    engine_provider: engines.EngineProvider
    engine_manager: engines.EngineManager | None
    service: service.TranscriptionService
    readiness: readiness.ReadinessMonitor
    pairing_config: runtime_config.RuntimeConfig
    config_path: Path

    def token_matches(self, supplied: str) -> bool:
        # Compared as bytes because `hmac.compare_digest` raises TypeError on
        # `str` arguments holding non-ASCII characters.
        if hmac.compare_digest(supplied.encode("utf-8"), self.settings.token.encode("utf-8")):
            return True
        return self.token_store.matches(supplied)

    def token_is_valid(self, authorization: str | None) -> bool:
        scheme, credentials = security.utils.get_authorization_scheme_param(authorization)
        if scheme.lower() != "bearer" or not credentials:
            return False
        return self.token_matches(credentials)


def get_context(request: Request) -> GatewayContext:
    ctx: GatewayContext = request.app.state.ctx
    return ctx


GatewayContextDependency = Annotated[GatewayContext, Depends(get_context)]
TokenCredentialsDependency = Annotated[
    security.HTTPAuthorizationCredentials | None, Depends(bearer_scheme)
]


def require_token(
    ctx: GatewayContextDependency,
    credentials: TokenCredentialsDependency,
) -> None:
    # credentials is None when the header is absent, empty, or carries a
    # non-bearer scheme; all three are the same 401 as a wrong token.
    presented_token = "" if credentials is None else credentials.credentials
    if not ctx.token_matches(presented_token):
        raise errors.APIProblem(
            status.HTTP_401_UNAUTHORIZED,
            "unauthorized",
            "A valid bearer token is required.",
        )
