from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Form, Query, Response
from fastapi.responses import HTMLResponse

from app.context import GatewayContextDependency, require_token
from app.pairing import (
    decode_pairing_payload,
    discover_gateway_base_urls,
    encode_pairing_payload,
    qr_svg_for_payload,
)
from app.pairing_view import (
    forget_pairing_url,
    pairing_html,
    resolve_pairing_token,
    resolve_pairing_url,
)

router = APIRouter(dependencies=[Depends(require_token)])
OptionalUrlQuery = Annotated[str | None, Query()]
OptionalTokenIdQuery = Annotated[str | None, Query()]
PairingLabelForm = Annotated[str, Form(min_length=1, max_length=100)]
OptionalUrlForm = Annotated[str | None, Form()]
PairingUrlQuery = Annotated[str, Query()]


@router.get("/v1/admin/pairing")
async def get_pairing(
    ctx: GatewayContextDependency,
    url: OptionalUrlQuery = None,
    token_id: OptionalTokenIdQuery = None,
) -> dict[str, Any]:
    selected = resolve_pairing_url(ctx, url)
    _, token, _, _ = resolve_pairing_token(ctx, token_id)
    payload = encode_pairing_payload(selected, token)
    # Round-trip so clients and tests share one format.
    decoded = decode_pairing_payload(payload)
    return {
        "version": decoded.version,
        "url": decoded.url,
        "payload": payload,
        "candidates": discover_gateway_base_urls(ctx.settings.port),
    }


@router.get("/v1/admin/pairing/qr.svg", response_class=Response)
async def get_pairing_qr(
    ctx: GatewayContextDependency,
    url: OptionalUrlQuery = None,
    token_id: OptionalTokenIdQuery = None,
) -> Response:
    selected = resolve_pairing_url(ctx, url)
    _, token, _, _ = resolve_pairing_token(ctx, token_id)
    payload = encode_pairing_payload(selected, token)
    svg = qr_svg_for_payload(payload)
    return Response(
        content=svg,
        media_type="image/svg+xml",
        headers={"Cache-Control": "no-store"},
    )


@router.get("/ui/partials/pairing", response_class=HTMLResponse)
async def ui_pairing(
    ctx: GatewayContextDependency,
    url: OptionalUrlQuery = None,
    token_id: OptionalTokenIdQuery = None,
) -> HTMLResponse:
    return HTMLResponse(pairing_html(ctx, url, token_id, persist=True))


@router.post("/ui/partials/pairing/tokens", response_class=HTMLResponse)
async def ui_create_pairing_token(
    label: PairingLabelForm,
    ctx: GatewayContextDependency,
    url: OptionalUrlForm = None,
) -> HTMLResponse:
    record, _ = ctx.token_store.create(label)
    return HTMLResponse(pairing_html(ctx, url, record.id, persist=True))


@router.post("/ui/partials/pairing/tokens/{token_id}/rotate", response_class=HTMLResponse)
async def ui_rotate_pairing_token(
    token_id: str,
    ctx: GatewayContextDependency,
    url: OptionalUrlForm = None,
) -> HTMLResponse:
    rotated = ctx.token_store.rotate(token_id)
    resolved_id = token_id if rotated is None else rotated[0].id
    return HTMLResponse(pairing_html(ctx, url, resolved_id, persist=True))


@router.delete("/ui/partials/pairing", response_class=HTMLResponse)
async def ui_forget_pairing(url: PairingUrlQuery, ctx: GatewayContextDependency) -> HTMLResponse:
    return HTMLResponse(forget_pairing_url(ctx, url))
