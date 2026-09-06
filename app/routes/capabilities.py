"""What this gateway can be asked for, so a client need not guess.

Deliberately descriptive rather than revealing: it names the modes and languages
a request may use and says nothing about where the runtime lives, which
executable serves it, or what the prompt says. A client talking to a gateway
that predates this endpoint gets a 404 and must then omit the new request
fields, because the session schema rejects unknown ones rather than ignoring
them.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.cleanup.base import RESOLVED_MODES
from app.context import VERSION, GatewayContextDependency, require_token
from app.schemas import CapabilitiesResponse, CleanupCapability
from app.text_styles import SUPPORTED_WRITING_STYLES

router = APIRouter(dependencies=[Depends(require_token)])

# Ordered for a picker rather than as a set, so two gateways list them the same.
STYLE_ORDER = ("raw", "clean", "formal", "casual", "very_casual", "excited")
OFF_MODE = "off"
CONSERVATIVE_MODE = "conservative"


@router.get("/v1/capabilities", response_model=CapabilitiesResponse)
async def capabilities(ctx: GatewayContextDependency) -> CapabilitiesResponse:
    return CapabilitiesResponse(
        version=VERSION,
        styles=[style for style in STYLE_ORDER if style in SUPPORTED_WRITING_STYLES],
        streaming_supported=_streaming_supported(ctx),
        cleanup=_cleanup_capability(ctx),
    )


def _streaming_supported(ctx: GatewayContextDependency) -> bool:
    engine = ctx.engine_provider.current()
    return bool(getattr(engine, "supports_streaming", False))


def _cleanup_capability(ctx: GatewayContextDependency) -> CleanupCapability:
    manager = ctx.cleanup
    if manager is None:
        return CleanupCapability(supported=False, enabled=False, default_mode="off", modes=["off"])
    report = manager.status()
    usable = report.runtime_available and report.model_installed
    return CleanupCapability(
        supported=usable,
        enabled=report.enabled and usable,
        # Always the resolved default, never `inherit`: `inherit` is something a
        # request says, not something a gateway is.
        default_mode=CONSERVATIVE_MODE if report.mode == CONSERVATIVE_MODE else OFF_MODE,
        modes=list(RESOLVED_MODES) if usable else [OFF_MODE],
        model_id=report.model_id,
        languages=list(report.languages),
        evaluated_languages=list(report.evaluated_languages),
        timeout_seconds=report.timeout_seconds,
    )
