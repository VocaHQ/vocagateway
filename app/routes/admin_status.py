from __future__ import annotations

from fastapi import APIRouter, Depends, Response
from fastapi.responses import HTMLResponse

from app.admin_queries import config_response, status_payload
from app.context import GatewayContextDependency, require_token
from app.diagnostics import build_diagnostics_bundle
from app.engine_state import engine_id
from app.fragments import about, engine, exposure, overview
from app.schemas import AdminStatusResponse, DiagnosticsBundle, EngineStatus, ReadinessStatus
from app.serializers import metrics_status

router = APIRouter(dependencies=[Depends(require_token)])


@router.get("/v1/admin/status", response_model=AdminStatusResponse)
async def get_admin_status(ctx: GatewayContextDependency) -> AdminStatusResponse:
    return await status_payload(ctx)


@router.get("/v1/admin/diagnostics", response_model=DiagnosticsBundle)
async def get_admin_diagnostics(
    response: Response, ctx: GatewayContextDependency
) -> DiagnosticsBundle:
    bundle = build_diagnostics_bundle(await status_payload(ctx), config_response(ctx))
    created_at = bundle.generated_at.strftime("%Y%m%dT%H%M%SZ")
    filename = f"vocagateway-diagnostics-{created_at}.json"
    response.headers["Content-Disposition"] = f'attachment; filename="{filename}"'
    return bundle


@router.get("/ui/partials/overview", response_class=HTMLResponse)
async def ui_overview(ctx: GatewayContextDependency) -> HTMLResponse:
    status = await status_payload(ctx)
    return HTMLResponse(overview.overview_fragment(status))


@router.get("/ui/partials/about", response_class=HTMLResponse)
async def ui_about(ctx: GatewayContextDependency) -> HTMLResponse:
    status = await status_payload(ctx)
    return HTMLResponse(about.about_fragment(status.version, status.commit))


@router.get("/ui/partials/operations", response_class=HTMLResponse)
async def ui_operations(ctx: GatewayContextDependency) -> HTMLResponse:
    # sample=True appends a ring-buffer point for sparklines (~every 5s poll).
    metrics = ctx.service.metrics.snapshot(sample=True)
    readiness_details = await ctx.readiness.details()
    return HTMLResponse(
        overview.operations_fragment(
            metrics_status(metrics),
            ReadinessStatus(
                probe_age_seconds=round(readiness_details.checked_age_seconds, 3),
                warmup_state=readiness_details.warmup_state,
                warmed_bytes=readiness_details.warmed_bytes,
            ),
        )
    )


@router.get("/ui/partials/engine-pill", response_class=HTMLResponse)
async def ui_engine_pill(ctx: GatewayContextDependency) -> HTMLResponse:
    state = await ctx.readiness.probe()
    return HTMLResponse(
        engine.engine_pill_fragment(
            EngineStatus(id=engine_id(ctx), name=state.name, ready=state.ready),
            bind_host=ctx.settings.bind_host,
            port=ctx.settings.port,
        )
    )


@router.get("/ui/partials/exposure-banner", response_class=HTMLResponse)
async def ui_exposure_banner(ctx: GatewayContextDependency) -> HTMLResponse:
    import platform

    return HTMLResponse(
        exposure.exposure_banner_fragment(
            ctx.settings.bind_host,
            is_mac=platform.system() == "Darwin",
        )
    )
