from __future__ import annotations

import platform
import time
from dataclasses import dataclass
from typing import Annotated

from fastapi import APIRouter, Depends, Form, Header, Query, Request
from fastapi.responses import HTMLResponse
from starlette.status import HTTP_409_CONFLICT, HTTP_422_UNPROCESSABLE_CONTENT

from app import admin_queries, audio, errors, pairing_view, serializers, service
from app.cleanup.base import MODE_CONSERVATIVE, MODE_OFF, CleanupOptions, CleanupStatus
from app.context import TOKEN_FILE_HINT, GatewayContextDependency, require_token
from app.fragments import cleanup, settings, test_panel, tokens
from app.fragments.engine import engine_update_fragment
from app.runtime_config import DEFAULT_IDLE_OFFLOAD_MINUTES
from app.schemas import (
    ConfigResponse,
    ConfigUpdateRequest,
    EngineStatus,
    SelectModelResponse,
    TestTranscriptionResponse,
)

router = APIRouter(dependencies=[Depends(require_token)])
TestLanguageQuery = Annotated[str, Query(pattern=r"^[A-Za-z-]+$|^auto$")]
# The mic test always states its choice rather than inheriting the gateway
# default, so a raw benchmark stays raw whatever the operator has enabled.
TestCleanupQuery = Annotated[str, Query(pattern=r"^(?:off|conservative)$")]
ContentTypeHeader = Annotated[str | None, Header()]
ContentLengthHeader = Annotated[int | None, Header()]
EngineForm = Annotated[str, Form()]
ComputeDeviceForm = Annotated[str, Form()]
ComputeTypeForm = Annotated[str, Form()]
CpuThreadsForm = Annotated[int, Form()]
IdleOffloadForm = Annotated[bool, Form()]
IdleOffloadMinutesForm = Annotated[int, Form()]


@dataclass
class _ConfigForm:
    engine: EngineForm
    compute_device: ComputeDeviceForm = "auto"
    compute_type: ComputeTypeForm = "auto"
    cpu_threads: CpuThreadsForm = 0
    idle_offload_enabled: IdleOffloadForm = False
    idle_offload_minutes: IdleOffloadMinutesForm = DEFAULT_IDLE_OFFLOAD_MINUTES


@router.get("/v1/admin/config", response_model=ConfigResponse)
async def get_config(ctx: GatewayContextDependency) -> ConfigResponse:
    return admin_queries.config_response(ctx)


@router.put("/v1/admin/config", response_model=SelectModelResponse)
async def update_config(
    body: ConfigUpdateRequest, ctx: GatewayContextDependency
) -> SelectModelResponse:
    engine_manager = ctx.engine_manager
    if engine_manager is None:
        raise errors.APIProblem(
            HTTP_409_CONFLICT, "engine_locked", "The engine was fixed at startup and cannot switch."
        )
    try:
        engine_manager.configure(
            body.engine,
            body.compute_device,
            body.compute_type,
            body.cpu_threads,
            body.idle_offload_enabled,
            body.idle_offload_minutes,
        )
    except ValueError as error:
        raise errors.APIProblem(
            HTTP_422_UNPROCESSABLE_CONTENT, "invalid_engine", str(error)
        ) from error
    await ctx.readiness.warmup()
    state = await ctx.readiness.probe()
    return SelectModelResponse(
        engine=EngineStatus(id=body.engine, name=state.name, ready=state.ready)
    )


@router.post("/v1/admin/test-transcription", response_model=TestTranscriptionResponse)
async def test_transcription(
    request: Request,
    ctx: GatewayContextDependency,
    language: TestLanguageQuery = "auto",
    cleanup_mode: TestCleanupQuery = MODE_OFF,
    content_type: ContentTypeHeader = None,
    content_length: ContentLengthHeader = None,
) -> TestTranscriptionResponse:
    max_bytes = ctx.settings.maximum_upload_bytes
    suffix = audio.validate_audio_upload_headers(content_type, content_length, max_bytes)
    final = await audio.save_streamed_upload(
        request.stream(),
        ctx.settings.data_dir / "test-uploads",
        f"test-{int(time.time() * 1000)}",
        suffix,
        max_bytes,
    )
    style, options = _test_processing(ctx, cleanup_mode)
    try:
        outcome = await ctx.service.transcribe_adhoc(final, language, style=style, cleanup=options)
    except BaseException:
        # Includes a failed transcription: the uploaded clip is removed either
        # way, so a failure never leaves a recording behind on disk.
        final.unlink(missing_ok=True)
        raise
    final.unlink(missing_ok=True)
    return _test_response(outcome)


def _test_processing(
    ctx: GatewayContextDependency, cleanup_mode: str
) -> tuple[str, CleanupOptions | None]:
    manager = ctx.cleanup
    if cleanup_mode == MODE_CONSERVATIVE and manager is not None:
        return service.ADHOC_CLEANUP_STYLE, manager.options(MODE_CONSERVATIVE)
    return service.RAW_STYLE, None


def _test_response(outcome: service.AdhocTranscription) -> TestTranscriptionResponse:
    timing = outcome.timing
    cleanup_outcome = outcome.cleanup
    # `disabled` is reported as nothing at all, so a raw benchmark's response is
    # the same shape it has always been.
    ran = cleanup_outcome is not None and cleanup_outcome.status is not CleanupStatus.DISABLED
    reported = serializers.cleanup_result(cleanup_outcome) if ran and cleanup_outcome else None
    return TestTranscriptionResponse(
        transcript=outcome.transcript,
        engine=outcome.engine,
        duration_ms=timing.total_ms,
        normalization_ms=timing.normalization_ms,
        model_load_ms=timing.model_load_ms,
        inference_ms=timing.inference_ms,
        audio_duration_ms=timing.audio_duration_ms,
        real_time_factor=timing.real_time_factor,
        peak_memory_mb=timing.peak_memory_mb,
        original_transcript=outcome.original_transcript,
        cleanup=reported,
        cleanup_ms=reported.duration_ms if reported else 0,
    )


@router.get("/ui/partials/settings", response_class=HTMLResponse)
async def ui_settings(ctx: GatewayContextDependency) -> HTMLResponse:
    paths = [
        ("Data directory", str(ctx.settings.data_dir)),
        ("Models directory", str(ctx.manager.models_dir)),
        ("WebUI config file", str(ctx.config_path)),
        ("Token file", TOKEN_FILE_HINT),
    ]
    return HTMLResponse(
        settings.settings_fragment(
            admin_queries.config_response(ctx),
            paths,
            ctx.settings.bind_host,
            ctx.settings.port,
            tokens.tokens_fragment_str(ctx),
            cleanup.cleanup_card(
                admin_queries.cleanup_config(ctx), admin_queries.cleanup_model_entries(ctx)
            ),
        )
    )


@router.put("/ui/partials/config", response_class=HTMLResponse)
async def ui_update_config(
    ctx: GatewayContextDependency,
    form: Annotated[_ConfigForm, Depends()],
) -> HTMLResponse:
    engine_manager = ctx.engine_manager
    if engine_manager is None:
        raise errors.APIProblem(
            HTTP_409_CONFLICT, "engine_locked", "The engine was fixed at startup and cannot switch."
        )
    try:
        engine_manager.configure(
            form.engine,
            form.compute_device,
            form.compute_type,
            form.cpu_threads,
            form.idle_offload_enabled,
            form.idle_offload_minutes,
        )
    except ValueError as error:
        raise errors.APIProblem(
            HTTP_422_UNPROCESSABLE_CONTENT, "invalid_engine", str(error)
        ) from error
    await ctx.readiness.warmup()
    state = await ctx.readiness.probe()
    return HTMLResponse(
        engine_update_fragment(
            EngineStatus(id=form.engine, name=state.name, ready=state.ready),
            "Engine preference saved. Memory policy updated.",
            bind_host=ctx.settings.bind_host,
            port=ctx.settings.port,
        )
    )


@router.get("/ui/partials/test", response_class=HTMLResponse)
async def ui_test(ctx: GatewayContextDependency) -> HTMLResponse:
    """Pair phone (once) and try the pipeline from this browser."""
    return HTMLResponse(
        test_panel.pair_and_test_fragment(
            pairing_view.pairing_html(ctx),
            ctx.settings.maximum_duration_seconds,
            bind_host=ctx.settings.bind_host,
            is_mac=platform.system() == "Darwin",
        )
    )
