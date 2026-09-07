"""Authenticated operator controls for transcript cleanup.

Kept in its own router, with its own partial-update contract, because cleanup
and speech recognition are separate decisions with separate lifecycles. Saving a
cleanup preference must never reset the engine, the compute device, or the ASR
idle-offload policy — which is exactly what routing it through the shared config
update would do.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from typing import Annotated

from fastapi import APIRouter, Depends, Form
from fastapi.responses import HTMLResponse
from starlette.status import (
    HTTP_404_NOT_FOUND,
    HTTP_409_CONFLICT,
    HTTP_422_UNPROCESSABLE_CONTENT,
    HTTP_507_INSUFFICIENT_STORAGE,
)

from app import admin_queries, errors, model_manager, serializers, service, text_styles
from app.cleanup import catalog
from app.cleanup.base import MODE_CONSERVATIVE
from app.cleanup.manager import CleanupManager, CleanupUpdate, preserve_implicit_model_selection
from app.context import GatewayContext, GatewayContextDependency, require_token
from app.fragments import cleanup as cleanup_fragment
from app.schemas import (
    CleanupConfigResponse,
    CleanupConfigUpdateRequest,
    CleanupModelEntry,
    CleanupPreviewRequest,
    CleanupPreviewResponse,
)

router = APIRouter(dependencies=[Depends(require_token)])
EnabledForm = Annotated[bool, Form()]
ModeForm = Annotated[str, Form()]
ModelForm = Annotated[str, Form()]
TimeoutForm = Annotated[float, Form()]
IdleMinutesForm = Annotated[int, Form()]


@dataclass
class _CleanupForm:
    """The settings form. Unlike the JSON contract, it always sends every field.

    An unchecked checkbox is simply absent from a form post, so `False` here
    means "the operator unticked it" rather than "leave it alone" — which is why
    the two contracts are separate rather than shared.
    """

    mode: ModeForm = "conservative"
    auto_language: ModelForm = ""
    timeout_seconds: TimeoutForm = 5.0
    idle_unload_minutes: IdleMinutesForm = 15
    enabled: EnabledForm = False
    idle_unload_enabled: EnabledForm = False

    def as_request(self) -> CleanupConfigUpdateRequest:
        return CleanupConfigUpdateRequest.model_validate(
            {
                "enabled": self.enabled,
                "mode": self.mode,
                "auto_language": self.auto_language,
                "timeout_seconds": self.timeout_seconds,
                "idle_unload_enabled": self.idle_unload_enabled,
                "idle_unload_minutes": self.idle_unload_minutes,
            }
        )


# Headroom over the artifact size for the staged download and the filesystem's
# own overhead. A download that fills the disk takes the gateway's database with
# it, so the check is deliberately generous.
DISK_HEADROOM = 1.15
UNKNOWN_MODEL_CODE = "unknown_model"
UNKNOWN_MODEL_MESSAGE = "That cleanup model is not in the catalog."


def require_cleanup(ctx: GatewayContextDependency) -> CleanupManager:
    manager = ctx.cleanup
    if manager is None:
        raise errors.APIProblem(
            HTTP_409_CONFLICT,
            "cleanup_locked",
            "This gateway was started with a fixed engine and cannot configure cleanup.",
        )
    return manager


CleanupManagerDependency = Annotated[CleanupManager, Depends(require_cleanup)]


@router.get("/v1/admin/cleanup", response_model=CleanupConfigResponse)
async def get_cleanup(ctx: GatewayContextDependency) -> CleanupConfigResponse:
    return admin_queries.cleanup_config(ctx)


@router.put("/v1/admin/cleanup", response_model=CleanupConfigResponse)
async def update_cleanup(
    body: CleanupConfigUpdateRequest,
    ctx: GatewayContextDependency,
    manager: CleanupManagerDependency,
) -> CleanupConfigResponse:
    _apply(manager, body)
    return admin_queries.cleanup_config(ctx)


@router.post("/v1/admin/cleanup/warmup", response_model=CleanupConfigResponse)
async def warm_cleanup(
    ctx: GatewayContextDependency, manager: CleanupManagerDependency
) -> CleanupConfigResponse:
    """Start loading the model, so the first corrected dictation is not cold.

    Returns as soon as the load is under way rather than holding the request
    open for it: a multi-gigabyte GGUF takes minutes, and the answer an
    operator needs is the `state` field, which reads `loading` until it reads
    `ready`.
    """
    await manager.warmup()
    return admin_queries.cleanup_config(ctx)


@router.post("/v1/admin/cleanup/preview", response_model=CleanupPreviewResponse)
async def preview_cleanup(
    body: CleanupPreviewRequest, ctx: GatewayContextDependency, manager: CleanupManagerDependency
) -> CleanupPreviewResponse:
    """Correct one piece of text supplied by the operator, and say what happened.

    The point is evidence. "Does this actually do anything" was answerable only
    by recording a clip and comparing two paragraphs by eye, which is a poor way
    to see a comma appear. This runs the same finalization the dictation path
    runs — same prompt, same validators, same fallback — on text typed into the
    settings page, so the answer takes one click.

    Nothing is stored: the text goes to the model and comes back, and no session
    row, log line, or diagnostic bundle sees either version.
    """
    final = await ctx.service.cleanup.finalize(
        body.text,
        style=service.ADHOC_CLEANUP_STYLE,
        language=body.language,
        options=manager.options(MODE_CONSERVATIVE),
    )
    return CleanupPreviewResponse(
        original=body.text,
        # The same deterministic pass the service runs first and independently,
        # so the panel can show what the model added rather than claiming the
        # rule-based capitalisation as its own work.
        without_cleanup=text_styles.apply_writing_style(
            body.text, service.ADHOC_CLEANUP_STYLE, body.language
        ),
        transcript=final.transcript,
        cleanup=serializers.cleanup_result(final.cleanup),
    )


@router.get("/v1/admin/cleanup/models", response_model=list[CleanupModelEntry])
async def cleanup_models(ctx: GatewayContextDependency) -> list[CleanupModelEntry]:
    return admin_queries.cleanup_model_entries(ctx)


@router.post("/ui/partials/cleanup/models/{model_id}", response_class=HTMLResponse)
async def ui_install_cleanup_model(
    model_id: str, ctx: GatewayContextDependency, manager: CleanupManagerDependency
) -> HTMLResponse:
    await download_cleanup_model(model_id, ctx, manager)
    return _page(ctx, "Downloading. This page refreshes as it lands.")


@router.delete("/ui/partials/cleanup/models/{model_id}", response_class=HTMLResponse)
async def ui_delete_cleanup_model(
    model_id: str, ctx: GatewayContextDependency, manager: CleanupManagerDependency
) -> HTMLResponse:
    await delete_cleanup_model(model_id, ctx, manager)
    return _page(ctx, "Model deleted.")


@router.post("/ui/partials/cleanup/warmup", response_class=HTMLResponse)
async def ui_warm_cleanup(
    ctx: GatewayContextDependency, manager: CleanupManagerDependency
) -> HTMLResponse:
    warmed = await manager.warmup()
    return HTMLResponse(
        cleanup_fragment.cleanup_status(
            admin_queries.cleanup_config(ctx), _warmup_note(manager, warmed=warmed)
        )
    )


@router.post("/v1/admin/cleanup/models/{model_id}/download", response_model=CleanupModelEntry)
async def download_cleanup_model(
    model_id: str, ctx: GatewayContextDependency, manager: CleanupManagerDependency
) -> CleanupModelEntry:
    selected = _known(model_id)
    _require_disk_space(manager, selected.size_bytes)
    preserve_implicit_model_selection(manager)
    try:
        manager.models.start_download(model_id)
    except model_manager.DownloadInProgressError as error:
        raise errors.APIProblem(HTTP_409_CONFLICT, "download_in_progress", str(error)) from error
    except model_manager.UnknownModelError as error:
        raise errors.APIProblem(HTTP_404_NOT_FOUND, "unknown_model", str(error)) from error
    return _entry(ctx, model_id)


@router.delete("/v1/admin/cleanup/models/{model_id}", response_model=CleanupModelEntry)
async def delete_cleanup_model(
    model_id: str, ctx: GatewayContextDependency, manager: CleanupManagerDependency
) -> CleanupModelEntry:
    _known(model_id)
    if manager.model_id == model_id and manager.host.is_running:
        raise errors.APIProblem(
            HTTP_409_CONFLICT,
            "model_in_use",
            "Stop or switch this cleanup model before deleting it.",
        )
    try:
        manager.models.delete(model_id)
    except model_manager.DownloadInProgressError as error:
        raise errors.APIProblem(HTTP_409_CONFLICT, "download_in_progress", str(error)) from error
    return _entry(ctx, model_id)


@router.get("/ui/partials/cleanup", response_class=HTMLResponse)
async def ui_cleanup(ctx: GatewayContextDependency) -> HTMLResponse:
    """The whole section, as the left-nav tab loads it."""
    return _page(ctx)


@router.get("/ui/partials/cleanup/library", response_class=HTMLResponse)
async def ui_cleanup_library(ctx: GatewayContextDependency) -> HTMLResponse:
    """Refresh downloads without replacing preview text or unsaved settings."""
    return HTMLResponse(
        cleanup_fragment.cleanup_library(admin_queries.cleanup_model_entries(ctx))
        + cleanup_fragment.cleanup_status(admin_queries.cleanup_config(ctx), out_of_band=True)
    )


@router.get("/ui/partials/cleanup/status", response_class=HTMLResponse)
async def ui_cleanup_status(ctx: GatewayContextDependency) -> HTMLResponse:
    """Just the runtime status strip.

    Swapping the whole page every two seconds during a load would clear whatever
    someone had typed into the try-it box.
    """
    return HTMLResponse(cleanup_fragment.cleanup_status(admin_queries.cleanup_config(ctx)))


@router.put("/ui/partials/cleanup/settings", response_class=HTMLResponse)
async def ui_update_cleanup(
    ctx: GatewayContextDependency,
    manager: CleanupManagerDependency,
    form: Annotated[_CleanupForm, Depends()],
) -> HTMLResponse:
    _apply(manager, form.as_request())
    return _page(ctx, "Settings saved.")


@router.put("/ui/partials/cleanup/select/{model_id}", response_class=HTMLResponse)
async def ui_select_cleanup_model(
    model_id: str, ctx: GatewayContextDependency, manager: CleanupManagerDependency
) -> HTMLResponse:
    """Choose which downloaded model does the correcting."""
    selected = _known(model_id)
    manager.configure(CleanupUpdate(model_id=model_id))
    return _page(ctx, f"{selected.label} is now the cleanup model.")


def _warmup_note(manager: CleanupManager, *, warmed: bool) -> str:
    """What to tell the operator, which is not the same as whether it is ready.

    A load that has only just started is neither a success nor a failure, and
    reporting it as "could not be loaded" would send someone debugging a
    working gateway.
    """
    if warmed:
        return "Model loaded and ready."
    if manager.loading:
        return "Loading the model. This panel refreshes when it is ready."
    return "The model could not be loaded."


def _page(ctx: GatewayContext, message: str = "") -> HTMLResponse:
    """The section, rebuilt from live state after every action."""
    return HTMLResponse(
        cleanup_fragment.cleanup_page(
            admin_queries.cleanup_config(ctx),
            admin_queries.cleanup_model_entries(ctx),
            message=message,
        )
    )


def _apply(manager: CleanupManager, body: CleanupConfigUpdateRequest) -> None:
    if body.model_id and catalog.cleanup_model(body.model_id) is None:
        raise errors.APIProblem(
            HTTP_422_UNPROCESSABLE_CONTENT,
            UNKNOWN_MODEL_CODE,
            UNKNOWN_MODEL_MESSAGE,
        )
    manager.configure(
        CleanupUpdate(
            enabled=body.enabled,
            mode=body.mode,
            model_id=body.model_id,
            timeout_seconds=body.timeout_seconds,
            idle_unload_enabled=body.idle_unload_enabled,
            idle_unload_minutes=body.idle_unload_minutes,
            auto_language=body.auto_language,
        )
    )


def _known(model_id: str) -> catalog.CleanupModel:
    selected = catalog.cleanup_model(model_id)
    if selected is None:
        raise errors.APIProblem(HTTP_404_NOT_FOUND, UNKNOWN_MODEL_CODE, UNKNOWN_MODEL_MESSAGE)
    if not selected.installable:
        # No pinned revision and digest means the exact bytes this entry was
        # reviewed against cannot be recognised, and unverifiable weights must
        # never reach a runtime launch.
        raise errors.APIProblem(
            HTTP_409_CONFLICT,
            "model_unpinned",
            "This cleanup model has no pinned checksum and cannot be installed.",
        )
    return selected


def _require_disk_space(manager: CleanupManager, size_bytes: int) -> None:
    target = manager.models.models_dir
    target.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(target).free
    if free_bytes < size_bytes * DISK_HEADROOM:
        raise errors.APIProblem(
            HTTP_507_INSUFFICIENT_STORAGE,
            "insufficient_disk_space",
            "There is not enough free disk space for this cleanup model.",
        )


def _entry(ctx: GatewayContext, model_id: str) -> CleanupModelEntry:
    matches = [entry for entry in admin_queries.cleanup_model_entries(ctx) if entry.id == model_id]
    if not matches:
        raise errors.APIProblem(HTTP_404_NOT_FOUND, UNKNOWN_MODEL_CODE, UNKNOWN_MODEL_MESSAGE)
    return matches[0]
