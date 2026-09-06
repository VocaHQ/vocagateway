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

from app import admin_queries, errors, model_manager
from app.cleanup import catalog
from app.cleanup.manager import CleanupManager, CleanupUpdate
from app.context import GatewayContext, GatewayContextDependency, require_token
from app.fragments import cleanup as cleanup_fragment
from app.schemas import CleanupConfigResponse, CleanupConfigUpdateRequest, CleanupModelEntry

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
    model_id: ModelForm = ""
    timeout_seconds: TimeoutForm = 5.0
    idle_unload_minutes: IdleMinutesForm = 15
    enabled: EnabledForm = False
    idle_unload_enabled: EnabledForm = False

    def as_request(self) -> CleanupConfigUpdateRequest:
        return CleanupConfigUpdateRequest.model_validate(
            {
                "enabled": self.enabled,
                "mode": self.mode,
                "model_id": self.model_id,
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
    """Load the model now, so the first corrected dictation is not a cold start.

    Bounded and separate from a request: a cold load that cannot meet a
    request's deadline returns the ASR result, and this is how the operator pays
    that cost once, deliberately, from the settings page.
    """
    await manager.warmup()
    return admin_queries.cleanup_config(ctx)


@router.get("/v1/admin/cleanup/models", response_model=list[CleanupModelEntry])
async def cleanup_models(ctx: GatewayContextDependency) -> list[CleanupModelEntry]:
    return admin_queries.cleanup_model_entries(ctx)


@router.post("/ui/partials/cleanup/models/{model_id}", response_class=HTMLResponse)
async def ui_install_cleanup_model(
    model_id: str, ctx: GatewayContextDependency, manager: CleanupManagerDependency
) -> HTMLResponse:
    await download_cleanup_model(model_id, ctx, manager)
    return _card(ctx, "Downloading. The card refreshes as it lands.")


@router.delete("/ui/partials/cleanup/models/{model_id}", response_class=HTMLResponse)
async def ui_delete_cleanup_model(
    model_id: str, ctx: GatewayContextDependency, manager: CleanupManagerDependency
) -> HTMLResponse:
    await delete_cleanup_model(model_id, ctx, manager)
    return _card(ctx, "Model deleted.")


@router.post("/ui/partials/cleanup/warmup", response_class=HTMLResponse)
async def ui_warm_cleanup(
    ctx: GatewayContextDependency, manager: CleanupManagerDependency
) -> HTMLResponse:
    warmed = await manager.warmup()
    note = "Model loaded and ready." if warmed else "The model could not be loaded."
    return _card(ctx, note)


@router.post("/v1/admin/cleanup/models/{model_id}/download", response_model=CleanupModelEntry)
async def download_cleanup_model(
    model_id: str, ctx: GatewayContextDependency, manager: CleanupManagerDependency
) -> CleanupModelEntry:
    selected = _known(model_id)
    _require_disk_space(manager, selected.size_bytes)
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
    return _card(ctx)


@router.put("/ui/partials/cleanup", response_class=HTMLResponse)
async def ui_update_cleanup(
    ctx: GatewayContextDependency,
    manager: CleanupManagerDependency,
    form: Annotated[_CleanupForm, Depends()],
) -> HTMLResponse:
    _apply(manager, form.as_request())
    return _card(ctx, "Transcript cleanup settings saved.")


def _card(ctx: GatewayContext, message: str = "") -> HTMLResponse:
    """The settings card, rebuilt from live state after every action."""
    return HTMLResponse(
        cleanup_fragment.cleanup_card(
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
