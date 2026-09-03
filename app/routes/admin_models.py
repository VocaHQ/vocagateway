from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Form, Query, Response
from fastapi.responses import HTMLResponse
from pydantic import BeforeValidator
from starlette.status import (
    HTTP_404_NOT_FOUND,
    HTTP_409_CONFLICT,
    HTTP_422_UNPROCESSABLE_CONTENT,
)

from app.admin_queries import filtered_model_entries, model_entries
from app.context import GatewayContext, GatewayContextDependency, require_token
from app.errors import APIProblem
from app.fragments.engine import engine_pill_oob
from app.fragments.models import models_fragment, models_list_fragment
from app.model_manager import DownloadInProgressError, UnknownModelError
from app.schemas import (
    AdminModelEntry,
    CustomDownloadRequest,
    DeleteResponse,
    DownloadResponse,
    EngineStatus,
    SelectModelResponse,
)

router = APIRouter(dependencies=[Depends(require_token)])
DOWNLOAD_IN_PROGRESS_CODE = "download_in_progress"


def _loose_bool(form_value: object) -> bool:
    """Treat missing/empty query/form values as false (Clear filters sends "")."""
    if form_value is None or form_value is False or form_value == "":
        return False
    if form_value is True:
        return True
    if isinstance(form_value, (int, float)):
        return form_value != 0
    if isinstance(form_value, str):
        lowered = form_value.strip().lower()
        if lowered in ("", "0", "false", "no", "off"):
            return False
        if lowered in ("1", "true", "yes", "on"):
            return True
    return bool(form_value)


# Repeated keys: family=Whisper&family=Parakeet (and the same for language/engine).
LanguageQ = Annotated[list[str] | None, Query()]
FamilyQ = Annotated[list[str] | None, Query()]
EngineQ = Annotated[list[str] | None, Query()]
LanguageF = Annotated[list[str] | None, Form()]
FamilyF = Annotated[list[str] | None, Form()]
EngineF = Annotated[list[str] | None, Form()]
BoolQ = Annotated[bool, BeforeValidator(_loose_bool), Query()]
BoolF = Annotated[bool, BeforeValidator(_loose_bool), Form()]
MaxSizeForm = Annotated[str, Form()]
CustomUrlForm = Annotated[str, Form()]
CustomDigestForm = Annotated[str, Form()]


def _models_list_html(
    ctx: GatewayContext,
    *,
    installed_only: bool = False,
    language: list[str] | None = None,
    family: list[str] | None = None,
    engine: list[str] | None = None,
    max_size: str = "",
    recommended_only: bool = False,
) -> HTMLResponse:
    languages = list(language or [])
    families = list(family or [])
    engines = list(engine or [])
    entries = filtered_model_entries(
        ctx,
        installed_only=installed_only,
        language=languages,
        family=families,
        engine=engines,
        max_size=max_size,
        recommended_only=recommended_only,
    )
    return HTMLResponse(
        models_list_fragment(
            entries,
            installed_only=installed_only,
            languages=languages,
            families=families,
            engines=engines,
            max_size=max_size,
            recommended_only=recommended_only,
        )
    )


@router.get("/v1/admin/models", response_model=list[AdminModelEntry])
async def get_admin_models(
    ctx: GatewayContextDependency,
    installed_only: BoolQ = False,
    language: LanguageQ = None,
    family: FamilyQ = None,
    engine: EngineQ = None,
    max_size: str = "",
    recommended_only: BoolQ = False,
) -> list[AdminModelEntry]:
    return filtered_model_entries(
        ctx,
        installed_only=installed_only,
        language=language,
        family=family,
        engine=engine,
        max_size=max_size,
        recommended_only=recommended_only,
    )


@router.post("/v1/admin/models/{model_id}/download", response_model=DownloadResponse)
async def start_model_download(model_id: str, ctx: GatewayContextDependency) -> DownloadResponse:
    try:
        state = ctx.manager.start_download(model_id)
    except UnknownModelError as error:
        raise APIProblem(
            HTTP_404_NOT_FOUND, "unknown_model", "This model is not in the catalog."
        ) from error
    except DownloadInProgressError as error:
        raise APIProblem(
            HTTP_409_CONFLICT, DOWNLOAD_IN_PROGRESS_CODE, "This model is already downloading."
        ) from error
    return DownloadResponse(model_id=model_id, status=state.status)


@router.post("/v1/admin/models/custom", response_model=DownloadResponse)
async def start_custom_download(
    body: CustomDownloadRequest, ctx: GatewayContextDependency
) -> DownloadResponse:
    try:
        state = ctx.manager.start_custom_download(body.url, body.sha256)
    except ValueError as error:
        # The URL and the digest are separate inputs, so they get separate
        # codes; a client cannot fix a bad digest by editing the URL.
        code = "invalid_model_digest" if "SHA-256" in str(error) else "invalid_model_url"
        raise APIProblem(HTTP_422_UNPROCESSABLE_CONTENT, code, str(error)) from error
    except DownloadInProgressError as error:
        raise APIProblem(HTTP_409_CONFLICT, DOWNLOAD_IN_PROGRESS_CODE, str(error)) from error
    return DownloadResponse(model_id=state.model_id, status=state.status)


@router.post("/v1/admin/models/{model_id}/cancel", response_model=DownloadResponse)
async def cancel_model_download(model_id: str, ctx: GatewayContextDependency) -> DownloadResponse:
    if not ctx.manager.cancel_download(model_id):
        raise APIProblem(
            HTTP_409_CONFLICT, "download_not_active", "This model is not currently downloading."
        )
    return DownloadResponse(model_id=model_id, status="cancelling")


@router.delete("/v1/admin/models/{model_id}", response_model=DeleteResponse)
async def delete_model(
    model_id: str, response: Response, ctx: GatewayContextDependency
) -> DeleteResponse:
    try:
        if ctx.engine_manager is not None:
            ctx.engine_manager.forget_if_active(model_id)
        deleted = ctx.manager.delete(model_id)
    except DownloadInProgressError as error:
        raise APIProblem(
            HTTP_409_CONFLICT, DOWNLOAD_IN_PROGRESS_CODE, "Cancel the download before deleting."
        ) from error
    if not deleted:
        response.status_code = HTTP_404_NOT_FOUND
    return DeleteResponse(deleted=deleted)


@router.post("/v1/admin/models/{model_id}/select", response_model=SelectModelResponse)
async def select_model(model_id: str, ctx: GatewayContextDependency) -> SelectModelResponse:
    engine_manager = ctx.engine_manager
    if engine_manager is None:
        raise APIProblem(
            HTTP_409_CONFLICT, "engine_locked", "The engine was fixed at startup and cannot switch."
        )
    try:
        engine_manager.select_model(model_id)
    except KeyError as error:
        raise APIProblem(
            HTTP_404_NOT_FOUND, "model_not_installed", "Download this model before selecting it."
        ) from error
    await ctx.readiness.warmup()
    state = await ctx.readiness.probe()
    return SelectModelResponse(
        engine=EngineStatus(
            id=engine_manager.runtime_config.engine, name=state.name, ready=state.ready
        )
    )


@router.get("/ui/partials/models", response_class=HTMLResponse)
async def ui_models(ctx: GatewayContextDependency) -> HTMLResponse:
    return HTMLResponse(models_fragment(model_entries(ctx)))


@router.get("/ui/partials/models-list", response_class=HTMLResponse)
async def ui_models_list(
    ctx: GatewayContextDependency,
    installed_only: BoolQ = False,
    language: LanguageQ = None,
    family: FamilyQ = None,
    engine: EngineQ = None,
    max_size: str = "",
    recommended_only: BoolQ = False,
) -> HTMLResponse:
    return _models_list_html(
        ctx,
        installed_only=installed_only,
        language=language,
        family=family,
        engine=engine,
        max_size=max_size,
        recommended_only=recommended_only,
    )


@router.post("/ui/partials/models/{model_id}/download", response_class=HTMLResponse)
async def ui_start_download(
    model_id: str,
    ctx: GatewayContextDependency,
    installed_only: BoolF = False,
    language: LanguageF = None,
    family: FamilyF = None,
    engine: EngineF = None,
    max_size: MaxSizeForm = "",
    recommended_only: BoolF = False,
) -> HTMLResponse:
    try:
        ctx.manager.start_download(model_id)
    except UnknownModelError as error:
        raise APIProblem(
            HTTP_404_NOT_FOUND, "unknown_model", "This model is not in the catalog."
        ) from error
    except DownloadInProgressError as error:
        raise APIProblem(
            HTTP_409_CONFLICT, DOWNLOAD_IN_PROGRESS_CODE, "This model is already downloading."
        ) from error
    return _models_list_html(
        ctx,
        installed_only=installed_only,
        language=language,
        family=family,
        engine=engine,
        max_size=max_size,
        recommended_only=recommended_only,
    )


@router.post("/ui/partials/models/custom", response_class=HTMLResponse)
async def ui_custom_download(
    url: CustomUrlForm,
    ctx: GatewayContextDependency,
    sha256: CustomDigestForm = "",
    installed_only: BoolF = False,
    language: LanguageF = None,
    family: FamilyF = None,
    engine: EngineF = None,
    max_size: MaxSizeForm = "",
    recommended_only: BoolF = False,
) -> HTMLResponse:
    try:
        ctx.manager.start_custom_download(url, sha256)
    except ValueError as error:
        code = "invalid_model_digest" if "SHA-256" in str(error) else "invalid_model_url"
        raise APIProblem(HTTP_422_UNPROCESSABLE_CONTENT, code, str(error)) from error
    except DownloadInProgressError as error:
        raise APIProblem(HTTP_409_CONFLICT, DOWNLOAD_IN_PROGRESS_CODE, str(error)) from error
    return _models_list_html(
        ctx,
        installed_only=installed_only,
        language=language,
        family=family,
        engine=engine,
        max_size=max_size,
        recommended_only=recommended_only,
    )


@router.post("/ui/partials/models/{model_id}/cancel", response_class=HTMLResponse)
async def ui_cancel_download(
    model_id: str,
    ctx: GatewayContextDependency,
    installed_only: BoolF = False,
    language: LanguageF = None,
    family: FamilyF = None,
    engine: EngineF = None,
    max_size: MaxSizeForm = "",
    recommended_only: BoolF = False,
) -> HTMLResponse:
    ctx.manager.cancel_download(model_id)
    return _models_list_html(
        ctx,
        installed_only=installed_only,
        language=language,
        family=family,
        engine=engine,
        max_size=max_size,
        recommended_only=recommended_only,
    )


@router.delete("/ui/partials/models/{model_id}", response_class=HTMLResponse)
async def ui_delete_model(
    model_id: str,
    ctx: GatewayContextDependency,
    installed_only: BoolF = False,
    language: LanguageF = None,
    family: FamilyF = None,
    engine: EngineF = None,
    max_size: MaxSizeForm = "",
    recommended_only: BoolF = False,
) -> HTMLResponse:
    try:
        if ctx.engine_manager is not None:
            ctx.engine_manager.forget_if_active(model_id)
        ctx.manager.delete(model_id)
    except DownloadInProgressError as error:
        raise APIProblem(
            HTTP_409_CONFLICT, DOWNLOAD_IN_PROGRESS_CODE, "Cancel the download before deleting."
        ) from error
    return _models_list_html(
        ctx,
        installed_only=installed_only,
        language=language,
        family=family,
        engine=engine,
        max_size=max_size,
        recommended_only=recommended_only,
    )


@router.post("/ui/partials/models/{model_id}/select", response_class=HTMLResponse)
async def ui_select_model(
    model_id: str,
    ctx: GatewayContextDependency,
    installed_only: BoolF = False,
    language: LanguageF = None,
    family: FamilyF = None,
    engine: EngineF = None,
    max_size: MaxSizeForm = "",
    recommended_only: BoolF = False,
) -> HTMLResponse:
    engine_manager = ctx.engine_manager
    if engine_manager is None:
        raise APIProblem(
            HTTP_409_CONFLICT, "engine_locked", "The engine was fixed at startup and cannot switch."
        )
    try:
        engine_manager.select_model(model_id)
    except KeyError as error:
        raise APIProblem(
            HTTP_404_NOT_FOUND, "model_not_installed", "Download this model before selecting it."
        ) from error
    await ctx.readiness.warmup()
    state = await ctx.readiness.probe()
    entries = filtered_model_entries(
        ctx,
        installed_only=installed_only,
        language=language,
        family=family,
        engine=engine,
        max_size=max_size,
        recommended_only=recommended_only,
    )
    return HTMLResponse(
        models_list_fragment(
            entries,
            installed_only=installed_only,
            languages=language,
            families=family,
            engines=engine,
            max_size=max_size,
            recommended_only=recommended_only,
        )
        + engine_pill_oob(
            EngineStatus(
                id=engine_manager.runtime_config.engine, name=state.name, ready=state.ready
            ),
            bind_host=ctx.settings.bind_host,
            port=ctx.settings.port,
        )
    )
