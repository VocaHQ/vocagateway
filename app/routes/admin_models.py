from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from fastapi import APIRouter, Depends, Form, Query, Response
from fastapi.responses import HTMLResponse
from pydantic import BeforeValidator
from starlette import status

from app import admin_queries, context, errors, model_manager, schemas
from app.fragments import engine as engine_fragments
from app.fragments import models as model_fragments

DOWNLOAD_IN_PROGRESS_CODE = "download_in_progress"
router = APIRouter(dependencies=[Depends(context.require_token)])


def _loose_bool(form_value: object) -> bool:
    """Treat missing/empty query/form values as false (Clear filters sends "")."""
    if not form_value:
        return False
    if isinstance(form_value, str):
        return form_value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(form_value)


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


@dataclass
class _ModelFilterQuery:
    installed_only: BoolQ = False
    language: LanguageQ = None
    family: FamilyQ = None
    engine: EngineQ = None
    max_size: str = ""
    recommended_only: BoolQ = False


@dataclass
class _ModelFilterForm:
    installed_only: BoolF = False
    language: LanguageF = None
    family: FamilyF = None
    engine: EngineF = None
    max_size: MaxSizeForm = ""
    recommended_only: BoolF = False


def _models_list_html(
    ctx: context.GatewayContext,
    filters: _ModelFilterQuery | _ModelFilterForm,
) -> HTMLResponse:
    entries = admin_queries.filtered_model_entries(
        ctx,
        installed_only=filters.installed_only,
        language=filters.language,
        family=filters.family,
        engine=filters.engine,
        max_size=filters.max_size,
        recommended_only=filters.recommended_only,
    )
    return HTMLResponse(
        model_fragments.models_list_fragment(
            entries,
            installed_only=filters.installed_only,
            languages=filters.language,
            families=filters.family,
            engines=filters.engine,
            max_size=filters.max_size,
            recommended_only=filters.recommended_only,
        )
    )


METHOD_GET = ("GET",)
METHOD_POST = ("POST",)
METHOD_DELETE = ("DELETE",)


class _AdminModelApiRoutes:
    def bind(self, target_router: APIRouter) -> None:
        target_router.add_api_route(
            "/v1/admin/models",
            self.get_admin_models,
            methods=list(METHOD_GET),
            response_model=list[schemas.AdminModelEntry],
        )
        target_router.add_api_route(
            "/v1/admin/models/{model_id}/download",
            self.start_model_download,
            methods=list(METHOD_POST),
            response_model=schemas.DownloadResponse,
        )
        target_router.add_api_route(
            "/v1/admin/models/custom",
            self.start_custom_download,
            methods=list(METHOD_POST),
            response_model=schemas.DownloadResponse,
        )
        target_router.add_api_route(
            "/v1/admin/models/{model_id}/cancel",
            self.cancel_model_download,
            methods=list(METHOD_POST),
            response_model=schemas.DownloadResponse,
        )
        target_router.add_api_route(
            "/v1/admin/models/{model_id}",
            self.delete_model,
            methods=list(METHOD_DELETE),
            response_model=schemas.DeleteResponse,
        )
        target_router.add_api_route(
            "/v1/admin/models/{model_id}/select",
            self.select_model,
            methods=list(METHOD_POST),
            response_model=schemas.SelectModelResponse,
        )

    async def get_admin_models(
        self,
        ctx: context.GatewayContextDependency,
        filters: Annotated[_ModelFilterQuery, Depends()],
    ) -> list[schemas.AdminModelEntry]:
        return admin_queries.filtered_model_entries(
            ctx,
            installed_only=filters.installed_only,
            language=filters.language,
            family=filters.family,
            engine=filters.engine,
            max_size=filters.max_size,
            recommended_only=filters.recommended_only,
        )

    async def start_model_download(
        self, model_id: str, ctx: context.GatewayContextDependency
    ) -> schemas.DownloadResponse:
        try:
            state = ctx.manager.start_download(model_id)
        except model_manager.UnknownModelError as error:
            raise errors.APIProblem(
                status.HTTP_404_NOT_FOUND, "unknown_model", "This model is not in the catalog."
            ) from error
        except model_manager.DownloadInProgressError as error:
            raise errors.APIProblem(
                status.HTTP_409_CONFLICT,
                DOWNLOAD_IN_PROGRESS_CODE,
                "This model is already downloading.",
            ) from error
        return schemas.DownloadResponse(model_id=model_id, status=state.status)

    async def start_custom_download(
        self, body: schemas.CustomDownloadRequest, ctx: context.GatewayContextDependency
    ) -> schemas.DownloadResponse:
        try:
            state = ctx.manager.start_custom_download(body.url, body.sha256)
        except ValueError as error:
            code = "invalid_model_digest" if "SHA-256" in str(error) else "invalid_model_url"
            raise errors.APIProblem(
                status.HTTP_422_UNPROCESSABLE_CONTENT, code, str(error)
            ) from error
        except model_manager.DownloadInProgressError as error:
            raise errors.APIProblem(
                status.HTTP_409_CONFLICT, DOWNLOAD_IN_PROGRESS_CODE, str(error)
            ) from error
        return schemas.DownloadResponse(model_id=state.model_id, status=state.status)

    async def cancel_model_download(
        self, model_id: str, ctx: context.GatewayContextDependency
    ) -> schemas.DownloadResponse:
        if not ctx.manager.cancel_download(model_id):
            raise errors.APIProblem(
                status.HTTP_409_CONFLICT,
                "download_not_active",
                "This model is not currently downloading.",
            )
        return schemas.DownloadResponse(model_id=model_id, status="cancelling")

    async def delete_model(
        self, model_id: str, response: Response, ctx: context.GatewayContextDependency
    ) -> schemas.DeleteResponse:
        if ctx.engine_manager is not None:
            ctx.engine_manager.forget_if_active(model_id)
        try:
            deleted = ctx.manager.delete(model_id)
        except model_manager.DownloadInProgressError as error:
            raise errors.APIProblem(
                status.HTTP_409_CONFLICT,
                DOWNLOAD_IN_PROGRESS_CODE,
                "Cancel the download before deleting.",
            ) from error
        if not deleted:
            response.status_code = status.HTTP_404_NOT_FOUND
        return schemas.DeleteResponse(deleted=deleted)

    async def select_model(
        self, model_id: str, ctx: context.GatewayContextDependency
    ) -> schemas.SelectModelResponse:
        engine_manager = ctx.engine_manager
        if engine_manager is None:
            raise errors.APIProblem(
                status.HTTP_409_CONFLICT,
                "engine_locked",
                "The engine was fixed at startup and cannot switch.",
            )
        try:
            engine_manager.select_model(model_id)
        except KeyError as error:
            raise errors.APIProblem(
                status.HTTP_404_NOT_FOUND,
                "model_not_installed",
                "Download this model before selecting it.",
            ) from error
        await ctx.readiness.warmup()
        state = await ctx.readiness.probe()
        return schemas.SelectModelResponse(
            engine=schemas.EngineStatus(
                id=engine_manager.runtime_config.engine, name=state.name, ready=state.ready
            )
        )


class _AdminModelUiRoutes:
    async def ui_models(self, ctx: context.GatewayContextDependency) -> HTMLResponse:
        return HTMLResponse(model_fragments.models_fragment(admin_queries.model_entries(ctx)))

    async def ui_models_list(
        self,
        ctx: context.GatewayContextDependency,
        filters: Annotated[_ModelFilterQuery, Depends()],
    ) -> HTMLResponse:
        return _models_list_html(ctx, filters)

    async def ui_start_download(
        self,
        model_id: str,
        ctx: context.GatewayContextDependency,
        filters: Annotated[_ModelFilterForm, Depends()],
    ) -> HTMLResponse:
        try:
            ctx.manager.start_download(model_id)
        except model_manager.UnknownModelError as error:
            raise errors.APIProblem(
                status.HTTP_404_NOT_FOUND, "unknown_model", "This model is not in the catalog."
            ) from error
        except model_manager.DownloadInProgressError as error:
            raise errors.APIProblem(
                status.HTTP_409_CONFLICT,
                DOWNLOAD_IN_PROGRESS_CODE,
                "This model is already downloading.",
            ) from error
        return _models_list_html(ctx, filters)

    async def ui_custom_download(
        self,
        url: CustomUrlForm,
        ctx: context.GatewayContextDependency,
        sha256: CustomDigestForm = "",
        filters: Annotated[_ModelFilterForm, Depends()] = None,  # type: ignore[assignment]
    ) -> HTMLResponse:
        active_filters = _ModelFilterForm() if filters is None else filters
        try:
            ctx.manager.start_custom_download(url, sha256)
        except ValueError as error:
            code = "invalid_model_digest" if "SHA-256" in str(error) else "invalid_model_url"
            raise errors.APIProblem(
                status.HTTP_422_UNPROCESSABLE_CONTENT, code, str(error)
            ) from error
        except model_manager.DownloadInProgressError as error:
            raise errors.APIProblem(
                status.HTTP_409_CONFLICT, DOWNLOAD_IN_PROGRESS_CODE, str(error)
            ) from error
        return _models_list_html(ctx, active_filters)

    async def ui_cancel_download(
        self,
        model_id: str,
        ctx: context.GatewayContextDependency,
        filters: Annotated[_ModelFilterForm, Depends()],
    ) -> HTMLResponse:
        ctx.manager.cancel_download(model_id)
        return _models_list_html(ctx, filters)

    async def ui_delete_model(
        self,
        model_id: str,
        ctx: context.GatewayContextDependency,
        filters: Annotated[_ModelFilterForm, Depends()],
    ) -> HTMLResponse:
        if ctx.engine_manager is not None:
            ctx.engine_manager.forget_if_active(model_id)
        try:
            ctx.manager.delete(model_id)
        except model_manager.DownloadInProgressError as error:
            raise errors.APIProblem(
                status.HTTP_409_CONFLICT,
                DOWNLOAD_IN_PROGRESS_CODE,
                "Cancel the download before deleting.",
            ) from error
        return _models_list_html(ctx, filters)

    async def ui_select_model(
        self,
        model_id: str,
        ctx: context.GatewayContextDependency,
        filters: Annotated[_ModelFilterForm, Depends()],
    ) -> HTMLResponse:
        engine_manager = ctx.engine_manager
        if engine_manager is None:
            raise errors.APIProblem(
                status.HTTP_409_CONFLICT,
                "engine_locked",
                "The engine was fixed at startup and cannot switch.",
            )
        try:
            engine_manager.select_model(model_id)
        except KeyError as error:
            raise errors.APIProblem(
                status.HTTP_404_NOT_FOUND,
                "model_not_installed",
                "Download this model before selecting it.",
            ) from error
        await ctx.readiness.warmup()
        state = await ctx.readiness.probe()
        entries = admin_queries.filtered_model_entries(
            ctx,
            installed_only=filters.installed_only,
            language=filters.language,
            family=filters.family,
            engine=filters.engine,
            max_size=filters.max_size,
            recommended_only=filters.recommended_only,
        )
        return HTMLResponse(
            model_fragments.models_list_fragment(
                entries,
                installed_only=filters.installed_only,
                languages=filters.language,
                families=filters.family,
                engines=filters.engine,
                max_size=filters.max_size,
                recommended_only=filters.recommended_only,
            )
            + engine_fragments.engine_pill_oob(
                schemas.EngineStatus(
                    id=engine_manager.runtime_config.engine, name=state.name, ready=state.ready
                ),
                bind_host=ctx.settings.bind_host,
                port=ctx.settings.port,
            )
        )


def _bind_routes(target_router: APIRouter) -> None:
    _AdminModelApiRoutes().bind(target_router)
    ui = _AdminModelUiRoutes()
    for path, endpoint, methods in (
        ("/ui/partials/models", ui.ui_models, METHOD_GET),
        ("/ui/partials/models-list", ui.ui_models_list, METHOD_GET),
        ("/ui/partials/models/{model_id}/download", ui.ui_start_download, METHOD_POST),
        ("/ui/partials/models/custom", ui.ui_custom_download, METHOD_POST),
        ("/ui/partials/models/{model_id}/cancel", ui.ui_cancel_download, METHOD_POST),
        ("/ui/partials/models/{model_id}", ui.ui_delete_model, METHOD_DELETE),
        ("/ui/partials/models/{model_id}/select", ui.ui_select_model, METHOD_POST),
    ):
        target_router.add_api_route(
            path, endpoint, methods=list(methods), response_class=HTMLResponse
        )


_bind_routes(router)
