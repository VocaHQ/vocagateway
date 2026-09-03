from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, Response, responses, staticfiles

from app import (
    audio,
    config,
    context,
    engines,
    errors,
    model_manager,
    readiness,
    runtime_config,
    schemas,
    service,
    storage,
    templating,
    tokens,
)
from app.models.base import AudioNormalizer, TranscriptionEngine
from app.routes import (
    admin_config,
    admin_models,
    admin_status,
    admin_tokens,
    health,
    pairing,
    sessions,
    streaming,
    transcriptions,
)

WEBUI_DIR = Path(__file__).parent / "webui"
SECURITY_CSP = (
    "default-src 'self'; script-src 'self'; "
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
    "font-src 'self' https://fonts.gstatic.com; "
    "img-src 'self' data:; connect-src 'self'; media-src 'self' blob:; "
    "object-src 'none'; base-uri 'none'; frame-ancestors 'none'; "
    "form-action 'self'"
)
ROUTERS = (
    health.router,
    sessions.router,
    transcriptions.router,
    streaming.router,
    admin_status.router,
    admin_tokens.router,
    admin_models.router,
    admin_config.router,
    pairing.router,
)


class _AppBuilder:
    def __init__(
        self,
        settings: config.Settings | None,
        engine: TranscriptionEngine | None,
        normalizer: AudioNormalizer | None,
        model_mgr: model_manager.ModelManager | None,
        run_cfg: runtime_config.RuntimeConfig | None,
    ) -> None:
        self.settings = settings
        self.engine = engine
        self.normalizer = normalizer
        self.model_manager = model_mgr
        self.runtime_config = run_cfg

    def build(self) -> FastAPI:
        ctx = self._build_context()
        openapi_url = "/openapi.json" if ctx.settings.debug else None
        app = FastAPI(
            title="VocaGateway",
            version=context.VERSION,
            docs_url=None,
            redoc_url=None,
            openapi_url=openapi_url,
            lifespan=_app_lifespan,
        )
        app.state.ctx = ctx
        app.middleware("http")(_browser_security_middleware)
        app.add_exception_handler(errors.APIProblem, _api_problem_handler)
        self._setup_routes(app, ctx)
        return app

    def _build_context(self) -> context.GatewayContext:
        cfg = self.settings or config.Settings.from_env()
        repo = storage.SessionRepository(cfg.data_dir / "sessions.sqlite3")
        repo.initialize()
        mgr = self.model_manager or model_manager.ModelManager(cfg.resolved_models_dir())
        setup = self._build_engine_setup(cfg, mgr)
        return self._assemble_context(cfg, repo, mgr, setup)

    def _build_engine_setup(
        self, cfg: config.Settings, mgr: model_manager.ModelManager
    ) -> tuple[engines.EngineProvider, engines.EngineManager | None, runtime_config.RuntimeConfig]:
        if self.engine is None:
            eng_mgr = engines.EngineManager(
                cfg,
                self.runtime_config or runtime_config.RuntimeConfig.load(cfg.config_path),
                cfg.config_path,
                mgr,
            )
            return eng_mgr, eng_mgr, eng_mgr.runtime_config
        return engines.StaticEngineProvider(self.engine), None, runtime_config.RuntimeConfig()

    def _assemble_context(
        self,
        cfg: config.Settings,
        repo: storage.SessionRepository,
        mgr: model_manager.ModelManager,
        setup: tuple[
            engines.EngineProvider,
            engines.EngineManager | None,
            runtime_config.RuntimeConfig,
        ],
    ) -> context.GatewayContext:
        provider, eng_mgr, pair_cfg = setup
        srv = service.TranscriptionService(
            cfg, repo, provider, self.normalizer or audio.FFmpegNormalizer()
        )
        return context.GatewayContext(
            settings=cfg,
            repository=repo,
            manager=mgr,
            token_store=tokens.TokenStore(cfg.data_dir / "device_tokens.json"),
            engine_provider=provider,
            engine_manager=eng_mgr,
            service=srv,
            readiness=readiness.ReadinessMonitor(provider),
            pairing_config=pair_cfg,
            config_path=cfg.config_path,
        )

    def _setup_routes(self, app: FastAPI, ctx: context.GatewayContext) -> None:
        for router in ROUTERS:
            app.include_router(router)
        app.add_api_route("/", _render_page, methods=["GET"], include_in_schema=False)
        if WEBUI_DIR.is_dir():
            app.mount("/assets", staticfiles.StaticFiles(directory=WEBUI_DIR), name="webui-assets")
        if ctx.settings.debug:
            app.add_api_route("/docs", _render_page, methods=["GET"], include_in_schema=False)


def create_app(
    settings: config.Settings | None = None,
    *,
    engine: TranscriptionEngine | None = None,
    normalizer: AudioNormalizer | None = None,
    model_manager: model_manager.ModelManager | None = None,
    runtime_config: runtime_config.RuntimeConfig | None = None,
) -> FastAPI:
    builder = _AppBuilder(
        settings=settings,
        engine=engine,
        normalizer=normalizer,
        model_mgr=model_manager,
        run_cfg=runtime_config,
    )
    return builder.build()


def select_engine(settings: config.Settings) -> TranscriptionEngine:
    """Resolve an engine purely from environment settings (CLI usage)."""
    if settings.engine not in runtime_config.VALID_ENGINES:
        raise RuntimeError("VOCAGATEWAY_ENGINE is not a supported engine.")
    mgr = model_manager.ModelManager(settings.resolved_models_dir())
    cfg = runtime_config.RuntimeConfig(engine=settings.engine)
    return engines.build_engine(settings, cfg, mgr)


@asynccontextmanager
async def _app_lifespan(app: FastAPI) -> Any:
    ctx: context.GatewayContext = app.state.ctx
    ctx.service.cleanup_expired()
    warmup_task = asyncio.create_task(ctx.readiness.warmup())
    app.state.warmup_task = warmup_task
    try:
        yield
    finally:
        if not warmup_task.done():
            warmup_task.cancel()
            with suppress(asyncio.CancelledError):
                await warmup_task
        await asyncio.to_thread(engines.close_engine, ctx.engine_provider.current())


async def _browser_security_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    response = await call_next(request)
    headers = response.headers
    headers["X-Content-Type-Options"] = "nosniff"
    headers["X-Frame-Options"] = "DENY"
    headers["Referrer-Policy"] = "no-referrer"
    headers["Permissions-Policy"] = "microphone=(self)"
    headers["Content-Security-Policy"] = SECURITY_CSP
    path = request.url.path
    if path == "/" or path.startswith(("/ui/", "/v1/")):
        headers["Cache-Control"] = "no-store"
    return response


async def _api_problem_handler(_: Request, exc: Exception) -> responses.JSONResponse:
    if isinstance(exc, errors.APIProblem):
        envelope = schemas.ErrorEnvelope(
            error=schemas.ErrorDetail(
                code=exc.code,
                message=exc.message,
                recoverable=exc.recoverable,
            )
        )
        return responses.JSONResponse(status_code=exc.status_code, content=envelope.model_dump())
    raise exc


async def _render_page(request: Request) -> Response:
    if request.url.path == "/docs":
        return responses.HTMLResponse(
            templating.render("docs/swagger.html", title=request.app.title)
        )
    return responses.FileResponse(WEBUI_DIR / "index.html")
