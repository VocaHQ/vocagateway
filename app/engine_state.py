from __future__ import annotations

from pathlib import Path

from app.catalog import CatalogModel
from app.context import GatewayContext
from app.system import detect_system, engine_runs_on

CATALOG_MODEL_ATTRIBUTES = (
    ("moonshine", "moonshine_model"),
    ("sherpa-onnx", "sherpa_model"),
    ("mlx-audio", "mlx_audio_model"),
)
PATH_MODEL_ATTRIBUTES = (
    "transcribe_model",
    "whisper_model",
    "whisperkit_model",
    "faster_whisper_model",
)


def engine_id(ctx: GatewayContext) -> str:
    if ctx.engine_manager is not None:
        return ctx.engine_manager.runtime_config.engine
    return "custom"


def available_engines(ctx: GatewayContext) -> list[str]:
    if ctx.engine_manager is None:
        return ["custom"]
    settings = ctx.settings
    system = detect_system(
        whisper_binary=settings.whisper_binary,
        whisperkit_binary=settings.whisperkit_binary,
        handy_binary=settings.handy_binary,
        vocamac_app=settings.vocamac_app,
        transcribe_binary=settings.transcribe_binary,
    )
    engines = [
        "auto",
        "sherpa-onnx",
        "faster-whisper",
        "moonshine",
        "whisper.cpp",
        "transcribe.cpp",
    ]
    # The desktop-app and Apple-native adapters only exist on a matching host.
    engines.extend(
        engine
        for engine in ("vocamac", "handy", "whisperkit", "mlx-audio")
        if engine_runs_on(
            engine,
            is_mac=system.os_name == "Darwin",
            is_apple_silicon=system.is_apple_silicon,
        )
    )
    return engines


def active_model_path(ctx: GatewayContext) -> Path | None:
    if ctx.engine_manager is None:
        return None
    config = ctx.engine_manager.runtime_config
    for engine_name, attribute_name in CATALOG_MODEL_ATTRIBUTES:
        if config.engine == engine_name:
            model_id = getattr(config, attribute_name)
            return ctx.manager.installed_path(model_id) if model_id else None
    for attribute_name in PATH_MODEL_ATTRIBUTES:
        model_path = getattr(config, attribute_name)
        if model_path:
            return Path(model_path)
    return None


def active_catalog_model(ctx: GatewayContext) -> CatalogModel | None:
    """The catalog entry behind the running engine, if it came from the catalog.

    sherpa-onnx and MLX engines hold their entry directly; the others are
    configured by path, so those are matched back through the installed list.
    Returns None for an imported model or when nothing is selected, which the
    clients read as "no restriction".
    """
    engine = ctx.engine_provider.current()
    held = getattr(engine, "catalog_model", None)
    if isinstance(held, CatalogModel):
        return held
    active = active_model_path(ctx)
    if active is None:
        return None
    for installed in ctx.manager.installed():
        if installed.path == active:
            return ctx.manager.catalog_model(installed.id)
    return None
