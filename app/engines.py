from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from pathlib import Path
from types import MappingProxyType
from typing import Protocol

from app import catalog, config, model_manager, runtime_config, system
from app.models import (
    base,
    faster_whisper,
    handy,
    mlx_audio,
    moonshine,
    sherpa_onnx,
    vocamac,
    whisper_cpp,
)
from app.models.whisperkit import WhisperKitEngine

ENGINE_VOCAMAC = "vocamac"
ENGINE_HANDY = "handy"
engine_runs_here = system.engine_runs_here
engine_requirement = system.engine_requirement

_MODEL_ATTRS = MappingProxyType(
    {
        catalog.ENGINE_SHERPA_ONNX: "sherpa_model",
        catalog.ENGINE_MLX_AUDIO: "mlx_audio_model",
        catalog.ENGINE_WHISPERKIT: "whisperkit_model",
        catalog.ENGINE_FASTER_WHISPER: "faster_whisper_model",
    }
)


class EngineProvider(Protocol):
    def current(self) -> base.TranscriptionEngine: ...

    def lease(self) -> AbstractAsyncContextManager[base.TranscriptionEngine]: ...


class StaticEngineProvider:
    """Wraps a fixed engine (used by tests and the cleanup CLI)."""

    def __init__(self, engine: base.TranscriptionEngine) -> None:
        self._engine = engine

    def current(self) -> base.TranscriptionEngine:
        return self._engine

    @asynccontextmanager
    async def lease(self) -> AsyncIterator[base.TranscriptionEngine]:
        yield self._engine


def close_engine(engine: base.TranscriptionEngine) -> None:
    """Release optional persistent resources owned by an engine."""
    unload = getattr(engine, "unload", None)
    if callable(unload):
        unload()
        return
    close = getattr(engine, "close", None)
    if callable(close):
        close()


class _ModelPathResolver:
    def __init__(
        self,
        settings: config.Settings,
        model_mgr: model_manager.ModelManager,
    ) -> None:
        self.settings = settings
        self.model_manager = model_mgr

    def resolve_path(self, engine: str, rc: runtime_config.RuntimeConfig) -> Path | None:
        configured = rc.whisper_model
        if engine == catalog.ENGINE_WHISPERKIT:
            configured = rc.whisperkit_model
        elif engine == catalog.ENGINE_FASTER_WHISPER:
            configured = rc.faster_whisper_model
        if configured:
            return Path(configured)
        installed = [model for model in self.model_manager.installed() if model.engine == engine]
        if installed:
            chosen = (
                min(installed, key=lambda model: model.size_bytes)
                if engine == catalog.ENGINE_FASTER_WHISPER
                else max(installed, key=lambda model: model.size_bytes)
            )
            return chosen.path
        return self.settings.whisper_model if engine == catalog.ENGINE_WHISPER_CPP else None

    def catalog_selection(
        self, model_id: str | None, engine: str
    ) -> tuple[Path | None, catalog.CatalogModel | None]:
        if model_id:
            return (
                self.model_manager.installed_path(model_id),
                self.model_manager.catalog_model(model_id),
            )
        installed = [model for model in self.model_manager.installed() if model.engine == engine]
        if not installed:
            return None, None
        chosen = min(installed, key=lambda model: model.size_bytes)
        return chosen.path, self.model_manager.catalog_model(chosen.id)

    def catalog_model_for_path(self, path: Path | None) -> catalog.CatalogModel | None:
        if path is None:
            return None
        for installed in self.model_manager.installed():
            if installed.path == path:
                return self.model_manager.catalog_model(installed.id)
        return None

    def apply_model(self, rc: runtime_config.RuntimeConfig, model_id: str, path_str: str) -> None:
        prefix = model_id.split(":", 1)[0]
        if prefix == catalog.ENGINE_MOONSHINE:
            cat_model = self.model_manager.catalog_model(model_id)
            if cat_model is None:
                raise KeyError(model_id)
            rc.engine = catalog.ENGINE_MOONSHINE
            rc.moonshine_model = model_id
            rc.moonshine_language = cat_model.language_code or "en"
            return
        rc.engine = prefix if prefix in runtime_config.VALID_ENGINES else catalog.ENGINE_WHISPER_CPP
        named = {catalog.ENGINE_SHERPA_ONNX, catalog.ENGINE_MLX_AUDIO}
        model_ref = model_id if prefix in named else path_str
        setattr(rc, _MODEL_ATTRS.get(prefix, "whisper_model"), model_ref)

    def forget_if_active(
        self, rc: runtime_config.RuntimeConfig, model_id: str, path_str: str
    ) -> bool:
        changed = False
        if model_id == rc.moonshine_model and rc.engine == catalog.ENGINE_MOONSHINE:
            rc.moonshine_model = "moonshine:en"
            rc.engine = runtime_config.AUTO_ENGINE
            changed = True
        if rc.whisper_model == path_str:
            rc.whisper_model = None
            changed = True
        if rc.whisperkit_model == path_str:
            rc.whisperkit_model = None
            changed = True
        if rc.faster_whisper_model == path_str:
            rc.faster_whisper_model = None
            changed = True
        if model_id == rc.sherpa_model and rc.engine == catalog.ENGINE_SHERPA_ONNX:
            rc.sherpa_model = None
            rc.engine = runtime_config.AUTO_ENGINE
            changed = True
        if model_id == rc.mlx_audio_model and rc.engine == catalog.ENGINE_MLX_AUDIO:
            rc.mlx_audio_model = None
            rc.engine = runtime_config.AUTO_ENGINE
            changed = True
        return changed

    def resolve_auto(self, rc: runtime_config.RuntimeConfig) -> base.TranscriptionEngine:
        voc = vocamac.VocaMacEngine(
            self.settings.whisperkit_binary,
            self.settings.vocamac_model,
            app_path=self.settings.vocamac_app,
        )
        if voc.is_available():
            return voc
        if self.settings.handy_binary.is_file():
            return handy.HandyEngine(
                self.settings.handy_binary,
                self.settings.handy_model,
                fallback_model=self.settings.handy_fallback_model,
            )
        return self._resolve_fallback(rc)

    def _resolve_fallback(self, rc: runtime_config.RuntimeConfig) -> base.TranscriptionEngine:
        wk_p = self.resolve_path(catalog.ENGINE_WHISPERKIT, rc)
        if wk_p is not None:
            return WhisperKitEngine(self.settings.whisperkit_binary, wk_p)
        mlx_sel = self.catalog_selection(rc.mlx_audio_model, catalog.ENGINE_MLX_AUDIO)
        if mlx_sel[0] is not None:
            return mlx_audio.MLXAudioEngine(mlx_sel[0], mlx_sel[1])
        shp_sel = self.catalog_selection(rc.sherpa_model, catalog.ENGINE_SHERPA_ONNX)
        if shp_sel[0] is not None:
            return sherpa_onnx.SherpaOnnxEngine(shp_sel[0], shp_sel[1], cpu_threads=rc.cpu_threads)
        fw_p = self.resolve_path(catalog.ENGINE_FASTER_WHISPER, rc)
        if fw_p is not None:
            return faster_whisper.FasterWhisperEngine(
                fw_p,
                device=rc.compute_device,
                compute_type=rc.compute_type,
                cpu_threads=rc.cpu_threads,
            )
        cpp_p = self.resolve_path(catalog.ENGINE_WHISPER_CPP, rc) or self.settings.whisper_model
        return whisper_cpp.WhisperCppEngine(
            self.settings.whisper_binary,
            cpp_p,
            self.catalog_model_for_path(cpp_p),
            cpu_threads=rc.cpu_threads,
            server_binary=self.settings.whisper_server_binary,
        )


class _EngineBuilder:
    def __init__(
        self,
        settings: config.Settings,
        model_mgr: model_manager.ModelManager,
    ) -> None:
        self.settings = settings
        self.model_manager = model_mgr
        self.resolver = _ModelPathResolver(settings, model_mgr)

    def build(self, rc: runtime_config.RuntimeConfig) -> base.TranscriptionEngine:
        engine_name = self.settings.engine
        if engine_name == runtime_config.AUTO_ENGINE:
            engine_name = rc.engine or runtime_config.AUTO_ENGINE
        if engine_name == runtime_config.AUTO_ENGINE:
            return self.resolver.resolve_auto(rc)
        return self._build_named(engine_name, rc)

    def swap(
        self,
        previous: base.TranscriptionEngine,
        rc: runtime_config.RuntimeConfig,
        config_path: Path,
    ) -> base.TranscriptionEngine:
        rc.save(config_path)
        new_engine = self.build(rc)
        close_engine(previous)
        return new_engine

    def validate(self, engine: str, device: str, compute_type: str, cpu_threads: int) -> None:
        err = self._check_error(engine) or self._check_perf(device, compute_type, cpu_threads)
        if err:
            raise ValueError(err)

    def _check_error(self, engine: str) -> str | None:
        if engine not in runtime_config.VALID_ENGINES:
            valid_names = ", ".join(runtime_config.VALID_ENGINES)
            return f"Engine must be one of: {valid_names}."
        if not engine_runs_here(engine):
            requirement = engine_requirement(engine)
            return f"The {engine} engine runs only on {requirement}."
        return None

    def _check_perf(self, device: str, compute_type: str, cpu_threads: int) -> str | None:
        if device not in {runtime_config.AUTO_ENGINE, "cpu", "cuda"}:
            return "Invalid compute device."
        valid_compute = {
            runtime_config.AUTO_ENGINE,
            "int8",
            "int8_float16",
            "float16",
            "float32",
        }
        if compute_type not in valid_compute:
            return "Invalid compute type."
        if not 0 <= cpu_threads <= runtime_config.MAXIMUM_CPU_THREADS:
            return "CPU threads must be between 0 and 256."
        return None

    def _build_named(
        self, engine: str, rc: runtime_config.RuntimeConfig
    ) -> base.TranscriptionEngine:
        if engine in {ENGINE_VOCAMAC, ENGINE_HANDY}:
            return (
                vocamac.VocaMacEngine(
                    self.settings.whisperkit_binary,
                    self.settings.vocamac_model,
                    app_path=self.settings.vocamac_app,
                )
                if engine == ENGINE_VOCAMAC
                else handy.HandyEngine(
                    self.settings.handy_binary,
                    self.settings.handy_model,
                    fallback_model=self.settings.handy_fallback_model,
                )
            )
        if engine in {catalog.ENGINE_SHERPA_ONNX, catalog.ENGINE_MLX_AUDIO}:
            sherpa = engine == catalog.ENGINE_SHERPA_ONNX
            model_id = rc.sherpa_model if sherpa else rc.mlx_audio_model
            sel = self.resolver.catalog_selection(model_id, engine)
            return (
                sherpa_onnx.SherpaOnnxEngine(sel[0], sel[1], cpu_threads=rc.cpu_threads)
                if sherpa
                else mlx_audio.MLXAudioEngine(sel[0], sel[1])
            )
        if engine == catalog.ENGINE_MOONSHINE:
            path = self.model_manager.installed_path(rc.moonshine_model)
            return moonshine.MoonshineEngine(path, rc.moonshine_language)
        if engine == catalog.ENGINE_FASTER_WHISPER:
            return faster_whisper.FasterWhisperEngine(
                self.resolver.resolve_path(engine, rc),
                device=rc.compute_device,
                compute_type=rc.compute_type,
                cpu_threads=rc.cpu_threads,
            )
        path = self.resolver.resolve_path(engine, rc)
        return (
            WhisperKitEngine(self.settings.whisperkit_binary, path)
            if engine == catalog.ENGINE_WHISPERKIT
            else whisper_cpp.WhisperCppEngine(
                self.settings.whisper_binary,
                path or self.settings.whisper_model,
                self.resolver.catalog_model_for_path(path or self.settings.whisper_model),
                cpu_threads=rc.cpu_threads,
                server_binary=self.settings.whisper_server_binary,
            )
        )


class EngineManager:
    """Builds and hot-swaps transcription engines from WebUI choices."""

    def __init__(
        self,
        settings: config.Settings,
        runtime_config: runtime_config.RuntimeConfig,
        config_path: Path,
        model_manager: model_manager.ModelManager,
    ) -> None:
        self.settings = settings
        self.runtime_config = runtime_config
        self.config_path = config_path
        self.model_manager = model_manager
        self._builder = _EngineBuilder(settings, model_manager)
        self._engine = self._builder.build(runtime_config)
        self._active_leases = 0
        self._last_used_at = time.monotonic()
        self._model_was_offloaded = False

    def current(self) -> base.TranscriptionEngine:
        return self._engine

    @asynccontextmanager
    async def lease(self) -> AsyncIterator[base.TranscriptionEngine]:
        self._active_leases += 1
        engine = self._engine
        try:
            yield engine
        finally:
            self._finish_lease()

    async def health(self) -> base.EngineHealth:
        return await self._engine.health()

    def select_model(self, model_id: str) -> None:
        path = self.model_manager.installed_path(model_id)
        if path is None:
            raise KeyError(model_id)
        self._builder.resolver.apply_model(self.runtime_config, model_id, str(path))
        self._engine = self._builder.swap(self._engine, self.runtime_config, self.config_path)
        self._model_was_offloaded = False

    def forget_if_active(self, model_id: str) -> None:
        path = self.model_manager.installed_path(model_id)
        if path is None:
            return
        if self._builder.resolver.forget_if_active(self.runtime_config, model_id, str(path)):
            self._engine = self._builder.swap(self._engine, self.runtime_config, self.config_path)

    def configure(
        self,
        engine: str,
        device: str | None = None,
        compute_type: str | None = None,
        cpu_threads: int | None = None,
        idle_offload_enabled: bool | None = None,
        idle_offload_minutes: int | None = None,
    ) -> None:
        rc = self.runtime_config
        dev = rc.compute_device if device is None else device
        ctype = rc.compute_type if compute_type is None else compute_type
        threads = rc.cpu_threads if cpu_threads is None else cpu_threads
        offload_enabled = (
            rc.idle_offload_enabled if idle_offload_enabled is None else idle_offload_enabled
        )
        offload_minutes = (
            rc.idle_offload_minutes if idle_offload_minutes is None else idle_offload_minutes
        )
        self._builder.validate(engine, dev, ctype, threads)
        if offload_minutes not in runtime_config.IDLE_OFFLOAD_MINUTES:
            valid_minutes = ", ".join(
                str(minutes) for minutes in runtime_config.IDLE_OFFLOAD_MINUTES
            )
            raise ValueError(f"Idle offload minutes must be one of: {valid_minutes}.")
        rc.engine = engine
        rc.compute_device = dev
        rc.compute_type = ctype
        rc.cpu_threads = threads
        rc.idle_offload_enabled = offload_enabled
        rc.idle_offload_minutes = offload_minutes
        if engine != catalog.ENGINE_WHISPER_CPP:
            rc.whisper_model = None
        if engine != catalog.ENGINE_WHISPERKIT:
            rc.whisperkit_model = None
        if engine != catalog.ENGINE_FASTER_WHISPER:
            rc.faster_whisper_model = None
        if engine != catalog.ENGINE_SHERPA_ONNX:
            rc.sherpa_model = None
        if engine != catalog.ENGINE_MLX_AUDIO:
            rc.mlx_audio_model = None
        self._engine = self._builder.swap(self._engine, rc, self.config_path)
        self._model_was_offloaded = False

    def update_performance(self, device: str, compute_type: str, cpu_threads: int) -> None:
        self.configure(self.runtime_config.engine, device, compute_type, cpu_threads)

    set_engine = configure

    @property
    def model_is_offloaded(self) -> bool:
        return self._model_was_offloaded

    def offload_if_idle(self, *, now: float | None = None) -> bool:
        rc = self.runtime_config
        engine = self._resident_engine()
        if not rc.idle_offload_enabled or engine is None or self._active_leases:
            return False
        current_time = time.monotonic() if now is None else now
        maximum_idle = rc.idle_offload_minutes * 60
        if current_time - self._last_used_at < maximum_idle:
            return False
        engine.unload()
        self._model_was_offloaded = True
        return True

    def _resident_engine(self) -> base.MemoryResidentEngine | None:
        engine = self._engine
        if isinstance(engine, base.MemoryResidentEngine) and engine.model_is_resident:
            return engine
        return None

    def _finish_lease(self) -> None:
        self._active_leases -= 1
        self._last_used_at = time.monotonic()
        if self._resident_engine() is not None:
            self._model_was_offloaded = False


def build_engine(
    settings: config.Settings,
    runtime_config: runtime_config.RuntimeConfig,
    model_manager: model_manager.ModelManager,
) -> base.TranscriptionEngine:
    return _EngineBuilder(settings, model_manager).build(runtime_config)
