from __future__ import annotations

from importlib import util as importlib_util
from types import MappingProxyType
from typing import Any

from app import schemas
from app.build_info import current_commit
from app.catalog import catalog_source_url, language_names, recommended_ids
from app.context import BOOTSTRAP_TOKEN_ID, TOKEN_FILE_HINT, VERSION, GatewayContext
from app.engine_state import active_model_path, available_engines, engine_id
from app.runtime_config import DEFAULT_IDLE_OFFLOAD_MINUTES
from app.serializers import metrics_status, model_covers
from app.system import detect_system

PYTHON_PACKAGE_PATH = "Python package"

SIZE_FILTER_CAPS: MappingProxyType[str, int] = MappingProxyType(
    {
        "100mb": 100_000_000,
        "300mb": 300_000_000,
        "800mb": 800_000_000,
        "1500mb": 1_500_000_000,
    }
)


class _SystemDependencyHelper:
    def __init__(self, ctx: GatewayContext) -> None:
        self.ctx = ctx
        self.settings = ctx.settings
        self.system = detect_system(
            whisper_binary=self.settings.whisper_binary,
            whisperkit_binary=self.settings.whisperkit_binary,
            handy_binary=self.settings.handy_binary,
            vocamac_app=self.settings.vocamac_app,
        )

    def build_dependencies(self) -> list[schemas.DependencyStatus]:
        is_mac = self.system.os_name == "Darwin"
        ffmpeg_hint = (
            "brew install ffmpeg" if is_mac else "Install FFmpeg with your Linux package manager"
        )
        whisper_hint = (
            "brew install whisper-cpp"
            if is_mac
            else "Included in Docker or build whisper.cpp from source"
        )
        deps = [
            schemas.DependencyStatus(
                name="FFmpeg",
                available=self.system.ffmpeg_path is not None,
                path=self.system.ffmpeg_path,
                install_hint=ffmpeg_hint,
            ),
            schemas.DependencyStatus(
                name="whisper.cpp CLI",
                available=self.system.whisper_cpp_path is not None,
                path=self.system.whisper_cpp_path,
                install_hint=whisper_hint,
            ),
        ]
        deps.extend(self._python_dependencies())
        deps.extend(self._app_dependencies(is_mac))
        return deps

    def build_checklist(
        self, dependencies: list[schemas.DependencyStatus], ready: bool
    ) -> schemas.SetupChecklist:
        return schemas.SetupChecklist(
            token_configured=True,
            ffmpeg_available=self.system.ffmpeg_path is not None,
            engine_binary_available=any(dep.available for dep in dependencies[1:]),
            model_installed=bool(self.ctx.manager.installed())
            or (self.ctx.engine_manager is not None and ready),
            engine_ready=ready,
        )

    def build_commit_status(self) -> schemas.CommitStatus | None:
        if self.ctx.settings.debug:
            commit = current_commit()
            if commit:
                return schemas.CommitStatus(
                    sha=commit.sha,
                    short_sha=commit.short_sha,
                    subject=commit.subject,
                    committed_at=commit.committed_at,
                )
        return None

    def os_summary(self) -> str:
        name = self.system.os_name
        version = self.system.os_version
        return f"{name} {version}"

    def _python_dependencies(self) -> list[schemas.DependencyStatus]:
        silicon = self.system.is_apple_silicon
        mlx_ready = silicon and importlib_util.find_spec("mlx_audio") is not None
        mlx_hint = (
            "Install vocagateway[apple]" if silicon else "Available only on Apple-silicon Macs"
        )
        return [
            schemas.DependencyStatus(
                name="faster-whisper",
                available=importlib_util.find_spec("faster_whisper") is not None,
                path=PYTHON_PACKAGE_PATH if importlib_util.find_spec("faster_whisper") else None,
                install_hint="Install vocagateway[engines] or use the Docker image",
            ),
            schemas.DependencyStatus(
                name="Moonshine Voice",
                available=importlib_util.find_spec("moonshine_voice") is not None,
                path=PYTHON_PACKAGE_PATH if importlib_util.find_spec("moonshine_voice") else None,
                install_hint="Install vocagateway[engines] or use the Docker image",
            ),
            schemas.DependencyStatus(
                name="sherpa-onnx",
                available=importlib_util.find_spec("sherpa_onnx") is not None,
                path=PYTHON_PACKAGE_PATH if importlib_util.find_spec("sherpa_onnx") else None,
                install_hint="Install vocagateway[engines] or use the Docker image",
            ),
            schemas.DependencyStatus(
                name="MLX Audio",
                available=mlx_ready,
                path=PYTHON_PACKAGE_PATH if mlx_ready else None,
                install_hint=mlx_hint,
            ),
        ]

    def _app_dependencies(self, is_mac: bool) -> list[schemas.DependencyStatus]:
        silicon = self.system.is_apple_silicon
        wk_hint = "brew install whisperkit-cli" if is_mac else "Available only on Apple platforms"
        vocamac_hint = (
            "https://github.com/jatinkrmalik/vocamac"
            if silicon
            else "Available only on Apple silicon Macs"
        )
        return [
            schemas.DependencyStatus(
                name="WhisperKit CLI",
                available=self.system.whisperkit_cli_path is not None,
                path=self.system.whisperkit_cli_path,
                install_hint=wk_hint,
            ),
            schemas.DependencyStatus(
                name="Handy app",
                available=self.system.handy_installed,
                path=str(self.settings.handy_binary) if self.system.handy_installed else None,
                install_hint="https://handy.computer" if is_mac else "Available only on macOS",
            ),
            schemas.DependencyStatus(
                name="VocaMac app",
                available=self.system.vocamac_installed,
                path=str(self.settings.vocamac_app) if self.system.vocamac_installed else None,
                install_hint=vocamac_hint,
            ),
        ]


_ModelState = tuple[str, float | None, str | None]


class _ModelEntryHelper:
    def __init__(self, ctx: GatewayContext) -> None:
        self.ctx = ctx
        self.system = detect_system(
            whisper_binary=ctx.settings.whisper_binary,
            whisperkit_binary=ctx.settings.whisperkit_binary,
            handy_binary=ctx.settings.handy_binary,
            vocamac_app=ctx.settings.vocamac_app,
        )
        self.recommended = recommended_ids(self.system)
        self.installed = {model.id: model for model in ctx.manager.installed()}
        self.active_path = active_model_path(ctx)

    def collect_entries(self) -> list[schemas.AdminModelEntry]:
        catalog_entries = [
            self.build_entry(model) for model in self.ctx.manager.catalog if self.is_visible(model)
        ]
        catalog_ids = {entry.id for entry in catalog_entries}
        retired_entries = [
            self.build_entry(model)
            for installed in self.ctx.manager.installed()
            if installed.retired
            and installed.id not in catalog_ids
            and (model := self.ctx.manager.catalog_model(installed.id)) is not None
        ]
        custom_entries = [
            self.build_custom_entry(custom)
            for custom in self.ctx.manager.installed()
            if custom.id.startswith("custom:")
        ]
        return catalog_entries + retired_entries + custom_entries

    def is_visible(self, model: Any) -> bool:
        if self.system.is_apple_silicon:
            return True
        return model.engine != "whisperkit" and not model.apple_silicon_only

    def build_entry(self, model: Any) -> schemas.AdminModelEntry:
        download = self.ctx.manager.download_state(model.id)
        inst = self.installed.get(model.id)
        resolution = self._resolve_state(download, inst)
        is_active = inst is not None and inst.path == self.active_path
        is_offloaded = bool(
            is_active
            and self.ctx.engine_manager is not None
            and self.ctx.engine_manager.model_is_offloaded
        )
        return schemas.AdminModelEntry(
            id=model.id,
            engine=model.engine,
            label=model.label,
            size_bytes=inst.size_bytes if inst else model.size_bytes,
            languages=model.languages,
            quality=model.quality,
            family=model.family,
            description=model.description,
            source=model.source,
            source_url=catalog_source_url(model),
            supports_streaming=model.supports_streaming,
            license_name=model.license_name,
            commercial_use=model.commercial_use,
            detects_language_automatically=model.detects_language_automatically,
            language_names=language_names(model.language_codes),
            language_codes=list(model.language_codes),
            state=resolution[0],
            active=is_active,
            offloaded=is_offloaded,
            recommended=model.id in self.recommended,
            progress=resolution[1],
            downloaded_bytes=download.downloaded_bytes if download else None,
            total_bytes=download.total_bytes if download else None,
            error=resolution[2],
            retired=model.retired,
            replacement_id=model.replacement_id,
            retirement_reason=model.retirement_reason,
        )

    def build_custom_entry(self, custom: Any) -> schemas.AdminModelEntry:
        key = custom.key
        return schemas.AdminModelEntry(
            id=custom.id,
            engine=custom.engine,
            label=f"Custom: {key}",
            size_bytes=custom.size_bytes,
            languages="Unknown",
            quality="Custom",
            family="Custom Whisper",
            description="User-provided local model.",
            source="Local file",
            state="installed",
            active=custom.path == self.active_path,
            offloaded=bool(
                custom.path == self.active_path
                and self.ctx.engine_manager is not None
                and self.ctx.engine_manager.model_is_offloaded
            ),
            recommended=False,
        )

    def filter_by_criteria(
        self,
        entries: list[schemas.AdminModelEntry],
        language: Any,
        family: Any,
        engine: Any,
        max_size: Any,
    ) -> list[schemas.AdminModelEntry]:
        matching = entries
        if language:
            codes = [language] if isinstance(language, str) else list(language)
            matching = [
                model for model in matching if any(model_covers(model, code) for code in codes)
            ]
        if family:
            fam_set = set([family] if isinstance(family, str) else family)
            matching = [model for model in matching if model.family in fam_set]
        if engine:
            eng_set = set([engine] if isinstance(engine, str) else engine)
            matching = [model for model in matching if model.engine in eng_set]
        cap = SIZE_FILTER_CAPS.get(str(max_size).strip().lower())
        if cap is not None:
            matching = [model for model in matching if model.size_bytes <= cap]
        return matching

    def _resolve_state(self, download: Any, inst: Any) -> _ModelState:
        if download and download.status == "downloading":
            progress = None
            if download.total_bytes:
                progress = round(download.downloaded_bytes / download.total_bytes, 4)
            return "downloading", progress, None
        if inst:
            return "installed", None, None
        if download and download.status == "failed":
            return "not_installed", None, download.error
        return "not_installed", None, None


async def status_payload(ctx: GatewayContext) -> schemas.AdminStatusResponse:
    helper = _SystemDependencyHelper(ctx)
    dependencies = helper.build_dependencies()
    readiness_details = await ctx.readiness.details()
    state = readiness_details.health
    metrics = ctx.service.metrics.snapshot(sample=True)
    return schemas.AdminStatusResponse(
        version=VERSION,
        commit=helper.build_commit_status(),
        engine=schemas.EngineStatus(id=engine_id(ctx), name=state.name, ready=state.ready),
        system=schemas.SystemStatus(
            os=helper.os_summary(),
            arch=helper.system.arch,
            chip=helper.system.chip,
            ram_gb=helper.system.ram_gb,
            is_apple_silicon=helper.system.is_apple_silicon,
            logical_cpus=helper.system.logical_cpus,
            effective_cpus=helper.system.effective_cpus,
            containerized=helper.system.containerized,
            accelerators=list(helper.system.accelerators),
            cpu_features=list(helper.system.cpu_features),
        ),
        dependencies=dependencies,
        paths=schemas.PathStatus(
            data_dir=str(ctx.settings.data_dir),
            models_dir=str(ctx.manager.models_dir),
            config_file=str(ctx.config_path),
            token_file=TOKEN_FILE_HINT,
        ),
        bind_host=ctx.settings.bind_host,
        port=ctx.settings.port,
        setup=helper.build_checklist(dependencies, state.ready),
        metrics=metrics_status(metrics),
        readiness=schemas.ReadinessStatus(
            probe_age_seconds=round(readiness_details.checked_age_seconds, 3),
            warmup_state=readiness_details.warmup_state,
            warmed_bytes=readiness_details.warmed_bytes,
        ),
    )


def model_entries(ctx: GatewayContext) -> list[schemas.AdminModelEntry]:
    return _ModelEntryHelper(ctx).collect_entries()


def filtered_model_entries(
    ctx: GatewayContext,
    installed_only: bool = False,
    language: str | list[str] | None = None,
    family: str | list[str] | None = None,
    **kwargs: Any,
) -> list[schemas.AdminModelEntry]:
    helper = _ModelEntryHelper(ctx)
    entries = model_entries(ctx)
    if installed_only:
        entries = [entry for entry in entries if entry.state == "installed"]
    entries = helper.filter_by_criteria(
        entries,
        language=language,
        family=family,
        engine=kwargs.get("engine"),
        max_size=kwargs.get("max_size", ""),
    )
    if kwargs.get("recommended_only", False):
        entries = [entry for entry in entries if entry.recommended]
    return entries


def token_entries(ctx: GatewayContext) -> list[schemas.DeviceTokenEntry]:
    entries = [
        schemas.DeviceTokenEntry(
            id=BOOTSTRAP_TOKEN_ID,
            label="Bootstrap token (VOCAGATEWAY_TOKEN / token file)",
            created_at=None,
            revocable=False,
        )
    ]
    entries.extend(
        schemas.DeviceTokenEntry(
            id=token.id, label=token.label, created_at=token.created_at, revocable=True
        )
        for token in ctx.token_store.all()
    )
    return entries


def config_response(ctx: GatewayContext) -> schemas.ConfigResponse:
    rc = ctx.engine_manager.runtime_config if ctx.engine_manager else None
    if rc:
        return schemas.ConfigResponse(
            engine=rc.engine,
            available_engines=available_engines(ctx),
            whisper_model=rc.whisper_model,
            whisperkit_model=rc.whisperkit_model,
            faster_whisper_model=rc.faster_whisper_model,
            moonshine_model=rc.moonshine_model,
            moonshine_language=rc.moonshine_language,
            sherpa_model=rc.sherpa_model,
            mlx_audio_model=rc.mlx_audio_model,
            compute_device=rc.compute_device,
            compute_type=rc.compute_type,
            cpu_threads=rc.cpu_threads,
            idle_offload_enabled=rc.idle_offload_enabled,
            idle_offload_minutes=rc.idle_offload_minutes,
        )
    return schemas.ConfigResponse(
        engine="custom",
        available_engines=available_engines(ctx),
        whisper_model=None,
        whisperkit_model=None,
        faster_whisper_model=None,
        moonshine_model="moonshine:en",
        moonshine_language="en",
        sherpa_model=None,
        mlx_audio_model=None,
        compute_device="auto",
        compute_type="auto",
        cpu_threads=0,
        idle_offload_enabled=False,
        idle_offload_minutes=DEFAULT_IDLE_OFFLOAD_MINUTES,
    )
