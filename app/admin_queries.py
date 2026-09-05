from __future__ import annotations

from dataclasses import dataclass
from importlib import util as importlib_util
from types import MappingProxyType
from typing import Any

from app import schemas
from app.build_info import current_commit
from app.catalog import catalog_source_url, language_names, recommended_ids
from app.config import Settings
from app.context import BOOTSTRAP_TOKEN_ID, TOKEN_FILE_HINT, VERSION, GatewayContext
from app.engine_state import active_model_path, available_engines, engine_id
from app.runtime_config import DEFAULT_IDLE_OFFLOAD_MINUTES
from app.serializers import metrics_status, model_covers
from app.system import SystemInfo, detect_system

PYTHON_PACKAGE_PATH = "Python package"
# The GGUF models in the catalog need a transcribe-cli this project never ships;
# the tile names the build and the override so a missing binary is actionable.
TRANSCRIBE_INSTALL_HINT = (
    "Build https://github.com/handy-computer/transcribe.cpp, then set "
    "VOCAGATEWAY_TRANSCRIBE_BINARY to its transcribe-cli"
)
PYTHON_ENGINE_INSTALL_HINT = "Install vocagateway[engines] or use the Docker image"


@dataclass(frozen=True, slots=True)
class _EngineTile:
    """An engine and the one runtime it needs, as the Overview panel shows it."""

    engine: str
    status: schemas.DependencyStatus


SIZE_FILTER_CAPS: MappingProxyType[str, int] = MappingProxyType(
    {
        "100mb": 100_000_000,
        "300mb": 300_000_000,
        "800mb": 800_000_000,
        "1500mb": 1_500_000_000,
    }
)


class _EngineRuntimes:
    """Per-engine runtime availability: one entry per engine the catalog ships.

    Both the Overview "Libraries & tools" tiles and the per-model warning read
    this, so a card can never claim a runtime the panel says is installed.
    Model weights are deliberately not part of the check: this answers "can this
    host run the engine at all", which is what a download decision turns on.
    """

    def __init__(self, system: SystemInfo, settings: Settings) -> None:
        self.system = system
        self.settings = settings
        self._tiles = {tile.engine: tile.status for tile in self._build()}

    def tiles(self) -> list[schemas.DependencyStatus]:
        return list(self._tiles.values())

    def missing(self, engine: str) -> schemas.DependencyStatus | None:
        """The unmet runtime behind an engine, or None when it can run here."""
        status = self._tiles.get(engine)
        return None if status is None or status.available else status

    def entry_fields(self, engine: str) -> dict[str, str | None]:
        """The AdminModelEntry warning fields for an engine, empty when it runs."""
        unmet = self.missing(engine)
        if unmet is None:
            return {}
        return {"runtime_requirement": unmet.name, "runtime_hint": unmet.install_hint}

    def _build(self) -> list[_EngineTile]:
        is_mac = self.system.os_name == "Darwin"
        silicon = self.system.is_apple_silicon
        return [
            *self._binary_tiles(is_mac),
            *self._package_tiles(silicon),
            *self._app_tiles(is_mac, silicon),
        ]

    def _binary_tiles(self, is_mac: bool) -> list[_EngineTile]:
        whisper_hint = (
            "brew install whisper-cpp"
            if is_mac
            else "Included in Docker or build whisper.cpp from source"
        )
        return [
            _EngineTile(
                "whisper.cpp",
                schemas.DependencyStatus(
                    name="whisper.cpp CLI",
                    available=self.system.whisper_cpp_path is not None,
                    path=self.system.whisper_cpp_path,
                    install_hint=whisper_hint,
                ),
            ),
            _EngineTile(
                "transcribe.cpp",
                schemas.DependencyStatus(
                    name="transcribe.cpp CLI",
                    available=self.system.transcribe_cli_path is not None,
                    path=self.system.transcribe_cli_path,
                    install_hint=TRANSCRIBE_INSTALL_HINT,
                ),
            ),
        ]

    def _package_tiles(self, silicon: bool) -> list[_EngineTile]:
        mlx_ready = silicon and importlib_util.find_spec("mlx_audio") is not None
        mlx_hint = (
            "Install vocagateway[apple]" if silicon else "Available only on Apple-silicon Macs"
        )
        tiles = [
            _EngineTile(engine, self._package_status(label, module))
            for engine, label, module in (
                ("faster-whisper", "faster-whisper", "faster_whisper"),
                ("moonshine", "Moonshine Voice", "moonshine_voice"),
                ("sherpa-onnx", "sherpa-onnx", "sherpa_onnx"),
            )
        ]
        tiles.append(
            _EngineTile(
                "mlx-audio",
                schemas.DependencyStatus(
                    name="MLX Audio",
                    available=mlx_ready,
                    path=PYTHON_PACKAGE_PATH if mlx_ready else None,
                    install_hint=mlx_hint,
                ),
            )
        )
        return tiles

    def _package_status(self, label: str, module: str) -> schemas.DependencyStatus:
        installed = importlib_util.find_spec(module) is not None
        return schemas.DependencyStatus(
            name=label,
            available=installed,
            path=PYTHON_PACKAGE_PATH if installed else None,
            install_hint=PYTHON_ENGINE_INSTALL_HINT,
        )

    def _app_tiles(self, is_mac: bool, silicon: bool) -> list[_EngineTile]:
        wk_hint = "brew install whisperkit-cli" if is_mac else "Available only on Apple platforms"
        vocamac_hint = (
            "https://github.com/jatinkrmalik/vocamac"
            if silicon
            else "Available only on Apple silicon Macs"
        )
        return [
            _EngineTile(
                "whisperkit",
                schemas.DependencyStatus(
                    name="WhisperKit CLI",
                    available=self.system.whisperkit_cli_path is not None,
                    path=self.system.whisperkit_cli_path,
                    install_hint=wk_hint,
                ),
            ),
            _EngineTile(
                "handy",
                schemas.DependencyStatus(
                    name="Handy app",
                    available=self.system.handy_installed,
                    path=str(self.settings.handy_binary) if self.system.handy_installed else None,
                    install_hint="https://handy.computer" if is_mac else "Available only on macOS",
                ),
            ),
            _EngineTile(
                "vocamac",
                schemas.DependencyStatus(
                    name="VocaMac app",
                    available=self.system.vocamac_installed,
                    path=str(self.settings.vocamac_app) if self.system.vocamac_installed else None,
                    install_hint=vocamac_hint,
                ),
            ),
        ]


class _SystemDependencyHelper:
    def __init__(self, ctx: GatewayContext) -> None:
        self.ctx = ctx
        self.settings = ctx.settings
        self.system = detect_system(
            whisper_binary=self.settings.whisper_binary,
            whisperkit_binary=self.settings.whisperkit_binary,
            handy_binary=self.settings.handy_binary,
            vocamac_app=self.settings.vocamac_app,
            transcribe_binary=self.settings.transcribe_binary,
        )
        self.runtimes = _EngineRuntimes(self.system, self.settings)

    def build_dependencies(self) -> list[schemas.DependencyStatus]:
        is_mac = self.system.os_name == "Darwin"
        ffmpeg_hint = (
            "brew install ffmpeg" if is_mac else "Install FFmpeg with your Linux package manager"
        )
        ffmpeg = schemas.DependencyStatus(
            name="FFmpeg",
            available=self.system.ffmpeg_path is not None,
            path=self.system.ffmpeg_path,
            install_hint=ffmpeg_hint,
        )
        return [ffmpeg, *self.runtimes.tiles()]

    def build_checklist(self, ready: bool) -> schemas.SetupChecklist:
        return schemas.SetupChecklist(
            token_configured=True,
            ffmpeg_available=self.system.ffmpeg_path is not None,
            engine_binary_available=any(tile.available for tile in self.runtimes.tiles()),
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


_ModelState = tuple[str, float | None, str | None]


class _ModelEntryHelper:
    def __init__(self, ctx: GatewayContext) -> None:
        self.ctx = ctx
        self.system = detect_system(
            whisper_binary=ctx.settings.whisper_binary,
            whisperkit_binary=ctx.settings.whisperkit_binary,
            handy_binary=ctx.settings.handy_binary,
            vocamac_app=ctx.settings.vocamac_app,
            transcribe_binary=ctx.settings.transcribe_binary,
        )
        self.runtimes = _EngineRuntimes(self.system, ctx.settings)
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
            **self.runtimes.entry_fields(model.engine),
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
            **self.runtimes.entry_fields(custom.engine),
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
        setup=helper.build_checklist(state.ready),
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
            transcribe_model=rc.transcribe_model,
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
