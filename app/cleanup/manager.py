"""Selection, lifecycle, and admission for the optional cleanup runtime.

This is the only object that owns a cleanup process. It decides which model is
selected, resolves an operator's environment overrides against their saved UI
choices, hands out bounded leases, releases the model when it has been idle, and
stops what it started at shutdown. Nothing here touches the speech engine: a
cleanup idle unload cannot change an ASR selection, and a busy cleanup runtime
cannot make `/health/ready` mean something different.

Two deployment shapes are supported, and they are not interchangeable. A
*managed* worker is a `llama-server` this gateway launched, so it can be warmed,
offloaded, and restarted. An *external* endpoint is a server an operator runs
themselves, typically for evaluation; the gateway will use it but makes no
promise about its lifecycle, because it does not own the process.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from app import config, model_manager, runtime_config
from app.cleanup import catalog
from app.cleanup.base import (
    ADMISSION_WAIT_SECONDS,
    MODE_INHERIT,
    MODE_OFF,
    PROMPT_VERSION,
    RESOLVED_MODES,
    CleanupOptions,
    CleanupRuntime,
)
from app.cleanup.host import WorkerHost
from app.cleanup.llama_server import LlamaServerRuntime
from app.cleanup.transport import Endpoint

SECONDS_PER_MINUTE = 60
# States the WebUI and the authenticated status endpoint report. Deliberately
# separate from the ASR warm-up vocabulary: the two are unrelated lifecycles.
STATE_DISABLED = "disabled"
STATE_UNAVAILABLE = "unavailable"
STATE_READY = "ready"
STATE_OFFLOADED = "offloaded"
STATE_ERROR = "error"
EXTERNAL_MODEL_NAME = "external"


@dataclass(frozen=True, slots=True)
class CleanupStatusReport:
    """What an operator is told about cleanup, with no prompts and no transcripts."""

    enabled: bool
    state: str
    mode: str
    model_id: str | None
    model_label: str | None
    model_installed: bool
    runtime_available: bool
    managed: bool
    timeout_seconds: float
    languages: tuple[str, ...]
    evaluated_languages: tuple[str, ...]
    idle_unload_enabled: bool
    idle_unload_minutes: int
    locked_settings: tuple[str, ...]
    detail: str = ""


class CleanupPreferences:
    """Environment overrides resolved against the operator's saved UI choices.

    Every setting is a two-layer decision, and the layers are not symmetric: an
    environment variable that is set wins and is reported as locked, while one
    that is unset must leave the saved choice alone rather than overwrite it
    with a default. That is why the override fields are tri-state.
    """

    def __init__(self, settings: config.Settings, run_config: runtime_config.RuntimeConfig) -> None:
        self.settings = settings
        self.runtime_config = run_config

    @property
    def enabled(self) -> bool:
        override = self.settings.cleanup_enabled
        return self.runtime_config.cleanup_enabled if override is None else override

    @property
    def default_mode(self) -> str:
        configured = self.settings.cleanup_mode or self.runtime_config.cleanup_mode
        if not self.enabled or configured not in RESOLVED_MODES:
            return MODE_OFF
        return configured

    @property
    def model_id(self) -> str | None:
        chosen = self.settings.cleanup_model or self.runtime_config.cleanup_model
        return chosen or None

    @property
    def timeout_seconds(self) -> float:
        override = self.settings.cleanup_timeout_seconds
        return self.runtime_config.cleanup_timeout_seconds if override is None else override

    @property
    def managed(self) -> bool:
        """True when the gateway owns the process, rather than merely using one."""
        return self.settings.cleanup_endpoint is None

    def locked_settings(self) -> tuple[str, ...]:
        """Settings an environment variable has taken away from the UI."""
        overrides = (
            ("enabled", self.settings.cleanup_enabled is not None),
            ("mode", bool(self.settings.cleanup_mode)),
            ("model", bool(self.settings.cleanup_model)),
            ("timeout_seconds", self.settings.cleanup_timeout_seconds is not None),
            ("endpoint", self.settings.cleanup_endpoint is not None),
        )
        return tuple(name for name, locked in overrides if locked)

    def resolve_mode(self, requested: str | None) -> str:
        """Collapse a request preference and the gateway default into one decision.

        An explicit opt-in never installs a model and never overrides an
        operator who turned cleanup off; it only declines to inherit a default
        that was already `conservative`.
        """
        if not self.enabled:
            return MODE_OFF
        if requested in (None, MODE_INHERIT):
            return self.default_mode
        return requested if requested in RESOLVED_MODES else MODE_OFF

    def options(self, requested: str | None) -> CleanupOptions:
        """The immutable snapshot one unit of work is pinned to."""
        return CleanupOptions(
            mode=self.resolve_mode(requested),
            model_id=self.model_id,
            prompt_version=PROMPT_VERSION,
            timeout_seconds=self.timeout_seconds,
        )

    def supported_languages(self) -> tuple[str, ...]:
        override = self.settings.cleanup_languages
        if override:
            return override
        selected = catalog.cleanup_model(self.model_id)
        return selected.candidate_languages if selected else catalog.CANDIDATE_LANGUAGES

    def evaluated_languages(self) -> tuple[str, ...]:
        """Languages this artifact has actually passed the release gates for.

        Reported apart from `supported_languages`, which is what cleanup will
        *run* for. An offered language is not a tested one, and the capability
        endpoint must never let a client confuse the two.
        """
        selected = catalog.cleanup_model(self.model_id)
        return selected.evaluated_languages if selected else ()

    def external_endpoint(self) -> Endpoint | None:
        configured = self.settings.cleanup_endpoint
        if configured is None:
            return None
        return Endpoint(configured[0], configured[1], self.settings.cleanup_api_key)


class CleanupManager:
    """Owns the cleanup runtime and every decision about whether to use it."""

    def __init__(
        self,
        settings: config.Settings,
        run_config: runtime_config.RuntimeConfig,
        config_path: Path,
        models: model_manager.ModelManager,
    ) -> None:
        self.preferences = CleanupPreferences(settings, run_config)
        self.runtime_config = run_config
        self.config_path = config_path
        self.models = models
        self.host = WorkerHost(settings, run_config)
        self._slot = asyncio.Semaphore(1)
        self._active_leases = 0
        self._last_used = time.monotonic()
        self._failure = ""

    @property
    def enabled(self) -> bool:
        return self.preferences.enabled

    @property
    def model_id(self) -> str | None:
        return self.preferences.model_id

    def options(self, requested: str | None) -> CleanupOptions:
        return self.preferences.options(requested)

    def supported_languages(self) -> tuple[str, ...]:
        return self.preferences.supported_languages()

    def model_path(self) -> Path | None:
        selected = catalog.cleanup_model(self.model_id)
        if selected is None:
            return None
        return self.models.installed_path(selected.id)

    @asynccontextmanager
    async def lease(self) -> AsyncIterator[CleanupRuntime | None]:
        """Admit one inference, or yield None rather than queueing behind others.

        Concurrency is one per runtime and the wait is bounded, so a burst of
        dictations degrades to the plain ASR result instead of building a queue
        whose tail would miss its deadline anyway.
        """
        if not await self._admit():
            yield None
            return
        self._active_leases += 1
        try:
            yield await self._runtime()
        finally:
            self._release()

    async def warmup(self) -> bool:
        """Load the model outside a request, so the first dictation is not cold."""
        if not self.enabled:
            return False
        runtime = await self._runtime()
        return runtime is not None and await runtime.available()

    def offload_if_idle(self, *, now: float | None = None) -> bool:
        """Release a managed worker after the configured idle period.

        Never while a lease is out, and never for a server the gateway does not
        own: stopping somebody else's process is not this object's business.
        """
        if not self._can_offload():
            return False
        idle_minutes = self.runtime_config.cleanup_idle_unload_minutes
        if (now or time.monotonic()) - self._last_used < idle_minutes * SECONDS_PER_MINUTE:
            return False
        self.host.stop(offloaded=True)
        return True

    async def shutdown(self) -> None:
        await asyncio.to_thread(self.host.stop)

    def status(self) -> CleanupStatusReport:
        preferences = self.preferences
        selected = catalog.cleanup_model(self.model_id)
        installed = self.model_path() is not None
        available = self.host.runtime_available()
        return CleanupStatusReport(
            enabled=preferences.enabled,
            state=self._state(installed, available),
            mode=preferences.default_mode,
            model_id=preferences.model_id,
            model_label=selected.label if selected else None,
            model_installed=installed,
            runtime_available=available,
            managed=preferences.managed,
            timeout_seconds=preferences.timeout_seconds,
            languages=preferences.supported_languages(),
            evaluated_languages=preferences.evaluated_languages(),
            idle_unload_enabled=self.runtime_config.cleanup_idle_unload_enabled,
            idle_unload_minutes=self.runtime_config.cleanup_idle_unload_minutes,
            locked_settings=preferences.locked_settings(),
            detail=self._failure or self.host.failure,
        )

    def configure(self, update: CleanupUpdate) -> None:
        """Apply a partial cleanup update and persist it atomically.

        Partial on purpose: changing whether transcripts are corrected must not
        reset the speech engine, the hardware options, or the ASR idle policy,
        which a whole-object save would do.
        """
        changed_model = (
            update.model_id is not None and update.model_id != self.runtime_config.cleanup_model
        )
        update.apply(self.runtime_config)
        self.runtime_config.save(self.config_path)
        if (changed_model or not self.enabled) and self._active_leases == 0:
            # Applies to new work only. An in-flight request keeps the runtime
            # it was admitted against; the swap happens once its lease drains.
            self.host.stop()

    async def _admit(self) -> bool:
        try:
            await asyncio.wait_for(self._slot.acquire(), timeout=ADMISSION_WAIT_SECONDS)
        except TimeoutError:
            return False
        return True

    def _release(self) -> None:
        self._active_leases -= 1
        self._last_used = time.monotonic()
        self._slot.release()

    async def _runtime(self) -> CleanupRuntime | None:
        if not self.enabled:
            return None
        external = self.preferences.external_endpoint()
        if external is not None:
            return LlamaServerRuntime(external, model_id=self.model_id or EXTERNAL_MODEL_NAME)
        selected_id = self.model_id
        model_file = self.model_path()
        if selected_id is None or model_file is None:
            return None
        return await self.host.runtime(selected_id, model_file)

    def _can_offload(self) -> bool:
        if not self.runtime_config.cleanup_idle_unload_enabled:
            return False
        if not self.preferences.managed or self._active_leases:
            return False
        return self.host.is_running

    def _state(self, installed: bool, runtime_available: bool) -> str:
        if not self.enabled:
            return STATE_DISABLED
        if self.host.failure:
            return STATE_ERROR
        if not runtime_available or not installed:
            return STATE_UNAVAILABLE
        return STATE_OFFLOADED if self.host.offloaded else STATE_READY


@dataclass(frozen=True, slots=True)
class CleanupUpdate:
    """A partial change to the cleanup block. `None` means "leave this alone"."""

    enabled: bool | None = None
    mode: str | None = None
    model_id: str | None = None
    timeout_seconds: float | None = None
    idle_unload_enabled: bool | None = None
    idle_unload_minutes: int | None = None

    def apply(self, run_config: runtime_config.RuntimeConfig) -> None:
        if self.enabled is not None:
            run_config.cleanup_enabled = self.enabled
        if self.mode is not None:
            run_config.cleanup_mode = self.mode
        if self.model_id is not None:
            run_config.cleanup_model = self.model_id or None
        if self.timeout_seconds is not None:
            run_config.cleanup_timeout_seconds = runtime_config.clamp_cleanup_timeout(
                self.timeout_seconds
            )
        if self.idle_unload_enabled is not None:
            run_config.cleanup_idle_unload_enabled = self.idle_unload_enabled
        if self.idle_unload_minutes is not None:
            run_config.cleanup_idle_unload_minutes = self.idle_unload_minutes


def build_manager(
    settings: config.Settings,
    run_config: runtime_config.RuntimeConfig,
    config_path: Path,
) -> CleanupManager:
    """Construct the manager with its own models directory and catalog.

    A separate `ModelManager` instance, not the speech one: cleanup artifacts
    live under their own directory with their own catalog, so an LLM can never
    turn up as a selectable speech engine or be sized into the ASR model list.
    """
    models = model_manager.ModelManager(
        settings.resolved_models_dir() / "cleanup",
        catalog=catalog.download_catalog(),
        retired_catalog=(),
    )
    return CleanupManager(settings, run_config, config_path, models)
