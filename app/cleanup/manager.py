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
from collections.abc import AsyncIterator, Callable
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
    CleanupReason,
    CleanupRuntime,
)
from app.cleanup.host import WorkerHost
from app.cleanup.llama_server import LlamaServerRuntime
from app.cleanup.profile import PROFILE_COMPACT, PROFILE_FULL, budget_for_window
from app.cleanup.transport import Endpoint

SECONDS_PER_MINUTE = 60
# States the WebUI and the authenticated status endpoint report. Deliberately
# separate from the ASR warm-up vocabulary: the two are unrelated lifecycles.
STATE_DISABLED = "disabled"
STATE_UNAVAILABLE = "unavailable"
STATE_LOADING = "loading"
STATE_READY = "ready"
STATE_OFFLOADED = "offloaded"
STATE_ERROR = "error"
EXTERNAL_MODEL_NAME = "external"


@dataclass(frozen=True, slots=True)
class Lease:
    """One admitted inference slot: a runtime to use, or the reason there is none.

    The reason travels with the lease because only the manager can tell the
    three apart — a full slot, a runtime still loading, and a model or
    executable that is simply not there. Inferring it afterwards from the model
    path reported "busy" for a missing `llama-server`, which points an operator
    at the wrong fix.
    """

    runtime: CleanupRuntime | None = None
    reason: CleanupReason | None = None


@dataclass(frozen=True, slots=True)
class CleanupStatusReport:
    """What an operator is told about cleanup, with no prompts and no transcripts."""

    enabled: bool
    # Whether an `inherit` request would be corrected: enabled, with a runtime
    # and a model to run it on. Reported apart from `enabled`, which is only the
    # operator's switch.
    usable: bool
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
    auto_language: str
    loading: bool
    idle_unload_enabled: bool
    idle_unload_minutes: int
    locked_settings: tuple[str, ...]
    detail: str = ""
    profile: str = ""
    profile_detail: str = ""


class CleanupPreferences:
    """Environment overrides resolved against the operator's saved UI choices.

    Every setting is a two-layer decision, and the layers are not symmetric: an
    environment variable that is set wins and is reported as locked, while one
    that is unset must leave the saved choice alone rather than overwrite it
    with a default. That is why the override fields are tri-state.
    """

    def __init__(
        self,
        settings: config.Settings,
        run_config: runtime_config.RuntimeConfig,
        # What "no model chosen" falls back to. Supplied by the manager, which
        # is the half of this pair that knows what is installed on disk.
        installed_default: Callable[[], str | None],
    ) -> None:
        self.settings = settings
        self.runtime_config = run_config
        self._installed_default = installed_default

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
        """The selected model, or the installed one when nothing was selected.

        Downloading a cleanup model is already the deliberate act that turns
        corrections on, and until a second one is installed there is nothing to
        choose between — so requiring a separate "select" click only produced a
        gateway that had everything it needed and still corrected nothing. An
        explicit choice, from the WebUI or from the environment, always wins;
        the fallback disappears the moment there is more than one candidate.
        """
        chosen = self.settings.cleanup_model or self.runtime_config.cleanup_model
        return chosen or self._installed_default()

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
            ("auto_language", self.settings.cleanup_auto_language is not None),
            ("endpoint", self.settings.cleanup_endpoint is not None),
            (
                "profile",
                (self.settings.cleanup_profile or "") in {PROFILE_COMPACT, PROFILE_FULL},
            ),
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

    @property
    def auto_language(self) -> str:
        """The language a transcript left on `auto` is corrected as, if any.

        Empty by default, and empty is a real answer: the gateway will not
        decide that Latin script means English, because most of the world's
        languages are written in it. An operator who knows what they dictate in
        says so once here, which is information nothing else in the pipeline
        has — the speech engines do not report a detected language.

        A value that is not on the allowlist is ignored rather than honoured,
        so narrowing the allowlist cannot leave this pointing somewhere unsupported.
        """
        override = self.settings.cleanup_auto_language
        configured = override or self.runtime_config.cleanup_auto_language
        code = configured.strip().lower()
        return code if code in {name.lower() for name in self.supported_languages()} else ""

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
        self.models = models
        self.preferences = CleanupPreferences(
            settings, run_config, lambda: only_installed_model(self.models)
        )
        self.runtime_config = run_config
        self.config_path = config_path
        self.host = WorkerHost(settings, run_config)
        self._slot = asyncio.Semaphore(1)
        self._active_leases = 0
        self._last_used = time.monotonic()
        self._failure = ""
        # Whether an operator-run endpoint's context window has been measured
        # and found big enough. A managed worker needs no such check: the
        # gateway passes its own `--ctx-size`.
        self._external_ok: bool | None = None
        self._external_window = 0

    @property
    def enabled(self) -> bool:
        return self.preferences.enabled

    @property
    def model_id(self) -> str | None:
        return self.preferences.model_id

    @property
    def usable(self) -> bool:
        """Whether this gateway can correct anything at all right now.

        The question a client's `inherit` turns on, and it is not the same as
        "is cleanup enabled": a gateway with the feature on and no model
        installed has nothing to inherit into. Answering it here rather than in
        the preferences keeps that a fact about the deployment — the runtime and
        the installed artifacts — instead of a setting.

        An operator-run endpoint is taken at its word. The gateway did not start
        that process and has no local binary or model file to look at, so
        refusing to use it because it cannot find one locally would be wrong.
        """
        if not self.preferences.managed:
            return self.enabled
        return self.enabled and self.host.runtime_available() and self.model_path() is not None

    def options(self, requested: str | None) -> CleanupOptions:
        """The resolved decision for one unit of work.

        `inherit` on a gateway that cannot correct anything resolves to `off`
        rather than to the configured default: the transcript is identical
        either way, and reporting it as a cleanup that was attempted and fell
        back would add a block to every packet on every deployment that has
        never installed a cleanup model. An explicit `conservative` is left
        alone — a client that asked for correction is owed the reason it did not
        happen.
        """
        if requested in (None, MODE_INHERIT) and not self.usable:
            return self.preferences.options(MODE_OFF)
        return self.preferences.options(requested)

    def supported_languages(self) -> tuple[str, ...]:
        return self.preferences.supported_languages()

    def auto_language(self) -> str:
        return self.preferences.auto_language

    def model_path(self) -> Path | None:
        selected = catalog.cleanup_model(self.model_id)
        if selected is None:
            return None
        return self.models.installed_path(selected.id)

    @asynccontextmanager
    async def lease(self) -> AsyncIterator[Lease]:
        """Admit one inference, or say why not rather than queueing behind others.

        Concurrency is one per runtime and the wait is bounded, so a burst of
        dictations degrades to the plain ASR result instead of building a queue
        whose tail would miss its deadline anyway.
        """
        if not await self._admit():
            yield Lease(reason=CleanupReason.BUSY)
            return
        self._active_leases += 1
        try:
            yield await self._lease()
        finally:
            self._release()

    async def warmup(self) -> bool:
        """Ask for the model to be loaded, and report whether it is ready yet.

        Deliberately does not wait. Loading a multi-gigabyte GGUF is minutes,
        and holding an HTTP request open for that long is a request that times
        out somewhere in the middle with nothing to show for it. The load runs
        in the background either way; the settings card polls until the state
        stops being `loading`.
        """
        if not self.enabled:
            return False
        runtime = await self._runtime()
        return runtime is not None and await runtime.available()

    @property
    def loading(self) -> bool:
        """Whether a managed worker is being loaded right now."""
        return self.preferences.managed and self.host.is_loading

    def offload_if_idle(self, *, now: float | None = None) -> bool:
        """Release a managed worker after the configured idle period.

        Never while a lease is out, and never for a server the gateway does not
        own: stopping somebody else's process is not this object's business.
        """
        offloadable = (
            self.runtime_config.cleanup_idle_unload_enabled
            and self.preferences.managed
            and not self._active_leases
            and self.host.is_running
        )
        if not offloadable:
            return False
        idle_minutes = self.runtime_config.cleanup_idle_unload_minutes
        if (now or time.monotonic()) - self._last_used < idle_minutes * SECONDS_PER_MINUTE:
            return False
        self.host.stop(offloaded=True)
        return True

    async def shutdown(self) -> None:
        await self.host.aclose()

    def status(self) -> CleanupStatusReport:
        preferences = self.preferences
        selected = catalog.cleanup_model(self.model_id)
        installed = self.model_path() is not None
        available = self.host.runtime_available()
        return CleanupStatusReport(
            enabled=preferences.enabled,
            usable=self.usable,
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
            auto_language=preferences.auto_language,
            loading=self.loading,
            idle_unload_enabled=self.runtime_config.cleanup_idle_unload_enabled,
            idle_unload_minutes=self.runtime_config.cleanup_idle_unload_minutes,
            locked_settings=preferences.locked_settings(),
            detail=self._failure or self.host.failure,
            profile=self.host.profile.id,
            profile_detail=self.host.profile_detail(),
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
        self._external_ok = None
        self._external_window = 0
        if (changed_model or not self.enabled) and self._active_leases == 0:
            # Applies to new work only. An in-flight request keeps the runtime
            # it was admitted against; the swap happens once its lease drains.
            self.host.stop()

    async def _external_runtime(self, endpoint: Endpoint) -> CleanupRuntime | None:
        """An operator-run server, once its context window has been vouched for.

        The gateway does not launch this one, so it never chose its `--ctx-size`.
        A window too small for the prompt does not fail loudly — it silently
        drops the front of the context, which is the system instruction — so it
        is measured once and cached rather than trusted or re-asked per request.
        A server that does not report one is used as before: an unknown window
        is not evidence of a bad one.
        """
        if self._external_ok is None:
            probe = LlamaServerRuntime(endpoint, model_id=self.model_id or EXTERNAL_MODEL_NAME)
            self._external_ok = await probe.context_is_sufficient()
            self._external_window = await probe.context_tokens() if self._external_ok else 0
        if not self._external_ok:
            return None
        return LlamaServerRuntime(
            endpoint,
            model_id=self.model_id or EXTERNAL_MODEL_NAME,
            budget=budget_for_window(self._external_window),
        )

    async def _lease(self) -> Lease:
        runtime = await self._runtime()
        if runtime is not None:
            return Lease(runtime=runtime)
        return Lease(reason=self._missing_reason())

    def _missing_reason(self) -> CleanupReason:
        """Why the slot was free but there is still nothing to run on.

        A load in progress is worth telling apart from an absent one: the first
        is fixed by waiting a moment, the second by installing something.
        """
        if self._external_ok is False:
            return CleanupReason.CONTEXT_TOO_SMALL
        return CleanupReason.MODEL_LOADING if self._loadable() else CleanupReason.MODEL_UNAVAILABLE

    def _loadable(self) -> bool:
        """Whether a managed worker could still turn up, given a moment."""
        if not self.preferences.managed or not self.enabled:
            return False
        if self.model_path() is None or self.host.failure:
            return False
        return self.host.runtime_available()

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
            return await self._external_runtime(external)
        selected_id = self.model_id
        model_file = self.model_path()
        if selected_id is None or model_file is None:
            return None
        return await self.host.runtime(selected_id, model_file)

    def _state(self, installed: bool, runtime_available: bool) -> str:
        """The one word the WebUI pill shows, and it has to match the checklist.

        The local runtime and model file are preconditions only for a worker
        this gateway would launch. An operator-run endpoint has neither on this
        host by definition, so judging it by them reported "Not available" for a
        deployment that corrects transcripts perfectly well — while the
        checklist beside it said every step was done.
        """
        if not self.enabled:
            return STATE_DISABLED
        if self.host.failure:
            return STATE_ERROR
        if self.preferences.managed and not (runtime_available and installed):
            return STATE_UNAVAILABLE
        if self.loading:
            return STATE_LOADING
        return STATE_OFFLOADED if self.host.offloaded else STATE_READY


def only_installed_model(models: model_manager.ModelManager) -> str | None:
    """The one installed cleanup artifact, when exactly one is installed.

    Read on demand rather than cached: a model finishes downloading in the
    background, and an operator watching the Cleanup tab has to see the
    checklist tick over without restarting the gateway.
    """
    installed = [
        model.id for model in catalog.CLEANUP_CATALOG if models.installed_path(model.id) is not None
    ]
    return installed[0] if len(installed) == 1 else None


def preserve_implicit_model_selection(manager: CleanupManager) -> None:
    """Persist a sole-model fallback before another download can remove it.

    With one installed model an empty saved selection deliberately resolves to
    that model. Once a second download completes there is no sole model, so
    persisting the effective choice here keeps an installation from silently
    turning cleanup off. An environment-selected model remains an environment
    decision and is never copied into the saved configuration.
    """
    if manager.preferences.settings.cleanup_model or manager.runtime_config.cleanup_model:
        return
    selected = manager.preferences.model_id
    if selected is None:
        return
    manager.runtime_config.cleanup_model = selected
    manager.runtime_config.save(manager.config_path)


@dataclass(frozen=True, slots=True)
class CleanupUpdate:
    """A partial change to the cleanup block. `None` means "leave this alone"."""

    enabled: bool | None = None
    mode: str | None = None
    model_id: str | None = None
    timeout_seconds: float | None = None
    idle_unload_enabled: bool | None = None
    idle_unload_minutes: int | None = None
    auto_language: str | None = None

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
        if self.auto_language is not None:
            run_config.cleanup_auto_language = self.auto_language.strip().lower()


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
