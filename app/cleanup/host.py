# Copyright (c) 2025-present VocaHQ, Inc.
# SPDX-License-Identifier: AGPL-3.0-only
# See the LICENSE file for details.

"""The gateway's ownership of a managed cleanup worker.

Split from the manager so that "which model is selected" and "is a process
running" stay separate questions. This half knows only about a `llama-server`
the gateway launched: it starts one on demand, keeps its start failure for the
status panel, and stops what it started. It never touches a server an operator
runs themselves, and it never touches the speech engine.

Loading is deliberately not something a request waits for. A multi-gigabyte
GGUF on a cold page cache takes minutes to reach the point where `/health`
turns green, and that is a model load rather than a hung worker. A request that
arrives cold starts the load in the background and takes the plain ASR result;
the next one finds the worker resident. Only an operator who asked for a
warm-up explicitly waits for the whole thing.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from app import config, runtime_config, system
from app.cleanup import catalog, worker
from app.cleanup.base import (
    MINIMUM_CONTEXT_TOKENS,
    CleanupRuntime,
    CleanupUnavailable,
    budget_for_context,
)
from app.cleanup.llama_server import LlamaServerRuntime
from app.cleanup.profile import CleanupLaunchProfile, describe_profile, resolve_profile


class WorkerHost:
    """One managed `llama-server` at a time, matched to the current selection."""

    def __init__(self, settings: config.Settings, run_config: runtime_config.RuntimeConfig) -> None:
        self.settings = settings
        self.runtime_config = run_config
        self.profile: CleanupLaunchProfile = resolve_profile(
            settings.cleanup_profile, _host_info(settings)
        )
        self.failure = ""
        self.offloaded = False
        self._worker: worker.LlamaServerWorker | None = None
        self._key: tuple[str, str] | None = None
        self._loading: asyncio.Task[None] | None = None

    def profile_detail(self) -> str:
        return describe_profile(self.profile, self.settings.cleanup_profile)

    @property
    def is_loading(self) -> bool:
        loading = self._loading
        return loading is not None and not loading.done()

    @property
    def is_running(self) -> bool:
        active = self._worker
        return active is not None and active.is_running

    def runtime_available(self) -> bool:
        """Whether a `llama-server` executable exists for the gateway to launch."""
        return worker.resolve_binary(self.settings.cleanup_binary) is not None

    async def runtime(self, model_id: str, model_file: Path) -> CleanupRuntime | None:
        """The runtime if it is resident, or None after starting a load for it.

        Nobody waits here. A transcription's budget is seconds and a load is
        minutes, and an operator pressing "Load model now" would only be
        holding an HTTP request open for a load that runs in the background
        regardless — so both callers get an answer immediately and the settings
        card polls until the state stops being `loading`.
        """
        binary = worker.resolve_binary(self.settings.cleanup_binary)
        if binary is None:
            return None
        active = self._ensure(binary, model_file, model_id)
        if not self._resident(active):
            return None
        # The budget follows the window this worker was actually launched with,
        # not the profile's nominal one: a model whose ceiling is lower than the
        # profile asked for gets a smaller window, and a runtime that believed
        # otherwise would pack pieces the server cannot hold.
        return LlamaServerRuntime(
            active.endpoint,
            model_id=model_id,
            budget=budget_for_context(active.context_tokens),
        )

    def stop(self, *, offloaded: bool = False) -> None:
        """Detach and terminate, without blocking the caller.

        Called from the event loop — by the idle sweep, by a settings save, and
        by a model switch — so the wait for the child happens off it.
        """
        active = self._detach(offloaded=offloaded)
        if active is not None:
            _reap(active)

    async def aclose(self) -> None:
        """Shutdown: detach, then actually wait for the child to be gone.

        The one caller that needs the process confirmed dead rather than merely
        asked to die, because the gateway is about to exit behind it.
        """
        active = self._detach(offloaded=False)
        if active is not None:
            await asyncio.to_thread(active.stop)

    def _detach(self, *, offloaded: bool) -> worker.LlamaServerWorker | None:
        loading = self._loading
        active = self._worker
        self._loading = None
        self._worker = None
        self._key = None
        self.offloaded = offloaded
        if loading is not None:
            # Cancelling propagates into `LlamaServerWorker._start`, which
            # terminates the child it adopted rather than leaving a model-sized
            # process behind an abandoned load. `Task.cancel` is only safe on
            # the loop thread, which is why nothing here is called from one of
            # its own worker threads.
            loading.cancel()
        return active

    def _resident(self, active: worker.LlamaServerWorker) -> bool:
        """Whether the worker is up, starting it in the background if not."""
        if active.is_running:
            self._succeeded()
            return True
        self._load_task(active)
        return False

    def _load_task(self, active: worker.LlamaServerWorker) -> asyncio.Task[None]:
        current = self._loading
        if current is not None and not current.done():
            return current
        task = asyncio.create_task(self._load(active))
        self._loading = task
        return task

    async def _load(self, active: worker.LlamaServerWorker) -> None:
        """Start the worker, recording a failure instead of raising one.

        Nothing awaits this task on the request path, so an exception here
        would be an unretrieved one. A failed load is a fallback for the
        request and a red badge for the operator, never a failed transcription.
        """
        try:
            await active.ensure_started()
        except CleanupUnavailable as error:
            self.failure = str(error)
            return
        self._succeeded()

    def _succeeded(self) -> None:
        self.failure = ""
        self.offloaded = False

    def _ensure(self, binary: Path, model_file: Path, model_id: str) -> worker.LlamaServerWorker:
        key = (model_id, str(model_file))
        current = self._worker
        if current is not None and self._key == key:
            return current
        self.stop()
        self._worker = worker.LlamaServerWorker(
            binary,
            model_file,
            context_tokens=_context_tokens(model_id, self.profile),
            cpu_threads=self.runtime_config.cpu_threads,
            profile=self.profile,
        )
        self._key = key
        return self._worker


def _reap(active: worker.LlamaServerWorker) -> None:
    """Terminate the worker without blocking whoever asked for it.

    Waiting for a child to die is a blocking call of up to several seconds, and
    every caller but shutdown is a coroutine on the event loop. The process
    reference is already detached by the time this runs, so nothing needs the
    result — only that the child actually goes away.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        active.stop()
        return
    # The result is deliberately dropped, so the failure it might carry is
    # consumed here rather than surfacing later as an unretrieved exception.
    loop.run_in_executor(None, active.stop).add_done_callback(lambda done: done.exception())


def _context_tokens(model_id: str, profile: CleanupLaunchProfile) -> int:
    """The window to launch with: the profile's, capped by the model's ceiling.

    A cap in both directions, and the direction matters. The compact profile
    exists to keep the KV cache from dwarfing a 0.6B model's weights on a
    4-8 GB host, so a catalog entry recording a 40k trained window must not be
    able to raise it; and the full profile must not ask for more window than
    the weights were trained for. An unknown model gets the floor.
    """
    selected = catalog.cleanup_model(model_id)
    ceiling = selected.maximum_context_tokens if selected else MINIMUM_CONTEXT_TOKENS
    return max(MINIMUM_CONTEXT_TOKENS, min(profile.context_tokens, ceiling))


def _host_info(settings: config.Settings) -> system.SystemInfo:
    return system.detect_system(
        whisper_binary=settings.whisper_binary,
        whisperkit_binary=settings.whisperkit_binary,
        handy_binary=settings.handy_binary,
        vocamac_app=settings.vocamac_app,
        cleanup_binary=settings.cleanup_binary,
    )
