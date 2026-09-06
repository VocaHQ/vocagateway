"""The gateway's ownership of a managed cleanup worker.

Split from the manager so that "which model is selected" and "is a process
running" stay separate questions. This half knows only about a `llama-server`
the gateway launched: it starts one on demand, keeps its start failure for the
status panel, and stops what it started. It never touches a server an operator
runs themselves, and it never touches the speech engine.
"""

from __future__ import annotations

from pathlib import Path

from app import config, runtime_config
from app.cleanup import catalog, worker
from app.cleanup.base import MINIMUM_CONTEXT_TOKENS, CleanupRuntime, CleanupUnavailable
from app.cleanup.llama_server import LlamaServerRuntime


class WorkerHost:
    """One managed `llama-server` at a time, matched to the current selection."""

    def __init__(self, settings: config.Settings, run_config: runtime_config.RuntimeConfig) -> None:
        self.settings = settings
        self.runtime_config = run_config
        self.failure = ""
        self.offloaded = False
        self._worker: worker.LlamaServerWorker | None = None
        self._key: tuple[str, str] | None = None

    @property
    def is_running(self) -> bool:
        active = self._worker
        return active is not None and active.is_running

    def runtime_available(self) -> bool:
        """Whether a `llama-server` executable exists for the gateway to launch."""
        return worker.resolve_binary(self.settings.cleanup_binary) is not None

    async def runtime(self, model_id: str, model_file: Path) -> CleanupRuntime | None:
        binary = worker.resolve_binary(self.settings.cleanup_binary)
        if binary is None:
            return None
        active = self._ensure(binary, model_file, model_id)
        try:
            await active.ensure_started()
        except CleanupUnavailable as error:
            # Remembered rather than raised: an unavailable corrector is a
            # fallback for this request and a red badge for the operator, never
            # a failed transcription.
            self.failure = str(error)
            self.stop()
            return None
        self.failure = ""
        self.offloaded = False
        return LlamaServerRuntime(active.endpoint, model_id=model_id)

    def stop(self, *, offloaded: bool = False) -> None:
        active = self._worker
        self._worker = None
        self._key = None
        self.offloaded = offloaded
        if active is not None:
            active.stop()

    def _ensure(self, binary: Path, model_file: Path, model_id: str) -> worker.LlamaServerWorker:
        key = (model_id, str(model_file))
        current = self._worker
        if current is not None and self._key == key:
            return current
        self.stop()
        self._worker = worker.LlamaServerWorker(
            binary,
            model_file,
            context_tokens=_context_tokens(model_id),
            cpu_threads=self.runtime_config.cpu_threads,
        )
        self._key = key
        return self._worker


def _context_tokens(model_id: str) -> int:
    selected = catalog.cleanup_model(model_id)
    declared = selected.context_tokens if selected else MINIMUM_CONTEXT_TOKENS
    return max(MINIMUM_CONTEXT_TOKENS, declared)
