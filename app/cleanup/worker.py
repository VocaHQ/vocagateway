"""A gateway-owned `llama-server` child process.

The gateway launches this itself so that enabling cleanup does not require an
operator to run and supervise a second daemon. Ownership is explicit and
narrow: only a process this object spawned is ever stopped or restarted, the
executable and model are validated before the spawn, the command is passed as
argv (never through a shell), and the listener is bound to loopback on an
ephemeral port that is never published.

An operator-run server used for evaluation is deliberately *not* represented
here. Nothing outside this class may promise gateway-controlled idle unloading,
because nothing outside this class owns the process.
"""

from __future__ import annotations

import asyncio
import os
import secrets
import shutil
import socket
import subprocess
import tempfile
import time
from contextlib import suppress
from pathlib import Path
from typing import IO

from app import system
from app.cleanup import transport
from app.cleanup.base import CleanupUnavailable
from app.cleanup.profile import COMPACT_PROFILE, CleanupLaunchProfile

LOOPBACK_HOST = "127.0.0.1"
SERVER_BINARY_NAME = "llama-server"
# A multi-gigabyte GGUF on a cold page cache takes a while to reach the point
# where /health turns green, and that wait is a model load rather than a hung
# worker. Cleanup requests do not wait on it: a cold start falls back to the
# ASR result while a bounded warm-up runs on its own.
START_TIMEOUT_SECONDS = 300.0
FIRST_POLL_SECONDS = 0.02
MAXIMUM_POLL_SECONDS = 0.25
STOP_TIMEOUT_SECONDS = 5.0
API_KEY_BYTES = 24
DIAGNOSTIC_TAIL_BYTES = 2048
MAXIMUM_DIAGNOSTIC_LENGTH = 400
# One slot. Cleanup admits a single inference per runtime, so a second slot
# would only split the KV cache without ever being used.
PARALLEL_SLOTS = 1
HEALTH_PROBE_SECONDS = 1.0


def resolve_binary(override: Path | None = None) -> Path | None:
    """Locate the `llama-server` to launch, preferring an explicit setting."""
    if override is not None:
        return override if _is_executable(override) else None
    found = shutil.which(SERVER_BINARY_NAME)
    return Path(found) if found else None


def _is_executable(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)


class LlamaServerWorker:
    """One resident `llama-server` holding exactly one cleanup model."""

    def __init__(
        self,
        binary: Path,
        model: Path,
        *,
        context_tokens: int | None = None,
        cpu_threads: int = 0,
        profile: CleanupLaunchProfile | None = None,
    ) -> None:
        self.binary = binary
        self.model = model
        self.profile = profile or COMPACT_PROFILE
        if context_tokens is None:
            self.context_tokens = self.profile.context_tokens
        else:
            self.context_tokens = context_tokens
        self.cpu_threads = cpu_threads
        # A credential of the worker's own, so the gateway authenticates to it
        # without ever forwarding a client's bearer token.
        self.api_key = secrets.token_urlsafe(API_KEY_BYTES)
        self._process: subprocess.Popen[bytes] | None = None
        self._port = 0
        self._stderr: IO[bytes] | None = None
        self._start_lock = asyncio.Lock()

    @property
    def is_running(self) -> bool:
        process = self._process
        return process is not None and process.poll() is None

    @property
    def endpoint(self) -> transport.Endpoint:
        return transport.Endpoint(LOOPBACK_HOST, self._port, self.api_key)

    def diagnostics(self) -> str:
        """The tail of the worker's stderr: its backend, its load, its errors.

        Never its requests. At default verbosity `llama-server` writes no
        prompt or response bodies, and the gateway passes nothing that would
        turn that on.
        """
        stderr = self._stderr
        if stderr is None:
            return ""
        try:
            size = stderr.seek(0, os.SEEK_END)
            stderr.seek(max(0, size - DIAGNOSTIC_TAIL_BYTES))
            captured = stderr.read()
        except OSError:
            return ""
        return captured.decode("utf-8", errors="replace").strip()[-MAXIMUM_DIAGNOSTIC_LENGTH:]

    async def ensure_started(self, *, budget: float = START_TIMEOUT_SECONDS) -> bool:
        """Start the worker unless it is already up. True when this call started it."""
        if self.is_running:
            return False
        async with self._start_lock:
            if self.is_running:
                return False
            self.stop()
            await self._start(budget)
            return True

    def stop(self, *, force: bool = False) -> None:
        process = self._process
        self._process = None
        self._port = 0
        if process is not None:
            _terminate(process, force=force)
        self._close_diagnostics()

    async def _start(self, budget: float) -> None:
        self._validate()
        port = _loopback_port()
        try:
            process = self._spawn(port)
        except OSError as error:
            raise CleanupUnavailable("The cleanup runtime could not be launched.") from error
        # Adopted before anything is awaited: recording it only once it answers
        # would leak a model-sized process whenever the start is cancelled.
        self._process = process
        self._port = port
        try:
            ready = await self._await_ready(process, budget)
        except BaseException:
            self.stop()
            raise
        if not ready:
            detail = self.diagnostics()
            self.stop()
            raise CleanupUnavailable(f"The cleanup runtime did not become ready. {detail}".strip())

    def _validate(self) -> None:
        if not _is_executable(self.binary):
            raise CleanupUnavailable("The configured cleanup runtime is not an executable file.")
        if not self.model.is_file():
            raise CleanupUnavailable("The selected cleanup model file is missing.")

    def _spawn(self, port: int) -> subprocess.Popen[bytes]:
        return subprocess.Popen(  # noqa: S603 - argv only, both entries validated above
            self._arguments(port),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=self._open_diagnostics(),
            # A clean environment: no inherited proxy variables to redirect the
            # loopback listener's own outbound calls, and no ambient tokens.
            env=_worker_environment(),
        )

    def _arguments(self, port: int) -> list[str]:
        return [
            str(self.binary),
            "--model",
            str(self.model),
            "--host",
            LOOPBACK_HOST,
            "--port",
            str(port),
            "--ctx-size",
            str(self.context_tokens),
            "--batch-size",
            str(self.profile.batch_tokens),
            "--ubatch-size",
            str(self.profile.ubatch_tokens),
            "--cache-type-k",
            self.profile.kv_cache_type,
            "--cache-type-v",
            self.profile.kv_cache_type,
            "--threads",
            str(system.inference_thread_count(self.cpu_threads)),
            "--parallel",
            str(PARALLEL_SLOTS),
            # The credential the gateway presents. Nothing else on the machine
            # is told it, so a local process that finds the port still cannot
            # spend the model's compute.
            "--api-key",
            self.api_key,
            # Required for `chat_template_kwargs`, which is how non-thinking
            # mode is requested at the template level rather than by a textual
            # hint in the prompt.
            "--jinja",
            # No browser surface on the port, and no slot state written to disk:
            # a cleanup prompt must not outlive the request in a cache file.
            "--no-webui",
        ]

    async def _await_ready(self, process: subprocess.Popen[bytes], budget: float) -> bool:
        """Poll `/health` until the model is loaded. False once the process exits."""
        limit = time.monotonic() + budget
        delay = FIRST_POLL_SECONDS
        while time.monotonic() < limit:
            if process.poll() is not None:
                return False
            if await _health_ok(self.endpoint):
                return True
            await asyncio.sleep(delay)
            delay = min(delay * 2, MAXIMUM_POLL_SECONDS)
        return False

    def _open_diagnostics(self) -> IO[bytes]:
        # Deliberately not a context manager: the capture lives as long as the
        # worker does and is read after it fails. `_close_diagnostics` and the
        # unlinked temporary file are what release it.
        self._close_diagnostics()
        self._stderr = tempfile.TemporaryFile()  # noqa: SIM115
        return self._stderr

    def _close_diagnostics(self) -> None:
        stderr = self._stderr
        self._stderr = None
        if stderr is not None:
            with suppress(OSError):
                stderr.close()


def _worker_environment() -> dict[str, str]:
    """The child's environment, with proxy and token variables stripped.

    The worker only ever talks to the gateway over loopback, so a `HTTP_PROXY`
    inherited from the operator's shell can do nothing useful and quite a lot of
    harm. `HOME` and `PATH` are kept because the runtime uses them to find its
    own accelerator libraries.
    """
    keep = (
        "PATH",
        "HOME",
        "TMPDIR",
        "LD_LIBRARY_PATH",
        "DYLD_LIBRARY_PATH",
        "CUDA_VISIBLE_DEVICES",
    )
    return {name: os.environ[name] for name in keep if name in os.environ}


async def _health_ok(endpoint: transport.Endpoint) -> bool:
    try:
        reply = await transport.get_json(endpoint, "/health", budget=HEALTH_PROBE_SECONDS)
    except (transport.TransportError, TimeoutError):
        return False
    return reply.status == transport.HTTP_OK


def _terminate(process: subprocess.Popen[bytes], *, force: bool) -> None:
    if process.poll() is not None:
        return
    if force:
        process.kill()
        process.wait()
        return
    process.terminate()
    try:
        process.wait(timeout=STOP_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind((LOOPBACK_HOST, 0))
        return int(listener.getsockname()[1])
